"""Tests for the shared side channel: record declaration, family slots, accumulation, finalization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from inspect import Parameter, signature
from typing import Self

import pytest
import torch
from torch import Tensor

from neurox.common import RecordBase, RecorderBase

_CPU = torch.device("cpu")
_ELSEWHERE = torch.device("meta")  # a second device every host has


class _Record(RecordBase):
    value: Tensor


class _Pair(RecordBase):
    """A record holding a second one, so the walk has a level to recurse into."""

    value: Tensor
    nested: _Record


class _SpyRecord(_Record):
    moves: list[torch.device]
    """Every device `to` was called with, in call order."""

    def to(self, device: torch.device) -> Self:
        self.moves.append(device)
        return super().to(device)


class _FamilyA(RecorderBase[_Record]):
    """One family root, owning the slot every class below it shares."""


class _FamilyB(RecorderBase[_Record]):
    """A second, independent family root."""


class _DerivedA(_FamilyA):
    """A member below A's root, holding no slot of its own."""


class _Tagged(_Record):
    """A record with a plain coordinate a gate reads without touching tensors."""

    tag: int


class _TaggedOnly(RecorderBase[_Tagged]):
    """A family whose admission rule lives in its submission hook."""

    @classmethod
    def _submit_impl(cls, record: _Tagged) -> None:
        if record.tag < 0:
            return
        cls._submit_record(record)


# === Record declaration ===


def test_a_record_declaring_its_own_init_is_refused_at_definition() -> None:
    with pytest.raises(TypeError, match=r"must declare dataclass fields, not __init__\(\)"):

        class _InvalidRecord(RecordBase):
            def __init__(self) -> None:
                pass


def test_a_record_declaring_a_post_init_is_refused_at_definition() -> None:
    with pytest.raises(TypeError, match=r"declares fields, not __post_init__\(\)"):

        class _InvalidRecord(RecordBase):
            def __post_init__(self) -> None:
                pass


def test_a_field_only_record_is_a_keyword_only_dataclass() -> None:
    parameters = signature(_Record).parameters.values()
    assert [parameter.name for parameter in parameters] == ["value"]
    assert all(parameter.kind is Parameter.KEYWORD_ONLY for parameter in parameters)


def test_a_record_refuses_positional_construction() -> None:
    construct: Callable[..., _Record] = _Record  # the positional call a checker would refuse outright
    with pytest.raises(TypeError, match="positional argument"):
        construct(torch.tensor(1.0))


def test_a_record_is_frozen_once_built() -> None:
    record = _Record(value=torch.tensor(1.0))
    field_name = "value"
    with pytest.raises(FrozenInstanceError):
        setattr(record, field_name, torch.tensor(2.0))


def test_a_record_equals_only_itself() -> None:
    """A record is a measurement event, not a value: two of equal content stay two entries."""
    value = torch.tensor(1.0)
    first = _Record(value=value)
    second = _Record(value=value)
    assert first == first
    assert first != second
    assert len({first, second}) == 2


# === Record moves ===


def test_a_record_with_nothing_to_detach_is_its_own_detached_form() -> None:
    """The walk rebuilds only what changes, so a graph-free record costs no copy."""
    record = _Record(value=torch.tensor(1.0))
    assert record.detach() is record


def test_a_record_already_on_the_device_is_returned_as_it_is() -> None:
    record = _Record(value=torch.tensor(1.0))
    assert record.to(_CPU) is record


def test_detaching_reaches_every_tensor_field_through_the_nesting() -> None:
    record = _Pair(
        value=torch.tensor([1.0], requires_grad=True),
        nested=_Record(value=torch.tensor([2.0], requires_grad=True)),
    )
    detached = record.detach()
    assert detached.value.requires_grad is False
    assert detached.nested.value.requires_grad is False


def test_a_move_reaches_every_tensor_field_through_the_nesting() -> None:
    record = _Pair(value=torch.tensor(1.0), nested=_Record(value=torch.tensor(2.0)))
    moved = record.to(_ELSEWHERE)
    assert moved.value.device == _ELSEWHERE
    assert moved.nested.value.device == _ELSEWHERE


# === Family slot ===


def test_one_recorder_of_a_family_collects_at_a_time() -> None:
    with _FamilyA(), pytest.raises(RuntimeError, match="only one _FamilyA"), _FamilyA():
        pass


def test_a_derived_recorder_shares_its_roots_slot() -> None:
    with _DerivedA() as derived:
        assert _FamilyA.active() is True
        assert _FamilyA.current() is derived
        _FamilyA.submit(_Record(value=torch.tensor(1.0)))
    assert [record.value.item() for record in derived.records] == [1.0]


@pytest.mark.parametrize(("outer", "inner"), [(_FamilyA, _DerivedA), (_DerivedA, _FamilyA)])
def test_a_root_and_its_subclass_are_mutually_exclusive(
    outer: type[_FamilyA],
    inner: type[_FamilyA],
) -> None:
    with outer(), pytest.raises(RuntimeError, match="only one _FamilyA"), inner():
        pass


def test_families_collect_side_by_side() -> None:
    """A record submitted to one family never reaches another's active recorder."""
    with _FamilyA() as a, _FamilyB() as b:
        _FamilyA.submit(_Record(value=torch.tensor(1.0)))
        _FamilyB.submit(_Record(value=torch.tensor(2.0)))
    assert [record.value.item() for record in a.records] == [1.0]
    assert [record.value.item() for record in b.records] == [2.0]


def test_activity_is_per_family() -> None:
    assert _FamilyA.active() is False
    with _FamilyA():
        assert _FamilyA.active() is True
        assert _FamilyB.active() is False
    assert _FamilyA.active() is False


def test_the_base_opens_no_family() -> None:
    """`RecorderBase` holds no slot: it can neither collect nor be asked."""
    with pytest.raises(TypeError, match="opens no recorder family"):
        RecorderBase()
    with pytest.raises(TypeError, match="opens no recorder family"):
        RecorderBase.active()


# === Collection ===


def test_submit_stores_the_record_detached() -> None:
    """The side channel stores a copy, so the submitted record itself is never the stored one."""
    value = torch.tensor([1.0, 2.0], requires_grad=True)
    submitted = _Record(value=value)
    with _FamilyA() as recorder:
        _FamilyA.submit(submitted)
    (record,) = recorder.records
    assert record is not submitted
    assert record.value.requires_grad is False
    assert record.value.grad_fn is None
    torch.testing.assert_close(record.value, value.detach())


def test_the_base_submit_keeps_every_record() -> None:
    """Admission is opt-in: a family that overrides nothing collects the whole stream."""
    with _FamilyA() as recorder:
        for value in (-1.0, 0.0, 1.0):
            _FamilyA.submit(_Record(value=torch.tensor(value)))
    assert [record.value.item() for record in recorder.records] == [-1.0, 0.0, 1.0]


def test_a_family_hook_applies_its_own_admission_rule() -> None:
    """The hook owns admission and hands survivors to `_submit_record`.

    The book therefore holds the family's subset and keeps it detached, exactly
    as the base would.
    """
    kept = _Tagged(value=torch.tensor([1.0], requires_grad=True), tag=0)
    with _TaggedOnly() as recorder:
        _TaggedOnly.submit(_Tagged(value=torch.tensor(2.0), tag=-1))
        _TaggedOnly.submit(kept)
    (record,) = recorder.records
    assert record is not kept
    torch.testing.assert_close(record.value, kept.value.detach())
    assert record.value.requires_grad is False


def test_submitting_outside_a_context_is_a_no_op() -> None:
    """An unclaimed record is dropped, not an error."""
    _FamilyA.submit(_Record(value=torch.tensor(1.0)))  # must not raise
    _TaggedOnly.submit(_Tagged(value=torch.tensor(1.0), tag=0))  # the family hook drops it too
    assert _FamilyA.active() is False
    assert _FamilyA.current() is None
    assert _TaggedOnly.active() is False


def test_re_entry_accumulates_into_one_book() -> None:
    """Records survive an exit, so successive measurements add to the same instance."""
    recorder = _FamilyA()
    with recorder:
        _FamilyA.submit(_Record(value=torch.tensor(1.0)))
    with recorder:
        _FamilyA.submit(_Record(value=torch.tensor(2.0)))
    assert [record.value.item() for record in recorder.records] == [1.0, 2.0]


def test_a_fresh_book_is_a_fresh_recorder() -> None:
    with _FamilyA():
        _FamilyA.submit(_Record(value=torch.tensor(1.0)))
    with _FamilyA() as second:
        pass
    assert second.records == ()


# === Finalization ===


def test_a_clean_exit_parks_the_records_on_the_declared_device() -> None:
    """The recording device is the emitter's; where records rest is the recorder's."""
    recorder = _FamilyA(sync_device=_ELSEWHERE)
    with recorder:
        _FamilyA.submit(_Record(value=torch.tensor(1.0)))
    (record,) = recorder.records
    assert record.value.device == _ELSEWHERE


def test_the_sweep_covers_the_whole_book_on_every_clean_exit() -> None:
    """No new-versus-old bookkeeping: an already-parked record is swept again."""
    recorder = _FamilyA(sync_device=_CPU)
    first = _SpyRecord(value=torch.tensor(1.0), moves=[])
    second = _SpyRecord(value=torch.tensor(2.0), moves=[])
    with recorder:
        _FamilyA.submit(first)
    assert first.moves == [_CPU]
    with recorder:
        _FamilyA.submit(second)
    assert first.moves == [_CPU, _CPU]
    assert second.moves == [_CPU]


@pytest.mark.parametrize("kwargs", [{}, {"sync_device": None}], ids=["default", "explicit_none"])
def test_no_sync_device_leaves_every_record_where_it_was_recorded(kwargs: dict[str, torch.device | None]) -> None:
    """Parking is opt-in: without a `sync_device` the sweep moves nothing.

    The default is that opt-out, so a recorder parks a book on one device only
    where a caller asked for it.
    """
    record = _SpyRecord(value=torch.tensor(1.0), moves=[])
    with _FamilyA(**kwargs) as recorder:
        _FamilyA.submit(record)
    assert recorder.records == (record,)
    assert record.moves == []


def test_a_raising_body_frees_the_slot_without_finalizing() -> None:
    """A failed measurement keeps its records raw: only a clean exit finalizes them."""
    recorder = _FamilyA(sync_device=_ELSEWHERE)
    record = _SpyRecord(value=torch.tensor(1.0), moves=[])
    with pytest.raises(RuntimeError, match="boom"), recorder:  # noqa: PT012
        _FamilyA.submit(record)
        raise RuntimeError("boom")
    assert _FamilyA.active() is False
    assert recorder.records == (record,)
    assert record.moves == []


# === Graph seams ===


@pytest.mark.parametrize("seam", ["current", "active", "submit", "_submit_record"])
def test_the_active_slot_is_reached_from_outside_every_graph(seam: str) -> None:
    """The slot is Python state a trace cannot guard on, so every seam touching it is a graph break."""
    assert getattr(RecorderBase, seam)._torchdynamo_disable


def test_the_base_submit_keeps_a_family_hook_outside_the_graph() -> None:
    """The stable public seam owns the graph break; family hooks need no decorator."""
    assert _TaggedOnly.submit._torchdynamo_disable
    assert not getattr(_TaggedOnly._submit_impl, "_torchdynamo_disable", False)
