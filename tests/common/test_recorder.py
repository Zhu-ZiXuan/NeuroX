"""Tests for the shared side channel: record declaration, family slots, accumulation, finalization.

A record declares fields and nothing else: the base turns every subclass into a
frozen, keyword-only dataclass, so detaching and parking a record are the base's
own field walk rather than a pair each family writes out. A recorder family is a
direct subclass of :class:`RecorderBase` together with everything below it. The
family shares one active slot, so at most one of its recorders collects at a
time while independent families collect side by side. The book belongs to the
instance: re-entering one recorder accumulates into it, and only a clean exit
parks what it holds on the recorder's device.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, field
from inspect import Parameter, signature
from typing import Self

import pytest
import torch
from torch import Tensor

from neurox.common import RecordBase, RecorderBase

_CPU = torch.device("cpu")
_ELSEWHERE = torch.device("meta")  # a second device every host has


class _Record(RecordBase):
    """Minimal record: one carried tensor, moved by the base's field walk.

    Attributes:
        value: The carried tensor.
    """

    value: Tensor


class _Pair(RecordBase):
    """A record holding a second one, so the walk has a level to recurse into.

    Attributes:
        value: The carried tensor.
        nested: The record held one level down.
    """

    value: Tensor
    nested: _Record


class _SpyRecord(_Record):
    """A record logging the moves it is asked for, on top of making them.

    Attributes:
        moves: Every device :meth:`to` was called with, in call order.
    """

    moves: list[torch.device]

    def to(self, device: torch.device) -> Self:
        self.moves.append(device)
        return super().to(device)


class _FamilyA(RecorderBase[_Record]):
    """One family root, owning the slot every class below it shares."""


class _FamilyB(RecorderBase[_Record]):
    """A second, independent family root."""


class _DerivedA(_FamilyA):
    """A member below A's root, holding no slot of its own."""


# === Record declaration ===


def test_a_record_declaring_its_own_init_is_refused_at_definition() -> None:
    """Construction belongs to the base's dataclass, so a hand-written one never gets defined."""
    with pytest.raises(TypeError, match=r"must declare dataclass fields, not __init__\(\)"):

        class _InvalidRecord(RecordBase):
            def __init__(self) -> None:
                pass


def test_a_record_declaring_a_post_init_is_refused_at_definition() -> None:
    """A record carries data and nothing else: there is no hook to run after it is built."""
    with pytest.raises(TypeError, match=r"declares fields, not __post_init__\(\)"):

        class _InvalidRecord(RecordBase):
            def __post_init__(self) -> None:
                pass


def test_a_field_specifier_is_refused_at_definition() -> None:
    """A record field is a plain annotation; per-field machinery has nothing to configure."""
    with pytest.raises(TypeError, match=r"a record declares plain fields only"):

        class _InvalidRecord(RecordBase):
            value: int = field(default=3)


def test_a_field_only_record_is_a_keyword_only_dataclass() -> None:
    """Declaring fields is the whole subclass: the base supplies the constructor."""
    parameters = signature(_Record).parameters.values()
    assert [parameter.name for parameter in parameters] == ["value"]
    assert all(parameter.kind is Parameter.KEYWORD_ONLY for parameter in parameters)


def test_a_record_refuses_positional_construction() -> None:
    """Keyword-only fields keep a record's call sites readable as its declaration grows."""
    construct: Callable[..., _Record] = _Record  # the positional call a checker would refuse outright
    with pytest.raises(TypeError, match="positional argument"):
        construct(torch.tensor(1.0))


def test_a_record_is_frozen_once_built() -> None:
    """A collected record is a measurement: nothing downstream may rewrite it."""
    record = _Record(value=torch.tensor(1.0))
    field_name = "value"
    with pytest.raises(FrozenInstanceError):
        setattr(record, field_name, torch.tensor(2.0))


def test_a_record_equals_only_itself() -> None:
    """A record is one measurement event, not a value: comparing two of them never merges them."""
    value = torch.tensor(1.0)
    first, second = _Record(value=value), _Record(value=value)
    assert first == first
    assert first != second
    assert len({first, second}) == 2


# === Record moves ===


def test_a_record_with_nothing_to_detach_is_its_own_detached_form() -> None:
    """The walk rebuilds only what changes, so a graph-free record costs no copy."""
    record = _Record(value=torch.tensor(1.0))
    assert record.detach() is record


def test_a_record_already_on_the_device_is_returned_as_it_is() -> None:
    """Same rule for parking: an unchanged field walk hands back the record itself."""
    record = _Record(value=torch.tensor(1.0))
    assert record.to(_CPU) is record


def test_detaching_reaches_every_tensor_field_through_the_nesting() -> None:
    """A tensor one level down is as much of the record as the top-level one."""
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
    """A second recorder of the same family refuses to open a rival book."""
    with _FamilyA(), pytest.raises(RuntimeError, match="only one _FamilyA"), _FamilyA():
        pass


def test_a_derived_recorder_shares_its_roots_slot() -> None:
    """The root's guard sees a subclass's recorder as its own."""
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
    """One slot per family: parent and child contend for it either way round."""
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
    """One family's open context leaves every other family inactive."""
    assert _FamilyA.active() is False
    with _FamilyA():
        assert _FamilyA.active() is True
        assert _FamilyB.active() is False
    assert _FamilyA.active() is False


def test_the_base_opens_no_family() -> None:
    """:class:`RecorderBase` holds no slot: it can neither collect nor be asked."""
    with pytest.raises(TypeError, match="opens no recorder family"):
        RecorderBase()
    with pytest.raises(TypeError, match="opens no recorder family"):
        RecorderBase.active()


# === Collection ===


def test_submit_stores_the_record_detached() -> None:
    """The side channel never carries autograd history out of the graph."""
    value = torch.tensor([1.0, 2.0], requires_grad=True)
    submitted = _Record(value=value)
    with _FamilyA() as recorder:
        _FamilyA.submit(submitted)
    (record,) = recorder.records
    assert record is not submitted
    assert record.value.requires_grad is False
    assert record.value.grad_fn is None
    torch.testing.assert_close(record.value, value.detach())


def test_submitting_outside_a_context_is_a_no_op() -> None:
    """An emitter's guard may be stale: an unclaimed record is dropped, not an error."""
    _FamilyA.submit(_Record(value=torch.tensor(1.0)))  # must not raise
    assert _FamilyA.active() is False
    assert _FamilyA.current() is None


def test_re_entry_accumulates_into_one_book() -> None:
    """Records survive an exit: successive measurements add to the same instance."""
    recorder = _FamilyA()
    with recorder:
        _FamilyA.submit(_Record(value=torch.tensor(1.0)))
    with recorder:
        _FamilyA.submit(_Record(value=torch.tensor(2.0)))
    assert [record.value.item() for record in recorder.records] == [1.0, 2.0]


def test_a_fresh_book_is_a_fresh_recorder() -> None:
    """Nothing a previous recorder collected reaches the next one."""
    with _FamilyA():
        _FamilyA.submit(_Record(value=torch.tensor(1.0)))
    with _FamilyA() as second:
        pass
    assert second.records == []


# === Finalization ===


def test_a_clean_exit_parks_the_records_on_the_declared_device() -> None:
    """The recording device is the emitter's; where records rest is the harness's."""
    recorder = _FamilyA(device=_ELSEWHERE)
    with recorder:
        _FamilyA.submit(_Record(value=torch.tensor(1.0)))
    (record,) = recorder.records
    assert record.value.device == _ELSEWHERE


def test_the_sweep_covers_the_whole_book_on_every_clean_exit() -> None:
    """No new-versus-old bookkeeping: an already-parked record is swept again, free."""
    recorder = _FamilyA()
    first, second = _SpyRecord(value=torch.tensor(1.0), moves=[]), _SpyRecord(value=torch.tensor(2.0), moves=[])
    with recorder:
        _FamilyA.submit(first)
    assert first.moves == [_CPU]
    with recorder:
        _FamilyA.submit(second)
    assert first.moves == [_CPU, _CPU]
    assert second.moves == [_CPU]


def test_device_none_leaves_every_record_where_it_was_recorded() -> None:
    """Opting out of the sweep leaves the records untouched, not moved to a default."""
    record = _SpyRecord(value=torch.tensor(1.0), moves=[])
    with _FamilyA(device=None) as recorder:
        _FamilyA.submit(record)
    assert recorder.records == [record]
    assert record.moves == []


def test_a_raising_body_frees_the_slot_without_finalizing() -> None:
    """A failed measurement keeps its records raw: only a clean exit closes a book."""
    recorder = _FamilyA()
    record = _SpyRecord(value=torch.tensor(1.0), moves=[])
    with pytest.raises(RuntimeError, match="boom"), recorder:
        _FamilyA.submit(record)
        raise RuntimeError("boom")
    assert _FamilyA.active() is False
    assert recorder.records == [record]
    assert record.moves == []


# === Graph seams ===


@pytest.mark.parametrize("seam", ["current", "active", "submit"])
def test_the_active_slot_is_reached_from_outside_every_graph(seam: str) -> None:
    """The slot is Python state a trace cannot guard on.

    Dynamo would fold an empty slot into the graph as a constant and install no
    guard on it, pinning a region first compiled outside any context to
    "inactive" ever after. Every seam touching the slot is therefore marked as
    a graph break, which is a property of the functions themselves.
    """
    assert getattr(RecorderBase, seam)._torchdynamo_disable
