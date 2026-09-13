"""Record moves, family slots, accumulation, and finalization."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from neurox.common.recorder import RecordBase, RecorderBase

_ELSEWHERE = torch.device("meta")  # a second device every host has


class _Record(RecordBase):
    value: Tensor


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


# === Family slot ===


def test_a_derived_recorder_shares_its_roots_slot() -> None:
    with _DerivedA() as derived:
        assert _FamilyA.active() is True
        assert _FamilyA.current() is derived
        _FamilyA.submit(_Record(value=torch.tensor(1.0)))
    assert [record.value.item() for record in derived.records] == [1.0]


@pytest.mark.parametrize(("outer", "inner"), [(_FamilyA, _FamilyA), (_FamilyA, _DerivedA), (_DerivedA, _FamilyA)])
def test_recorders_in_one_family_are_mutually_exclusive(
    outer: type[_FamilyA],
    inner: type[_FamilyA],
) -> None:
    with outer(), pytest.raises(RuntimeError, match="only one _FamilyA"), inner():
        pass


def test_families_collect_side_by_side() -> None:
    """A record submitted to one family never reaches another's active recorder."""
    assert _FamilyA.active() is False
    assert _FamilyB.active() is False
    with _FamilyA() as a:
        assert _FamilyA.active() is True
        assert _FamilyB.active() is False
        with _FamilyB() as b:
            assert _FamilyA.active() is True
            assert _FamilyB.active() is True
            _FamilyA.submit(_Record(value=torch.tensor(1.0)))
            _FamilyB.submit(_Record(value=torch.tensor(2.0)))
        assert _FamilyA.active() is True
        assert _FamilyB.active() is False
    assert _FamilyA.active() is False
    assert _FamilyB.active() is False
    assert [record.value.item() for record in a.records] == [1.0]
    assert [record.value.item() for record in b.records] == [2.0]


# === Collection ===


def test_submit_stores_the_record_detached() -> None:
    value = torch.tensor([1.0, 2.0], requires_grad=True)
    submitted = _Record(value=value)
    with _FamilyA() as recorder:
        _FamilyA.submit(submitted)
    (record,) = recorder.records
    assert record.value.requires_grad is False
    assert record.value.grad_fn is None
    torch.testing.assert_close(record.value, value.detach())


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
    torch.testing.assert_close(record.value, kept.value.detach())
    assert record.value.requires_grad is False


def test_submitting_outside_a_context_is_a_no_op() -> None:
    """An unclaimed record is dropped, not an error."""
    _FamilyA.submit(_Record(value=torch.tensor(1.0)))  # must not raise
    _TaggedOnly.submit(_Tagged(value=torch.tensor(1.0), tag=0))  # the family hook drops it too
    assert _FamilyA.active() is False
    assert _FamilyA.current() is None
    assert _TaggedOnly.active() is False


def test_records_include_history_and_the_open_batch() -> None:
    recorder = _FamilyA()
    first = _Record(value=torch.tensor(1.0))
    second = _Record(value=torch.tensor(2.0))
    with recorder:
        _FamilyA.submit(first)
    with recorder:
        assert [item.value.item() for item in recorder.records] == [1.0]
        _FamilyA.submit(second)
        assert [item.value.item() for item in recorder.records] == [1.0, 2.0]
    assert [item.value.item() for item in recorder.records] == [1.0, 2.0]
    with recorder:
        pass
    assert [item.value.item() for item in recorder.records] == [1.0, 2.0]


def test_rejected_nested_entry_preserves_the_open_batch() -> None:
    recorder = _FamilyA()
    first = _Record(value=torch.tensor(1.0))
    second = _Record(value=torch.tensor(2.0))
    with recorder:
        _FamilyA.submit(first)
        with pytest.raises(RuntimeError, match="only one _FamilyA"), recorder:
            pass
        _FamilyA.submit(second)
    assert [item.value.item() for item in recorder.records] == [1.0, 2.0]


# === Finalization ===


def test_a_clean_exit_parks_the_records_on_the_declared_device() -> None:
    """The recording device is the emitter's; where records rest is the recorder's."""
    recorder = _FamilyA(sync_device=_ELSEWHERE)
    with recorder:
        _FamilyA.submit(_Record(value=torch.tensor(1.0)))
    (record,) = recorder.records
    assert record.value.device == _ELSEWHERE


@pytest.mark.parametrize("kwargs", [{}, {"sync_device": None}], ids=["default", "explicit_none"])
def test_no_sync_device_leaves_every_record_where_it_was_recorded(kwargs: dict[str, torch.device | None]) -> None:
    """Parking is opt-in: without a `sync_device` the sweep moves nothing.

    The default is that opt-out, so a recorder parks a book on one device only
    where a caller asked for it.
    """
    record = _Record(value=torch.tensor(1.0))
    with _FamilyA(**kwargs) as recorder:
        _FamilyA.submit(record)
    assert [item.value.item() for item in recorder.records] == [1.0]
    assert recorder.records[0].value.device == torch.device("cpu")


def test_a_raising_body_frees_the_slot_without_finalizing() -> None:
    """A failed measurement keeps its records raw: only a clean exit finalizes them."""
    recorder = _FamilyA(sync_device=_ELSEWHERE)
    record = _Record(value=torch.tensor(1.0))
    with pytest.raises(RuntimeError, match="boom"), recorder:  # noqa: PT012
        _FamilyA.submit(record)
        raise RuntimeError("boom")
    assert _FamilyA.active() is False
    assert [item.value.item() for item in recorder.records] == [1.0]
    assert recorder.records[0].value.device == torch.device("cpu")


def test_a_failed_batch_is_merged_once_and_survives_re_entry() -> None:
    recorder = _FamilyA()
    first = _Record(value=torch.tensor(1.0))
    failed = _Record(value=torch.tensor(2.0))
    last = _Record(value=torch.tensor(3.0))
    with recorder:
        _FamilyA.submit(first)
    with pytest.raises(RuntimeError, match="boom"), recorder:  # noqa: PT012
        _FamilyA.submit(failed)
        raise RuntimeError("boom")
    assert [item.value.item() for item in recorder.records] == [1.0, 2.0]
    with recorder:
        _FamilyA.submit(last)
    assert [item.value.item() for item in recorder.records] == [1.0, 2.0, 3.0]
