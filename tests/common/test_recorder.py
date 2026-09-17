"""Recorder families isolate collection and preserve batches across rejected or failed contexts."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from neurox.common.recorder import RecordBase, RecorderBase

_ELSEWHERE = torch.device("meta")


class _Record(RecordBase):
    value: Tensor


class _FamilyA(RecorderBase[_Record]):
    pass


class _FamilyB(RecorderBase[_Record]):
    pass


class _DerivedA(_FamilyA):
    pass


class _Tagged(_Record):
    tag: int


class _TaggedOnly(RecorderBase[_Tagged]):
    @classmethod
    def _submit_impl(cls, record: _Tagged) -> None:
        if record.tag < 0:
            return
        cls._submit_record(record)


def _run_failing_batch(recorder: _FamilyA, record: _Record) -> None:
    with recorder:
        _FamilyA.submit(record)
        raise RuntimeError("failed batch")


def test_derived_recorders_share_their_family_slot_without_interfering_with_other_families() -> None:
    with _DerivedA() as a:
        _FamilyA.submit(_Record(value=torch.tensor(1.0)))
        with _FamilyB() as b:
            _FamilyB.submit(_Record(value=torch.tensor(2.0)))
        _FamilyA.submit(_Record(value=torch.tensor(3.0)))
    assert not _FamilyA.active()
    assert not _FamilyB.active()
    assert [record.value.item() for record in a.records] == [1.0, 3.0]
    assert [record.value.item() for record in b.records] == [2.0]


def test_a_family_admission_hook_retains_only_accepted_records_and_detaches_them() -> None:
    kept = _Tagged(value=torch.tensor([1.0], requires_grad=True), tag=0)
    with _TaggedOnly() as recorder:
        _TaggedOnly.submit(_Tagged(value=torch.tensor(2.0), tag=-1))
        _TaggedOnly.submit(kept)
    (record,) = recorder.records
    torch.testing.assert_close(record.value, kept.value.detach())
    assert not record.value.requires_grad
    assert kept.value.requires_grad


@pytest.mark.parametrize(
    ("outer", "inner"),
    [(_FamilyA, _FamilyA), (_FamilyA, _DerivedA), (_DerivedA, _FamilyA)],
    ids=["same-instance", "base-to-derived", "derived-to-base"],
)
def test_rejected_nested_entry_preserves_the_active_batch(outer: type[_FamilyA], inner: type[_FamilyA]) -> None:
    recorder = outer()
    nested = recorder if outer is inner else inner()
    with recorder:
        _FamilyA.submit(_Record(value=torch.tensor(1.0)))
        with pytest.raises(RuntimeError, match="only one"), nested:
            pass
        _FamilyA.submit(_Record(value=torch.tensor(2.0)))
    assert [record.value.item() for record in recorder.records] == [1.0, 2.0]


def test_failed_exit_keeps_raw_records_until_the_next_clean_exit() -> None:
    recorder = _FamilyA(sync_device=_ELSEWHERE)
    record = _Record(value=torch.tensor(1.0))
    with pytest.raises(RuntimeError, match="failed batch"):
        _run_failing_batch(recorder, record)
    assert not _FamilyA.active()
    assert [item.value.item() for item in recorder.records] == [1.0]
    assert recorder.records[0].value.device == record.value.device

    with recorder:
        pass
    assert recorder.records[0].value.device == _ELSEWHERE


def test_history_and_open_batches_accumulate_once_across_failed_and_empty_contexts() -> None:
    recorder = _FamilyA()
    with recorder:
        _FamilyA.submit(_Record(value=torch.tensor(1.0)))
    with pytest.raises(RuntimeError, match="failed batch"):
        _run_failing_batch(recorder, _Record(value=torch.tensor(2.0)))
    assert [record.value.item() for record in recorder.records] == [1.0, 2.0]

    with recorder:
        assert [record.value.item() for record in recorder.records] == [1.0, 2.0]
        _FamilyA.submit(_Record(value=torch.tensor(3.0)))
        assert [record.value.item() for record in recorder.records] == [1.0, 2.0, 3.0]
    with recorder:
        pass
    assert [record.value.item() for record in recorder.records] == [1.0, 2.0, 3.0]
