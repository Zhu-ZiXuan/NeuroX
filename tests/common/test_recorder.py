"""Recorder families isolate batches and merge only after successful execution."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from torch import Tensor

from neurox.common.recorder import RecorderBase
from neurox.primitive.analog import AdcProber

_ELSEWHERE = torch.device("meta")


class _FamilyA(RecorderBase[Tensor, Tensor, Sequence[Tensor]]):
    def __init__(self, *, sync_device: torch.device | None = None) -> None:
        super().__init__(sync_device=sync_device)
        self.events: list[str] = []

    @property
    def result(self) -> Sequence[Tensor]:
        return self._history_records

    def submit(self, record: Tensor) -> None:
        self._submit_record(record.detach())

    def _merge_records(self, records: Sequence[Tensor]) -> Sequence[Tensor]:
        assert self.current() is None
        self.events.append("merge")
        return records

    def _sync_history(self, records: Sequence[Tensor]) -> Sequence[Tensor]:
        self.events.append("sync")
        return [record.to(self._sync_device) for record in records] if self._sync_device is not None else records


class _DerivedA(_FamilyA):
    pass


def _run_failing_batch(recorder: _FamilyA) -> None:
    with recorder:
        recorder.submit(torch.tensor(1.0))
        raise RuntimeError("failed batch")


def test_derived_recorders_share_their_family_slot_without_interfering_with_other_families() -> None:
    first = torch.tensor(1.0, requires_grad=True)
    with _DerivedA() as a:
        assert _FamilyA.current() is a
        assert _DerivedA.current() is a
        a.submit(first)
        with AdcProber() as b:
            assert _FamilyA.current() is a
            assert AdcProber.current() is b
            b.submit_current(i_in__uA=torch.tensor(2.0))
        a.submit(torch.tensor(3.0))
    assert not _FamilyA.active()
    assert not AdcProber.active()
    assert [record.item() for record in a.result] == [1.0, 3.0]
    assert [record.input_value().item() for record in b.result] == [2.0]


@pytest.mark.parametrize(
    ("outer", "inner"),
    [(_FamilyA, _FamilyA), (_FamilyA, _DerivedA), (_DerivedA, _FamilyA)],
    ids=["same-instance", "base-to-derived", "derived-to-base"],
)
def test_rejected_nested_entry_preserves_the_active_batch(outer: type[_FamilyA], inner: type[_FamilyA]) -> None:
    recorder = outer()
    nested = recorder if outer is inner else inner()
    with recorder:
        recorder.submit(torch.tensor(1.0))
        with pytest.raises(RuntimeError, match="only one"), nested:
            pass
        assert _FamilyA.current() is recorder
        recorder.submit(torch.tensor(2.0))
    assert [record.item() for record in recorder.result] == [1.0, 2.0]


def test_execution_failure_discards_the_batch_without_merging_or_syncing() -> None:
    recorder = _FamilyA(sync_device=_ELSEWHERE)
    with pytest.raises(RuntimeError, match="failed batch"):
        _run_failing_batch(recorder)
    assert not _FamilyA.active()
    assert recorder.events == []

    with recorder:
        pass
    assert not recorder.result
    assert recorder.events == ["merge", "sync"]


def test_records_are_merged_before_device_synchronization() -> None:
    recorder = _FamilyA(sync_device=_ELSEWHERE)
    with recorder:
        recorder.submit(torch.tensor(1.0))
    assert recorder.events == ["merge", "sync"]
    assert recorder.result[0].device == _ELSEWHERE


def test_merging_failure_releases_the_family_and_skips_sync() -> None:
    class _Rejecting(_FamilyA):
        def _merge_records(self, records: Sequence[Tensor]) -> Sequence[Tensor]:
            raise ValueError("merging failed")

    recorder = _Rejecting(sync_device=_ELSEWHERE)
    with pytest.raises(ValueError, match="merging failed"), recorder:
        recorder.submit(torch.tensor(1.0))
    assert not _FamilyA.active()
    assert recorder.events == []

    with _FamilyA() as fresh:
        fresh.submit(torch.tensor(2.0))
    assert [record.item() for record in fresh.result] == [2.0]
