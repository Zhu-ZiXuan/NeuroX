"""Recorder families isolate batches and merge only after successful execution."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

import pytest
import torch
from torch import Tensor

from neurox.common.recorder import RecorderBase
from neurox.primitive.analog import AdcProber


class _FamilyA(RecorderBase[Tensor, Tensor, Sequence[Tensor]]):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    @property
    def result(self) -> Sequence[Tensor]:
        return self._history_records

    @RecorderBase.submission
    def submit(self, record: Tensor) -> None:
        self._submit_record(self._export_tensor(record))

    def _merge_records(self, records: Sequence[Tensor]) -> Sequence[Tensor]:
        assert self.current() is None
        self.events.append("merge")
        return records


class _DerivedA(_FamilyA):
    pass


def test_derived_recorders_share_their_family_slot_without_interfering_with_other_families() -> None:
    first = torch.tensor(1.0, requires_grad=True)
    with _DerivedA() as a:
        assert _FamilyA.current() is a
        assert _DerivedA.current() is a
        a.submit(first)
        with AdcProber() as b:
            assert _FamilyA.current() is a
            assert AdcProber.current() is b
            b.submit_current(torch.tensor(2.0))
        a.submit(torch.tensor(3.0))
    assert not _FamilyA.active()
    assert not AdcProber.active()
    assert [record.item() for record in a.result] == [1.0, 3.0]
    assert not a.result[0].requires_grad
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


def test_execution_failure_discards_exports_and_allows_reuse(device: torch.device) -> None:
    recorder = _FamilyA()
    # Exercise cleanup while a CUDA export may still be in flight.
    first = torch.arange(65536, dtype=torch.float32, device=device)

    def fail_batch() -> None:
        with recorder:
            recorder.submit(first)
            raise RuntimeError("failed batch")

    with pytest.raises(RuntimeError, match="failed batch"):
        fail_batch()
    assert not _FamilyA.active()
    assert recorder.events == []
    assert not recorder.result

    with recorder:
        pass
    assert not recorder.result
    assert recorder.events == ["merge"]

    with recorder:
        recorder.submit(first + 1)
    torch.testing.assert_close(recorder.result, [torch.arange(65536, dtype=torch.float32) + 1])


def test_merging_failure_releases_the_family_without_retaining_history() -> None:
    class _Rejecting(_FamilyA):
        def _merge_records(self, records: Sequence[Tensor]) -> Sequence[Tensor]:
            raise ValueError("merging failed")

    recorder = _Rejecting()
    with pytest.raises(ValueError, match="merging failed"), recorder:
        recorder.submit(torch.tensor(1.0))
    assert not _FamilyA.active()
    assert recorder.events == []
    assert not recorder.result

    with _FamilyA() as fresh:
        fresh.submit(torch.tensor(2.0))
    assert [record.item() for record in fresh.result] == [2.0]


def test_compiled_snapshots_are_ready_in_order_across_contexts_and_copying(device: torch.device) -> None:
    class _ReadOnMerge(_FamilyA):
        def _merge_records(self, records: Sequence[Tensor]) -> Sequence[Tensor]:
            # The first permitted CPU read must see complete snapshots.
            torch.testing.assert_close(records, expected_batch)
            return super()._merge_records(records)

    @torch.compile(dynamic=False, fullgraph=True)
    def emit(value):
        recorder = _FamilyA.current()
        if recorder is not None:
            recorder.submit(value)
        value.add_(1000)

    recorder = _ReadOnMerge()
    expected = []
    for offset, count in ((0, 12), (100, 3)):
        expected_batch = [
            (torch.arange(12, dtype=torch.float64).reshape(3, 4) + offset + i).T for i in reversed(range(count))
        ]
        with recorder:
            for value in expected_batch:
                emit(value.to(device).clone())
        expected.extend(expected_batch)
        torch.testing.assert_close(recorder.result, expected)

    copied = deepcopy(recorder)
    expected_batch = [torch.full_like(expected[0], 7.0)]
    with copied:
        emit(expected_batch[0].to(device).clone())
    torch.testing.assert_close(recorder.result, expected)
    torch.testing.assert_close(copied.result, expected + expected_batch)
