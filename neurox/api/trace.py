"""Collection of structured diagnostic snapshots without PPA aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor

from neurox.common.recorder import RecorderBase


@dataclass(frozen=True, eq=False)
class TraceRecord:
    source: str
    """Stable operation or algorithm name chosen by the emitting owner."""
    values: dict[str, Tensor]
    """Named tensor snapshots, retaining position and history axes and dtypes."""


class TraceCollector(RecorderBase[TraceRecord, TraceRecord, tuple[TraceRecord, ...]]):
    """Collect diagnostics explicitly requested from a solver or algorithm.

    Enter outside compiled execution. This context does not enable solver
    tracing or relax convergence checks. Completed records are CPU snapshots;
    raw dtype, invalid-value sentinels, and nested history axes are retained.
    Each scoped loop invocation contributes one record per static submission
    site, with the loop's declared record axis. Histories are never summed.
    Failed contexts discard their records, as with other recorder families.
    """

    @property
    def result(self) -> tuple[TraceRecord, ...]:
        return tuple(self._history_records)

    def submit(self, values: dict[str, Tensor], *, source: str) -> None:
        """Save a fixed set of tensor fields under one diagnostic identity.

        Keys and each field's shape stay fixed across a collected scan/map.
        Numerical state is not changed by submission. Repeated calls retain
        separate records instead of merging observations by source name. Field
        names must be nonempty and contain no newline.
        """
        if not values:
            raise ValueError("A trace record must contain at least one tensor")
        if any(not key or "\n" in key for key in values):
            raise ValueError("Trace field names must be nonempty and contain no newline")
        self._submit_values(list(values.values()), keys="\n".join(values), source=source)

    @RecorderBase.submission(payloads=("values",))
    def _submit_values(self, values: list[Tensor], *, keys: str, source: str) -> None:
        self._submit_record(
            TraceRecord(
                source=source,
                values={key: self._export_tensor(value) for key, value in zip(keys.split("\n"), values, strict=True)},
            )
        )

    def _merge_records(self, records: Sequence[TraceRecord]) -> Sequence[TraceRecord]:
        return records
