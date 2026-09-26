"""Derive named PPA data while preserving every collected sample axis."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Literal, overload

from torch import Tensor

from .profiler import ProfileItem

type _StaticMetric = Literal["area", "leakage"]
type _SampleMetric = Literal["dynamic_energy", "static_energy", "total_energy", "working_duration", "powered_duration"]


@dataclass(eq=False, kw_only=True)
class ReportItem:
    area__um2: float | None
    """Local hardware area, excluding child hardware; counted once."""
    leakage__uW: float | None
    """Local hardware leakage, excluding child hardware."""
    dynamic_energy__fJ: Tensor | None
    """Collected energy with all sample axes preserved.
    Shape: `[*sample]`."""
    static_energy__fJ: Tensor | None
    """Leakage energy over each sample's powered interval.
    Shape: `[*sample]`."""
    working_duration__ns: Tensor | None
    """Basic-operation duration at each sample position, using the nearest timed
    ancestor when this name has no timing of its own; not a task completion time.
    Shape: `[*sample]`."""
    powered_duration__ns: Tensor | None
    """Supply-on duration used to calculate static energy; defaults to the
    recorded working window unless a reporting override applies.
    Shape: `[*sample]`."""


def profile_result_to_report_items(
    result: Mapping[str, ProfileItem],
    *,
    powered_duration__ns: Mapping[str, Tensor] | None = None,
) -> dict[str, ReportItem]:
    """Derive named static energy from already assembled profiling tensors.

    Input observations have already been synchronized at profiler context exit.
    Energy and duration tensors retain their axes, dtype, and device. A name
    without timing inherits the nearest timed ancestor's durations, which must
    describe the same sample positions. Powered intervals default to recorded
    working durations, with explicit overrides applied before ancestor lookup.
    Overrides leave working durations unchanged.

    Static costs already include each module's physical instance count. The
    caller ensures they describe the hardware represented by one dynamic-energy
    position. Missing data stays `None`; zero remains a known value.

    Input tensors, including inherited timing, are shared without modification;
    treat results as read-only.

    Args:
        result: Completed named profiler observations with the intended hardware
            scope.
        powered_duration__ns: Exact name-to-tensor replacements for the powered
            window lookup. Names absent from the input may define an ancestor
            window. The nearest name with a recorded duration or override wins;
            a child's recorded duration takes precedence over ancestor
            overrides. Tensors must match the corresponding sample layout and
            device. Zero represents a zero-length powered interval.

    Returns:
        A mapping from names to report items, preserving observation axes. Input
        observation tensors are shared and should be treated as read-only.
    """
    # --- Index directly submitted working windows ---

    durations_by_name = {
        name: item.working_duration__ns for name, item in result.items() if item.working_duration__ns is not None
    }

    powered_durations_by_name = durations_by_name.copy()
    if powered_duration__ns is not None:
        powered_durations_by_name.update(powered_duration__ns)

    # --- Preserve working durations and derive energy from powered intervals ---

    data: dict[str, ReportItem] = {}
    for name, item in result.items():
        duration_name = _nearest_name(name, durations_by_name)
        duration = None if duration_name is None else durations_by_name[duration_name]
        powered_name = _nearest_name(name, powered_durations_by_name)
        powered_duration = None if powered_name is None else powered_durations_by_name[powered_name]
        data[name] = ReportItem(
            area__um2=item.area__um2,
            leakage__uW=item.leakage__uW,
            dynamic_energy__fJ=item.dynamic_energy__fJ,
            static_energy__fJ=None
            if powered_duration is None or item.leakage__uW is None
            else powered_duration * item.leakage__uW,
            working_duration__ns=duration,
            powered_duration__ns=powered_duration,
        )
    return data


class Reporter:
    """Read named PPA quantities from a completed profiler result.

    Construct after the profiler context exits. The reporter retains each name
    and observation axis; it does not sum a model, average batches, or infer a
    schedule. Use `breakdown` to select a metric and perform
    application-specific aggregation afterwards. Static costs must already
    describe the physical scope represented by each observation.

    A name without its own duration inherits the nearest timed ancestor's
    window. `powered_duration__ns` overrides powered windows by exact name
    before ancestor lookup; a child's own recorded duration takes precedence
    over an ancestor window. Overrides must match the corresponding observation
    layout and device. They change leakage-energy calculation without changing
    recorded work.

    Missing quantities remain `None`; zero is a known value. Total energy
    combines available dynamic and static contributions, even if only one is
    available. Input observation tensors are shared, so treat report tensors as
    read-only. Later profiler collection is not incorporated into an existing
    reporter.

    Args:
        result: Completed named profiler observations with the intended hardware
            scope.
        powered_duration__ns: Optional exact-name powered-window overrides
            matching the observation layout and device.
    """

    def __init__(
        self,
        result: Mapping[str, ProfileItem],
        *,
        powered_duration__ns: Mapping[str, Tensor] | None = None,
    ) -> None:
        self._data = profile_result_to_report_items(result, powered_duration__ns=powered_duration__ns)

    # === Public API ===

    @property
    def data(self) -> Mapping[str, ReportItem]:
        """Prepared named data; treat items and tensors as read-only."""
        return self._data

    @overload
    def breakdown(self, metric: _StaticMetric) -> dict[str, float | None]: ...

    @overload
    def breakdown(self, metric: _SampleMetric) -> dict[str, Tensor | None]: ...

    def breakdown(self, metric: _StaticMetric | _SampleMetric) -> dict[str, float | None] | dict[str, Tensor | None]:
        """Select one metric without changing names or sample dimensions.

        Area and leakage use um2 and uW; energies use fJ, durations use ns.
        Total energy adds available dynamic and static contributions elementwise;
        their individual metrics retain `None` for any missing contribution.

        Raises:
            ValueError: The metric name is not supported.
        """
        if metric not in (
            "area",
            "leakage",
            "dynamic_energy",
            "static_energy",
            "total_energy",
            "working_duration",
            "powered_duration",
        ):
            raise ValueError(f"unsupported report metric {metric!r}")
        if metric == "area":
            return {name: item.area__um2 for name, item in self._data.items()}
        if metric == "leakage":
            return {name: item.leakage__uW for name, item in self._data.items()}
        values: dict[str, Tensor | None] = {}
        for name, item in self._data.items():
            match metric:
                case "dynamic_energy":
                    values[name] = item.dynamic_energy__fJ
                case "static_energy":
                    values[name] = item.static_energy__fJ
                case "working_duration":
                    values[name] = item.working_duration__ns
                case "powered_duration":
                    values[name] = item.powered_duration__ns
                case "total_energy":
                    dynamic, static = item.dynamic_energy__fJ, item.static_energy__fJ
                    values[name] = static if dynamic is None else dynamic if static is None else dynamic + static
        return values


def _nearest_name(name: str, names: Collection[str]) -> str | None:
    while True:
        if name in names:
            return name
        if not name:
            return None
        name = name.rpartition(".")[0]
