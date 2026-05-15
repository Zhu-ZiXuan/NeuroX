"""Side-channel hardware profiler for NeuroX circuit-level simulation.

``NeuroxProfiler`` is a thread-local context manager that captures hardware
metrics emitted by physical modules during a forward pass.  Numerical
functions stay clean — energy / latency / area flow through the
``ProfiledModule._log_dynamic`` side channel rather than the return path —
and the profiler is responsible for the central aggregation:

* **Runtime events** (dynamic energy + latency) are appended per physical
  module call.  Total runtime latency is the sum across events.
* **Static records** (area + leakage power) are computed at fabrication
  time and stored on each ``ProfiledModule`` (``_inst_area__um2``,
  ``_inst_leakage__uW``).  ``analyze_static(model)`` walks the model
  tree and sums them.
* **Leakage energy** is *not* logged per-module.  The profiler derives it
  centrally as ``leakage_power__uW * total_latency__ns`` in ``summary``.

Outside an active context every ``_log_dynamic`` call is a silent no-op,
so physical modules can be exercised without paying any logging cost.
"""

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Self

import torch.nn as nn

from .profiled_module import ProfiledModule


@dataclass(frozen=True)
class RuntimeEvent:
    """One dynamic event from a physical module's primary execution call.

    Attributes:
        qualified_name: Hierarchical module identifier (e.g.
            ``fc1.macro.xbar.core.tia``).
        module_type: Short class-name tag of the emitting module.
        dynamic_energy__fJ: Switching energy attributed to this call [fJ].
        latency__ns: Runtime latency contribution for this call [ns].
    """

    qualified_name: str
    module_type: str
    dynamic_energy__fJ: float
    latency__ns: float


@dataclass(frozen=True)
class StaticRecord:
    """Per-module static-metric snapshot built from fabrication-time state.

    Attributes:
        qualified_name: Hierarchical module identifier.
        module_type: Short class-name tag of the module.
        area__um2: Module-local total area = ``area_per_inst * inst_count``.
        leakage_power__uW: Module-local leakage power total.
    """

    qualified_name: str
    module_type: str
    area__um2: float
    leakage_power__uW: float


@dataclass
class StaticMetrics:
    """Aggregated static hardware metrics for one model.

    ``leakage_power__uW`` is the static total; ``latency__ns`` is the
    *dynamic* total runtime latency the profiler observed (default
    ``0.0`` when produced outside a runtime context).
    Leakage *energy* is the product ``leakage_power__uW * latency__ns``
    computed centrally in :meth:`NeuroxProfiler.summary`.
    """

    area__um2: float = 0.0
    leakage_power__uW: float = 0.0
    latency__ns: float = 0.0

    @property
    def leakage_energy__fJ(self) -> float:
        """Derived ``leakage_power__uW * latency__ns`` — central computation."""
        return self.leakage_power__uW * self.latency__ns


@dataclass
class ProfilerReport:
    """Combined runtime + static report bundle."""

    events: list[RuntimeEvent] = field(default_factory=list)
    static_records: list[StaticRecord] = field(default_factory=list)
    static: StaticMetrics = field(default_factory=StaticMetrics)

    @property
    def total_dynamic_energy__fJ(self) -> float:
        return sum(e.dynamic_energy__fJ for e in self.events)

    @property
    def total_latency__ns(self) -> float:
        return sum(e.latency__ns for e in self.events)


class NeuroxProfiler:
    """Context manager that captures physical-module side-channel events.

    Within a ``with NeuroxProfiler() as profiler:`` block, every
    ``ProfiledModule._log_dynamic`` call appends to ``events``.  Outside
    a context the calls are no-ops, so production code pays no cost when
    profiling is not enabled.

    Example::

        with NeuroxProfiler() as profiler:
            output = model(input_data)
        report = profiler.report(model)
        print(profiler.summary(report.static))

    Attributes:
        events: Runtime events appended during the active context.
    """

    _local = threading.local()

    def __init__(self) -> None:
        self.events: list[RuntimeEvent] = []

    def __enter__(self) -> Self:
        self.events.clear()
        NeuroxProfiler._local.current = self
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        NeuroxProfiler._local.current = None

    @classmethod
    def get_current(cls) -> Self | None:
        """Return the active profiler for this thread, or ``None``."""
        return getattr(cls._local, "current", None)

    # ----------------------------------------------------------------
    # Side-channel entry point (called by ProfiledModule._log_dynamic)
    # ----------------------------------------------------------------

    def _append_runtime_event(
        self,
        *,
        qualified_name: str,
        module_type: str,
        dynamic_energy__fJ: float,
        latency__ns: float,
    ) -> None:
        """Append one runtime event — used exclusively by the mixin."""
        self.events.append(
            RuntimeEvent(
                qualified_name=qualified_name,
                module_type=module_type,
                dynamic_energy__fJ=dynamic_energy__fJ,
                latency__ns=latency__ns,
            )
        )

    # --------------------- Runtime aggregations ---------------------

    @property
    def total_dynamic_energy__fJ(self) -> float:
        """Sum of dynamic energy across all runtime events."""
        return sum(e.dynamic_energy__fJ for e in self.events)

    @property
    def total_latency__ns(self) -> float:
        """Sum of latency contributions across all runtime events."""
        return sum(e.latency__ns for e in self.events)

    @property
    def energy_by_name(self) -> dict[str, float]:
        """Dynamic energy grouped by qualified module name."""
        out: dict[str, float] = {}
        for e in self.events:
            out[e.qualified_name] = out.get(e.qualified_name, 0.0) + e.dynamic_energy__fJ
        return out

    @property
    def energy_by_type(self) -> dict[str, float]:
        """Dynamic energy grouped by module class name."""
        out: dict[str, float] = {}
        for e in self.events:
            out[e.module_type] = out.get(e.module_type, 0.0) + e.dynamic_energy__fJ
        return out

    @property
    def latency_by_name(self) -> dict[str, float]:
        """Runtime latency grouped by qualified module name."""
        out: dict[str, float] = {}
        for e in self.events:
            out[e.qualified_name] = out.get(e.qualified_name, 0.0) + e.latency__ns
        return out

    # --------------------- Static aggregation -----------------------

    @staticmethod
    def collect_static(model: nn.Module) -> list[StaticRecord]:
        """Build a per-module ``StaticRecord`` list by walking ``model``.

        Only modules inheriting :class:`ProfiledModule` contribute; the
        per-instance area and leakage are already pre-multiplied by the
        replica count via :meth:`ProfiledModule._record_inst_count`.
        """
        out: list[StaticRecord] = []
        for module in model.modules():
            if isinstance(module, ProfiledModule):
                out.append(
                    StaticRecord(
                        qualified_name=module.qualified_name,
                        module_type=module.module_type,
                        area__um2=module.inst_area__um2,
                        leakage_power__uW=module.inst_leakage__uW,
                    )
                )
        return out

    @staticmethod
    def analyze_static(model: nn.Module) -> StaticMetrics:
        """Aggregate static metrics across all ``ProfiledModule`` instances.

        Returns total area and leakage *power* only — leakage *energy*
        is derived in :meth:`summary` from the dynamic-latency total.
        """
        area = 0.0
        leakage = 0.0
        for module in model.modules():
            if isinstance(module, ProfiledModule):
                area += module.inst_area__um2
                leakage += module.inst_leakage__uW
        return StaticMetrics(area__um2=area, leakage_power__uW=leakage, latency__ns=0.0)

    @staticmethod
    def analyze_model(model: nn.Module) -> ProfilerReport:
        """Build a static-only ``ProfilerReport`` (no runtime events).

        Useful for area/leakage analysis without running the model.
        """
        records = NeuroxProfiler.collect_static(model)
        report = ProfilerReport(static_records=records)
        report.static = NeuroxProfiler.analyze_static(model)
        return report

    # --------------------------- Report -----------------------------

    def report(self, model: nn.Module) -> ProfilerReport:
        """Bundle this context's runtime events with the model's static state."""
        static = NeuroxProfiler.analyze_static(model)
        static.latency__ns = self.total_latency__ns
        return ProfilerReport(
            events=list(self.events),
            static_records=NeuroxProfiler.collect_static(model),
            static=static,
        )

    def summary(
        self,
        *,
        static: StaticMetrics | None = None,
        extras: Mapping[str, str | int | float] | None = None,
    ) -> str:
        """Format a concise multi-line summary.

        When ``static`` is supplied, leakage *energy* is derived centrally
        as ``static.leakage_power__uW * self.total_latency__ns`` — the
        only place in the system where leakage energy is computed.
        """
        lines: list[str] = []
        if extras is not None:
            for k, v in extras.items():
                lines.append(f"{k}: {v}")
        lines.append(f"dynamic_energy_total_fJ: {self.total_dynamic_energy__fJ:.4f}")
        lines.append(f"latency_total_ns: {self.total_latency__ns:.4f}")
        if static is not None:
            leakage_energy__fJ = static.leakage_power__uW * self.total_latency__ns
            lines.append(f"area_total_um2: {static.area__um2:.4f}")
            lines.append(f"leakage_power_total_uW: {static.leakage_power__uW:.4f}")
            lines.append(f"leakage_energy_total_fJ: {leakage_energy__fJ:.4f}")
        return "\n".join(lines)
