"""Side-channel hardware profiler for NeuroX circuit-level simulation.

See also:
    docs/internals/common/profiler.md
"""

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import TracebackType
from typing import Self

import torch
import torch.nn as nn
from torch import Tensor

from .circuit import CircuitBase


@dataclass(frozen=True)
class EnergyEvent:
    """One dynamic-energy event from a physical module's primary execution call.

    Attributes:
        qualified_name: Hierarchical module identifier
            (``<layer>.<owner>...<leaf>``).
        module_type: Short class-name tag of the emitting module.
        dynamic_energy__fJ: Switching energy attributed to this call.
    """

    qualified_name: str
    module_type: str
    dynamic_energy__fJ: float


@dataclass(frozen=True)
class LatencyEvent:
    """One latency event from a physical module's primary execution call.

    Attributes:
        qualified_name: Hierarchical module identifier
            (``<layer>.<owner>...<leaf>``).
        module_type: Short class-name tag of the emitting module.
        latency__ns: Latency contribution attributed to this call.
    """

    qualified_name: str
    module_type: str
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

    Attributes:
        area__um2: Sum of per-instance area across all profiled modules.
        leakage_power__uW: Sum of per-instance leakage power.
    """

    area__um2: float = 0.0
    leakage_power__uW: float = 0.0


@dataclass
class ProfilerReport:
    """Combined runtime + static report bundle.

    ``total_latency__ns`` is the event-sum total — sequential-execution
    assumption — and feeds ``leakage_energy__fJ`` integration. It is NOT
    a wall-clock pipeline latency.

    Attributes:
        energy_events: Dynamic-energy events captured during the profiler context.
        latency_events: Latency events captured during the profiler context.
        static_records: Per-module static-metric snapshots.
        static: Aggregated static metrics across the whole model.
    """

    energy_events: list[EnergyEvent] = field(default_factory=list)
    latency_events: list[LatencyEvent] = field(default_factory=list)
    static_records: list[StaticRecord] = field(default_factory=list)
    static: StaticMetrics = field(default_factory=StaticMetrics)

    @property
    def total_dynamic_energy__fJ(self) -> float:
        return sum(e.dynamic_energy__fJ for e in self.energy_events)

    @property
    def total_latency__ns(self) -> float:
        return sum(e.latency__ns for e in self.latency_events)

    @property
    def leakage_energy__fJ(self) -> float:
        """Derived ``static.leakage_power__uW × total_latency__ns``."""
        return self.static.leakage_power__uW * self.total_latency__ns


class NeuroxProfiler:
    """Context manager that captures physical-module side-channel events.

    Canonical usage — record inside the ``with`` block, read totals /
    call ``report()`` after the block exits::

        with NeuroxProfiler() as profiler:
            model(...)
        # _finalize runs in __exit__ — exactly one batched GPU→CPU
        # sync per quantity. Reading aggregations or calling report()
        # mid-with is unsupported.
        print(profiler.total_dynamic_energy__fJ, profiler.total_latency__ns)
        report = profiler.report(model)

    Leaves that own a dynamic profile model call
    ``_log_dynamic_energy(tensor)`` and / or ``_log_latency(tensor)``
    independently inside their primary method; leaves without a
    dynamic model (devices, dynamics-less analog blocks) emit nothing.
    Pending tensors stay on the recording device until ``_finalize``
    drains them. Reading any aggregation property after exit is a
    pure-CPU field access.

    Attributes:
        energy_events: Resolved dynamic-energy events (populated by ``_finalize``).
        latency_events: Resolved latency events (populated by ``_finalize``).
    """

    _local = threading.local()

    def __init__(self) -> None:
        self.energy_events: list[EnergyEvent] = []
        self.latency_events: list[LatencyEvent] = []
        # Recording buffers: 0-D tensors on the recording device; one
        # batched stack→cpu→tolist sync per quantity at _finalize.
        self._pending_energy: list[tuple[str, str, Tensor]] = []
        self._pending_latency: list[tuple[str, str, Tensor]] = []
        # Cached aggregations (populated by _finalize at __exit__).
        self._total_dynamic_energy__fJ: float = 0.0
        self._total_latency__ns: float = 0.0
        self._energy_by_name: dict[str, float] = {}
        self._energy_by_type: dict[str, float] = {}
        self._latency_by_name: dict[str, float] = {}

    def __enter__(self) -> Self:
        # Fresh recording state on every entry.
        self.energy_events = []
        self.latency_events = []
        self._pending_energy = []
        self._pending_latency = []
        self._total_dynamic_energy__fJ = 0.0
        self._total_latency__ns = 0.0
        self._energy_by_name = {}
        self._energy_by_type = {}
        self._latency_by_name = {}
        NeuroxProfiler._local.current = self
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        NeuroxProfiler._local.current = None
        # Auto-finalize on clean exit; on an exception the partial state
        # is not useful and a stray GPU sync could mask the original error.
        if exc_type is None:
            self._finalize()

    @classmethod
    def get_current(cls) -> Self | None:
        """Return the active profiler for this thread, or ``None``."""
        return getattr(cls._local, "current", None)

    # ----------------------------------------------------------------
    # Side-channel entry points (called by ProfileMixin._log_*)
    # ----------------------------------------------------------------

    def _record_energy(
        self,
        *,
        qualified_name: str,
        module_type: str,
        dynamic_energy__fJ: Tensor,
    ) -> None:
        """Stash a 0-D energy reduction into the energy pending buffer."""
        self._pending_energy.append((qualified_name, module_type, dynamic_energy__fJ.detach().sum()))

    def _record_latency(
        self,
        *,
        qualified_name: str,
        module_type: str,
        latency__ns: Tensor,
    ) -> None:
        """Stash a 0-D latency reduction into the latency pending buffer."""
        self._pending_latency.append((qualified_name, module_type, latency__ns.detach().sum()))

    # ----------------------------------------------------------------
    # Finalization — one batched sync per quantity, invoked by __exit__
    # ----------------------------------------------------------------

    def _finalize(self) -> None:
        """Drain pending energy + latency buffers and populate aggregations."""
        if self._pending_energy:
            stacked_e = torch.stack([t for _, _, t in self._pending_energy])
            energies = stacked_e.cpu().tolist()
            for (name, mtype, _t), energy in zip(self._pending_energy, energies, strict=True):
                self.energy_events.append(
                    EnergyEvent(qualified_name=name, module_type=mtype, dynamic_energy__fJ=energy)
                )
                self._total_dynamic_energy__fJ += energy
                self._energy_by_name[name] = self._energy_by_name.get(name, 0.0) + energy
                self._energy_by_type[mtype] = self._energy_by_type.get(mtype, 0.0) + energy
            self._pending_energy = []

        if self._pending_latency:
            stacked_l = torch.stack([t for _, _, t in self._pending_latency])
            latencies = stacked_l.cpu().tolist()
            for (name, mtype, _t), latency in zip(self._pending_latency, latencies, strict=True):
                self.latency_events.append(LatencyEvent(qualified_name=name, module_type=mtype, latency__ns=latency))
                self._total_latency__ns += latency
                self._latency_by_name[name] = self._latency_by_name.get(name, 0.0) + latency
            self._pending_latency = []

    # --------------------- Runtime aggregations ---------------------
    #
    # All properties below are pure cached-field reads after
    # ``_finalize`` has run (i.e. after the ``with`` block exits).

    @property
    def total_dynamic_energy__fJ(self) -> float:
        """Sum of dynamic energy across all energy events."""
        return self._total_dynamic_energy__fJ

    @property
    def total_latency__ns(self) -> float:
        """Sum of latency contributions across all latency events."""
        return self._total_latency__ns

    @property
    def energy_by_name(self) -> dict[str, float]:
        """Dynamic energy grouped by qualified module name [fJ]."""
        return self._energy_by_name

    @property
    def energy_by_type(self) -> dict[str, float]:
        """Dynamic energy grouped by module class name [fJ]."""
        return self._energy_by_type

    @property
    def latency_by_name(self) -> dict[str, float]:
        """Runtime latency grouped by qualified module name [ns]."""
        return self._latency_by_name

    # --------------------- Static aggregation -----------------------

    @staticmethod
    def collect_static(model: nn.Module) -> list[StaticRecord]:
        """Build a per-module ``StaticRecord`` list by walking ``model``.

        Only :class:`CircuitBase` instances are collected — that's the
        layer that carries the static-PPA contract; pure ``ProfileMixin``
        subclasses (if any) emit dynamic events but don't have static
        PPA fields. Devices (RRAM / MOSFET / Selector) are not circuits
        and don't contribute to this report; their physical contribution
        rolls up into the owning circuit's PPA.
        """
        return [
            StaticRecord(
                qualified_name=module.qualified_name,
                module_type=module.module_type,
                area__um2=module.inst_area__um2,
                leakage_power__uW=module.inst_leakage__uW,
            )
            for module in model.modules()
            if isinstance(module, CircuitBase)
        ]

    @staticmethod
    def analyze_static(model: nn.Module) -> StaticMetrics:
        """Aggregate total area and leakage power across all circuits."""
        area = 0.0
        leakage = 0.0
        for module in model.modules():
            if isinstance(module, CircuitBase):
                area += module.inst_area__um2
                leakage += module.inst_leakage__uW
        return StaticMetrics(area__um2=area, leakage_power__uW=leakage)

    @staticmethod
    def analyze_model(model: nn.Module) -> ProfilerReport:
        """Build a static-only ``ProfilerReport`` (no runtime events)."""
        return ProfilerReport(
            static_records=NeuroxProfiler.collect_static(model),
            static=NeuroxProfiler.analyze_static(model),
        )

    # --------------------------- Report -----------------------------

    def report(self, model: nn.Module) -> ProfilerReport:
        """Bundle this context's runtime events with the model's static state.

        Must be called after the ``with`` block exits — at that point
        ``__exit__`` has invoked ``_finalize`` and every aggregation is
        ready. Calling ``report()`` from inside the ``with`` block yields
        an incomplete report (pending events are still on the recording
        device).
        """
        return ProfilerReport(
            energy_events=list(self.energy_events),
            latency_events=list(self.latency_events),
            static_records=NeuroxProfiler.collect_static(model),
            static=NeuroxProfiler.analyze_static(model),
        )

    def summary(
        self,
        *,
        static: StaticMetrics | None = None,
        extras: Mapping[str, str | int | float] | None = None,
    ) -> str:
        """Format a concise multi-line summary.

        When ``static`` is supplied, leakage energy is added as
        ``static.leakage_power__uW * self.total_latency__ns``.
        """
        lines: list[str] = []
        if extras is not None:
            for k, v in extras.items():
                lines.append(f"{k}: {v}")
        lines.append(f"dynamic_energy_total_fJ: {self._total_dynamic_energy__fJ:.4f}")
        lines.append(f"latency_total_ns: {self._total_latency__ns:.4f}")
        if static is not None:
            leakage_energy__fJ = static.leakage_power__uW * self._total_latency__ns
            lines.append(f"area_total_um2: {static.area__um2:.4f}")
            lines.append(f"leakage_power_total_uW: {static.leakage_power__uW:.4f}")
            lines.append(f"leakage_energy_total_fJ: {leakage_energy__fJ:.4f}")
        return "\n".join(lines)
