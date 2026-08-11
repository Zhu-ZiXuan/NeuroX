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

from .base import ModuleBase
from .mixin import ProfileMixin

_UNROOTED_PREFIX = "<unrooted>."


def neurox_roots(model: nn.Module) -> list[ModuleBase]:
    """Collect the outermost NeuroX modules ``model`` holds.

    The walk stops descending at the first :class:`ModuleBase` it meets, so a
    root covers its own NeuroX children instead of listing them beside it. A
    ``model`` that is itself a NeuroX module is the single root; a plain
    container or a third-party wrapper may hold several. Roots are deduplicated
    by identity, so a module bound under two parents — a tied or shared layer —
    is reported once, at its first appearance: a consumer summing over roots
    would otherwise count its hardware twice.

    Args:
        model: Tree to walk — a NeuroX module, or any ``nn.Module`` holding
            some.

    Returns:
        The outermost NeuroX modules, in child order.
    """
    if isinstance(model, ModuleBase):
        return [model]
    # dict keys preserve child order; nn.Module hashes by identity.
    roots: dict[ModuleBase, None] = {}
    for child in model.children():
        for root in neurox_roots(child):
            roots.setdefault(root, None)
    return list(roots)


def _energy_scalars(events: list["EnergyEvent"]) -> list[float]:
    """Total each event's energy tensor to a host float in one device sync.

    Event tensors keep their per-unit-operation layout on device; every scalar
    consumer (totals, per-name and per-type grouping) reduces through here so a
    whole report costs a single host transfer. Call it once per immutable event
    list and reuse the result — every call is a fresh sync.
    """
    if not events:
        return []
    return torch.stack([e.dynamic_energy__fJ.sum() for e in events]).cpu().tolist()


@dataclass(frozen=True)
class EnergyEvent:
    """One dynamic-energy event from a physical module's primary execution call.

    Attributes:
        module: The emitting module. A module never knows its own name — the
            hierarchical name is a property of the tree, resolved against a
            root by :meth:`ProfilerReport.energy_by_name`.
        module_type: Short class-name tag of the emitting module.
        dynamic_energy__fJ: Switching energy attributed to this call. Only the
            caller's leading dims survive, so each element is the energy of one
            unit operation; with no caller leading dims this is a 0-dim scalar.
            Excluded from equality: a multi-element tensor comparison is not a
            ``bool``, so an event compares on emitter identity and channel.
            Shape: ``[*caller_leading]``.
        channel: Optional sub-branch label the emitter passed to
            ``_record_dynamic_energy``; ``None`` for an un-channelled event.
    """

    module: ProfileMixin
    module_type: str
    dynamic_energy__fJ: Tensor = field(compare=False)
    channel: str | None = None


@dataclass(frozen=True)
class StaticRecord:
    """Per-module static-metric record built from fabrication-time state.

    Attributes:
        qualified_name: Hierarchical name, as the walked root names the module.
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

    Energy totals cover every unit operation the measurement executed.

    Attributes:
        energy_events: Dynamic-energy events captured during the profiler context.
        static_records: Per-module static-metric records.
        static: Aggregated static metrics across the whole model.
        qualified_names: Hierarchical dotted name of every profiled module the
            reported root reaches, as the root's own traversal names it. The
            root itself is named ``""``.
        energy_scalars__fJ: Per-event scalar totals, positionally aligned with
            ``energy_events``. :meth:`NeuroxProfiler.report` hands over the list
            the profiler already synced at context exit; left ``None`` it is
            computed on first read and cached, so every scalar view costs at
            most one host sync for the life of the report.
    """

    energy_events: list[EnergyEvent] = field(default_factory=list)
    static_records: list[StaticRecord] = field(default_factory=list)
    static: StaticMetrics = field(default_factory=StaticMetrics)
    qualified_names: dict[ProfileMixin, str] = field(default_factory=dict)
    energy_scalars__fJ: list[float] | None = None

    def _synced_scalars__fJ(self) -> list[float]:
        """Return the per-event scalar totals, syncing once if not already held."""
        if self.energy_scalars__fJ is None:
            self.energy_scalars__fJ = _energy_scalars(self.energy_events)
        return self.energy_scalars__fJ

    @property
    def total_dynamic_energy__fJ(self) -> float:
        """Dynamic energy summed over every event and every unit operation [fJ]."""
        return sum(self._synced_scalars__fJ(), 0.0)

    def name_of(self, module: ProfileMixin) -> str:
        """Return the module's root-relative name or an unrooted label."""
        name = self.qualified_names.get(module)
        return f"{_UNROOTED_PREFIX}{module.module_type}" if name is None else name

    @property
    def energy_by_name(self) -> dict[str, float]:
        """Dynamic energy grouped by qualified module name [fJ].

        A channelled event groups under ``"<module dotted name>.<channel>"``
        instead of the bare module name, so a composite's distinct billed
        branches appear as separate rows. Each row is a scalar total over the
        grouped events' unit operations.
        """
        by_name: dict[str, float] = {}
        for e, energy in zip(self.energy_events, self._synced_scalars__fJ(), strict=True):
            name = self.name_of(e.module)
            if e.channel is not None:
                name = f"{name}.{e.channel}"
            by_name[name] = by_name.get(name, 0.0) + energy
        return by_name


class NeuroxProfiler:
    """Context manager that captures physical-module side-channel events.

    Read totals or call :meth:`report` after leaving the context::

        with NeuroxProfiler() as profiler:
            model(...)
        print(profiler.total_dynamic_energy__fJ)
        report = profiler.report(model)

    Dynamic-energy payloads follow one repo-wide reduction rule: sum every axis
    past the caller's leading dims, keep the caller's leading dims. Everything
    after that prefix — the emitter's internal work axes (digit, phase, serial
    round, output, instance, …) — is summed, so an event tensor element is the
    energy of one unit operation rather than a figure already collapsed across
    the batch. A linear unit called with ``[G, T, B, input_num]`` under
    ``leading_rank=3`` therefore yields ``[G, T, B]``. A payload built by
    expanding a constant costs nothing extra: the expanded view holds no
    storage and reducing over its stride-0 axes allocates only the kept prefix.

    ``leading_rank`` is a property of the measurement, not of any module — only
    the measurement site can say how many leading dims its caller owns. It is a
    reporting-resolution knob, not a physical quantity: getting it wrong never
    changes a total (summation is summation, and every scalar view reduces the
    event to a scalar anyway), only how finely the per-unit-operation view
    resolves. There is no runtime check on it.

    Args:
        leading_rank: Number of leading dims the caller owns, ``0`` when the
            measured call has none.

    Attributes:
        energy_events: Captured dynamic-energy events, in emission order.
    """

    _local = threading.local()

    def __init__(self, *, leading_rank: int = 0) -> None:
        if leading_rank < 0:
            raise ValueError(f"leading_rank must be non-negative; got {leading_rank}")
        self._leading_rank = leading_rank
        self.energy_events: list[EnergyEvent] = []
        self._pending_energy: list[tuple[ProfileMixin, str | None, Tensor]] = []
        self._energy_scalars__fJ: list[float] = []
        self._total_dynamic_energy__fJ: float = 0.0

    def __enter__(self) -> Self:
        if NeuroxProfiler.get_current() is not None:
            raise RuntimeError("only one NeuroxProfiler may be active at a time")
        self.energy_events = []
        self._pending_energy = []
        self._energy_scalars__fJ = []
        self._total_dynamic_energy__fJ = 0.0
        NeuroxProfiler._local.current = self
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        NeuroxProfiler._local.current = None
        if exc_type is None:
            self._finalize()

    @classmethod
    def get_current(cls) -> Self | None:
        """Return the active profiler for this thread, or ``None``."""
        return getattr(cls._local, "current", None)

    @property
    def leading_rank(self) -> int:
        """Number of caller leading dims every energy payload keeps."""
        return self._leading_rank

    def _record_dynamic_energy(
        self,
        *,
        module: ProfileMixin,
        dynamic_energy__fJ: Tensor,
        channel: str | None = None,
    ) -> None:
        energy = self._lay_out_dynamic_energy(dynamic_energy__fJ=dynamic_energy__fJ)
        self._pending_energy.append((module, channel, energy))

    def _lay_out_dynamic_energy(self, *, dynamic_energy__fJ: Tensor) -> Tensor:
        """Reduce one energy payload onto ``[*caller_leading]``.

        Sums every axis past the caller's leading dims, whatever the emitter put
        there, and keeps the caller's leading dims untouched.
        """
        rank = self.leading_rank
        energy = dynamic_energy__fJ.detach()
        ndim = energy.ndim
        reduced = tuple(range(rank, ndim))
        if len(reduced) == ndim:
            energy = energy.sum()
        elif reduced:
            energy = energy.sum(dim=reduced)
        return energy

    def _finalize(self) -> None:
        for module, channel, energy in self._pending_energy:
            self.energy_events.append(
                EnergyEvent(
                    module=module,
                    module_type=module.module_type,
                    dynamic_energy__fJ=energy,
                    channel=channel,
                )
            )
        self._pending_energy = []
        self._energy_scalars__fJ = _energy_scalars(self.energy_events)
        self._total_dynamic_energy__fJ = sum(self._energy_scalars__fJ, 0.0)

    @property
    def total_dynamic_energy__fJ(self) -> float:
        """Sum of dynamic energy across all energy events and unit operations [fJ].

        Deliberately scalar: it reads as the total energy of this measurement,
        with no ambiguity about which axis a figure belongs to. Synced once at
        context exit, so reading it is a pure CPU field read.
        """
        return self._total_dynamic_energy__fJ

    @property
    def energy_by_type(self) -> dict[str, float]:
        """Dynamic energy grouped by module class name [fJ]."""
        by_type: dict[str, float] = {}
        for e, energy in zip(self.energy_events, self._energy_scalars__fJ, strict=True):
            by_type[e.module_type] = by_type.get(e.module_type, 0.0) + energy
        return by_type

    @staticmethod
    def collect_static(model: nn.Module) -> list[StaticRecord]:
        """Build a per-module ``StaticRecord`` list by walking ``model``.

        Static PPA is collected from :class:`ProfileMixin` hosts whose
        ``is_profile_target`` is ``True``.
        """
        return [
            StaticRecord(
                qualified_name=name,
                module_type=module.module_type,
                area__um2=module.area__um2,
                leakage_power__uW=module.leakage__uW,
            )
            for name, module in model.named_modules()
            if isinstance(module, ProfileMixin) and module.is_profile_target
        ]

    @staticmethod
    def analyze_static(model: nn.Module) -> StaticMetrics:
        """Aggregate total area and leakage power across all profiled modules."""
        area = 0.0
        leakage = 0.0
        for module in model.modules():
            if isinstance(module, ProfileMixin) and module.is_profile_target:
                area += module.area__um2
                leakage += module.leakage__uW
        return StaticMetrics(area__um2=area, leakage_power__uW=leakage)

    @staticmethod
    def analyze_model(model: nn.Module) -> ProfilerReport:
        """Build a static-only ``ProfilerReport`` (no runtime events)."""
        return ProfilerReport(
            static_records=NeuroxProfiler.collect_static(model),
            static=NeuroxProfiler.analyze_static(model),
        )

    def report(self, model: nn.Module) -> ProfilerReport:
        """Bundle this context's runtime events with the model's static state.

        Call this method after leaving the profiler context. ``model`` is the
        root used to derive hierarchical module names.
        """
        return ProfilerReport(
            energy_events=list(self.energy_events),
            static_records=NeuroxProfiler.collect_static(model),
            static=NeuroxProfiler.analyze_static(model),
            qualified_names={
                module: name for name, module in model.named_modules() if isinstance(module, ProfileMixin)
            },
            energy_scalars__fJ=list(self._energy_scalars__fJ),
        )

    def summary(
        self,
        *,
        static: StaticMetrics | None = None,
        extras: Mapping[str, str | int | float] | None = None,
    ) -> str:
        """Format a concise multi-line summary.

        Only the profiler's own totals appear here. A duration is asked of a
        module family through that family's own ``latency__ns``, which takes
        the arguments that family needs, so a caller that wants one states it
        among ``extras``. No static energy is derived either: the window
        leakage integrates over is a duty-cycle property a scheme declares in
        its own config.

        Args:
            static: Static totals to append.
            extras: Leading ``key: value`` lines.

        Returns:
            The summary text.
        """
        lines: list[str] = []
        if extras is not None:
            for k, v in extras.items():
                lines.append(f"{k}: {v}")
        lines.append(f"dynamic_energy_total_fJ: {self.total_dynamic_energy__fJ:.4f}")
        if static is not None:
            lines.append(f"area_total_um2: {static.area__um2:.4f}")
            lines.append(f"leakage_power_total_uW: {static.leakage_power__uW:.4f}")
        return "\n".join(lines)
