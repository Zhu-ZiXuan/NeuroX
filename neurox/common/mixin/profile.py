"""Shared mixin for profile-event emission."""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import Tensor


class ProfileMixin:
    """Emit a host's dynamic PPA events and aggregate its static PPA.

    Two emit hooks — ``_log_dynamic_energy`` and ``_log_latency`` — attribute a
    leaf's own runtime energy and latency to the active profiler, which records
    the emitting module itself. Naming is not the emitter's business: the
    reporting side derives every hierarchical name from the module tree. For
    dynamic PPA the mixin only routes a caller-built tensor, so every emit is a
    pure side channel — a no-op outside a profiler that never alters host
    numerics. For static PPA it owns the aggregation: ``area__um2`` and
    ``leakage__uW`` scale the host-provided per-instance data by ``inst_count``.
    The collector half — capture, batched sync, aggregation, the static walk —
    lives in the profiler.

    ``is_profile_target`` gates a host's whole profiling role, both sides at
    once. Static: the profiler's static walk collects area / leakage only from
    hosts where it is ``True``. Dynamic: a non-target must never emit — both
    emit hooks raise if called on one. A non-reporter whose silicon rolls up
    into an owner's budget overrides it to ``False``, and is then absent from
    the static walk and forbidden to emit.

    Host requirements:
        - Inherit ``nn.Module`` alongside this mixin, and be reachable from
          the reported root as a registered child — an emitter the traversal
          cannot reach has no name and reports under an ``<unrooted>``
          placeholder. Holding an emitter in a plain container instead of
          binding it as an attribute hides it from the traversal.
        - Call ``_log_dynamic_energy`` / ``_log_latency`` at the end of the
          primary method, after all kernel math, once the output tensor
          exists, and at most once each per logical operation.
        - Own the emit decision: the hooks apply no value-based gating, so call
          each only for a quantity this leaf models, and guard a conditional
          emit at the call site — pushing a zero through unconditionally
          records a spurious event.
        - Set the bare ``_area_per_inst__um2`` and ``_leakage_per_inst__uW``
          per-instance data (a leaf, in its own ``__init__``) and expose
          ``inst_count``; unset per-instance data raises ``AttributeError`` on
          the first static-PPA read.
    """

    # Host ModuleBase leaf sets these bare per-inst data; the mixin aggregates by inst_count.
    _area_per_inst__um2: float
    _leakage_per_inst__uW: float

    # Two-way profiling gate: static walk collects PPA only where True; a non-target must never emit (both hooks raise). Roll-up non-reporters override to False.
    is_profile_target: ClassVar[bool] = True

    @property
    def inst_count(self) -> int:
        raise NotImplementedError

    @property
    def module_type(self) -> str:
        """Short class-name tag (``type(self).__name__``) recorded on each event."""
        return type(self).__name__

    @property
    def area__um2(self) -> float:
        """Total static area: per-instance area scaled by ``inst_count``."""
        return self._area_per_inst__um2 * self.inst_count

    @property
    def leakage__uW(self) -> float:
        """Total static leakage: per-instance leakage scaled by ``inst_count``."""
        return self._leakage_per_inst__uW * self.inst_count

    @torch.compiler.disable
    def _log_dynamic_energy(self, dynamic_energy__fJ: Tensor) -> None:
        """Record one dynamic-energy event to the active profiler (no-op outside one).

        Caller-side construction: ``dynamic_energy__fJ = torch.full_like(y, per_op_energy__fJ)``
        for element-wise per-op energy, or any tensor reduced from the
        leaf's own physical model.

        Args:
            dynamic_energy__fJ: Per-op switching energy tensor.
        """
        if not self.is_profile_target:
            raise RuntimeError(f"{type(self).__name__} is not a profile target but emitted a dynamic-energy event")
        from neurox.common.profiler import NeuroxProfiler  # local import: avoid cycle

        profiler = NeuroxProfiler.get_current()
        if profiler is None:
            return
        profiler._record_energy(module=self, dynamic_energy__fJ=dynamic_energy__fJ)

    @torch.compiler.disable
    def _log_latency(self, latency__ns: Tensor) -> None:
        """Record one latency event to the active profiler (no-op outside one).

        Caller-side construction: ``latency = torch.tensor(per_op_latency__ns *
        serial_op_count, device=..., dtype=...)``, where ``per_op_latency__ns``
        is leaf-defined — read from ``self.config.latency_per_op__ns`` for
        fixed-latency leaves, derived from runtime parameters for
        parametric leaves (e.g. a per-conversion latency built from a
        bit count and a clock period).

        Args:
            latency__ns: Per-op latency contribution tensor.
        """
        if not self.is_profile_target:
            raise RuntimeError(f"{type(self).__name__} is not a profile target but emitted a latency event")
        from neurox.common.profiler import NeuroxProfiler  # local import: avoid cycle

        profiler = NeuroxProfiler.get_current()
        if profiler is None:
            return
        profiler._record_latency(module=self, latency__ns=latency__ns)
