"""Shared mixin for profile-event emission.

See also:
    docs/modules/profiler/README.md
"""

from __future__ import annotations

import torch
from torch import Tensor


class ProfileMixin:
    """Profile event emitter — name + identity + per-quantity log entries.

    Carries no PPA awareness. Subclasses (typically via
    :class:`CircuitBase`) build their own ``Tensor`` for each quantity
    they emit (dynamic energy and/or per-op latency) and route them
    through the two independent log entries below. Energy and latency
    are independent runtime quantities — each leaf decides whether to
    emit one, the other, both, or neither based on its physical model.

    Attributes:
        qualified_name: Hierarchical instance name.
        module_type: Short class-name tag included in every event.
    """

    def __init__(self, name: str) -> None:
        self._neurox_name = name

    @property
    def qualified_name(self) -> str:
        return self._neurox_name

    @property
    def module_type(self) -> str:
        return type(self).__name__

    @torch.compiler.disable
    def _log_dynamic_energy(self, dynamic_energy__fJ: Tensor) -> None:
        """Record one dynamic-energy event to the active profiler (no-op outside one).

        Caller-side construction: ``dynamic_energy__fJ = torch.full_like(y, per_op_energy__fJ)``
        for element-wise per-op energy, or any tensor reduced from the
        leaf's own physical model. The profiler does ``.detach().sum()``
        on entry and batches the GPU→CPU sync once at ``_finalize``.

        Args:
            dynamic_energy__fJ: Per-op switching energy tensor [fJ].
        """
        from neurox.common.profiler import NeuroxProfiler  # local import: avoid cycle

        profiler = NeuroxProfiler.get_current()
        if profiler is None:
            return
        profiler._record_energy(
            qualified_name=self._neurox_name,
            module_type=self.module_type,
            dynamic_energy__fJ=dynamic_energy__fJ,
        )

    @torch.compiler.disable
    def _log_latency(self, latency__ns: Tensor) -> None:
        """Record one latency event to the active profiler (no-op outside one).

        Caller-side construction: ``latency = torch.tensor(per_op_latency__ns *
        serial_op_count, device=..., dtype=...)``, where ``per_op_latency__ns``
        is leaf-defined — read from ``self.config.latency_per_op__ns`` for
        fixed-latency leaves, derived from runtime parameters for
        parametric leaves (e.g. SAR ADC's ``(bits + 1) × clk_period``).

        Args:
            latency__ns: Per-op latency contribution tensor [ns].
        """
        from neurox.common.profiler import NeuroxProfiler  # local import: avoid cycle

        profiler = NeuroxProfiler.get_current()
        if profiler is None:
            return
        profiler._record_latency(
            qualified_name=self._neurox_name,
            module_type=self.module_type,
            latency__ns=latency__ns,
        )
