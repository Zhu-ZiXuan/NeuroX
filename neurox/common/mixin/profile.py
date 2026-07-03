"""Shared mixin for profile-event emission.

See also:
    docs/internals/common/mixin/profile.md
"""

from __future__ import annotations

import torch
from torch import Tensor


class ProfileMixin:
    """Emit a host's dynamic PPA events into the active profiler.

    A host gets a hierarchical dotted instance name plus two emit hooks —
    ``_log_dynamic_energy`` and ``_log_latency`` — through which a leaf
    attributes its own runtime energy and latency to the active profiler. The
    mixin is physics-agnostic: it carries no PPA fields and only routes a
    caller-built tensor, so every emit is a pure side channel — a no-op outside
    a profiler that never alters host numerics. The collector half — capture,
    batched sync, aggregation, the static walk — lives in the profiler.

    Host requirements:
        - Inherit ``nn.Module`` alongside this mixin, so a profiled instance
          lives in the module tree the collector walks.
        - Thread the owner-supplied ``name`` to this mixin's ``__init__``
          (typically via ``super().__init__``). The parent composes it as a
          dotted hierarchical path; the mixin stores it verbatim and neither
          validates nor transforms it, so the owner owns uniqueness — a
          duplicated or omitted prefix yields colliding names that silently
          merge two emitters.
    """

    def __init__(self, name: str) -> None:
        self._neurox_name = name

    @property
    def qualified_name(self) -> str:
        """Hierarchical dotted instance name, fixed at construction.

        Carried as the emitter identity in every event.
        """
        return self._neurox_name

    @property
    def module_type(self) -> str:
        """Short class-name tag (``type(self).__name__``) emitted with the name.

        Lets the collector group by type as well as by name; it is the concrete
        class, so renaming the class changes the grouping key.
        """
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
        parametric leaves (e.g. a per-conversion latency built from a
        bit count and a clock period).

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
