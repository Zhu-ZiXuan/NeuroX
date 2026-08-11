"""Shared mixin for profile-event emission."""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn
from torch import Tensor


class ProfileMixin:
    """Add static PPA properties and profile-event hooks to an ``nn.Module``.

    Energy payloads follow the profiler's single reduction rule — sum every axis
    past the caller's leading dims, keep the caller's leading dims; see
    :class:`neurox.common.profiler.NeuroxProfiler`. An emitter declares nothing
    about its own axes: it lays the caller's leading dims out first and puts its
    internal work axes after them.

    Host requirements:
        - Also inherit :class:`torch.nn.Module`.
        - Expose ``inst_count``.
        - A reporting module implements ``_area_per_inst__um2`` and
          ``_leakage_per_inst__uW``; a module whose silicon is counted at its
          owner implements neither.
        - A non-target must not emit profile events.
    """

    is_profile_target: ClassVar[bool] = True

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if not issubclass(cls, nn.Module):
            raise TypeError(f"{cls.__qualname__} must also inherit torch.nn.Module")

    @property
    def inst_count(self) -> int:
        raise NotImplementedError

    @property
    def _area_per_inst__um2(self) -> float:
        raise NotImplementedError

    @property
    def _leakage_per_inst__uW(self) -> float:
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
    def _is_dynamic_energy_profile_active(self) -> bool:
        """Return whether the active profiler requests dynamic energy."""
        from neurox.common.profiler import NeuroxProfiler

        profiler = NeuroxProfiler.get_current()
        return profiler is not None

    @torch.compiler.disable
    def _record_dynamic_energy(self, dynamic_energy__fJ: Tensor, *, channel: str | None = None) -> None:
        """Record one dynamic-energy event to the active profiler (no-op outside one).

        Everything past the caller's leading dims is summed, so the payload's
        own work axes need no declaration. The payload MUST carry the caller's
        leading dims at their true extents; a size-1 stand-in is a contract
        violation, not a broadcast request.

        The energy tensor is the emitter's to build. A flat per-op lump is a
        0-dim constant expanded onto the billed layout: the expanded view holds
        no storage and the profiler's reduction over its stride-0 axes
        allocates only ``[*caller_leading]``, so nothing is materialized.

        Args:
            dynamic_energy__fJ: Per-op dynamic energy [fJ].
                Shape: ``[*caller_leading, ...]``.
            channel: Optional energy-branch label.
        """
        if not self.is_profile_target:
            raise RuntimeError(f"{type(self).__name__} is not a profile target but emitted a dynamic-energy event")
        from neurox.common.profiler import NeuroxProfiler

        profiler = NeuroxProfiler.get_current()
        if profiler is None:
            return
        profiler._record_dynamic_energy(module=self, dynamic_energy__fJ=dynamic_energy__fJ, channel=channel)
