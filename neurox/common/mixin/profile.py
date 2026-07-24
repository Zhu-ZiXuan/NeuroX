"""Shared mixin for profile-event emission."""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn
from torch import Tensor


class ProfileMixin:
    """Add static PPA properties and profile-event hooks to an ``nn.Module``.

    Host requirements:
        - Also inherit :class:`torch.nn.Module`.
        - Expose ``inst_count``.
        - A profile target must initialize
          ``_area_per_inst__um2`` and ``_leakage_per_inst__uW`` before static
          metrics are read.
        - A non-target must not emit profile events.
    """

    _area_per_inst__um2: float
    _leakage_per_inst__uW: float

    is_profile_target: ClassVar[bool] = True

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if not issubclass(cls, nn.Module):
            raise TypeError(f"{cls.__qualname__} must also inherit torch.nn.Module")

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
    def _record_dynamic_energy(self, dynamic_energy__fJ: Tensor, *, channel: str | None = None) -> None:
        """Record one dynamic-energy event to the active profiler (no-op outside one).

        Args:
            dynamic_energy__fJ: Per-op switching energy tensor.
            channel: Optional energy-branch label.
        """
        if not self.is_profile_target:
            raise RuntimeError(f"{type(self).__name__} is not a profile target but emitted a dynamic-energy event")
        from neurox.common.profiler import NeuroxProfiler

        profiler = NeuroxProfiler.get_current()
        if profiler is None:
            return
        profiler._record_dynamic_energy(module=self, dynamic_energy__fJ=dynamic_energy__fJ, channel=channel)

    @torch.compiler.disable
    def _record_latency(self, latency__ns: Tensor) -> None:
        """Record one latency event to the active profiler (no-op outside one).

        Args:
            latency__ns: Per-op latency contribution tensor.
        """
        if not self.is_profile_target:
            raise RuntimeError(f"{type(self).__name__} is not a profile target but emitted a latency event")
        from neurox.common.profiler import NeuroxProfiler

        profiler = NeuroxProfiler.get_current()
        if profiler is None:
            return
        profiler._record_latency(module=self, latency__ns=latency__ns)
