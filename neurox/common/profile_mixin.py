"""Shared mixin for profile-record emission."""

from __future__ import annotations

from typing import ClassVar, final

import torch.nn as nn
from torch import Tensor

from neurox.api.profiler import Profiler


class ProfileMixin:
    """Add static PPA properties and profile-record hooks to an `nn.Module`.

    A host must also inherit `torch.nn.Module`. A module whose silicon is
    counted at its owner sets `is_profile_target = False` and implements
    neither `_area_per_inst__um2` nor `_leakage_per_inst__uW`; every other host
    implements both.
    """

    is_profile_target: ClassVar[bool] = True

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if not issubclass(cls, nn.Module):
            raise TypeError(f"{cls.__qualname__} must also inherit torch.nn.Module")
        if not cls.is_profile_target:
            declared = [name for name in ("_area_per_inst__um2", "_leakage_per_inst__uW") if name in cls.__dict__]
            if declared:
                raise TypeError(f"{cls.__qualname__} sets is_profile_target = False but declares {', '.join(declared)}")

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
    def qualified_name(self) -> str:
        raise NotImplementedError

    @property
    @final
    def area__um2(self) -> float:
        if not self.is_profile_target:
            raise RuntimeError(f"{type(self).__qualname__} is not a profile target and reports no area")
        return self._area_per_inst__um2 * self.inst_count

    @property
    @final
    def leakage__uW(self) -> float:
        if not self.is_profile_target:
            raise RuntimeError(f"{type(self).__qualname__} is not a profile target and reports no leakage")
        return self._leakage_per_inst__uW * self.inst_count

    @final
    def _is_dynamic_energy_profile_active(self) -> bool:
        return Profiler.active()

    @final
    def _record_dynamic_energy(self, dynamic_energy__fJ: Tensor, *, channel: str | None = None) -> None:
        """Submit this call's dynamic energy to the active profiler.

        A profile target emits nothing outside a profiling context. Compute
        billed energy under `torch.no_grad()` or an equivalent guard.

        Preserve the caller's leading axes at their full extents; reassemble
        chunked results before submitting them. The profiler retains those
        axes and sums all trailing axes, so include each billed physical
        instance and access exactly once. Constant energy may be expanded
        from a scalar without materializing the billed layout.

        Args:
            dynamic_energy__fJ: Energy laid out over the caller's full leading
                extents and billed trailing axes. Expanded views are supported;
                singleton dimensions do not request implicit broadcasting.
                Shape: `[*caller_leading, ...]`.
            channel: Optional virtual submodule to bill under.

        Raises:
            RuntimeError: The emitter is not a profile target, or it carries no
                name stamp while a profiler is collecting.
        """
        if not self.is_profile_target:
            raise RuntimeError(f"{type(self).__name__} is not a profile target but emitted dynamic energy")
        ledger = Profiler.current()
        if ledger is None:
            return
        Profiler.submit(
            ledger.lay_out(
                qualified_name=self.qualified_name,
                dynamic_energy__fJ=dynamic_energy__fJ,
                channel=channel,
            )
        )
