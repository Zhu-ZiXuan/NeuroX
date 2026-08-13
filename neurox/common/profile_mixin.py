"""Shared mixin for profile-record emission."""

from __future__ import annotations

from typing import ClassVar

import torch.nn as nn
from torch import Tensor

from .profiler import Profiler


class ProfileMixin:
    """Add static PPA properties and profile-record hooks to an `nn.Module`.

    A host must also inherit `torch.nn.Module`. A module whose silicon is
    counted at its owner implements neither `_area_per_inst__um2` nor
    `_leakage_per_inst__uW`.
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
    def qualified_name(self) -> str:
        """Hierarchical name the module's tree stamped onto it.

        Raises:
            RuntimeError: No tree has stamped this module yet.
        """
        try:
            return self.__qualified_name
        except AttributeError:
            raise RuntimeError(
                f"{type(self).__name__} carries no name stamp; "
                "call neurox.common.tree.stamp_names(model) once the model is assembled"
            ) from None

    def _stamp(self, name: str) -> None:
        """Record the hierarchical name a tree walk gave this module."""
        self.__qualified_name = name

    @property
    def area__um2(self) -> float:
        return self._area_per_inst__um2 * self.inst_count

    @property
    def leakage__uW(self) -> float:
        return self._leakage_per_inst__uW * self.inst_count

    def _is_dynamic_energy_profile_active(self) -> bool:
        """Return whether a profiler is collecting dynamic energy."""
        return Profiler.active()

    def _record_dynamic_energy(self, dynamic_energy__fJ: Tensor, *, channel: str | None = None) -> None:
        """Record one dynamic-energy tensor to the active profiler (no-op outside one).

        The tensor MUST carry the caller's leading dims at their true extents;
        a size-1 stand-in is a contract violation, not a broadcast request.

        The energy tensor is the emitter's to build. A flat per-op lump is a
        0-dim constant expanded onto the billed layout: the expanded view holds
        no storage and the fold over its stride-0 axes allocates only
        `[*caller_leading]`, so nothing is materialized.

        Args:
            dynamic_energy__fJ: Dynamic energy of this call as the emitter
                billed it. Shape: `[*caller_leading, ...]`.
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
