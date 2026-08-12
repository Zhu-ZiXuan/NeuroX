"""Shared mixin for profile-record emission."""

from __future__ import annotations

from typing import ClassVar

import torch.nn as nn
from torch import Tensor

from .profiler import Profiler


class ProfileMixin:
    """Add static PPA properties and profile-record hooks to an ``nn.Module``.

    An emitted energy tensor follows the profiler's single reduction rule — sum
    every axis past the caller's leading dims, keep the caller's leading dims; see
    :class:`neurox.common.profiler.Profiler`. An emitter declares nothing about
    its own axes: it lays the caller's leading dims out first and puts its
    internal work axes after them.

    A record carries the emitter's name, which the module holds only once the
    tree stamped it — see :func:`neurox.common.tree.stamp_names`.

    Host requirements:
        - Also inherit :class:`torch.nn.Module`.
        - Expose ``inst_count``.
        - A reporting module implements ``_area_per_inst__um2`` and
          ``_leakage_per_inst__uW``; a module whose silicon is counted at its
          owner implements neither.
        - A non-target must not emit dynamic energy.
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
            RuntimeError: The module was never stamped, so no tree has named it
                yet.
        """
        try:
            return self.__qualified_name
        except AttributeError:
            raise RuntimeError(
                f"{type(self).__name__} carries no name stamp; "
                "call neurox.common.tree.stamp_names(model) once the model is assembled"
            ) from None

    def _stamp(self, name: str) -> None:
        """Record the hierarchical name a tree walk gave this module.

        Called by :func:`neurox.common.tree.stamp_names` only; a later call
        overwrites, which is how a rewired model is renamed.

        Args:
            name: Hierarchical name, verbatim from the walk.
        """
        self.__qualified_name = name

    @property
    def area__um2(self) -> float:
        """Total static area: per-instance area scaled by ``inst_count``."""
        return self._area_per_inst__um2 * self.inst_count

    @property
    def leakage__uW(self) -> float:
        """Total static leakage: per-instance leakage scaled by ``inst_count``."""
        return self._leakage_per_inst__uW * self.inst_count

    def _is_dynamic_energy_profile_active(self) -> bool:
        """Return whether a profiler is collecting dynamic energy.

        Note:
            The answer comes from outside the caller's graph, so an emitter
            gating its billing work on it breaks the graph there and gets the
            live answer on every call rather than the one that held when the
            region was first compiled.
        """
        return Profiler.active()

    def _record_dynamic_energy(self, dynamic_energy__fJ: Tensor, *, channel: str | None = None) -> None:
        """Record one dynamic-energy tensor to the active profiler (no-op outside one).

        This hook derives no physics and owns no layout rule: it invokes the
        active ledger's declared layout, which folds everything past the
        caller's leading dims, so the emitter's own work axes need no
        declaration. The tensor MUST carry the caller's leading dims at their
        true extents; a size-1 stand-in is a contract violation, not a broadcast
        request.

        The energy tensor is the emitter's to build. A flat per-op lump is a
        0-dim constant expanded onto the billed layout: the expanded view holds
        no storage and the layout's reduction over its stride-0 axes allocates
        only ``[*caller_leading]``, so nothing is materialized.

        Args:
            dynamic_energy__fJ: Per-op dynamic energy [fJ].
                Shape: ``[*caller_leading, ...]``.
            channel: Optional virtual submodule name to bill under.

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
