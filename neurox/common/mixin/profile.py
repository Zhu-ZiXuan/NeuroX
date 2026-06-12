"""Shared mixin for profile-capable physical modules.

See also:
    docs/dev/modules/profiler/README.md
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


class ProfileMixin:
    """Mixin for any physical module that emits profiler events.

    Subclasses must:

    * pass ``name`` into ``ProfileMixin.__init__`` from their own init;
    * set ``self._inst_shape: tuple[int, ...]`` in their own init;
    * expose ``area_per_inst__um2``, ``leakage_per_inst__uW``,
      ``latency_per_op__ns`` properties;
    * call ``self._log_static()`` once at the end of the most-derived
      concrete ``__init__``, after ``self.config`` and ``self._inst_shape``
      are set;
    * call ``self._log_dynamic(dyn_energy__fJ, latency__ns)`` at the end
      of each primary execution method.

    Attributes:
        qualified_name: Hierarchical instance name.
        module_type: Short class-name tag included in every event.
    """

    _inst_shape: tuple[int, ...]

    def __init__(self, name: str) -> None:
        self._neurox_name = name
        # `_inst_count` / `_inst_area__um2` / `_inst_leakage__uW` are intentionally
        # NOT pre-initialised — they are populated by ``_log_static()`` and accessing
        # them earlier surfaces as ``AttributeError`` so a forgotten call is loud.

    @property
    def qualified_name(self) -> str:
        return self._neurox_name

    @property
    def module_type(self) -> str:
        return type(self).__name__

    @property
    def inst_shape(self) -> tuple[int, ...]:
        """Per-instance fabrication shape declared at construction.

        Public read-only view over ``self._inst_shape`` — tools and
        callers should consult this instead of poking at the private
        backing attribute.
        """
        return self._inst_shape

    @property
    def inst_count(self) -> int:
        """Total fabrication instance count; set by :meth:`_log_static`."""
        return self._inst_count

    @property
    def inst_area__um2(self) -> float:
        """Cached ``area_per_inst__um2 * inst_count``; set by :meth:`_log_static`."""
        return self._inst_area__um2

    @property
    def inst_leakage__uW(self) -> float:
        """Cached ``leakage_per_inst__uW * inst_count``; set by :meth:`_log_static`."""
        return self._inst_leakage__uW

    def _log_static(self) -> None:
        """Cache static PPA from ``self._inst_shape`` × per-instance values."""
        count = math.prod(self._inst_shape)
        self._inst_count = count
        self._inst_area__um2 = self.area_per_inst__um2 * count  # type: ignore[attr-defined]
        self._inst_leakage__uW = self.leakage_per_inst__uW * count  # type: ignore[attr-defined]

    @torch.compiler.disable
    def _log_dynamic(self, dynamic_energy__fJ: float | Tensor, latency__ns: float = 0.0) -> None:
        """Append a runtime event to the active profiler (no-op outside one).

        Tensor energies are summed and ``.item()``-coerced internally.
        """
        from neurox.common.profiler import NeuroxProfiler  # local import: avoid cycle

        profiler = NeuroxProfiler.get_current()
        if profiler is None:
            return
        if isinstance(dynamic_energy__fJ, torch.Tensor):
            dynamic_energy__fJ = float(dynamic_energy__fJ.detach().sum().item())
        profiler._append_runtime_event(
            qualified_name=self._neurox_name,
            module_type=self.module_type,
            dynamic_energy__fJ=dynamic_energy__fJ,
            latency__ns=latency__ns,
        )
