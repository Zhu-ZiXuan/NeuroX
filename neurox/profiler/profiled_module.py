"""Shared mixin for profile-capable physical modules.

See also:
    docs/dev/modules/profiler/README.md
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


class ProfiledModule:
    """Mixin for any physical module that emits profiler events.

    Subclasses must:

    * pass ``name`` into ``ProfiledModule.__init__`` from their own init;
    * expose ``area_per_inst__um2``, ``leakage_per_inst__uW``,
      ``latency_per_op__ns`` properties;
    * call ``self._record_inst_count(n)`` whenever the physical replica
      count changes (typically in ``fabricate``);
    * call ``self._log_dynamic(dyn_energy__fJ, latency__ns)`` at the end
      of each primary execution method.

    Attributes:
        qualified_name: Hierarchical instance name.
        module_type: Short class-name tag included in every event.
    """

    def __init__(self, name: str) -> None:
        self._neurox_name = name
        self._inst_count = 0
        self._inst_area__um2 = 0.0
        self._inst_leakage__uW = 0.0

    @property
    def qualified_name(self) -> str:
        return self._neurox_name

    @property
    def module_type(self) -> str:
        return type(self).__name__

    @property
    def inst_count(self) -> int:
        """Number of physical replicas recorded by ``_record_inst_count``."""
        return self._inst_count

    @property
    def inst_area__um2(self) -> float:
        """Cached ``area_per_inst__um2 * inst_count`` (fJ-free total)."""
        return self._inst_area__um2

    @property
    def inst_leakage__uW(self) -> float:
        """Cached ``leakage_per_inst__uW * inst_count``."""
        return self._inst_leakage__uW

    def _record_inst_count(self, n: int | Sequence[int]) -> None:
        """Pre-multiply per-instance area and leakage by the replica count.

        Accepts an integer count or a shape tuple (product of the shape).
        """
        count = n if isinstance(n, int) else math.prod(n)
        self._inst_count = count
        self._inst_area__um2 = self.area_per_inst__um2 * count  # type: ignore[attr-defined]
        self._inst_leakage__uW = self.leakage_per_inst__uW * count  # type: ignore[attr-defined]

    @torch.compiler.disable
    def _log_dynamic(self, dynamic_energy__fJ: float | Tensor, latency__ns: float = 0.0) -> None:
        """Append a runtime event to the active profiler (no-op outside one).

        Tensor energies are summed and ``.item()``-coerced internally.
        """
        from .profiler import NeuroxProfiler  # local import: avoid cycle

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
