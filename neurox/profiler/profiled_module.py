"""Shared mixin for profile-capable physical modules.

The profiler is a logger-style side channel: numerical functions return clean
numerical results, while dynamic and static hardware metrics are emitted as
side effects via this mixin.  ``ProfiledModule`` provides the four contract
methods every physical module needs to participate:

1. ``__init__(name)`` — accept a hierarchical qualified name (e.g.
   ``fc1.macro.xbar.core.tia``) propagated downward by composite parents.
2. ``_record_inst_count(n)`` — called from ``fabricate(...)`` (or from the
   parent at fabricate time for modules without a per-op shape) with the
   number of physical replicas.  The mixin pre-multiplies per-instance area
   and leakage so ``analyze_static`` only has to sum.
3. ``_log_dynamic(dyn_energy__fJ, latency__ns)`` — append a runtime event
   to the active profiler, no-op when no profiler is bound.
4. expose ``area_per_inst__um2``, ``leakage_per_inst__uW``, and
   ``latency_per_op__ns`` for static aggregation.

The ``@torch.compiler.disable`` decorator on ``_log_dynamic`` lets dynamo
graph-break around the call site without aborting compilation of the
surrounding kernel — energy/latency capture in eager mode, transparent skip
inside compiled regions.
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

    Subclasses (typically also ``nn.Module``) must:

    * pass ``name`` into ``ProfiledModule.__init__`` from their own init;
    * expose ``area_per_inst__um2``, ``leakage_per_inst__uW``, and
      ``latency_per_op__ns`` (usually delegated to a frozen config);
    * call ``self._record_inst_count(n)`` whenever the number of physical
      instances changes (typically inside ``fabricate``);
    * call ``self._log_dynamic(dyn_energy__fJ, latency__ns)`` at the end
      of their primary execution method.

    Composite modules that have no extra contribution beyond their children
    still inherit this mixin so they can thread ``name`` to children;
    their ``_log_dynamic`` calls are simply omitted.

    Attributes:
        qualified_name: Hierarchical instance name, set at construction.
        module_type: Short class-name tag included in every event.
    """

    def __init__(self, name: str) -> None:
        self._neurox_name: str = name
        self._inst_count: int = 0
        self._inst_area__um2: float = 0.0
        self._inst_leakage__uW: float = 0.0

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

        Accepts either an integer count or a shape tuple; the shape's
        product is taken for tuple inputs (matching the *P fabricate
        shape used across the codebase).
        """
        count = n if isinstance(n, int) else math.prod(n)
        self._inst_count = count
        self._inst_area__um2 = self.area_per_inst__um2 * count  # type: ignore[attr-defined]
        self._inst_leakage__uW = self.leakage_per_inst__uW * count  # type: ignore[attr-defined]

    @torch.compiler.disable
    def _log_dynamic(self, dynamic_energy__fJ: float | Tensor, latency__ns: float = 0.0) -> None:
        """Append a runtime event to the active profiler (no-op outside one).

        Dynamo-disabled so call sites inside compiled regions graph-break
        cleanly around the side channel while the surrounding kernel still
        compiles.  Tensor energies are summed and ``.item()``-coerced here
        so callers can pass per-batch tensors without an extra reduction.
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
