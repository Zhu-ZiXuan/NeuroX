"""1T1R array unit — concrete 1T1R cell and its pure-array core.

The cell and core co-vary by array type: the core is built for its cell. The
shared :class:`~neurox.xbar.cell.XbarCell` ABC, the :class:`~neurox.xbar.Xbar`
ABC, and the parallel-rail solver stay at the layer level.
"""

from .cell import (
    XbarCell1T1R,
    XbarCell1T1RConfig,
    XbarCell1T1RDCOP,
    XbarCell1T1RPolicy,
    XbarCell1T1RResiduals,
    XbarCell1T1RSnap,
)
from .core import Core1T1R, Core1T1RConfig, Core1T1RPolicy, CoreSteadyState

__all__ = [
    "Core1T1R",
    "Core1T1RConfig",
    "Core1T1RPolicy",
    "CoreSteadyState",
    "XbarCell1T1R",
    "XbarCell1T1RConfig",
    "XbarCell1T1RDCOP",
    "XbarCell1T1RPolicy",
    "XbarCell1T1RResiduals",
    "XbarCell1T1RSnap",
]
