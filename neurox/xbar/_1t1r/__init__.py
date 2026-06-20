"""1T1R cell topology — condensed branch cell, physical core, and offset xbar.

The DC solver is topology-agnostic and lives in :mod:`neurox.xbar.solver`
(``Solver`` / ``NestedSolver``); the 1T1R cell supplies the per-call branch
the solver drives.
"""

from .cell import (
    XbarCell1T1R,
    XbarCell1T1RConfig,
    XbarCell1T1RDCOP,
    XbarCell1T1RPolicy,
    XbarCell1T1RResiduals,
    XbarCell1T1RSnap,
)
from .circuit_core import (
    CircuitCore1T1R,
    CircuitCore1T1RConfig,
    CircuitCore1T1RPolicy,
)
from .offset import (
    Offset1T1RXbar,
    Offset1T1RXbarConfig,
    Offset1T1RXbarPolicy,
)

__all__ = [
    "CircuitCore1T1R",
    "CircuitCore1T1RConfig",
    "CircuitCore1T1RPolicy",
    "Offset1T1RXbar",
    "Offset1T1RXbarConfig",
    "Offset1T1RXbarPolicy",
    "XbarCell1T1R",
    "XbarCell1T1RConfig",
    "XbarCell1T1RDCOP",
    "XbarCell1T1RPolicy",
    "XbarCell1T1RResiduals",
    "XbarCell1T1RSnap",
]
