"""1T1R cell topology — physical core, offset xbar, and DC solver.

Subpackage holds every implementation file specific to the 1T1R cell
topology so the top-level :mod:`neurox.xbar` namespace stays small.
Python rejects ``1t1r`` as a package name (identifiers cannot start
with a digit), so this directory is named ``_1t1r``; the leading
underscore conventionally signals "internal organisation".

Users should reach the public names through :mod:`neurox.xbar` rather
than this subpackage path.
"""

from .core_1t1r import Core1T1R, Core1T1RConfig, Core1T1ROutput
from .offset_1t1r import Offset1T1RXbar, Offset1T1RXbarConfig
from .solver_1t1r import NewtonRaphsonSolver1T1R, SolverResult

__all__ = [
    "Core1T1R",
    "Core1T1RConfig",
    "Core1T1ROutput",
    "NewtonRaphsonSolver1T1R",
    "Offset1T1RXbar",
    "Offset1T1RXbarConfig",
    "SolverResult",
]
