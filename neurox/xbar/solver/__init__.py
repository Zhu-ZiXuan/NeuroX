"""Topology-agnostic SL/BL IR-drop DC solver — framework, nested impl, and primitives.

The :class:`Solver` base, its config / DCOP / residual containers, and the
:class:`NestedSolver` impl drive the two wire ladders and the two clamp
boundaries; the cell and both clamp drivers are per-call parameters of
:meth:`Solver.solve_dc`, so any SL/BL topology whose cell condenses to one
branch reuses the same solver. The :class:`ClampDriver` role — the
structural contract each boundary clamp satisfies — lives here beside its
only consumer (the solver). The shared numerical primitives
(:func:`col_driver_current`, :func:`col_wire_kcl_residual`,
:func:`solve_block_tridiagonal`) are re-exported here so
``from neurox.xbar.solver import ...`` resolves both the framework and the
primitives. Less-used primitives stay importable from
:mod:`neurox.xbar.solver.primitives`.
"""

from .base import Solver, SolverConfig, SolverDCOP, SolverResiduals
from .clamp import ClampDriver
from .nested import NestedSolver, NestedSolverConfig
from .primitives import (
    col_driver_current,
    col_wire_kcl_residual,
    solve_block_tridiagonal,
)

__all__ = [
    "ClampDriver",
    "NestedSolver",
    "NestedSolverConfig",
    "Solver",
    "SolverConfig",
    "SolverDCOP",
    "SolverResiduals",
    "col_driver_current",
    "col_wire_kcl_residual",
    "solve_block_tridiagonal",
]
