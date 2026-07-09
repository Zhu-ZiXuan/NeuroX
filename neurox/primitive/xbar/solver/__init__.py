"""Topology-agnostic SL/BL IR-drop DC solver — framework, nested impl, and numerical helpers.

The :class:`Solver` base, its config / DCOP / residual containers, and the
:class:`NestedParallelRailSolver` impl drive the two wire ladders and the two clamp
boundaries; the cell and both clamp drivers are per-call parameters of
:meth:`Solver.solve_dc`, so any SL/BL topology whose cell condenses to one
branch reuses the same solver. The :class:`ClampDriver` role — the
structural contract each boundary clamp satisfies — lives here beside its
only consumer (the solver). The numerical solves
(:func:`solve_block_tridiagonal` and its siblings) stay in
:mod:`neurox.primitive.xbar.solver._linalg`, and the wire-KCL builders
(:func:`col_driver_current`, :func:`col_wire_kcl_residual`) in
:mod:`neurox.primitive.xbar.solver._wire_kcl`; the package surface re-exports
neither, so import them straight from those modules. The leading-batch chunking helpers
(:func:`classify_leading_positions`, :func:`iter_chunks`,
:func:`reassemble_chunks`, :class:`ChunkSpec`) — the per-chunk peak-memory
mechanism any solver-driven core uses to stream its leading batch — are
re-exported here too.
"""

from .base import Solver, SolverConfig, SolverDCOP, SolverResiduals
from .chunking import (
    ChunkSpec,
    classify_leading_positions,
    iter_chunks,
    reassemble_chunks,
)
from .clamp import ClampDriver
from .nested import NestedParallelRailSolver, NestedParallelRailSolverConfig

__all__ = [
    "ChunkSpec",
    "ClampDriver",
    "NestedParallelRailSolver",
    "NestedParallelRailSolverConfig",
    "Solver",
    "SolverConfig",
    "SolverDCOP",
    "SolverResiduals",
    "classify_leading_positions",
    "iter_chunks",
    "reassemble_chunks",
]
