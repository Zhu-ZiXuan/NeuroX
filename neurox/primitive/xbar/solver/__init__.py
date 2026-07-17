"""Topology-agnostic SL/BL IR-drop DC solver: framework, nested impl, and numerical helpers."""

from .base import Solver, SolverConfig, SolverDcop, SolverResiduals
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
    "SolverDcop",
    "SolverResiduals",
    "classify_leading_positions",
    "iter_chunks",
    "reassemble_chunks",
]
