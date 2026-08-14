"""Shared interfaces for SL/BL IR-drop DC solvers.

See Also:
    docs/internals/primitive/xbar/solver/base.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

from neurox.common import ConfigBase, TensorDataClassBase
from neurox.primitive.xbar.cell import XbarCell, XbarCellDcop, XbarCellSnap

from .clamp import ClampDriver, ClampSnap


class SolverConfig(ConfigBase, ABC):
    """Base class for fixed DC-solver parameters."""

    def validate(self) -> None:
        """Validate parameter ranges; the base accepts every value."""


class SolverDcop[CellDcopT: XbarCellDcop](TensorDataClassBase):
    """Complete steady-state solution of one DC solve."""

    i_bl_driver: Tensor
    """BL driver current [uA]. Shape: `[..., num_col]`."""
    i_sl_driver: Tensor
    """SL driver current [uA]. Shape: `[..., num_col]`."""
    v_bl_node: Tensor
    """BL node voltages [V]. Shape: `[..., num_col, num_row]`."""
    v_sl_node: Tensor
    """SL node voltages [V]. Shape: `[..., num_col, num_row]`."""
    cell: CellDcopT
    """Condensed cell DC working point at the converged node voltages,
    including the internal node voltage."""
    v_bl_clamp: Tensor
    """BL clamp voltages [V]. Shape: `[..., num_col]`."""
    v_sl_drive: Tensor
    """SL drive voltages [V]. Shape: `[..., num_col]`."""


class Solver(ABC):
    """Base class for SL/BL IR-drop DC solvers.

    Args:
        config: Fixed numerical parameters for the concrete solver.
    """

    @abstractmethod
    def __init__(self, *, config: SolverConfig) -> None:
        raise NotImplementedError

    @abstractmethod
    def solve_dc[
        CellSnapT: XbarCellSnap,
        CellDcopT: XbarCellDcop,
        BLSnapT: ClampSnap,
        SLSnapT: ClampSnap,
    ](
        self,
        *,
        bl_segment_r__MOhm: float,
        sl_segment_r__MOhm: float,
        cell: XbarCell[Any, Any, CellSnapT, CellDcopT],
        cell_snap: CellSnapT,
        bl_driver: ClampDriver[BLSnapT],
        bl_driver_snap: BLSnapT,
        sl_driver: ClampDriver[SLSnapT],
        sl_driver_snap: SLSnapT,
    ) -> SolverDcop[CellDcopT]:
        """Solve the fabricated tile for one cell snap.

        The rail lattice is uniform: one link resistance describes a whole
        rail, the driver's own link to the node at index 0 included.

        Args:
            bl_segment_r__MOhm: BL rail resistance of one lattice link.
            sl_segment_r__MOhm: SL rail resistance of one lattice link.
            cell: Condensed cell branch model.
            cell_snap: Per-solve cell snap bundling the device snaps and the
                per-cell word-line drive at `[..., col, row]`.
            bl_driver: BL clamp driver.
            bl_driver_snap: Per-solve BL driver snap.
            sl_driver: SL clamp driver.
            sl_driver_snap: Per-solve SL driver snap.

        Returns:
            Complete steady-state solution for the current VMM.
        """
        raise NotImplementedError
