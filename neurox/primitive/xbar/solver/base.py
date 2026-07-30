"""Shared interfaces for SL/BL IR-drop DC solvers.

See also:
    docs/internals/primitive/xbar/solver.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from torch import Tensor

from neurox.common import ConfigBase
from neurox.primitive.xbar.cell import XbarCell, XbarCellDcop, XbarCellSnap

from .clamp import ClampDriver, ClampSnap

CellSnapT = TypeVar("CellSnapT", bound=XbarCellSnap)
CellDCOPT = TypeVar("CellDCOPT", bound=XbarCellDcop)
BLSnapT = TypeVar("BLSnapT", bound=ClampSnap)
SLSnapT = TypeVar("SLSnapT", bound=ClampSnap)


class SolverConfig(ConfigBase, ABC):
    """Base class for fixed DC-solver parameters."""

    def validate(self) -> None:
        """Hook for subclasses to enforce parameter ranges."""


@dataclass(frozen=True)
class SolverDcop(Generic[CellDCOPT]):
    """Complete steady-state solution of one DC solve.

    Attributes:
        i_bl_driver: BL driver current [uA].
            Shape: ``[..., num_col]``.
        i_sl_driver: SL driver current [uA].
            Shape: ``[..., num_col]``.
        v_bl_node: BL node voltages [V].
            Shape: ``[..., num_col, num_row]``.
        v_sl_node: SL node voltages [V].
            Shape: ``[..., num_col, num_row]``.
        cell: Condensed cell DC working point at the converged node
            voltages, including the internal node voltage.
        v_bl_clamp: BL clamp voltages [V].
            Shape: ``[..., num_col]``.
        v_sl_drive: SL drive voltages [V].
            Shape: ``[..., num_col]``.
    """

    i_bl_driver: Tensor
    i_sl_driver: Tensor
    v_bl_node: Tensor
    v_sl_node: Tensor
    cell: CellDCOPT
    v_bl_clamp: Tensor
    v_sl_drive: Tensor


class Solver(ABC):
    """Base class for SL/BL IR-drop DC solvers.

    Args:
        config: Fixed numerical parameters for the concrete solver.
    """

    @abstractmethod
    def __init__(self, *, config: SolverConfig) -> None:
        raise NotImplementedError

    @abstractmethod
    def solve_dc(
        self,
        *,
        bl_segment_r__MOhm: Tensor,
        sl_segment_r__MOhm: Tensor,
        bl_segment_g__uS: Tensor,
        sl_segment_g__uS: Tensor,
        cell: XbarCell[Any, Any, CellSnapT, CellDCOPT],
        cell_snap: CellSnapT,
        bl_driver: ClampDriver[BLSnapT],
        bl_driver_snap: BLSnapT,
        sl_driver: ClampDriver[SLSnapT],
        sl_driver_snap: SLSnapT,
    ) -> SolverDcop[CellDCOPT]:
        """Solve the fabricated tile for one cell snap.

        Args:
            bl_segment_r__MOhm: BL segment resistances; index 0 is
                driver-to-first.
                Shape: ``[num_row]``.
            sl_segment_r__MOhm: SL segment resistances; index 0 is
                driver-to-first.
                Shape: ``[num_row]``.
            bl_segment_g__uS: BL segment conductances, reciprocal of
                ``bl_segment_r__MOhm``.
                Shape: ``[num_row]``.
            sl_segment_g__uS: SL segment conductances, reciprocal of
                ``sl_segment_r__MOhm``.
                Shape: ``[num_row]``.
            cell: Condensed cell branch model.
            cell_snap: Per-solve cell snap bundling the device snaps and the
                per-row control-line (WL) drive.
            bl_driver: BL clamp driver.
            bl_driver_snap: Per-solve BL driver snap.
            sl_driver: SL clamp driver.
            sl_driver_snap: Per-solve SL driver snap.

        Returns:
            Complete steady-state solution for the current VMM.
        """
        raise NotImplementedError
