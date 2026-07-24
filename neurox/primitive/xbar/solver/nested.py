"""Block-nested DC solver for a crossbar array with PARALLEL BL/SL rails.

The two array rails (BL, SL) run side by side along the row axis; the
gate/control line is a driven boundary, so the columns are independent and
batched.

See also:
    docs/reference/primitive/xbar/solver/nested.md
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, ClassVar, Generic, Self, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.common.prober import Prober
from neurox.primitive.xbar.cell import XbarCell, XbarCellDcop, XbarCellSnap

from ._linalg import block_solve, solve_block_tridiagonal
from ._wire_kcl import col_driver_current, col_wire_kcl_residual
from .base import Solver, SolverConfig, SolverDcop
from .clamp import ClampDriver, ClampSnap

CellSnapT = TypeVar("CellSnapT", bound=XbarCellSnap)
CellDCOPT = TypeVar("CellDCOPT", bound=XbarCellDcop)
BLSnapT = TypeVar("BLSnapT", bound=ClampSnap)
SLSnapT = TypeVar("SLSnapT", bound=ClampSnap)


def _detach_dataclass_tensors(obj: object) -> object:
    """Detach every tensor in a nested dataclass."""
    if isinstance(obj, Tensor):
        return obj.detach()
    if is_dataclass(obj) and not isinstance(obj, type):
        changes = {f.name: _detach_dataclass_tensors(getattr(obj, f.name)) for f in fields(obj)}
        return replace(obj, **changes)
    return obj


@dataclass(frozen=True)
class SolverObservation(Generic[CellDCOPT]):
    """Converged operating point plus KCL residuals from one DC solve.

    Attributes:
        dcop: The converged :class:`SolverDcop`; its cell working point is
            carried opaquely as-is.
        wire_bl__uA: BL wire KCL residual per node.
            Shape: ``[..., num_col, num_row]``.
        wire_sl__uA: SL wire KCL residual per node.
            Shape: ``[..., num_col, num_row]``.
        clamp_bl__V: ``|bl_driver(I_BL_port) - V_BL_clamp|`` per column.
            Shape: ``[..., num_col]``.
        clamp_sl__V: ``|sl_driver(I_SL_port) - V_SL_drive|`` per column.
            Shape: ``[..., num_col]``.
    """

    dcop: SolverDcop[CellDCOPT]
    wire_bl__uA: Tensor
    wire_sl__uA: Tensor
    clamp_bl__V: Tensor
    clamp_sl__V: Tensor

    def detach(self) -> Self:
        """Return an equivalent observation with detached tensors."""
        detached_dcop = _detach_dataclass_tensors(self.dcop)
        assert isinstance(detached_dcop, SolverDcop)
        return replace(
            self,
            dcop=detached_dcop,
            wire_bl__uA=self.wire_bl__uA.detach(),
            wire_sl__uA=self.wire_sl__uA.detach(),
            clamp_bl__V=self.clamp_bl__V.detach(),
            clamp_sl__V=self.clamp_sl__V.detach(),
        )


class SolverProber(Prober[SolverObservation[XbarCellDcop]]):
    """Capture converged solver states and residuals."""

    _active_stack: ClassVar[list[Prober[SolverObservation[XbarCellDcop]]]] = []

    @classmethod
    def _stack(cls) -> list[Prober[SolverObservation[XbarCellDcop]]]:
        return cls._active_stack


class NestedParallelRailSolverConfig(SolverConfig):
    """Workload-tuned numerical knobs for :class:`NestedParallelRailSolver`.

    Attributes:
        n_outer: Outer Newton iterations on the per-column clamp voltage.
            Each outer step takes one implicit-Jacobian Newton step on
            V_clamp and then runs ``n_inner`` inner array Newton steps at
            the updated V_clamp (clamp-first ordering).
        n_inner: Inner Newton iterations on the wire / cell coupled state
            at a frozen V_clamp boundary.
    """

    n_outer: int
    n_inner: int

    def validate(self) -> None:
        super().validate()
        self._require_pos(self.n_outer, "n_outer")
        self._require_pos(self.n_inner, "n_inner")


class NestedParallelRailSolver(Solver):
    """Block Gauss-Seidel + implicit-Newton DC solver for a parallel BL/SL tile.

    The cell grid uses ``[..., num_col, num_row]``: the last axis is the
    wire-ladder / IR-drop direction and the second-to-last axis indexes the
    independent columns.

    Args:
        config: Fixed nested-solver iteration counts.
    """

    _MAX_OUTER_STEP__V: float = 0.10
    _MAX_INNER_STEP__V: float = 0.05

    def __init__(self, *, config: NestedParallelRailSolverConfig) -> None:
        self._config = config

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
            bl_segment_r__MOhm: 1-D BL segment resistances; index 0
                is driver-to-first.
            sl_segment_r__MOhm: 1-D SL segment resistances; index 0
                is driver-to-first.
            bl_segment_g__uS: BL segment conductances.
            sl_segment_g__uS: SL segment conductances.
            cell: Condensed cell branch model.
            cell_snap: Per-solve cell snap bundling the device snaps and the
                per-cell control-line (WL) drive.
            bl_driver: BL clamp driver.
            bl_driver_snap: Per-solve BL driver snap.
            sl_driver: SL clamp driver.
            sl_driver_snap: Per-solve SL driver snap.

        Returns:
            Complete steady-state solution for the current VMM.
        """
        dcop = self._solve_dc_compiled(
            bl_segment_r__MOhm=bl_segment_r__MOhm,
            sl_segment_r__MOhm=sl_segment_r__MOhm,
            bl_segment_g__uS=bl_segment_g__uS,
            sl_segment_g__uS=sl_segment_g__uS,
            cell=cell,
            cell_snap=cell_snap,
            bl_driver=bl_driver,
            bl_driver_snap=bl_driver_snap,
            sl_driver=sl_driver,
            sl_driver_snap=sl_driver_snap,
        )
        if SolverProber.active():
            observation = self._converged_observation(
                dcop=dcop,
                bl_segment_g__uS=bl_segment_g__uS,
                sl_segment_g__uS=sl_segment_g__uS,
                bl_driver=bl_driver,
                bl_driver_snap=bl_driver_snap,
                sl_driver=sl_driver,
                sl_driver_snap=sl_driver_snap,
            )
            SolverProber.submit(observation)
        return dcop

    @torch.compile(dynamic=False)
    def _solve_dc_compiled(
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
        """Run the fixed-shape nested solve."""

        # --- 1: build wire-Jacobian templates ---

        # Shape: [num_row]
        bl_wire_diag_tmpl = bl_segment_g__uS + F.pad(bl_segment_g__uS[1:], (0, 1))
        # Shape: [num_row]
        sl_wire_diag_tmpl = sl_segment_g__uS + F.pad(sl_segment_g__uS[1:], (0, 1))
        # Shape: [num_row-1]
        bl_wire_offdiag = -bl_segment_g__uS[1:]
        # Shape: [num_row-1]
        sl_wire_offdiag = -sl_segment_g__uS[1:]
        # Shape: []
        bl_driver_segment_g = bl_segment_g__uS[0]
        # Shape: []
        sl_driver_segment_g = sl_segment_g__uS[0]

        # --- 2: initialize the cell at the reference clamps ---

        v_bl_seed = bl_driver_snap.v_ref__V
        v_sl_seed = sl_driver_snap.v_ref__V
        # Shape: [..., num_col, num_row]
        i_cell, _g_bl_init, _g_sl_init = cell.solve_branch(v_bl_seed.unsqueeze(-1), v_sl_seed.unsqueeze(-1), cell_snap)
        *_batch, num_col, num_row = i_cell.shape
        if not (num_col > 1):
            raise ValueError(f"require: num_col ({num_col}) > 1")
        if not (num_row > 1):
            raise ValueError(f"require: num_row ({num_row}) > 1")

        # --- 3: initialize the clamp voltages ---

        # Shape: [..., num_col, num_row] -> [..., num_col]
        i_bl_seed__uA = i_cell.sum(dim=-1)
        # Shape: [..., num_col, num_row] -> [..., num_col]
        i_sl_seed__uA = -i_cell.sum(dim=-1)
        # Shape: [..., num_col]
        v_bl_clamp__V, _ = bl_driver.solve_clamp(i_bl_seed__uA, bl_driver_snap, v_clamp_init__V=None)
        v_sl_drive__V, _ = sl_driver.solve_clamp(i_sl_seed__uA, sl_driver_snap, v_clamp_init__V=None)

        # --- 4: initialize wire nodes with first-order IR drop ---

        # Shape: [..., num_col, num_row]
        v_bl_node, v_sl_node = self._wire_ir_drop_seed(
            i_cell,
            v_bl_clamp__V,
            v_sl_drive__V,
            bl_segment_r__MOhm,
            sl_segment_r__MOhm,
            i_cell.ndim,
            num_row,
        )

        # --- 5: refresh the cell at the wire-node seed ---

        i_cell, di_dvbl, di_dvsl = cell.solve_branch(v_bl_node, v_sl_node, cell_snap)

        # --- 6: solve coupled clamps and wire nodes ---

        max_inner_step__V = self._MAX_INNER_STEP__V
        max_outer_step__V = self._MAX_OUTER_STEP__V
        n_inner = self._config.n_inner

        for _ in range(self._config.n_outer):
            # Coupled 2×2 Newton step on ``(V_BL_clamp, V_SL_drive)``.
            # K = ∂V_node[0]/∂V_clamp captures cross-rail cell coupling.
            g_cell_bl_eff = di_dvbl
            g_cell_sl_eff = -di_dvsl
            # Shape: [..., num_col, 2, 2]
            k_inner_2x2 = self._compute_k_inner_coupled_2x2(
                v_bl_node,
                g_cell_bl_eff,
                g_cell_sl_eff,
                bl_wire_diag_tmpl,
                sl_wire_diag_tmpl,
                bl_wire_offdiag,
                sl_wire_offdiag,
                bl_driver_segment_g,
                sl_driver_segment_g,
            )

            # First-segment port current and BL / SL clamp-driver targets.
            # Shape: [..., num_col]
            i_bl_port__uA = (v_bl_clamp__V - v_bl_node.select(-1, 0)) * bl_driver_segment_g
            i_sl_port__uA = (v_sl_drive__V - v_sl_node.select(-1, 0)) * sl_driver_segment_g
            v_bl_target__V, r_bl_driver__MOhm = bl_driver.solve_clamp(
                i_bl_port__uA,
                bl_driver_snap,
                v_clamp_init__V=v_bl_clamp__V,
            )
            v_sl_target__V, r_sl_driver__MOhm = sl_driver.solve_clamp(
                i_sl_port__uA,
                sl_driver_snap,
                v_clamp_init__V=v_sl_drive__V,
            )

            # Outer Newton on F_outer(V_clamp) = V_target(V_clamp) - V_clamp:
            # each clamp driver maps its first-segment port current to a
            # target clamp, and the 2×2 dF/dV_clamp couples the driver slope,
            # the port-current sensitivity, and K_inner's V_node[0] response.
            # Shape: [..., num_col]
            f_bl_outer = v_bl_target__V - v_bl_clamp__V
            f_sl_outer = v_sl_target__V - v_sl_drive__V
            rg_bl = r_bl_driver__MOhm * bl_driver_segment_g
            rg_sl = r_sl_driver__MOhm * sl_driver_segment_g
            k00 = k_inner_2x2[..., 0, 0]
            k01 = k_inner_2x2[..., 0, 1]
            k10 = k_inner_2x2[..., 1, 0]
            k11 = k_inner_2x2[..., 1, 1]
            df_row0 = torch.stack(
                [rg_bl * (1.0 - k00) - 1.0, -rg_bl * k01],
                dim=-1,
            )
            df_row1 = torch.stack(
                [-rg_sl * k10, rg_sl * (1.0 - k11) - 1.0],
                dim=-1,
            )
            # Shape: [..., num_col, 2, 2]
            df_outer = torch.stack([df_row0, df_row1], dim=-2)
            # Shape: [..., num_col, 2]
            f_outer = torch.stack([f_bl_outer, f_sl_outer], dim=-1)
            # Solve 2×2 system per column: δ = -inv(df_outer) · f_outer.
            delta_2 = block_solve(df_outer, -f_outer.unsqueeze(-1)).squeeze(-1)
            delta_bl = delta_2[..., 0].clamp(min=-max_outer_step__V, max=max_outer_step__V)
            delta_sl = delta_2[..., 1].clamp(min=-max_outer_step__V, max=max_outer_step__V)
            v_bl_clamp__V = v_bl_clamp__V + delta_bl
            v_sl_drive__V = v_sl_drive__V + delta_sl

            # Solve the inner wire state at the updated clamp voltages.
            # Shape: [..., num_col] -> [..., num_col, 1]
            v_bl_clamp_grid__V = v_bl_clamp__V.unsqueeze(-1)
            v_sl_drive_grid__V = v_sl_drive__V.unsqueeze(-1)

            for _ in range(n_inner):
                # Shape: [..., num_col, num_row]
                i_cell, di_dvbl, di_dvsl = cell.solve_branch(v_bl_node, v_sl_node, cell_snap)
                g_cell_bl_eff = di_dvbl
                g_cell_sl_eff = -di_dvsl
                # Shape: [..., num_col, num_row]
                f_bl_kcl = col_wire_kcl_residual(v_bl_node, v_bl_clamp_grid__V, bl_segment_g__uS, i_cell)
                f_sl_kcl = col_wire_kcl_residual(v_sl_node, v_sl_drive_grid__V, sl_segment_g__uS, -i_cell)
                dv_bl_node, dv_sl_node = self._wire_newton_coupled_block2x2(
                    v_bl_node,
                    f_bl_kcl,
                    f_sl_kcl,
                    g_cell_bl_eff,
                    g_cell_sl_eff,
                    bl_wire_diag_tmpl,
                    sl_wire_diag_tmpl,
                    bl_wire_offdiag,
                    sl_wire_offdiag,
                )
                dv_bl_node = dv_bl_node.clamp(min=-max_inner_step__V, max=max_inner_step__V)
                dv_sl_node = dv_sl_node.clamp(min=-max_inner_step__V, max=max_inner_step__V)
                v_bl_node = v_bl_node + dv_bl_node
                v_sl_node = v_sl_node + dv_sl_node

        # --- 7: refresh the cell and boundary currents ---

        cell_dcop = cell.solve_dc(v_bl_node, v_sl_node, cell_snap)

        # Shape: [..., num_col] -> [..., num_col, 1]
        v_bl_clamp_grid__V = v_bl_clamp__V.unsqueeze(-1)
        # Shape: [..., num_col] -> [..., num_col, 1]
        v_sl_drive_grid__V = v_sl_drive__V.unsqueeze(-1)
        i_bl_driver = col_driver_current(v_bl_node, v_bl_clamp_grid__V, bl_segment_g__uS)
        i_sl_driver = col_driver_current(v_sl_node, v_sl_drive_grid__V, sl_segment_g__uS)

        return SolverDcop(
            i_bl_driver=i_bl_driver,
            i_sl_driver=i_sl_driver,
            v_bl_node=v_bl_node,
            v_sl_node=v_sl_node,
            cell=cell_dcop,
            v_bl_clamp=v_bl_clamp__V,
            v_sl_drive=v_sl_drive__V,
        )

    def _converged_observation(
        self,
        *,
        dcop: SolverDcop[CellDCOPT],
        bl_segment_g__uS: Tensor,
        sl_segment_g__uS: Tensor,
        bl_driver: ClampDriver[BLSnapT],
        bl_driver_snap: BLSnapT,
        sl_driver: ClampDriver[SLSnapT],
        sl_driver_snap: SLSnapT,
    ) -> SolverObservation[CellDCOPT]:
        """Compute wire and clamp residuals at a converged point."""
        wire_bl_res, wire_sl_res = self._compute_wire_residuals(dcop, bl_segment_g__uS, sl_segment_g__uS)

        # Clamp residual: |driver(I_port) - V_clamp| at the converged
        # operating point. Zero at the outer Newton fixed point.
        bl_driver_segment_g = bl_segment_g__uS[0]
        sl_driver_segment_g = sl_segment_g__uS[0]
        i_bl_port_final = (dcop.v_bl_clamp - dcop.v_bl_node.select(-1, 0)) * bl_driver_segment_g
        i_sl_port_final = (dcop.v_sl_drive - dcop.v_sl_node.select(-1, 0)) * sl_driver_segment_g
        v_bl_target_final, _ = bl_driver.solve_clamp(
            i_bl_port_final,
            bl_driver_snap,
            v_clamp_init__V=dcop.v_bl_clamp,
        )
        v_sl_target_final, _ = sl_driver.solve_clamp(
            i_sl_port_final,
            sl_driver_snap,
            v_clamp_init__V=dcop.v_sl_drive,
        )
        return SolverObservation(
            dcop=dcop,
            wire_bl__uA=wire_bl_res,
            wire_sl__uA=wire_sl_res,
            clamp_bl__V=(v_bl_target_final - dcop.v_bl_clamp).abs(),
            clamp_sl__V=(v_sl_target_final - dcop.v_sl_drive).abs(),
        )

    @staticmethod
    def _compute_wire_residuals(
        dcop: SolverDcop[CellDCOPT],
        bl_segment_g__uS: Tensor,
        sl_segment_g__uS: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Absolute BL / SL wire KCL residuals at the converged DCOP.

        BL uses ``+i`` (drained from BL), SL uses ``-i`` (injected into SL).
        """
        v_bl_clamp_grid__V = dcop.v_bl_clamp.unsqueeze(-1)
        v_sl_drive_grid__V = dcop.v_sl_drive.unsqueeze(-1)
        i_cell = dcop.cell.i__uA
        wire_bl_res = col_wire_kcl_residual(dcop.v_bl_node, v_bl_clamp_grid__V, bl_segment_g__uS, i_cell).abs()
        wire_sl_res = col_wire_kcl_residual(dcop.v_sl_node, v_sl_drive_grid__V, sl_segment_g__uS, -i_cell).abs()
        return wire_bl_res, wire_sl_res

    def solve_array_fixed_clamp(
        self,
        *,
        v_bl_clamp__V: Tensor,
        v_sl_drive__V: Tensor,
        bl_segment_r__MOhm: Tensor,
        sl_segment_r__MOhm: Tensor,
        bl_segment_g__uS: Tensor,
        sl_segment_g__uS: Tensor,
        cell: XbarCell[Any, Any, CellSnapT, CellDCOPT],
        cell_snap: CellSnapT,
    ) -> SolverDcop[CellDCOPT]:
        """Run only the inner array Newton loop at FIXED clamp boundaries.

        Args:
            v_bl_clamp__V: BL clamp voltage held fixed throughout the
                solve. Shape: ``[..., num_col]``.
            v_sl_drive__V: SL drive voltage held fixed. Shape:
                ``[..., num_col]``.
            bl_segment_r__MOhm: BL segment resistances.
            sl_segment_r__MOhm: SL segment resistances.
            bl_segment_g__uS: BL segment conductances.
            sl_segment_g__uS: SL segment conductances.
            cell: Condensed cell branch model.
            cell_snap: Per-solve cell snapshot.

        Returns:
            Steady-state solution at the fixed clamp voltages.
        """
        dcop = self._solve_array_fixed_clamp_impl(
            v_bl_clamp__V=v_bl_clamp__V,
            v_sl_drive__V=v_sl_drive__V,
            bl_segment_r__MOhm=bl_segment_r__MOhm,
            sl_segment_r__MOhm=sl_segment_r__MOhm,
            bl_segment_g__uS=bl_segment_g__uS,
            sl_segment_g__uS=sl_segment_g__uS,
            cell=cell,
            cell_snap=cell_snap,
        )
        if SolverProber.active():
            wire_bl_res, wire_sl_res = self._compute_wire_residuals(dcop, bl_segment_g__uS, sl_segment_g__uS)
            clamp_zero = torch.zeros_like(dcop.v_bl_clamp)
            SolverProber.submit(
                SolverObservation(
                    dcop=dcop,
                    wire_bl__uA=wire_bl_res,
                    wire_sl__uA=wire_sl_res,
                    clamp_bl__V=clamp_zero,
                    clamp_sl__V=clamp_zero,
                ),
            )
        return dcop

    def _solve_array_fixed_clamp_impl(
        self,
        *,
        v_bl_clamp__V: Tensor,
        v_sl_drive__V: Tensor,
        bl_segment_r__MOhm: Tensor,
        sl_segment_r__MOhm: Tensor,
        bl_segment_g__uS: Tensor,
        sl_segment_g__uS: Tensor,
        cell: XbarCell[Any, Any, CellSnapT, CellDCOPT],
        cell_snap: CellSnapT,
    ) -> SolverDcop[CellDCOPT]:
        """Run the fixed-clamp inner solve."""
        bl_wire_diag_tmpl = bl_segment_g__uS + F.pad(bl_segment_g__uS[1:], (0, 1))
        bl_wire_offdiag = -bl_segment_g__uS[1:]
        sl_wire_diag_tmpl = sl_segment_g__uS + F.pad(sl_segment_g__uS[1:], (0, 1))
        sl_wire_offdiag = -sl_segment_g__uS[1:]

        # Shape: [..., num_col] -> [..., num_col, 1]
        v_bl_clamp_grid__V = v_bl_clamp__V.unsqueeze(-1)
        # Shape: [..., num_col] -> [..., num_col, 1]
        v_sl_drive_grid__V = v_sl_drive__V.unsqueeze(-1)
        i_seed, _g_bl_init, _g_sl_init = cell.solve_branch(v_bl_clamp_grid__V, v_sl_drive_grid__V, cell_snap)
        num_row = i_seed.shape[-1]

        # IR-drop wire seed: BL propagates ``+i``, SL propagates ``-i``.
        v_bl_node, v_sl_node = self._wire_ir_drop_seed(
            i_seed,
            v_bl_clamp__V,
            v_sl_drive__V,
            bl_segment_r__MOhm,
            sl_segment_r__MOhm,
            i_seed.ndim,
            num_row,
        )

        max_inner_step__V = self._MAX_INNER_STEP__V
        for _ in range(self._config.n_inner):
            i_cell, di_dvbl, di_dvsl = cell.solve_branch(v_bl_node, v_sl_node, cell_snap)
            g_cell_bl_eff = di_dvbl
            g_cell_sl_eff = -di_dvsl
            # BL wire KCL uses ``+i``; SL wire KCL uses ``-i``.
            f_bl_kcl = col_wire_kcl_residual(v_bl_node, v_bl_clamp_grid__V, bl_segment_g__uS, i_cell)
            f_sl_kcl = col_wire_kcl_residual(v_sl_node, v_sl_drive_grid__V, sl_segment_g__uS, -i_cell)
            dv_bl_node, dv_sl_node = self._wire_newton_coupled_block2x2(
                v_bl_node,
                f_bl_kcl,
                f_sl_kcl,
                g_cell_bl_eff,
                g_cell_sl_eff,
                bl_wire_diag_tmpl,
                sl_wire_diag_tmpl,
                bl_wire_offdiag,
                sl_wire_offdiag,
            )
            dv_bl_node = dv_bl_node.clamp(min=-max_inner_step__V, max=max_inner_step__V)
            dv_sl_node = dv_sl_node.clamp(min=-max_inner_step__V, max=max_inner_step__V)
            v_bl_node = v_bl_node + dv_bl_node
            v_sl_node = v_sl_node + dv_sl_node

        cell_dcop = cell.solve_dc(v_bl_node, v_sl_node, cell_snap)

        i_bl_driver = col_driver_current(v_bl_node, v_bl_clamp_grid__V, bl_segment_g__uS)
        i_sl_driver = col_driver_current(v_sl_node, v_sl_drive_grid__V, sl_segment_g__uS)

        return SolverDcop(
            i_bl_driver=i_bl_driver,
            i_sl_driver=i_sl_driver,
            v_bl_node=v_bl_node,
            v_sl_node=v_sl_node,
            cell=cell_dcop,
            v_bl_clamp=v_bl_clamp__V,
            v_sl_drive=v_sl_drive__V,
        )

    @staticmethod
    def _wire_ir_drop_seed(
        i_cell: Tensor,
        v_bl_clamp__V: Tensor,
        v_sl_drive__V: Tensor,
        bl_segment_r__MOhm: Tensor,
        sl_segment_r__MOhm: Tensor,
        ndim: int,
        num_row: int,
    ) -> tuple[Tensor, Tensor]:
        """First-order IR-drop seed for the wire ladders.

        Args:
            i_cell: Signed cell branch currents [uA].
            v_bl_clamp__V: BL clamp voltages.
            v_sl_drive__V: SL drive voltages.
            bl_segment_r__MOhm: BL segment resistances.
            sl_segment_r__MOhm: SL segment resistances.
            ndim: Rank of the cell-current tensor.
            num_row: Number of wire nodes per column.

        Returns:
            Initial BL and SL node voltages [V].
        """
        shape_broadcast = [1] * ndim
        shape_broadcast[-1] = num_row
        bl_segment_r_b = bl_segment_r__MOhm.view(shape_broadcast)
        sl_segment_r_b = sl_segment_r__MOhm.view(shape_broadcast)

        v_bl_clamp_grid = v_bl_clamp__V.unsqueeze(-1)
        v_sl_drive_grid = v_sl_drive__V.unsqueeze(-1)

        i_bl_down = torch.flip(torch.cumsum(torch.flip(i_cell, [-1]), -1), [-1])
        v_bl_node = v_bl_clamp_grid - torch.cumsum(i_bl_down * bl_segment_r_b, dim=-1)

        i_sl_inject = -i_cell
        i_sl_down = torch.flip(torch.cumsum(torch.flip(i_sl_inject, [-1]), -1), [-1])
        v_sl_node = v_sl_drive_grid - torch.cumsum(i_sl_down * sl_segment_r_b, dim=-1)
        return v_bl_node, v_sl_node

    @staticmethod
    def _wire_newton_coupled_block2x2(
        v_bl_node: Tensor,
        f_bl_kcl: Tensor,
        f_sl_kcl: Tensor,
        g_cell_bl_eff: Tensor,
        g_cell_sl_eff: Tensor,
        bl_wire_diag_tmpl: Tensor,
        sl_wire_diag_tmpl: Tensor,
        bl_wire_offdiag: Tensor,
        sl_wire_offdiag: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Coupled BL/SL wire Newton step at frozen V_clamp / V_SL_drive.

        Args:
            v_bl_node: BL node voltages [V], shape
                ``[..., num_col, num_row]``.
            f_bl_kcl: BL KCL residuals [uA], same shape as ``v_bl_node``.
            f_sl_kcl: SL KCL residuals [uA], same shape as ``v_bl_node``.
            g_cell_bl_eff: BL-side cell derivatives [uS].
            g_cell_sl_eff: Negated SL-side cell derivatives [uS].
            bl_wire_diag_tmpl: BL wire diagonal, shape ``[num_row]``.
            sl_wire_diag_tmpl: SL wire diagonal, shape ``[num_row]``.
            bl_wire_offdiag: BL wire off-diagonal, shape ``[num_row - 1]``.
            sl_wire_offdiag: SL wire off-diagonal, shape ``[num_row - 1]``.

        Returns:
            BL and SL Newton voltage steps [V], each shaped like
            ``v_bl_node``.
        """
        num_axis = v_bl_node.shape[-1]
        shape_broadcast = [1] * v_bl_node.ndim
        shape_broadcast[-1] = num_axis

        # Per-row diagonal entries.
        # Shape: [..., num_col, num_row]
        bl_diag_node = bl_wire_diag_tmpl.view(shape_broadcast) + g_cell_bl_eff
        sl_diag_node = sl_wire_diag_tmpl.view(shape_broadcast) + g_cell_sl_eff
        cross_to_bl_from_sl = -g_cell_sl_eff  # ∂F_BL/∂V_SL
        cross_to_sl_from_bl = -g_cell_bl_eff  # ∂F_SL/∂V_BL

        # Assemble per-row 2×2 diagonal block.
        # Shape: [..., num_col, num_row, 2, 2]
        diag_row_top = torch.stack([bl_diag_node, cross_to_bl_from_sl], dim=-1)
        diag_row_bot = torch.stack([cross_to_sl_from_bl, sl_diag_node], dim=-1)
        diag_blocks = torch.stack([diag_row_top, diag_row_bot], dim=-2)

        # Off-diagonal blocks (BL-BL and SL-SL wire only).
        # sub_k[0] is unused; sup_k[N-1] is unused — pad with zeros.
        # Shape: [num_row]
        sub_bl_pad = F.pad(bl_wire_offdiag, (1, 0)).view(shape_broadcast)
        sub_sl_pad = F.pad(sl_wire_offdiag, (1, 0)).view(shape_broadcast)
        sup_bl_pad = F.pad(bl_wire_offdiag, (0, 1)).view(shape_broadcast)
        sup_sl_pad = F.pad(sl_wire_offdiag, (0, 1)).view(shape_broadcast)
        # Broadcast to [..., num_col, num_row]
        sub_bl_full = sub_bl_pad.expand_as(v_bl_node)
        sub_sl_full = sub_sl_pad.expand_as(v_bl_node)
        sup_bl_full = sup_bl_pad.expand_as(v_bl_node)
        sup_sl_full = sup_sl_pad.expand_as(v_bl_node)
        zero_full = torch.zeros_like(sub_bl_full)
        # 2×2 diagonal blocks: diag(-bl_offdiag, -sl_offdiag)
        # Shape: [..., num_col, num_row, 2, 2]
        sub_row_top = torch.stack([sub_bl_full, zero_full], dim=-1)
        sub_row_bot = torch.stack([zero_full, sub_sl_full], dim=-1)
        sub_blocks = torch.stack([sub_row_top, sub_row_bot], dim=-2)
        sup_row_top = torch.stack([sup_bl_full, zero_full], dim=-1)
        sup_row_bot = torch.stack([zero_full, sup_sl_full], dim=-1)
        sup_blocks = torch.stack([sup_row_top, sup_row_bot], dim=-2)

        # RHS = [-F_BL, -F_SL] stacked.
        # Shape: [..., num_col, num_row, 2]
        rhs = torch.stack([-f_bl_kcl, -f_sl_kcl], dim=-1)

        # Solve. Block tensors take last 3 dims [..., num_row, 2, 2];
        # rhs takes last 2 dims [..., num_row, 2]. The column dim sits in the
        # leading batch.
        delta = solve_block_tridiagonal(sub_blocks, diag_blocks, sup_blocks, rhs)
        # Unpack into (dv_bl, dv_sl).
        return delta[..., 0], delta[..., 1]

    @staticmethod
    def _compute_k_inner_coupled_2x2(
        v_bl_node: Tensor,
        g_cell_bl_eff: Tensor,
        g_cell_sl_eff: Tensor,
        bl_wire_diag_tmpl: Tensor,
        sl_wire_diag_tmpl: Tensor,
        bl_wire_offdiag: Tensor,
        sl_wire_offdiag: Tensor,
        bl_driver_segment_g__uS: Tensor,
        sl_driver_segment_g__uS: Tensor,
    ) -> Tensor:
        """Compute the 2×2 ``K_inner = ∂V_array[0] / ∂V_clamp`` per column.

        Args:
            v_bl_node: BL node voltages [V], shape
                ``[..., num_col, num_row]``.
            g_cell_bl_eff: BL-side cell derivatives [uS].
            g_cell_sl_eff: Negated SL-side cell derivatives [uS].
            bl_wire_diag_tmpl: BL wire diagonal, shape ``[num_row]``.
            sl_wire_diag_tmpl: SL wire diagonal, shape ``[num_row]``.
            bl_wire_offdiag: BL wire off-diagonal, shape ``[num_row - 1]``.
            sl_wire_offdiag: SL wire off-diagonal, shape ``[num_row - 1]``.
            bl_driver_segment_g__uS: First BL segment conductance.
            sl_driver_segment_g__uS: First SL segment conductance.

        Returns:
            Clamp-to-port-node sensitivity, shape
            ``[..., num_col, 2, 2]``.
        """
        num_row = v_bl_node.shape[-1]
        shape_broadcast = [1] * v_bl_node.ndim
        shape_broadcast[-1] = num_row

        # --- 1: build the block-2x2 inner Jacobian ---

        bl_diag_node = bl_wire_diag_tmpl.view(shape_broadcast) + g_cell_bl_eff
        sl_diag_node = sl_wire_diag_tmpl.view(shape_broadcast) + g_cell_sl_eff
        cross_to_bl_from_sl = -g_cell_sl_eff  # ∂F_BL/∂V_SL
        cross_to_sl_from_bl = -g_cell_bl_eff  # ∂F_SL/∂V_BL
        # Shape: [..., num_col, num_row, 2, 2]
        diag_blocks = torch.stack(
            [
                torch.stack([bl_diag_node, cross_to_bl_from_sl], dim=-1),
                torch.stack([cross_to_sl_from_bl, sl_diag_node], dim=-1),
            ],
            dim=-2,
        )

        sub_bl_pad = F.pad(bl_wire_offdiag, (1, 0)).view(shape_broadcast)
        sub_sl_pad = F.pad(sl_wire_offdiag, (1, 0)).view(shape_broadcast)
        sup_bl_pad = F.pad(bl_wire_offdiag, (0, 1)).view(shape_broadcast)
        sup_sl_pad = F.pad(sl_wire_offdiag, (0, 1)).view(shape_broadcast)
        sub_bl_full = sub_bl_pad.expand_as(v_bl_node)
        sub_sl_full = sub_sl_pad.expand_as(v_bl_node)
        sup_bl_full = sup_bl_pad.expand_as(v_bl_node)
        sup_sl_full = sup_sl_pad.expand_as(v_bl_node)
        zero_full = torch.zeros_like(sub_bl_full)
        sub_blocks = torch.stack(
            [
                torch.stack([sub_bl_full, zero_full], dim=-1),
                torch.stack([zero_full, sub_sl_full], dim=-1),
            ],
            dim=-2,
        )
        sup_blocks = torch.stack(
            [
                torch.stack([sup_bl_full, zero_full], dim=-1),
                torch.stack([zero_full, sup_sl_full], dim=-1),
            ],
            dim=-2,
        )

        # --- 2: build boundary-forcing basis vectors ---

        # BL-clamp forcing is nonzero only at the first BL wire node.
        zeros_node = torch.zeros_like(v_bl_node)
        # Per-row BL-only basis: g_BL_seg[0] at row 0, 0 elsewhere.
        g_bl_at_row0_per_row = F.pad(
            bl_driver_segment_g__uS.expand(*v_bl_node.shape[:-1], 1),
            (0, num_row - 1),
        )
        g_sl_at_row0_per_row = F.pad(
            sl_driver_segment_g__uS.expand(*v_bl_node.shape[:-1], 1),
            (0, num_row - 1),
        )
        # Shape: [..., num_col, num_row] -> [..., num_col, num_row, 2]
        rhs_bl_basis = torch.stack([g_bl_at_row0_per_row, zeros_node], dim=-1)
        # Shape: [..., num_col, num_row] -> [..., num_col, num_row, 2]
        rhs_sl_basis = torch.stack([zeros_node, g_sl_at_row0_per_row], dim=-1)

        # --- 3: solve and extract row-zero responses ---

        # Shape: [..., num_col, num_row, 2]
        u_bl = solve_block_tridiagonal(sub_blocks, diag_blocks, sup_blocks, rhs_bl_basis)
        u_sl = solve_block_tridiagonal(sub_blocks, diag_blocks, sup_blocks, rhs_sl_basis)

        # K[:, 0] from BL-basis solve at row 0; K[:, 1] from SL-basis solve at row 0.
        # Shape: [..., num_col, 2]
        k_col_bl = u_bl.select(-2, 0)
        k_col_sl = u_sl.select(-2, 0)
        # Stack into [..., num_col, 2, 2] with K[:, j] as columns.
        return torch.stack([k_col_bl, k_col_sl], dim=-1)
