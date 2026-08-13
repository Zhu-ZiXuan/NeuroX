"""Block-nested DC solver for a crossbar array with PARALLEL BL/SL rails.

The two array rails (BL, SL) run side by side along the row axis; the
gate/control line is a driven boundary, so the columns are independent and
batched.

Each rail is a UNIFORM ladder — one link resistance joins every pair of
adjacent nodes and the same link joins the clamp driver to the node at index
0 — so a whole rail enters the solve as one scalar. The single distinguished
node is the ladder's open end at the last row, which has no link onward.

See also:
    docs/reference/primitive/xbar/solver/nested.md
    docs/internals/primitive/xbar/solver/nested.md
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.common import RecordBase, RecorderBase
from neurox.primitive.xbar.cell import XbarCell, XbarCellDcop, XbarCellSnap

from ._linalg import block_solve, solve_block_tridiagonal_2x2_uniform
from ._wire_kcl import col_driver_current, col_wire_kcl_residual
from .base import Solver, SolverConfig, SolverDcop
from .clamp import ClampDriver, ClampSnap

CellSnapT = TypeVar("CellSnapT", bound=XbarCellSnap)
CellDCOPT = TypeVar("CellDCOPT", bound=XbarCellDcop)
BLSnapT = TypeVar("BLSnapT", bound=ClampSnap)
SLSnapT = TypeVar("SLSnapT", bound=ClampSnap)


def _ladder_self_g(g_cell_eff: Tensor, segment_g__uS: float) -> Tensor:
    """Self-conductance of every node of one uniform wire ladder.

    A node's own conductance is its cell branch plus every rail link attached
    to it. An interior node has two — the one back towards the driver and the
    one onward — while the last row is the ladder's open end and has only the
    first. Row 0 is NOT distinguished: its driver-side link is one standard
    lattice pitch like any other.

    Args:
        g_cell_eff: Per-node cell branch derivative [uS].
            Shape: `[..., num_col, num_row]`.
        segment_g__uS: Rail conductance of one lattice link.

    Returns:
        Per-node self-conductance [uS].
        Shape: `[..., num_col, num_row]`.
    """
    # Shape: [..., num_col, num_row]
    return torch.cat(
        (g_cell_eff[..., :-1] + 2.0 * segment_g__uS, g_cell_eff[..., -1:] + segment_g__uS),
        dim=-1,
    )


def _wire_diag_blocks(
    g_cell_bl_eff: Tensor,
    g_cell_sl_eff: Tensor,
    bl_segment_g__uS: float,
    sl_segment_g__uS: float,
) -> Tensor:
    """Per-row 2×2 diagonal blocks of the coupled BL/SL wire Jacobian.

    The rail entries are each ladder's per-node self-conductance; the cross
    entries are the cell branch alone, which is what couples the two rails at
    a node. The off-diagonal blocks of the same Jacobian are not built here —
    they are the constant `diag(-g_BL, -g_SL)` the block solver takes as a
    pair of scalars.

    Args:
        g_cell_bl_eff: BL-side cell derivatives [uS].
            Shape: `[..., num_col, num_row]`.
        g_cell_sl_eff: Negated SL-side cell derivatives [uS].
            Shape: `[..., num_col, num_row]`.
        bl_segment_g__uS: BL rail conductance of one lattice link.
        sl_segment_g__uS: SL rail conductance of one lattice link.

    Returns:
        Diagonal blocks, rail-major within each block.
        Shape: `[..., num_col, num_row, 2, 2]`.
    """
    # Shape: [..., num_col, num_row]
    bl_diag_node = _ladder_self_g(g_cell_bl_eff, bl_segment_g__uS)
    sl_diag_node = _ladder_self_g(g_cell_sl_eff, sl_segment_g__uS)
    # ∂F_BL/∂V_SL on the top row, ∂F_SL/∂V_BL on the bottom.
    # Shape: [..., num_col, num_row, 2, 2]
    return torch.stack(
        (
            torch.stack((bl_diag_node, -g_cell_sl_eff), dim=-1),
            torch.stack((-g_cell_bl_eff, sl_diag_node), dim=-1),
        ),
        dim=-2,
    )


class SolverRecord(RecordBase, Generic[CellDCOPT]):
    """Converged operating point plus KCL residuals from one DC solve."""

    emitter: str
    """Fixed name of the solve entry point that emitted the record, a solver
    being a plain numerical object rather than a module."""
    dcop: SolverDcop[CellDCOPT]
    """The converged operating point; being a dataclass, the record's field
    walk reaches its tensors, the nested cell working point included."""
    wire_bl__uA: Tensor
    """BL wire KCL residual per node. Shape: `[..., num_col, num_row]`."""
    wire_sl__uA: Tensor
    """SL wire KCL residual per node. Shape: `[..., num_col, num_row]`."""
    clamp_bl__V: Tensor
    """`|bl_driver(I_BL_port) - V_BL_clamp|` per column. Shape: `[..., num_col]`."""
    clamp_sl__V: Tensor
    """`|sl_driver(I_SL_port) - V_SL_drive|` per column. Shape: `[..., num_col]`."""


class SolverProber(RecorderBase[SolverRecord[XbarCellDcop]]):
    """Capture converged solver states and residuals."""


class NestedParallelRailSolverConfig(SolverConfig):
    """Workload-tuned numerical knobs for `NestedParallelRailSolver`."""

    n_outer: int
    """Outer Newton iterations on the per-column clamp voltage. Each step
    takes one implicit-Jacobian Newton step on V_clamp and then runs `n_inner`
    inner array Newton steps at the updated V_clamp."""
    n_inner: int
    """Inner Newton iterations on the wire / cell coupled state at a frozen
    V_clamp boundary."""

    def validate(self) -> None:
        super().validate()
        self._require_pos(self.n_outer, "n_outer")
        self._require_pos(self.n_inner, "n_inner")


class NestedParallelRailSolver(Solver):
    """Block Gauss-Seidel + implicit-Newton DC solver for a parallel BL/SL tile.

    Every cell-grid tensor runs the wire ladder / IR-drop direction along the
    last axis and indexes the independent columns along the second-to-last.

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
        bl_segment_r__MOhm: float,
        sl_segment_r__MOhm: float,
        cell: XbarCell[Any, Any, CellSnapT, CellDCOPT],
        cell_snap: CellSnapT,
        bl_driver: ClampDriver[BLSnapT],
        bl_driver_snap: BLSnapT,
        sl_driver: ClampDriver[SLSnapT],
        sl_driver_snap: SLSnapT,
    ) -> SolverDcop[CellDCOPT]:
        """Solve the fabricated tile for one cell snap.

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
        dcop = self._solve_dc_compiled(
            bl_segment_r__MOhm=bl_segment_r__MOhm,
            sl_segment_r__MOhm=sl_segment_r__MOhm,
            cell=cell,
            cell_snap=cell_snap,
            bl_driver=bl_driver,
            bl_driver_snap=bl_driver_snap,
            sl_driver=sl_driver,
            sl_driver_snap=sl_driver_snap,
        )
        if SolverProber.active():
            record = self._converged_record(
                dcop=dcop,
                bl_segment_g__uS=1.0 / bl_segment_r__MOhm,
                sl_segment_g__uS=1.0 / sl_segment_r__MOhm,
                bl_driver=bl_driver,
                bl_driver_snap=bl_driver_snap,
                sl_driver=sl_driver,
                sl_driver_snap=sl_driver_snap,
            )
            SolverProber.submit(record)
        return dcop

    @torch.compile(dynamic=False)
    def _solve_dc_compiled(
        self,
        *,
        bl_segment_r__MOhm: float,
        sl_segment_r__MOhm: float,
        cell: XbarCell[Any, Any, CellSnapT, CellDCOPT],
        cell_snap: CellSnapT,
        bl_driver: ClampDriver[BLSnapT],
        bl_driver_snap: BLSnapT,
        sl_driver: ClampDriver[SLSnapT],
        sl_driver_snap: SLSnapT,
    ) -> SolverDcop[CellDCOPT]:
        """Run the fixed-shape nested solve."""

        # --- 1: read the lattice link as a conductance ---

        # One uS is exactly one reciprocal MOhm, so no unit factor enters.
        bl_g__uS = 1.0 / bl_segment_r__MOhm
        sl_g__uS = 1.0 / sl_segment_r__MOhm

        # --- 2: initialize the cell at the reference clamps ---

        v_bl_seed = bl_driver_snap.v_ref__V
        v_sl_seed = sl_driver_snap.v_ref__V
        # Shape: [..., num_col, num_row]
        i_cell, _g_bl_init, _g_sl_init = cell.solve_branch(v_bl_seed.unsqueeze(-1), v_sl_seed.unsqueeze(-1), cell_snap)

        # --- 3: initialize the clamp voltages ---

        # Shape: [..., num_col, num_row] -> [..., num_col]
        i_bl_seed__uA = i_cell.sum(dim=-1)
        # Shape: [..., num_col, num_row] -> [..., num_col]
        i_sl_seed__uA = -i_cell.sum(dim=-1)
        # Shape: [..., num_col]
        v_bl_clamp__V, _ = bl_driver.solve_clamp(i_bl_seed__uA, bl_driver_snap, v_clamp_init__V=None)
        v_sl_drive__V, _ = sl_driver.solve_clamp(i_sl_seed__uA, sl_driver_snap, v_clamp_init__V=None)

        # --- 4: initialize wire nodes with first-order IR drop ---

        # The grid lift of the clamp boundaries is carried through the outer
        # loop and refreshed on every clamp update.
        # Shape: [..., num_col] -> [..., num_col, 1]
        v_bl_clamp_grid__V = v_bl_clamp__V.unsqueeze(-1)
        v_sl_drive_grid__V = v_sl_drive__V.unsqueeze(-1)

        # Shape: [..., num_col, num_row]
        v_bl_node, v_sl_node = self._wire_ir_drop_seed(
            i_cell,
            v_bl_clamp_grid__V,
            v_sl_drive_grid__V,
            bl_segment_r__MOhm,
            sl_segment_r__MOhm,
        )

        # --- 5: refresh the cell at the wire-node seed ---

        i_cell, di_dvbl, di_dvsl = cell.solve_branch(v_bl_node, v_sl_node, cell_snap)

        # --- 6: solve coupled clamps and wire nodes ---

        max_inner_step__V = self._MAX_INNER_STEP__V
        max_outer_step__V = self._MAX_OUTER_STEP__V
        n_inner = self._config.n_inner

        for _ in range(self._config.n_outer):
            # Coupled 2×2 Newton step on `(V_BL_clamp, V_SL_drive)`.
            # K = ∂V_node[0]/∂V_clamp captures cross-rail cell coupling.
            g_cell_bl_eff = di_dvbl
            g_cell_sl_eff = -di_dvsl
            # Shape: [..., num_col, 2, 2]
            k_inner_2x2 = self._compute_k_inner_coupled_2x2(
                g_cell_bl_eff,
                g_cell_sl_eff,
                bl_g__uS,
                sl_g__uS,
            )

            # Port current through the driver's own link, and the BL / SL
            # clamp-driver targets it implies.
            # Shape: [..., num_col]
            i_bl_port__uA = (v_bl_clamp__V - v_bl_node.select(-1, 0)) * bl_g__uS
            i_sl_port__uA = (v_sl_drive__V - v_sl_node.select(-1, 0)) * sl_g__uS
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
            # each clamp driver maps the port current through its own link to
            # a target clamp, and the 2×2 dF/dV_clamp couples the driver slope,
            # the port-current sensitivity, and K_inner's V_node[0] response.
            # Shape: [..., num_col]
            f_bl_outer = v_bl_target__V - v_bl_clamp__V
            f_sl_outer = v_sl_target__V - v_sl_drive__V
            rg_bl = r_bl_driver__MOhm * bl_g__uS
            rg_sl = r_sl_driver__MOhm * sl_g__uS
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
                f_bl_kcl = col_wire_kcl_residual(v_bl_node, v_bl_clamp_grid__V, bl_g__uS, i_cell)
                f_sl_kcl = col_wire_kcl_residual(v_sl_node, v_sl_drive_grid__V, sl_g__uS, -i_cell)
                dv_bl_node, dv_sl_node = self._wire_newton_coupled_block2x2(
                    f_bl_kcl,
                    f_sl_kcl,
                    g_cell_bl_eff,
                    g_cell_sl_eff,
                    bl_g__uS,
                    sl_g__uS,
                )
                dv_bl_node = dv_bl_node.clamp(min=-max_inner_step__V, max=max_inner_step__V)
                dv_sl_node = dv_sl_node.clamp(min=-max_inner_step__V, max=max_inner_step__V)
                v_bl_node = v_bl_node + dv_bl_node
                v_sl_node = v_sl_node + dv_sl_node

        # --- 7: refresh the cell and boundary currents ---

        cell_dcop = cell.solve_dc(v_bl_node, v_sl_node, cell_snap)

        i_bl_driver = col_driver_current(v_bl_node, v_bl_clamp_grid__V, bl_g__uS)
        i_sl_driver = col_driver_current(v_sl_node, v_sl_drive_grid__V, sl_g__uS)

        return SolverDcop(
            i_bl_driver=i_bl_driver,
            i_sl_driver=i_sl_driver,
            v_bl_node=v_bl_node,
            v_sl_node=v_sl_node,
            cell=cell_dcop,
            v_bl_clamp=v_bl_clamp__V,
            v_sl_drive=v_sl_drive__V,
        )

    def _converged_record(
        self,
        *,
        dcop: SolverDcop[CellDCOPT],
        bl_segment_g__uS: float,
        sl_segment_g__uS: float,
        bl_driver: ClampDriver[BLSnapT],
        bl_driver_snap: BLSnapT,
        sl_driver: ClampDriver[SLSnapT],
        sl_driver_snap: SLSnapT,
    ) -> SolverRecord[CellDCOPT]:
        """Build the `solve_dc` record: residuals at a converged point."""
        wire_bl_res, wire_sl_res = self._compute_wire_residuals(dcop, bl_segment_g__uS, sl_segment_g__uS)

        # Clamp residual: |driver(I_port) - V_clamp| at the converged
        # operating point. Zero at the outer Newton fixed point.
        i_bl_port_final = (dcop.v_bl_clamp - dcop.v_bl_node.select(-1, 0)) * bl_segment_g__uS
        i_sl_port_final = (dcop.v_sl_drive - dcop.v_sl_node.select(-1, 0)) * sl_segment_g__uS
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
        return SolverRecord(
            emitter="NestedParallelRailSolver.solve_dc",
            dcop=dcop,
            wire_bl__uA=wire_bl_res,
            wire_sl__uA=wire_sl_res,
            clamp_bl__V=(v_bl_target_final - dcop.v_bl_clamp).abs(),
            clamp_sl__V=(v_sl_target_final - dcop.v_sl_drive).abs(),
        )

    @staticmethod
    def _compute_wire_residuals(
        dcop: SolverDcop[CellDCOPT],
        bl_segment_g__uS: float,
        sl_segment_g__uS: float,
    ) -> tuple[Tensor, Tensor]:
        """Absolute BL / SL wire KCL residuals at the converged DCOP.

        BL uses `+i` (drained from BL), SL uses `-i` (injected into SL).
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
        bl_segment_r__MOhm: float,
        sl_segment_r__MOhm: float,
        cell: XbarCell[Any, Any, CellSnapT, CellDCOPT],
        cell_snap: CellSnapT,
    ) -> SolverDcop[CellDCOPT]:
        """Run only the inner array Newton loop at FIXED clamp boundaries.

        Args:
            v_bl_clamp__V: BL clamp voltage held fixed throughout the solve.
                Shape: `[..., num_col]`.
            v_sl_drive__V: SL drive voltage held fixed.
                Shape: `[..., num_col]`.
            bl_segment_r__MOhm: BL rail resistance of one lattice link.
            sl_segment_r__MOhm: SL rail resistance of one lattice link.
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
            cell=cell,
            cell_snap=cell_snap,
        )
        if SolverProber.active():
            wire_bl_res, wire_sl_res = self._compute_wire_residuals(
                dcop,
                1.0 / bl_segment_r__MOhm,
                1.0 / sl_segment_r__MOhm,
            )
            clamp_zero = torch.zeros_like(dcop.v_bl_clamp)
            SolverProber.submit(
                SolverRecord(
                    emitter="NestedParallelRailSolver.solve_array_fixed_clamp",
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
        bl_segment_r__MOhm: float,
        sl_segment_r__MOhm: float,
        cell: XbarCell[Any, Any, CellSnapT, CellDCOPT],
        cell_snap: CellSnapT,
    ) -> SolverDcop[CellDCOPT]:
        """Run the fixed-clamp inner solve."""
        bl_g__uS = 1.0 / bl_segment_r__MOhm
        sl_g__uS = 1.0 / sl_segment_r__MOhm

        # Shape: [..., num_col] -> [..., num_col, 1]
        v_bl_clamp_grid__V = v_bl_clamp__V.unsqueeze(-1)
        # Shape: [..., num_col] -> [..., num_col, 1]
        v_sl_drive_grid__V = v_sl_drive__V.unsqueeze(-1)
        i_seed, _g_bl_init, _g_sl_init = cell.solve_branch(v_bl_clamp_grid__V, v_sl_drive_grid__V, cell_snap)

        # IR-drop wire seed: BL propagates `+i`, SL propagates `-i`.
        v_bl_node, v_sl_node = self._wire_ir_drop_seed(
            i_seed,
            v_bl_clamp_grid__V,
            v_sl_drive_grid__V,
            bl_segment_r__MOhm,
            sl_segment_r__MOhm,
        )

        max_inner_step__V = self._MAX_INNER_STEP__V
        for _ in range(self._config.n_inner):
            i_cell, di_dvbl, di_dvsl = cell.solve_branch(v_bl_node, v_sl_node, cell_snap)
            g_cell_bl_eff = di_dvbl
            g_cell_sl_eff = -di_dvsl
            # BL wire KCL uses `+i`; SL wire KCL uses `-i`.
            f_bl_kcl = col_wire_kcl_residual(v_bl_node, v_bl_clamp_grid__V, bl_g__uS, i_cell)
            f_sl_kcl = col_wire_kcl_residual(v_sl_node, v_sl_drive_grid__V, sl_g__uS, -i_cell)
            dv_bl_node, dv_sl_node = self._wire_newton_coupled_block2x2(
                f_bl_kcl,
                f_sl_kcl,
                g_cell_bl_eff,
                g_cell_sl_eff,
                bl_g__uS,
                sl_g__uS,
            )
            dv_bl_node = dv_bl_node.clamp(min=-max_inner_step__V, max=max_inner_step__V)
            dv_sl_node = dv_sl_node.clamp(min=-max_inner_step__V, max=max_inner_step__V)
            v_bl_node = v_bl_node + dv_bl_node
            v_sl_node = v_sl_node + dv_sl_node

        cell_dcop = cell.solve_dc(v_bl_node, v_sl_node, cell_snap)

        i_bl_driver = col_driver_current(v_bl_node, v_bl_clamp_grid__V, bl_g__uS)
        i_sl_driver = col_driver_current(v_sl_node, v_sl_drive_grid__V, sl_g__uS)

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
        v_bl_clamp_grid__V: Tensor,
        v_sl_drive_grid__V: Tensor,
        bl_segment_r__MOhm: float,
        sl_segment_r__MOhm: float,
    ) -> tuple[Tensor, Tensor]:
        """First-order IR-drop seed for the wire ladders.

        Every link carries the same resistance, so the drop accumulated down
        a ladder is that one resistance times the running sum of the currents
        its links carry. The two rails share that running sum: BL drains the
        cell current and SL injects the very same current back.

        Args:
            i_cell: Signed cell branch currents [uA].
                Shape: `[..., num_col, num_row]`.
            v_bl_clamp_grid__V: BL clamp voltages on the cell grid.
                Shape: `[..., num_col, 1]`.
            v_sl_drive_grid__V: SL drive voltages on the cell grid.
                Shape: `[..., num_col, 1]`.
            bl_segment_r__MOhm: BL rail resistance of one lattice link.
            sl_segment_r__MOhm: SL rail resistance of one lattice link.

        Returns:
            Initial BL and SL node voltages [V].
            Shape: `[..., num_col, num_row]`.
        """
        # Current in the link that feeds node k: everything drawn at k and beyond.
        # Shape: [..., num_col, num_row]
        i_link = torch.flip(torch.cumsum(torch.flip(i_cell, [-1]), -1), [-1])
        # Shape: [..., num_col, num_row]
        i_cumulative = torch.cumsum(i_link, dim=-1)
        return (
            v_bl_clamp_grid__V - bl_segment_r__MOhm * i_cumulative,
            v_sl_drive_grid__V + sl_segment_r__MOhm * i_cumulative,
        )

    @staticmethod
    def _wire_newton_coupled_block2x2(
        f_bl_kcl: Tensor,
        f_sl_kcl: Tensor,
        g_cell_bl_eff: Tensor,
        g_cell_sl_eff: Tensor,
        bl_segment_g__uS: float,
        sl_segment_g__uS: float,
    ) -> tuple[Tensor, Tensor]:
        """Coupled BL/SL wire Newton step at frozen V_clamp / V_SL_drive.

        Args:
            f_bl_kcl: BL KCL residuals [uA].
                Shape: `[..., num_col, num_row]`.
            f_sl_kcl: SL KCL residuals [uA].
                Shape: `[..., num_col, num_row]`.
            g_cell_bl_eff: BL-side cell derivatives [uS].
                Shape: `[..., num_col, num_row]`.
            g_cell_sl_eff: Negated SL-side cell derivatives [uS].
                Shape: `[..., num_col, num_row]`.
            bl_segment_g__uS: BL rail conductance of one lattice link.
            sl_segment_g__uS: SL rail conductance of one lattice link.

        Returns:
            BL and SL Newton voltage steps [V], one tensor each.
            Shape: `[..., num_col, num_row]`.
        """
        # Shape: [..., num_col, num_row, 2, 2]
        diag_blocks = _wire_diag_blocks(g_cell_bl_eff, g_cell_sl_eff, bl_segment_g__uS, sl_segment_g__uS)
        # RHS = [-F_BL, -F_SL] stacked.
        # Shape: [..., num_col, num_row, 2]
        rhs = torch.stack((-f_bl_kcl, -f_sl_kcl), dim=-1)

        # The block solver claims the row axis as its N axis and the per-node
        # rail pair as its 2x2 block, so the column dim sits in its leading
        # batch. Neighbouring rows couple through their shared rail link
        # alone, which is the constant off-block it takes as two scalars.
        # Shape: [..., num_col, num_row, 2]
        delta = solve_block_tridiagonal_2x2_uniform(
            diag_blocks,
            rhs,
            off_block=(-bl_segment_g__uS, -sl_segment_g__uS),
        )
        # Unpack into (dv_bl, dv_sl).
        return delta[..., 0], delta[..., 1]

    @staticmethod
    def _compute_k_inner_coupled_2x2(
        g_cell_bl_eff: Tensor,
        g_cell_sl_eff: Tensor,
        bl_segment_g__uS: float,
        sl_segment_g__uS: float,
    ) -> Tensor:
        """Compute the 2x2 `K_inner = dV_array[0] / dV_clamp` per column.

        A clamp reaches the array through its own rail link alone, so its
        forcing is that link's conductance at row 0 and nothing anywhere else.
        The frozen array being linear in that forcing, each solve runs on the
        bare row-0 unit vector and the link conductance scales the extracted
        row-0 response.

        Args:
            g_cell_bl_eff: BL-side cell derivatives [uS].
                Shape: `[..., num_col, num_row]`.
            g_cell_sl_eff: Negated SL-side cell derivatives [uS].
                Shape: `[..., num_col, num_row]`.
            bl_segment_g__uS: BL rail conductance of one lattice link.
            sl_segment_g__uS: SL rail conductance of one lattice link.

        Returns:
            Clamp-to-port-node sensitivity.
            Shape: `[..., num_col, 2, 2]`.
        """
        # Shape: [..., num_col, num_row, 2, 2]
        diag_blocks = _wire_diag_blocks(g_cell_bl_eff, g_cell_sl_eff, bl_segment_g__uS, sl_segment_g__uS)
        off_block = (-bl_segment_g__uS, -sl_segment_g__uS)

        # Unit forcing at row 0, one rail at a time.
        # Shape: [..., num_col, num_row]
        zeros_node = torch.zeros_like(g_cell_bl_eff)
        unit_row0 = F.pad(torch.ones_like(g_cell_bl_eff[..., :1]), (0, g_cell_bl_eff.shape[-1] - 1))

        # Shape: [..., num_col, num_row, 2]
        u_bl = solve_block_tridiagonal_2x2_uniform(
            diag_blocks,
            torch.stack((unit_row0, zeros_node), dim=-1),
            off_block=off_block,
        )
        u_sl = solve_block_tridiagonal_2x2_uniform(
            diag_blocks,
            torch.stack((zeros_node, unit_row0), dim=-1),
            off_block=off_block,
        )

        # K[:, 0] is the BL-forced row-0 response, K[:, 1] the SL-forced one,
        # each carrying the conductance of the link that forced it.
        # Shape: [..., num_col, 2]
        k_col_bl = u_bl.select(-2, 0) * bl_segment_g__uS
        k_col_sl = u_sl.select(-2, 0) * sl_segment_g__uS
        # Stack into [..., num_col, 2, 2] with K[:, j] as columns.
        return torch.stack((k_col_bl, k_col_sl), dim=-1)
