"""Block-nested DC solver for a crossbar array with PARALLEL BL/SL rails.

The two array rails (BL, SL) run side by side along a single shared series
axis; the gate/control line is a driven boundary, so the parallel (per-driver)
lines are independent and batched. Orthogonal BL/SL meshes are a future,
separate solver.

See also:
    docs/reference/xbar/solver.md
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.xbar.cell import XbarCell, XbarCellDCOP, XbarCellSnap

from .base import Solver, SolverConfig, SolverDCOP, SolverResiduals
from .clamp import ClampDriver, ClampSnap
from .primitives import (
    col_driver_current,
    col_wire_kcl_residual,
    solve_block_tridiagonal,
)

# ---------------------------------------------------------------------------
# Per-call method-generic type vars
# ---------------------------------------------------------------------------

# Bound only inside the solve-method signatures so mypy infers them per
# call and the solver class itself stays non-generic. ``CellSnapT`` /
# ``CellDCOPT`` carry the cell's own bounds (mirroring ``XbarCell``); the
# driver snaps carry ``ClampSnap`` (mirroring ``ClampDriver``), which the
# solver relies on to read each driver snap's injected ``v_ref__V`` seed.
CellSnapT = TypeVar("CellSnapT", bound=XbarCellSnap)
CellDCOPT = TypeVar("CellDCOPT", bound=XbarCellDCOP)
BLSnapT = TypeVar("BLSnapT", bound=ClampSnap)
SLSnapT = TypeVar("SLSnapT", bound=ClampSnap)

# ---------------------------------------------------------------------------
# Trailing-axis normalization (non-canonical layouts)
# ---------------------------------------------------------------------------

_T = TypeVar("_T")


def _swap_trailing_axes(obj: _T) -> _T:
    """Swap the two trailing axes of every rank ≥ 2 tensor in a snap / DCOP.

    Maps a ``series_axis == -2`` caller grid ``[..., num_series, num_line]`` to
    the canonical ``[..., num_line, num_series]`` (and back). Frozen dataclasses
    are rebuilt field-wise with :func:`dataclasses.replace`; nested dataclass
    fields recurse; rank ≤ 1 tensors (per-line / per-series 1-D state) and
    non-tensor fields pass through unchanged. Used only on the ``-2`` path; the
    canonical ``-1`` path never calls this (byte-identical fast path).
    """
    if isinstance(obj, Tensor):
        return obj.transpose(-1, -2) if obj.ndim >= 2 else obj
    if hasattr(type(obj), "__dataclass_fields__"):
        swapped = {f.name: _swap_trailing_axes(getattr(obj, f.name)) for f in fields(obj)}
        return replace(obj, **swapped)
    return obj


# ---------------------------------------------------------------------------
# Solver config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NestedParallelRailSolverConfig(SolverConfig):
    """Workload-tuned numerical knobs for :class:`NestedParallelRailSolver`.

    Carries only fields that need re-calibration per chip preset. The
    Newton damping caps are method-intrinsic safety constants and live
    on the solver class itself.

    Attributes:
        n_outer: Outer Newton iterations on the per-column clamp voltage.
            Each outer step takes one implicit-Jacobian Newton step on
            V_clamp and then runs ``n_inner`` inner array Newton steps at
            the updated V_clamp (clamp-first ordering).
        n_inner: Inner Newton iterations on the wire / cell coupled state
            at a frozen V_clamp boundary. ``n_inner = 1`` is often enough
            since the outer Gauss-Seidel sweep drives the per-cell
            convergence — see ``solver_calibrate.nested``.
    """

    n_outer: int
    n_inner: int

    def validate(self) -> None:
        super().validate()
        self._require_pos(self.n_outer, "n_outer")
        self._require_pos(self.n_inner, "n_inner")


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


@Solver.register_key(NestedParallelRailSolverConfig)
class NestedParallelRailSolver(Solver):
    """Block Gauss-Seidel + implicit-Newton DC solver for a parallel BL/SL tile.

    Canonical layout: of the two trailing axes of the cell grid
    ``[..., num_line, num_series]``, the LAST axis is the series axis (the wire
    ladder / IR-drop direction) and the second-to-last is the parallel
    (per-driver) line axis. ``series_axis`` (a construction-time int, default
    ``-1``) names the caller's series axis; the body always runs in this
    canonical orientation, with entry-time normalization for non-canonical
    callers.
    """

    MAX_OUTER_STEP__V: float = 0.10
    MAX_INNER_STEP__V: float = 0.05

    def __init__(self, *, config: NestedParallelRailSolverConfig, series_axis: int = -1) -> None:
        """Bind the solver to its fixed iteration / step knobs and layout axis.

        Args:
            config: Fixed iteration / step knobs. The cell and the two
                clamp drivers are supplied per call to :meth:`solve_dc`,
                not here — the solver is stateless except for ``config``.
            series_axis: Construction-time index (``-1`` or ``-2``) of the
                caller's series axis among the two trailing cell-grid axes.
                ``-1`` is the canonical no-op fast path; ``-2`` permutes the
                two trailing axes to canonical at entry and back at exit. A
                plain Python int, hence a compile-time constant under
                ``@torch.compile(dynamic=False)`` — no dynamic guard, no
                recompile.
        """
        if series_axis not in (-1, -2):
            raise ValueError(f"require: series_axis in (-1, -2), got {series_axis}")
        self.config = config
        self._series_axis = series_axis

    # ---------------------------------------------------------------
    # Public entry point: full nested solve
    # ---------------------------------------------------------------

    # Compiled as a fixed-shape regional leaf. ``solve_array`` (an eager island)
    # feeds it one chunk at a time at a constant ``solve_chunk_size`` leading,
    # so a single graph is built once and reused across every chunk, VMM, and
    # caller instance (``inline_inbuilt_nn_modules`` lifts the device buffers as
    # shape-guarded inputs — verified one shared graph across instances).
    # ``dynamic=False`` pins the unrolled iteration counts (n_outer, num_series) as
    # compile-time constants. The compiled block-tridiagonal Thomas sweep runs
    # fastest and leanest, and with a uniform chunk shape its one long cold
    # compile happens once and is cached.
    @torch.compile(dynamic=False)
    def solve_dc(
        self,
        *,
        bl_segment_r__MOhm: Tensor,
        sl_segment_r__MOhm: Tensor,
        bl_segment_g__uS: Tensor,
        sl_segment_g__uS: Tensor,
        cell: XbarCell[CellSnapT, CellDCOPT],
        cell_snap: CellSnapT,
        bl_driver: ClampDriver[BLSnapT],
        bl_driver_snap: BLSnapT,
        sl_driver: ClampDriver[SLSnapT],
        sl_driver_snap: SLSnapT,
        compute_residuals: bool = False,
    ) -> SolverDCOP[CellDCOPT]:
        """Solve the fabricated tile for one cell snap.

        Normalizes the caller's layout to canonical (series axis last) before
        the block-tridiagonal body. ``self._series_axis == -1`` is the
        canonical no-op fast path (byte-identical to the bare body); ``-2``
        swaps the two trailing cell-grid axes of ``cell_snap`` in and the grid
        outputs back. ``_series_axis`` is a Python int, so the branch is a
        compile-time constant — the ``-1`` path compiles to exactly the
        canonical graph with no permute.

        Args:
            bl_segment_r__MOhm: 1-D BL segment resistances; index 0
                is driver-to-first.
            sl_segment_r__MOhm: 1-D SL segment resistances; index 0
                is driver-to-first.
            bl_segment_g__uS: BL segment conductances.
            sl_segment_g__uS: SL segment conductances.
            cell: Pluggable cell; owns the device branch and condenses any
                internal node.
            cell_snap: Per-solve cell snap bundling the device snaps and the
                per-cell control-line (WL) drive.
            bl_driver: BL clamp driver.
            bl_driver_snap: Per-solve BL driver snap.
            sl_driver: SL clamp driver.
            sl_driver_snap: Per-solve SL driver snap.
            compute_residuals: When True, populate
                :attr:`SolverDCOP.residuals` after the exit-state refresh;
                when False (hot path) leaves it as ``None``.

        Returns:
            Complete steady-state solution for the current VMM (in the
            caller's original layout).
        """
        if self._series_axis == -1:
            return self._solve_dc_canonical(
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
                compute_residuals=compute_residuals,
            )
        # series_axis == -2: caller grid is [..., num_series, num_line].
        # Per-line driver snaps ([..., num_line]) are already canonical; only
        # the cell grid's two trailing axes need swapping in, and the grid
        # outputs back.
        dcop = self._solve_dc_canonical(
            bl_segment_r__MOhm=bl_segment_r__MOhm,
            sl_segment_r__MOhm=sl_segment_r__MOhm,
            bl_segment_g__uS=bl_segment_g__uS,
            sl_segment_g__uS=sl_segment_g__uS,
            cell=cell,
            cell_snap=_swap_trailing_axes(cell_snap),
            bl_driver=bl_driver,
            bl_driver_snap=bl_driver_snap,
            sl_driver=sl_driver,
            sl_driver_snap=sl_driver_snap,
            compute_residuals=compute_residuals,
        )
        return self._restore_grid_layout(dcop)

    def _restore_grid_layout(self, dcop: SolverDCOP[CellDCOPT]) -> SolverDCOP[CellDCOPT]:
        """Swap the grid-shaped DCOP fields back to the caller's ``-2`` layout.

        Grid fields (``v_bl_node`` / ``v_sl_node``, the cell DCOP, and the
        wire residuals) carry the two trailing array axes and are transposed
        back; per-line fields (``i_*_driver`` / ``v_bl_clamp`` / ``v_sl_drive``,
        the clamp residuals) keep the parallel axis last and pass through.
        """
        residuals = dcop.residuals
        if residuals is not None:
            residuals = SolverResiduals(
                wire_bl__uA=residuals.wire_bl__uA.transpose(-1, -2),
                wire_sl__uA=residuals.wire_sl__uA.transpose(-1, -2),
                clamp_bl__V=residuals.clamp_bl__V,
                clamp_sl__V=residuals.clamp_sl__V,
            )
        return SolverDCOP(
            i_bl_driver=dcop.i_bl_driver,
            i_sl_driver=dcop.i_sl_driver,
            v_bl_node=dcop.v_bl_node.transpose(-1, -2),
            v_sl_node=dcop.v_sl_node.transpose(-1, -2),
            cell=_swap_trailing_axes(dcop.cell),
            v_bl_clamp=dcop.v_bl_clamp,
            v_sl_drive=dcop.v_sl_drive,
            residuals=residuals,
        )

    def _solve_dc_canonical(
        self,
        *,
        bl_segment_r__MOhm: Tensor,
        sl_segment_r__MOhm: Tensor,
        bl_segment_g__uS: Tensor,
        sl_segment_g__uS: Tensor,
        cell: XbarCell[CellSnapT, CellDCOPT],
        cell_snap: CellSnapT,
        bl_driver: ClampDriver[BLSnapT],
        bl_driver_snap: BLSnapT,
        sl_driver: ClampDriver[SLSnapT],
        sl_driver_snap: SLSnapT,
        compute_residuals: bool = False,
    ) -> SolverDCOP[CellDCOPT]:
        """Canonical-layout nested solve (series axis last).

        Verbatim block-tridiagonal body. Called directly on the ``-1`` fast
        path and around the trailing-axis swap on the ``-2`` path. Inputs are
        canonical ``[..., num_line, num_series]`` cell grids with per-line
        ``[..., num_line]`` driver snaps; outputs are canonical too. See
        :meth:`solve_dc` for the argument contract.
        """

        # --- Per-solve wire Jacobian templates ---

        # Shape: [num_series]
        bl_wire_diag_tmpl = bl_segment_g__uS + F.pad(bl_segment_g__uS[1:], (0, 1))
        sl_wire_diag_tmpl = sl_segment_g__uS + F.pad(sl_segment_g__uS[1:], (0, 1))
        # Shape: [num_series-1]
        bl_wire_offdiag = -bl_segment_g__uS[1:]
        sl_wire_offdiag = -sl_segment_g__uS[1:]
        # Shape: []
        bl_driver_segment_g = bl_segment_g__uS[0]
        sl_driver_segment_g = sl_segment_g__uS[0]

        # --- Warm start phase 1: condensed cell seed at ref clamp ---

        # The cell condenses its own internal node; the solver seeds only
        # the BL / SL node voltages. The reference-clamp seed rides in each
        # driver snap (the injected, post-noise tap); unsqueezing the
        # trailing axis aligns ``(*lead, num_line)`` with the
        # ``(*lead, num_line, num_series)`` branch grid so the cell broadcasts
        # its branch against the fabricated device grid and reports the
        # full branch shape. A scalar snap value also broadcasts here.
        v_bl_seed = bl_driver_snap.v_ref__V
        v_sl_seed = sl_driver_snap.v_ref__V
        # Shape: [..., num_line, num_series]
        i_cell, _g_bl_init, _g_sl_init = cell.solve_branch(
            v_bl_seed.unsqueeze(-1), v_sl_seed.unsqueeze(-1), cell_snap
        )
        *_batch, num_col, num_row = i_cell.shape
        if not (num_col > 1):
            raise ValueError(f"require: num_col ({num_col}) > 1")
        if not (num_row > 1):
            raise ValueError(f"require: num_row ({num_row}) > 1")

        # --- First clamp-driver evaluation from cell-sum seed ---

        # BL clamp sees the cell current drained from BL; SL drive sees the
        # same current pushed into SL (the cell self-converges so one
        # signed branch current serves both rails).
        # Shape: [..., num_line, num_series] -> [..., num_line]
        i_bl_seed__uA = i_cell.sum(dim=-1)
        i_sl_seed__uA = -i_cell.sum(dim=-1)
        # Shape: [..., num_line]
        v_bl_clamp__V, _ = bl_driver.solve_clamp(i_bl_seed__uA, bl_driver_snap, v_clamp_init__V=None)
        v_sl_drive__V, _ = sl_driver.solve_clamp(i_sl_seed__uA, sl_driver_snap, v_clamp_init__V=None)

        # --- Warm start phase 2: first-order IR-drop wire seed ---

        # BL ladder propagates ``+i`` (drained from BL); SL ladder
        # propagates ``-i`` (injected into SL).
        # Shape: [..., num_line, num_series]
        v_bl_node, v_sl_node = self._wire_ir_drop_seed(
            i_cell,
            v_bl_clamp__V,
            v_sl_drive__V,
            bl_segment_r__MOhm,
            sl_segment_r__MOhm,
            i_cell.ndim,
            num_row,
        )

        # --- Pre-loop cell refresh ---

        # First outer step's V_clamp Jacobian needs the signed branch
        # derivatives and the cell current at the POST-IR-drop seed; the
        # scalar warm start above reported them at the PRE-IR-drop state.
        i_cell, di_dvbl, di_dvsl = cell.solve_branch(v_bl_node, v_sl_node, cell_snap)

        # --- Outer V_clamp Newton  ×  n_outer ---

        # Clamp-FIRST ordering per outer step:
        #   (a) implicit-Jacobian Newton step on V_clamp using current
        #       g_cell_eff for K_inner
        #   (b) inner array Newton (n_inner damped steps) at the NEW
        #       V_clamp so the loop exits with V_array converged for the
        #       latest V_clamp boundary (no final consistency refresh
        #       needed).

        max_inner_step__V = self.MAX_INNER_STEP__V
        max_outer_step__V = self.MAX_OUTER_STEP__V
        n_inner = self.config.n_inner

        for _ in range(self.config.n_outer):
            # (a) Coupled 2×2 outer Newton step on (V_BL_clamp, V_SL_drive).
            # Implicit-function-theorem K = ∂V_node[0]/∂V_clamp is a 2×2
            # matrix per column; one block-tridiagonal solve per basis
            # direction reads off its columns. K captures the BL ↔ SL
            # cross-coupling through the cell — a variable-SL chip can
            # have non-negligible cross terms.
            # Signed-to-magnitude adaptation: the wire Jacobian is built
            # from the BL-side branch conductance ``di_dvbl`` (>= 0) and
            # the SL-side magnitude ``-di_dvsl`` (>= 0, since ``di_dvsl``
            # <= 0). Feeding these keeps the assembled block-2×2 Jacobian
            # identical in value.
            g_cell_bl_eff = di_dvbl
            g_cell_sl_eff = -di_dvsl
            # Shape: [..., num_line, 2, 2]
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
            # Shape: [..., num_line]
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
            # Shape: [..., num_line]
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
            # Shape: [..., num_line, 2, 2]
            df_outer = torch.stack([df_row0, df_row1], dim=-2)
            # Shape: [..., num_line, 2]
            f_outer = torch.stack([f_bl_outer, f_sl_outer], dim=-1)
            # Solve 2×2 system per column: δ = -inv(df_outer) · f_outer.
            delta_2 = torch.linalg.solve(df_outer, -f_outer.unsqueeze(-1)).squeeze(-1)
            delta_bl = delta_2[..., 0].clamp(min=-max_outer_step__V, max=max_outer_step__V)
            delta_sl = delta_2[..., 1].clamp(min=-max_outer_step__V, max=max_outer_step__V)
            v_bl_clamp__V = v_bl_clamp__V + delta_bl
            v_sl_drive__V = v_sl_drive__V + delta_sl

            # (b) inner array Newton at the NEW V_clamp.
            # Shape: [..., num_line] -> [..., num_line, 1]
            v_bl_clamp_grid__V = v_bl_clamp__V.unsqueeze(-1)
            v_sl_drive_grid__V = v_sl_drive__V.unsqueeze(-1)

            for _ in range(n_inner):
                # Shape: [..., num_line, num_series]
                i_cell, di_dvbl, di_dvsl = cell.solve_branch(v_bl_node, v_sl_node, cell_snap)
                g_cell_bl_eff = di_dvbl
                g_cell_sl_eff = -di_dvsl
                # The cell self-converges its internal node, so one signed
                # branch current ``i`` serves both rails: BL wire KCL uses
                # ``+i`` (current drained from BL), SL wire KCL uses ``-i``
                # (current injected into SL).
                # Shape: [..., num_line, num_series]
                f_bl_kcl = col_wire_kcl_residual(v_bl_node, v_bl_clamp_grid__V, bl_segment_g__uS, i_cell)
                f_sl_kcl = col_wire_kcl_residual(v_sl_node, v_sl_drive_grid__V, sl_segment_g__uS, -i_cell)
                # Coupled block-2×2 wire Newton — captures BL/SL cross terms
                # ``∂F_BL/∂V_SL = -g_cell_sl_eff`` and ``∂F_SL/∂V_BL = -g_cell_bl_eff``.
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

        # --- Exit-state cell refresh ---

        # The last inner step updated V_BL / V_SL but the cell working
        # point still carries the pre-update wire state. One full cell
        # solve re-aligns it (and condenses the internal node + optional
        # per-cell KCL residual) so the returned DCOP is self-consistent.
        cell_dcop = cell.solve_dc(v_bl_node, v_sl_node, cell_snap, compute_residuals=compute_residuals)
        i_cell = cell_dcop.i__uA

        # Boundary currents on the converged wire + clamp state.
        v_bl_clamp_grid__V = v_bl_clamp__V.unsqueeze(-1)
        v_sl_drive_grid__V = v_sl_drive__V.unsqueeze(-1)
        i_bl_driver = col_driver_current(v_bl_node, v_bl_clamp_grid__V, bl_segment_g__uS)
        i_sl_driver = col_driver_current(v_sl_node, v_sl_drive_grid__V, sl_segment_g__uS)

        # --- Optional solver-owned residual diagnostics -------------------

        residuals: SolverResiduals | None
        if compute_residuals:
            # Wire residuals: BL uses ``+i`` (drained from BL), SL uses
            # ``-i`` (injected into SL).
            wire_bl_res = col_wire_kcl_residual(v_bl_node, v_bl_clamp_grid__V, bl_segment_g__uS, i_cell).abs()
            wire_sl_res = col_wire_kcl_residual(v_sl_node, v_sl_drive_grid__V, sl_segment_g__uS, -i_cell).abs()
            # Clamp residual: |driver(I_port) - V_clamp| at the converged
            # operating point. Zero at the outer Newton fixed point.
            i_bl_port_final = (v_bl_clamp__V - v_bl_node.select(-1, 0)) * bl_driver_segment_g
            i_sl_port_final = (v_sl_drive__V - v_sl_node.select(-1, 0)) * sl_driver_segment_g
            v_bl_target_final, _ = bl_driver.solve_clamp(
                i_bl_port_final,
                bl_driver_snap,
                v_clamp_init__V=v_bl_clamp__V,
            )
            v_sl_target_final, _ = sl_driver.solve_clamp(
                i_sl_port_final,
                sl_driver_snap,
                v_clamp_init__V=v_sl_drive__V,
            )
            clamp_bl_res = (v_bl_target_final - v_bl_clamp__V).abs()
            clamp_sl_res = (v_sl_target_final - v_sl_drive__V).abs()
            residuals = SolverResiduals(
                wire_bl__uA=wire_bl_res,
                wire_sl__uA=wire_sl_res,
                clamp_bl__V=clamp_bl_res,
                clamp_sl__V=clamp_sl_res,
            )
        else:
            residuals = None

        return SolverDCOP(
            i_bl_driver=i_bl_driver,
            i_sl_driver=i_sl_driver,
            v_bl_node=v_bl_node,
            v_sl_node=v_sl_node,
            cell=cell_dcop,
            v_bl_clamp=v_bl_clamp__V,
            v_sl_drive=v_sl_drive__V,
            residuals=residuals,
        )

    # ---------------------------------------------------------------
    # Public entry point: inner-only solve (calibration tool)
    # ---------------------------------------------------------------

    def solve_array_fixed_clamp(
        self,
        *,
        v_bl_clamp__V: Tensor,
        v_sl_drive__V: Tensor,
        bl_segment_r__MOhm: Tensor,
        sl_segment_r__MOhm: Tensor,
        bl_segment_g__uS: Tensor,
        sl_segment_g__uS: Tensor,
        cell: XbarCell[CellSnapT, CellDCOPT],
        cell_snap: CellSnapT,
        compute_residuals: bool = False,
    ) -> SolverDCOP[CellDCOPT]:
        """Run only the inner array Newton loop at FIXED clamp boundaries.

        Debug entry point — bypasses the outer V_clamp Newton entirely so
        the inner sub-problem can be inspected in isolation (clamps
        pinned at the supplied values, no BL / SL clamp-driver feedback).
        Layout normalization mirrors :meth:`solve_dc`: ``-1`` runs the
        canonical body directly; ``-2`` swaps the two trailing cell-grid axes
        in and the grid outputs back (per-line clamps stay as-is).

        Args:
            v_bl_clamp__V: BL clamp voltage held fixed throughout the
                solve. Shape: ``[..., num_line]``.
            v_sl_drive__V: SL drive voltage held fixed. Shape:
                ``[..., num_line]``.
            cell: Pluggable cell; owns the device branch and condenses any
                internal node.
            cell_snap: Per-solve cell snap. Clamp-driver snaps and outer
                driver state are NOT touched.
            (other args): same as :meth:`solve_dc`.
            compute_residuals: Populate ``residuals`` if True.

        Returns:
            ``SolverDCOP`` with the inner solution; ``i_bl_driver``
            and ``i_sl_driver`` are computed from the held clamp values
            so the caller can inspect inner-port currents.
        """
        if self._series_axis == -1:
            return self._solve_array_fixed_clamp_canonical(
                v_bl_clamp__V=v_bl_clamp__V,
                v_sl_drive__V=v_sl_drive__V,
                bl_segment_r__MOhm=bl_segment_r__MOhm,
                sl_segment_r__MOhm=sl_segment_r__MOhm,
                bl_segment_g__uS=bl_segment_g__uS,
                sl_segment_g__uS=sl_segment_g__uS,
                cell=cell,
                cell_snap=cell_snap,
                compute_residuals=compute_residuals,
            )
        dcop = self._solve_array_fixed_clamp_canonical(
            v_bl_clamp__V=v_bl_clamp__V,
            v_sl_drive__V=v_sl_drive__V,
            bl_segment_r__MOhm=bl_segment_r__MOhm,
            sl_segment_r__MOhm=sl_segment_r__MOhm,
            bl_segment_g__uS=bl_segment_g__uS,
            sl_segment_g__uS=sl_segment_g__uS,
            cell=cell,
            cell_snap=_swap_trailing_axes(cell_snap),
            compute_residuals=compute_residuals,
        )
        return self._restore_grid_layout(dcop)

    def _solve_array_fixed_clamp_canonical(
        self,
        *,
        v_bl_clamp__V: Tensor,
        v_sl_drive__V: Tensor,
        bl_segment_r__MOhm: Tensor,
        sl_segment_r__MOhm: Tensor,
        bl_segment_g__uS: Tensor,
        sl_segment_g__uS: Tensor,
        cell: XbarCell[CellSnapT, CellDCOPT],
        cell_snap: CellSnapT,
        compute_residuals: bool = False,
    ) -> SolverDCOP[CellDCOPT]:
        """Canonical-layout inner-only solve (series axis last).

        Verbatim inner Newton body. See :meth:`solve_array_fixed_clamp` for
        the argument contract.
        """
        bl_wire_diag_tmpl = bl_segment_g__uS + F.pad(bl_segment_g__uS[1:], (0, 1))
        bl_wire_offdiag = -bl_segment_g__uS[1:]
        sl_wire_diag_tmpl = sl_segment_g__uS + F.pad(sl_segment_g__uS[1:], (0, 1))
        sl_wire_offdiag = -sl_segment_g__uS[1:]

        # Condensed cell warm start with the supplied clamp as the seed.
        # The grid clamp broadcasts against the cell's fabricated device
        # grid; the returned branch current carries the full
        # ``[..., num_line, num_series]`` shape.
        # Shape: [..., num_line] -> [..., num_line, 1]
        v_bl_clamp_grid__V = v_bl_clamp__V.unsqueeze(-1)
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

        max_inner_step__V = self.MAX_INNER_STEP__V
        for _ in range(self.config.n_inner):
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

        # Final cell refresh so the returned cell working point aligns with
        # the returned wire state.
        cell_dcop = cell.solve_dc(v_bl_node, v_sl_node, cell_snap, compute_residuals=compute_residuals)
        i_cell = cell_dcop.i__uA

        i_bl_driver = col_driver_current(v_bl_node, v_bl_clamp_grid__V, bl_segment_g__uS)
        i_sl_driver = col_driver_current(v_sl_node, v_sl_drive_grid__V, sl_segment_g__uS)

        residuals: SolverResiduals | None
        if compute_residuals:
            # Wire residuals: BL uses ``+i``, SL uses ``-i``.
            wire_bl_res = col_wire_kcl_residual(v_bl_node, v_bl_clamp_grid__V, bl_segment_g__uS, i_cell).abs()
            wire_sl_res = col_wire_kcl_residual(v_sl_node, v_sl_drive_grid__V, sl_segment_g__uS, -i_cell).abs()
            # Inner-only path: clamps are PINNED inputs, not solved → clamp
            # residual is conceptually 0. Fill with zeros for shape parity.
            clamp_zero = torch.zeros_like(v_bl_clamp__V)
            residuals = SolverResiduals(
                wire_bl__uA=wire_bl_res,
                wire_sl__uA=wire_sl_res,
                clamp_bl__V=clamp_zero,
                clamp_sl__V=clamp_zero,
            )
        else:
            residuals = None

        return SolverDCOP(
            i_bl_driver=i_bl_driver,
            i_sl_driver=i_sl_driver,
            v_bl_node=v_bl_node,
            v_sl_node=v_sl_node,
            cell=cell_dcop,
            v_bl_clamp=v_bl_clamp__V,
            v_sl_drive=v_sl_drive__V,
            residuals=residuals,
        )

    # ---------------------------------------------------------------
    # Wire helpers
    # ---------------------------------------------------------------

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

        The cell condenses to one signed branch current ``i``: the BL
        ladder propagates ``+i`` (drained from BL) and the SL ladder
        propagates ``-i`` (injected into SL). The seed accumulates the
        downstream current at each segment and subtracts the cumulative
        IR drop from the boundary voltage.
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

        Builds and solves the coupled block-2×2 tridiagonal wire system:
        the per-node diagonal block carries the cell's BL / SL
        cross-coupling and the sub / super blocks are diagonal (BL and SL
        are independent ladders, no cross-rail wire coupling). For an
        SL-grounded chip the coupling is numerically tiny, reducing to
        near-independent BL / SL solves; a variable-SL chip keeps the full
        linearisation through the same code path.

        Shape conventions:
          * ``v_bl_node``, ``f_bl_kcl``, ``f_sl_kcl``, ``g_*_eff``:
            ``[..., num_line, num_series]``.
          * ``bl/sl_wire_diag_tmpl``: ``[num_series]``.
          * ``bl/sl_wire_offdiag``: ``[num_series - 1]``.

        Returns ``(dv_bl, dv_sl)``, each shape ``[..., num_line, num_series]``.
        """
        num_axis = v_bl_node.shape[-1]
        shape_broadcast = [1] * v_bl_node.ndim
        shape_broadcast[-1] = num_axis

        # Per-row diagonal entries.
        # Shape: [..., num_line, num_series]
        bl_diag_node = bl_wire_diag_tmpl.view(shape_broadcast) + g_cell_bl_eff
        sl_diag_node = sl_wire_diag_tmpl.view(shape_broadcast) + g_cell_sl_eff
        cross_to_bl_from_sl = -g_cell_sl_eff  # ∂F_BL/∂V_SL
        cross_to_sl_from_bl = -g_cell_bl_eff  # ∂F_SL/∂V_BL

        # Assemble per-row 2×2 diagonal block.
        # Shape: [..., num_line, num_series, 2, 2]
        diag_row_top = torch.stack([bl_diag_node, cross_to_bl_from_sl], dim=-1)
        diag_row_bot = torch.stack([cross_to_sl_from_bl, sl_diag_node], dim=-1)
        diag_blocks = torch.stack([diag_row_top, diag_row_bot], dim=-2)

        # Off-diagonal blocks (BL-BL and SL-SL wire only).
        # sub_k[0] is unused; sup_k[N-1] is unused — pad with zeros.
        # Shape: [num_series]
        sub_bl_pad = F.pad(bl_wire_offdiag, (1, 0)).view(shape_broadcast)
        sub_sl_pad = F.pad(sl_wire_offdiag, (1, 0)).view(shape_broadcast)
        sup_bl_pad = F.pad(bl_wire_offdiag, (0, 1)).view(shape_broadcast)
        sup_sl_pad = F.pad(sl_wire_offdiag, (0, 1)).view(shape_broadcast)
        # Broadcast to [..., num_line, num_series]
        sub_bl_full = sub_bl_pad.expand_as(v_bl_node)
        sub_sl_full = sub_sl_pad.expand_as(v_bl_node)
        sup_bl_full = sup_bl_pad.expand_as(v_bl_node)
        sup_sl_full = sup_sl_pad.expand_as(v_bl_node)
        zero_full = torch.zeros_like(sub_bl_full)
        # 2×2 diagonal blocks: diag(-bl_offdiag, -sl_offdiag)
        # Shape: [..., num_line, num_series, 2, 2]
        sub_row_top = torch.stack([sub_bl_full, zero_full], dim=-1)
        sub_row_bot = torch.stack([zero_full, sub_sl_full], dim=-1)
        sub_blocks = torch.stack([sub_row_top, sub_row_bot], dim=-2)
        sup_row_top = torch.stack([sup_bl_full, zero_full], dim=-1)
        sup_row_bot = torch.stack([zero_full, sup_sl_full], dim=-1)
        sup_blocks = torch.stack([sup_row_top, sup_row_bot], dim=-2)

        # RHS = [-F_BL, -F_SL] stacked.
        # Shape: [..., num_line, num_series, 2]
        rhs = torch.stack([-f_bl_kcl, -f_sl_kcl], dim=-1)

        # Solve. Block tensors take last 3 dims [..., num_series, 2, 2];
        # rhs takes last 2 dims [..., num_series, 2]. The line dim sits in
        # the leading batch.
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

        The implicit-function-theorem sensitivity of the port-adjacent
        node voltages to the clamp pair. Each column is one block-
        tridiagonal solve of the coupled inner wire Jacobian ``J_inner``
        (the same assembled by ``_wire_newton_coupled_block2x2``) against a
        node-0 boundary-forcing basis vector, read off at row 0. The two
        basis solves (BL and SL) are mathematically independent — pack them
        only if a future block-tridiagonal kernel exposes a multi-RHS
        interface.

        Returns ``K`` of shape ``[..., num_line, 2, 2]``.
        """
        num_row = v_bl_node.shape[-1]
        shape_broadcast = [1] * v_bl_node.ndim
        shape_broadcast[-1] = num_row

        # --- Build the block-2×2 inner Jacobian (same as the wire Newton) ---

        bl_diag_node = bl_wire_diag_tmpl.view(shape_broadcast) + g_cell_bl_eff
        sl_diag_node = sl_wire_diag_tmpl.view(shape_broadcast) + g_cell_sl_eff
        cross_to_bl_from_sl = -g_cell_sl_eff  # ∂F_BL/∂V_SL
        cross_to_sl_from_bl = -g_cell_bl_eff  # ∂F_SL/∂V_BL
        # Shape: [..., num_line, num_series, 2, 2]
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

        # --- Boundary-forcing basis vectors ---

        # RHS for ``V_BL_CL`` perturbation: e_0 with BL component = g_BL_seg[0],
        # SL component = 0. RHS shape ``[..., num_line, num_series, 2]``.
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
        rhs_bl_basis = torch.stack([g_bl_at_row0_per_row, zeros_node], dim=-1)
        rhs_sl_basis = torch.stack([zeros_node, g_sl_at_row0_per_row], dim=-1)

        # --- Solve and extract row-0 responses ---

        # Shape: [..., num_line, num_series, 2]
        u_bl = solve_block_tridiagonal(sub_blocks, diag_blocks, sup_blocks, rhs_bl_basis)
        u_sl = solve_block_tridiagonal(sub_blocks, diag_blocks, sup_blocks, rhs_sl_basis)

        # K[:, 0] from BL-basis solve at row 0; K[:, 1] from SL-basis solve at row 0.
        # Shape: [..., num_line, 2]
        k_col_bl = u_bl.select(-2, 0)
        k_col_sl = u_sl.select(-2, 0)
        # Stack into [..., num_line, 2, 2] with K[:, j] as columns.
        return torch.stack([k_col_bl, k_col_sl], dim=-1)
