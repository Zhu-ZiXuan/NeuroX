"""Block-nested DC solver for a 1T1R crossbar array.

See also:
    docs/reference/xbar/_1t1r/solver.md
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.analog import Driver, DriverSnapshot
from neurox.analog.tia import TIA, TIASnapshot
from neurox.device import NMOS, RRAM, NMOSSnapshot, RRAMSnapshot
from neurox.xbar.solver import (
    col_driver_current,
    col_wire_kcl_residual,
    solve_block_tridiagonal,
)

from .solver import Solver1T1R, Solver1T1RConfig, Solver1T1RDCOP, Solver1T1RResiduals

# ---------------------------------------------------------------------------
# Solver config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NestedSolver1T1RConfig(Solver1T1RConfig):
    """Workload-tuned numerical knobs for :class:`NestedSolver1T1R`.

    Carries only fields that need re-calibration per chip preset. The
    Newton damping caps are method-intrinsic safety constants and live
    on the solver class itself.

    Attributes:
        n_outer: Outer Newton iterations on the per-column clamp voltage.
            Each outer step takes one implicit-Jacobian Newton step on
            V_clamp and then runs ``n_inner`` inner array Newton steps at
            the updated V_clamp (clamp-first ordering).
        n_inner: Inner Newton iterations on the wire / cell coupled state
            at a frozen V_clamp boundary. The inner system is an M-matrix
            with a unique solution per V_clamp; ``n_inner = 1`` is often
            enough since the outer Gauss-Seidel sweep drives the per-cell
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


@Solver1T1R.register_key(NestedSolver1T1RConfig)
class NestedSolver1T1R(Solver1T1R):
    """Block Gauss-Seidel + implicit-Newton DC solver for a 1T1R tile.

    See ``docs/reference/xbar/_1t1r/solver.md`` for algorithm
    and convergence rationale.
    """

    MAX_OUTER_STEP__V: float = 0.10
    MAX_INNER_STEP__V: float = 0.05

    def __init__(
        self,
        *,
        config: NestedSolver1T1RConfig,
        rram: RRAM,
        nmos: NMOS,
        bl_driver: TIA,
        sl_driver: Driver,
    ) -> None:
        """Bind the solver to its 1T1R device and boundary-driver instances.

        Args:
            config: Fixed iteration / step knobs.
            rram: Programmed RRAM device model.
            nmos: Fabricated access-NMOS model.
            bl_driver: BL clamp driver (typically the OpAmpTIA).
            sl_driver: SL clamp driver (typically an ideal clamp).
        """
        self.config = config
        self.rram = rram
        self.nmos = nmos
        self.bl_driver = bl_driver
        self.sl_driver = sl_driver

    # ---------------------------------------------------------------
    # Public entry point: full nested solve
    # ---------------------------------------------------------------

    def solve_dc(
        self,
        *,
        v_wl_drive__V: Tensor,
        bl_segment_r__MOhm: Tensor,
        sl_segment_r__MOhm: Tensor,
        bl_segment_g__uS: Tensor,
        sl_segment_g__uS: Tensor,
        rram_snapshot: RRAMSnapshot,
        nmos_snapshot: NMOSSnapshot,
        bl_driver_snapshot: TIASnapshot,
        sl_driver_snapshot: DriverSnapshot,
        compute_residuals: bool = False,
    ) -> Solver1T1RDCOP:
        """Solve the fabricated 1T1R tile for one WL-drive tensor.

        Args:
            v_wl_drive__V: WL drive voltage tensor [V]. Shape: [..., 1, num_row].
            bl_segment_r__MOhm: 1-D BL segment resistances [MOhm]; index 0
                is driver-to-first.
            sl_segment_r__MOhm: 1-D SL segment resistances [MOhm]; index 0
                is driver-to-first.
            bl_segment_g__uS: BL segment conductances [uS].
            sl_segment_g__uS: SL segment conductances [uS].
            rram_snapshot: Per-solve RRAM snapshot.
            nmos_snapshot: Per-solve NMOS snapshot.
            bl_driver_snapshot: Per-solve BL driver snapshot.
            sl_driver_snapshot: Per-solve SL driver snapshot.
            compute_residuals: When True, populate
                :attr:`Solver1T1RDCOP.residuals` after the exit-state
                refresh; when False (hot path) leaves it as ``None``.

        Returns:
            Complete steady-state solution for the current VMM.
        """

        # --- Per-solve wire Jacobian templates ---

        # Shape: [num_row]
        bl_wire_diag_tmpl = bl_segment_g__uS + F.pad(bl_segment_g__uS[1:], (0, 1))
        sl_wire_diag_tmpl = sl_segment_g__uS + F.pad(sl_segment_g__uS[1:], (0, 1))
        # Shape: [num_row-1]
        bl_wire_offdiag = -bl_segment_g__uS[1:]
        sl_wire_offdiag = -sl_segment_g__uS[1:]
        # Shape: []
        bl_driver_segment_g = bl_segment_g__uS[0]
        sl_driver_segment_g = sl_segment_g__uS[0]

        # Shape: [..., num_col, num_row]
        rram_state_g_snapshot = rram_snapshot.g__uS
        *batch, num_col, num_row = rram_state_g_snapshot.shape
        if not (num_col > 1):
            raise ValueError(f"require: num_col ({num_col}) > 1")
        if not (num_row > 1):
            raise ValueError(f"require: num_row ({num_row}) > 1")
        # Shape: [..., num_col, num_row]
        # Shape: [..., 1, num_row] -> [..., num_col, num_row]
        v_wl_drive_grid__V = v_wl_drive__V.expand(*batch, num_col, num_row)

        # --- Warm start phase 1: Padé cell seed at ref clamp ---

        v_bl_clamp_ref__V = self.bl_driver.v_ref__V
        v_sl_drive_ref__V = self.sl_driver.v_ref__V
        # Shape: [..., num_col, num_row]
        v_bl_node_seed = torch.full_like(rram_state_g_snapshot, v_bl_clamp_ref__V)
        v_sl_node_seed = torch.full_like(rram_state_g_snapshot, v_sl_drive_ref__V)
        # Shape: [..., num_col, num_row]
        i_r, i_n, _g_bl_init, _g_sl_init, v_x_node = self._solve_cell_pade_warm_start(
            v_bl_node_seed,
            v_sl_node_seed,
            v_wl_drive_grid__V,
            rram_state_g_snapshot,
            rram_snapshot,
            nmos_snapshot,
        )

        # --- First clamp-driver evaluation from cell-sum seed ---

        # BL clamp sees the sum of RRAM currents drained from BL;
        # SL drive sees the sum of NMOS currents pushed into SL.
        # Shape: [..., num_col, num_row] -> [..., num_col]
        i_bl_seed__uA = i_r.sum(dim=-1)
        i_sl_seed__uA = -i_n.sum(dim=-1)
        # Shape: [..., num_col]
        v_bl_clamp__V, _ = self.bl_driver.solve_clamp(i_bl_seed__uA, bl_driver_snapshot, v_clamp_init__V=None)
        v_sl_drive__V, _ = self.sl_driver.solve_clamp(i_sl_seed__uA, sl_driver_snapshot, v_clamp_init__V=None)

        # --- Warm start phase 2: first-order IR-drop wire seed ---

        # BL ladder propagates the RRAM current; SL ladder propagates the
        # NMOS current. At the Padé seed I_R ≈ I_N already.
        # Shape: [..., num_col, num_row]
        v_bl_node, v_sl_node = self._wire_ir_drop_seed(
            i_r,
            i_n,
            v_bl_clamp__V,
            v_sl_drive__V,
            bl_segment_r__MOhm,
            sl_segment_r__MOhm,
            rram_state_g_snapshot.ndim,
            num_row,
        )

        # --- Pre-loop cell refresh ---

        # First outer step's V_clamp Jacobian needs g_cell_eff and the
        # per-rail cell currents at the POST-IR-drop seed; the Padé warm
        # start above returned them at the PRE-IR-drop state.
        i_r, i_n, g_cell_bl_eff, g_cell_sl_eff, v_x_node = self._solve_cell_newton_once(
            v_bl_node,
            v_sl_node,
            v_wl_drive_grid__V,
            v_x_node,
            rram_snapshot,
            nmos_snapshot,
        )

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

            # First-segment port current and TIA / SL-driver targets.
            # Shape: [..., num_col]
            i_bl_port__uA = (v_bl_clamp__V - v_bl_node.select(-1, 0)) * bl_driver_segment_g
            i_sl_port__uA = (v_sl_drive__V - v_sl_node.select(-1, 0)) * sl_driver_segment_g
            v_bl_target__V, r_bl_driver__MOhm = self.bl_driver.solve_clamp(
                i_bl_port__uA,
                bl_driver_snapshot,
                v_clamp_init__V=v_bl_clamp__V,
            )
            v_sl_target__V, r_sl_driver__MOhm = self.sl_driver.solve_clamp(
                i_sl_port__uA,
                sl_driver_snapshot,
                v_clamp_init__V=v_sl_drive__V,
            )

            # Outer Newton on F_outer(V_clamp) = V_target(V_clamp) − V_clamp.
            #   V_BL_target = bl_driver(g_BL_seg[0]·(V_BL_CL − V_BL[0](V_clamp)))
            #   V_SL_target = sl_driver(g_SL_seg[0]·(V_SL_DR − V_SL[0](V_clamp)))
            # dV_target/dV_clamp expands as the small-signal slope of the
            # driver times the port-current sensitivity, with K_inner
            # carrying the V_node[0] response.
            #
            # dF/dV_clamp (2×2 per col):
            #   [[ r_bl·g_bl·(1 − K[0,0]) − 1,   −r_bl·g_bl·K[0,1]    ],
            #    [ −r_sl·g_sl·K[1,0],            r_sl·g_sl·(1 − K[1,1]) − 1 ]]
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
            # Solve 2×2 system per column: delta = -inv(df_outer) · f_outer.
            delta_2 = torch.linalg.solve(df_outer, -f_outer.unsqueeze(-1)).squeeze(-1)
            delta_bl = delta_2[..., 0].clamp(min=-max_outer_step__V, max=max_outer_step__V)
            delta_sl = delta_2[..., 1].clamp(min=-max_outer_step__V, max=max_outer_step__V)
            v_bl_clamp__V = v_bl_clamp__V + delta_bl
            v_sl_drive__V = v_sl_drive__V + delta_sl

            # (b) inner array Newton at the NEW V_clamp.
            # Shape: [..., num_col] -> [..., num_col, 1]
            v_bl_clamp_grid__V = v_bl_clamp__V.unsqueeze(-1)
            v_sl_drive_grid__V = v_sl_drive__V.unsqueeze(-1)

            for _ in range(n_inner):
                # Shape: [..., num_col, num_row]
                i_r, i_n, g_cell_bl_eff, g_cell_sl_eff, v_x_node = self._solve_cell_newton_once(
                    v_bl_node,
                    v_sl_node,
                    v_wl_drive_grid__V,
                    v_x_node,
                    rram_snapshot,
                    nmos_snapshot,
                )
                # BL wire KCL uses ``I_R`` (RRAM is the device draining BL);
                # SL wire KCL uses ``I_N`` (NMOS is the device feeding SL).
                # During iteration ``I_R != I_N``; the cell-KCL residual
                # ``F_X = I_N - I_R`` is what the outer Newton drives to 0.
                # Shape: [..., num_col, num_row]
                f_bl_kcl = col_wire_kcl_residual(v_bl_node, v_bl_clamp_grid__V, bl_segment_g__uS, i_r)
                f_sl_kcl = col_wire_kcl_residual(v_sl_node, v_sl_drive_grid__V, sl_segment_g__uS, -i_n)
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

        # The last inner step updated V_BL / V_SL but the cell-side
        # currents and V_X still carry the pre-update wire state. One
        # cell solve re-aligns them so the returned DCOP is self-consistent.
        i_r, i_n, _, _, v_x_node = self._solve_cell_newton_once(
            v_bl_node,
            v_sl_node,
            v_wl_drive_grid__V,
            v_x_node,
            rram_snapshot,
            nmos_snapshot,
        )
        # ``i_cell`` reported in the DCOP is the RRAM current at the cell
        # (the BL-side reading); at convergence ``I_R == I_N`` so the
        # choice is cosmetic for downstream consumers.
        i_cell = i_r

        # Boundary currents on the converged wire + clamp state.
        v_bl_clamp_grid__V = v_bl_clamp__V.unsqueeze(-1)
        v_sl_drive_grid__V = v_sl_drive__V.unsqueeze(-1)
        i_bl_driver = col_driver_current(v_bl_node, v_bl_clamp_grid__V, bl_segment_g__uS)
        i_sl_driver = col_driver_current(v_sl_node, v_sl_drive_grid__V, sl_segment_g__uS)

        # --- Optional residual diagnostics --------------------------------

        residuals: Solver1T1RResiduals | None
        if compute_residuals:
            cell_res = (i_n - i_r).abs()
            # Wire residuals: BL uses I_R (drained by RRAM), SL uses I_N
            # (delivered by NMOS) — the device actually connected to each wire.
            wire_bl_res = col_wire_kcl_residual(v_bl_node, v_bl_clamp_grid__V, bl_segment_g__uS, i_r).abs()
            wire_sl_res = col_wire_kcl_residual(v_sl_node, v_sl_drive_grid__V, sl_segment_g__uS, -i_n).abs()
            # Clamp residual: |TIA(I_port) − V_clamp| at the converged operating
            # point. Zero at the outer Newton fixed point.
            i_bl_port_final = (v_bl_clamp__V - v_bl_node.select(-1, 0)) * bl_driver_segment_g
            i_sl_port_final = (v_sl_drive__V - v_sl_node.select(-1, 0)) * sl_driver_segment_g
            v_bl_target_final, _ = self.bl_driver.solve_clamp(
                i_bl_port_final,
                bl_driver_snapshot,
                v_clamp_init__V=v_bl_clamp__V,
            )
            v_sl_target_final, _ = self.sl_driver.solve_clamp(
                i_sl_port_final,
                sl_driver_snapshot,
                v_clamp_init__V=v_sl_drive__V,
            )
            clamp_bl_res = (v_bl_target_final - v_bl_clamp__V).abs()
            clamp_sl_res = (v_sl_target_final - v_sl_drive__V).abs()
            residuals = Solver1T1RResiduals(
                cell__uA=cell_res,
                wire_bl__uA=wire_bl_res,
                wire_sl__uA=wire_sl_res,
                clamp_bl__V=clamp_bl_res,
                clamp_sl__V=clamp_sl_res,
            )
        else:
            residuals = None

        return Solver1T1RDCOP(
            i_bl_driver=i_bl_driver,
            i_sl_driver=i_sl_driver,
            v_bl_node=v_bl_node,
            v_sl_node=v_sl_node,
            v_x_node=v_x_node,
            i_cell=i_cell,
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
        v_wl_drive__V: Tensor,
        v_bl_clamp__V: Tensor,
        v_sl_drive__V: Tensor,
        bl_segment_r__MOhm: Tensor,
        sl_segment_r__MOhm: Tensor,
        bl_segment_g__uS: Tensor,
        sl_segment_g__uS: Tensor,
        rram_snapshot: RRAMSnapshot,
        nmos_snapshot: NMOSSnapshot,
        compute_residuals: bool = False,
    ) -> Solver1T1RDCOP:
        """Run only the inner array Newton loop at FIXED clamp boundaries.

        Debug entry point — bypasses the outer V_clamp Newton entirely so
        the inner sub-problem can be inspected in isolation (clamps
        pinned at the supplied values, no TIA / SL-driver feedback).

        Args:
            v_wl_drive__V: WL drive voltage, same shape contract as
                :meth:`solve_dc`.
            v_bl_clamp__V: BL clamp voltage held fixed throughout the
                solve. Shape: ``[..., num_col]``.
            v_sl_drive__V: SL drive voltage held fixed. Shape:
                ``[..., num_col]``.
            (other args): same as :meth:`solve_dc`. Driver snapshots and
                outer driver state are NOT touched.
            compute_residuals: Populate ``residuals`` if True.

        Returns:
            ``Solver1T1RDCOP`` with the inner solution; ``i_bl_driver``
            and ``i_sl_driver`` are computed from the held clamp values
            so the caller can inspect inner-port currents.
        """
        bl_wire_diag_tmpl = bl_segment_g__uS + F.pad(bl_segment_g__uS[1:], (0, 1))
        bl_wire_offdiag = -bl_segment_g__uS[1:]
        sl_wire_diag_tmpl = sl_segment_g__uS + F.pad(sl_segment_g__uS[1:], (0, 1))
        sl_wire_offdiag = -sl_segment_g__uS[1:]

        rram_state_g_snapshot = rram_snapshot.g__uS
        *batch, num_col, num_row = rram_state_g_snapshot.shape
        v_wl_drive_grid__V = v_wl_drive__V.expand(*batch, num_col, num_row)

        # Padé warm start with the supplied clamp as the seed.
        v_bl_clamp_grid__V = v_bl_clamp__V.unsqueeze(-1)
        v_sl_drive_grid__V = v_sl_drive__V.unsqueeze(-1)
        v_bl_node_seed = v_bl_clamp_grid__V.expand_as(rram_state_g_snapshot)
        v_sl_node_seed = v_sl_drive_grid__V.expand_as(rram_state_g_snapshot)
        i_r_seed, i_n_seed, _g_bl_init, _g_sl_init, v_x_node = self._solve_cell_pade_warm_start(
            v_bl_node_seed,
            v_sl_node_seed,
            v_wl_drive_grid__V,
            rram_state_g_snapshot,
            rram_snapshot,
            nmos_snapshot,
        )

        # IR-drop wire seed: BL propagates I_R, SL propagates I_N.
        v_bl_node, v_sl_node = self._wire_ir_drop_seed(
            i_r_seed,
            i_n_seed,
            v_bl_clamp__V,
            v_sl_drive__V,
            bl_segment_r__MOhm,
            sl_segment_r__MOhm,
            rram_state_g_snapshot.ndim,
            num_row,
        )

        max_inner_step__V = self.MAX_INNER_STEP__V
        for _ in range(self.config.n_inner):
            i_r, i_n, g_cell_bl_eff, g_cell_sl_eff, v_x_node = self._solve_cell_newton_once(
                v_bl_node,
                v_sl_node,
                v_wl_drive_grid__V,
                v_x_node,
                rram_snapshot,
                nmos_snapshot,
            )
            # BL wire KCL uses I_R; SL wire KCL uses I_N.
            f_bl_kcl = col_wire_kcl_residual(v_bl_node, v_bl_clamp_grid__V, bl_segment_g__uS, i_r)
            f_sl_kcl = col_wire_kcl_residual(v_sl_node, v_sl_drive_grid__V, sl_segment_g__uS, -i_n)
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

        # Final cell refresh so the returned cell-side quantities align
        # with the returned wire state.
        i_r, i_n, _, _, v_x_node = self._solve_cell_newton_once(
            v_bl_node,
            v_sl_node,
            v_wl_drive_grid__V,
            v_x_node,
            rram_snapshot,
            nmos_snapshot,
        )
        i_cell = i_r

        i_bl_driver = col_driver_current(v_bl_node, v_bl_clamp_grid__V, bl_segment_g__uS)
        i_sl_driver = col_driver_current(v_sl_node, v_sl_drive_grid__V, sl_segment_g__uS)

        residuals: Solver1T1RResiduals | None
        if compute_residuals:
            cell_res = (i_n - i_r).abs()
            # Wire residuals: BL uses I_R (drained by RRAM), SL uses I_N.
            wire_bl_res = col_wire_kcl_residual(v_bl_node, v_bl_clamp_grid__V, bl_segment_g__uS, i_r).abs()
            wire_sl_res = col_wire_kcl_residual(v_sl_node, v_sl_drive_grid__V, sl_segment_g__uS, -i_n).abs()
            # Inner-only path: clamps are PINNED inputs, not solved → clamp
            # residual is conceptually 0. Fill with zeros for shape parity.
            clamp_zero = torch.zeros_like(v_bl_clamp__V)
            residuals = Solver1T1RResiduals(
                cell__uA=cell_res,
                wire_bl__uA=wire_bl_res,
                wire_sl__uA=wire_sl_res,
                clamp_bl__V=clamp_zero,
                clamp_sl__V=clamp_zero,
            )
        else:
            residuals = None

        return Solver1T1RDCOP(
            i_bl_driver=i_bl_driver,
            i_sl_driver=i_sl_driver,
            v_bl_node=v_bl_node,
            v_sl_node=v_sl_node,
            v_x_node=v_x_node,
            i_cell=i_cell,
            v_bl_clamp=v_bl_clamp__V,
            v_sl_drive=v_sl_drive__V,
            residuals=residuals,
        )

    # ---------------------------------------------------------------
    # Per-cell helpers (equations are part of the device physics,
    # not the iteration strategy).
    # ---------------------------------------------------------------

    def _solve_cell_pade_warm_start(
        self,
        v_bl_node: Tensor,
        v_sl_node: Tensor,
        v_wl_drive_grid: Tensor,
        rram_state_g: Tensor,
        rram_snapshot: RRAMSnapshot,
        nmos_snapshot: NMOSSnapshot,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        v_cell_bl_to_sl = v_bl_node - v_sl_node
        dc_nmos_seed = self.nmos.solve_dc(v_wl_drive_grid, v_bl_node, v_sl_node, nmos_snapshot)
        v_rram_drop_init = dc_nmos_seed.did_dvd__uS * v_cell_bl_to_sl / (dc_nmos_seed.did_dvd__uS + rram_state_g)
        v_x_node_init = v_bl_node - v_rram_drop_init
        return self._solve_cell_newton_once(
            v_bl_node,
            v_sl_node,
            v_wl_drive_grid,
            v_x_node_init,
            rram_snapshot,
            nmos_snapshot,
        )

    def _solve_cell_newton_once(
        self,
        v_bl_node: Tensor,
        v_sl_node: Tensor,
        v_wl_drive_grid__V: Tensor,
        v_x_node_init: Tensor,
        rram_snapshot: RRAMSnapshot,
        nmos_snapshot: NMOSSnapshot,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run one Newton step on V_X and re-evaluate cell-side quantities.

        Returns ``(i_r, i_n, g_cell_bl_eff, g_cell_sl_eff, v_x_node)``
        where ``i_r`` is the RRAM current (the device draining BL) and
        ``i_n`` is the NMOS current (the device feeding SL). At cell
        convergence ``i_r == i_n``; during iteration the difference is
        the cell-KCL residual ``F_X``.
        """
        dc_nmos_at_init = self.nmos.solve_dc(v_wl_drive_grid__V, v_x_node_init, v_sl_node, nmos_snapshot)
        dc_rram_at_init = self.rram.solve_dc(v_bl_node - v_x_node_init, rram_snapshot)
        f_cell_residual = dc_nmos_at_init.ids__uA - dc_rram_at_init.i__uA
        dFcell_dVx = dc_nmos_at_init.did_dvd__uS + dc_rram_at_init.di_dv__uS
        v_x_node = v_x_node_init - f_cell_residual / dFcell_dVx

        dc_nmos_final = self.nmos.solve_dc(v_wl_drive_grid__V, v_x_node, v_sl_node, nmos_snapshot)
        dc_rram_final = self.rram.solve_dc(v_bl_node - v_x_node, rram_snapshot)
        i_n = dc_nmos_final.ids__uA
        i_r = dc_rram_final.i__uA
        denom = dc_nmos_final.did_dvd__uS + dc_rram_final.di_dv__uS
        g_cell_bl_eff = dc_nmos_final.did_dvd__uS * dc_rram_final.di_dv__uS / denom
        g_cell_sl_eff = (-dc_nmos_final.did_dvs__uS) * dc_rram_final.di_dv__uS / denom
        return i_r, i_n, g_cell_bl_eff, g_cell_sl_eff, v_x_node

    # ---------------------------------------------------------------
    # Wire helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _wire_ir_drop_seed(
        i_r: Tensor,
        i_n: Tensor,
        v_bl_clamp__V: Tensor,
        v_sl_drive__V: Tensor,
        bl_segment_r__MOhm: Tensor,
        sl_segment_r__MOhm: Tensor,
        ndim: int,
        num_row: int,
    ) -> tuple[Tensor, Tensor]:
        """First-order IR-drop seed for the wire ladders.

        BL ladder propagates the RRAM current ``i_r`` (drained from BL by
        the RRAM); SL ladder propagates the NMOS current ``i_n``
        (delivered to SL by the NMOS). The seed accumulates the
        downstream current at each segment and subtracts the cumulative
        IR drop from the boundary voltage.
        """
        shape_broadcast = [1] * ndim
        shape_broadcast[-1] = num_row
        bl_segment_r_b = bl_segment_r__MOhm.view(shape_broadcast)
        sl_segment_r_b = sl_segment_r__MOhm.view(shape_broadcast)

        v_bl_clamp_grid = v_bl_clamp__V.unsqueeze(-1)
        v_sl_drive_grid = v_sl_drive__V.unsqueeze(-1)

        i_bl_down = torch.flip(torch.cumsum(torch.flip(i_r, [-1]), -1), [-1])
        v_bl_node = v_bl_clamp_grid - torch.cumsum(i_bl_down * bl_segment_r_b, dim=-1)

        i_sl_inject = -i_n
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

        Builds a block-2×2 tridiagonal system whose per-row Jacobian
        block carries the cell's BL ↔ SL cross-coupling:

            ┌                                                          ┐
            │ bl_wire_diag[k] + g_bl_eff[k]      −g_sl_eff[k]          │
            │ −g_bl_eff[k]                       sl_wire_diag[k] + g_sl_eff[k] │
            └                                                          ┘

        Off-diagonal blocks are diagonal 2×2 with ``−bl_offdiag`` /
        ``−sl_offdiag`` on their respective rails (BL and SL wires are
        independent ladders, no cross-rail wire coupling).

        For an SL-grounded chip both ``g_sl_eff`` and the SL wire's
        contribution to the BL drop are tiny, so the coupling reduces
        numerically to ~independent BL / SL solves. A variable-SL chip
        retains the full linearisation through the same code path.

        Shape conventions:
          * ``v_bl_node``, ``f_bl_kcl``, ``f_sl_kcl``, ``g_*_eff``:
            ``[..., num_col, num_row]``.
          * ``bl/sl_wire_diag_tmpl``: ``[num_row]``.
          * ``bl/sl_wire_offdiag``: ``[num_row - 1]``.

        Returns ``(dv_bl, dv_sl)``, each shape ``[..., num_col, num_row]``.
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
        # rhs takes last 2 dims [..., num_row, 2]. The col dim sits in
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

        The outer Newton solves for ``(V_BL_clamp, V_SL_drive)``
        simultaneously, so the implicit-function-theorem Jacobian
        ``dV_node[0]/dV_clamp`` is a 2×2 matrix:

            K = [[ ∂V_BL[0]/∂V_BL_CL,  ∂V_BL[0]/∂V_SL_DR ],
                 [ ∂V_SL[0]/∂V_BL_CL,  ∂V_SL[0]/∂V_SL_DR ]]

        Derivation: at the inner-converged state, ``J_inner · u = b``
        for a fixed ``V_clamp`` perturbation. The boundary forcing
        ``b`` for a unit ``V_BL_CL`` perturbation is
        ``e_0_BL · g_BL_seg[0]`` (only F_BL at row 0 sees the change);
        analogously for ``V_SL_DR``. So:

            K[:, 0] = g_BL_seg[0] · (J_inner⁻¹ · e_0_BL)[0, :]
            K[:, 1] = g_SL_seg[0] · (J_inner⁻¹ · e_0_SL)[0, :]

        Implemented as **two** block-tridiagonal solves, one per basis
        vector (``e_0_BL`` and ``e_0_SL``). The two solves are
        mathematically independent — pack them only if a future
        block-tridiagonal kernel exposes a multi-RHS interface.

        ``J_inner`` is the **coupled** block-2×2 wire Jacobian: same
        Jacobian used by ``_wire_newton_coupled_block2x2``, including
        the BL ↔ SL cell cross-coupling.

        Returns ``K`` of shape ``[..., num_col, 2, 2]``.
        """
        num_row = v_bl_node.shape[-1]
        shape_broadcast = [1] * v_bl_node.ndim
        shape_broadcast[-1] = num_row

        # --- Build the block-2×2 inner Jacobian (same as the wire Newton) ---

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

        # --- Boundary-forcing basis vectors ---

        # RHS for ``V_BL_CL`` perturbation: e_0 with BL component = g_BL_seg[0],
        # SL component = 0. RHS shape ``[..., num_col, num_row, 2]``.
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

        # Shape of each solve result: [..., num_col, num_row, 2]
        u_bl = solve_block_tridiagonal(sub_blocks, diag_blocks, sup_blocks, rhs_bl_basis)
        u_sl = solve_block_tridiagonal(sub_blocks, diag_blocks, sup_blocks, rhs_sl_basis)

        # K[:, 0] from BL-basis solve at row 0; K[:, 1] from SL-basis solve at row 0.
        # Shape: [..., num_col, 2]
        k_col_bl = u_bl.select(-2, 0)
        k_col_sl = u_sl.select(-2, 0)
        # Stack into [..., num_col, 2, 2] with K[:, j] as columns.
        return torch.stack([k_col_bl, k_col_sl], dim=-1)
