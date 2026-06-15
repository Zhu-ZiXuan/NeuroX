"""Fully-coupled-Jacobian DC solver for a 1T1R crossbar array.

See ``docs/reference/xbar/_1t1r/solver.md`` for the
unknown vector, residual definition, Schur-elimination of the boundary
scalars, and convergence rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.analog import Driver, DriverSnapshot
from neurox.analog.tia import TIA, TIASnapshot
from neurox.device import RRAM, RRAMSnapshot
from neurox.device.nmos import NMOS, NMOSSnapshot
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
class FullJacobianSolver1T1RConfig(Solver1T1RConfig):
    """Workload-tuned numerical knobs for :class:`FullJacobianSolver1T1R`.

    Carries only the iteration count. The Newton damping cap is a
    method-intrinsic safety constant and lives on the solver class.

    Attributes:
        n_newton: Outer Newton iterations on the full coupled system
            ``(V_BL, V_SL, V_X, V_BL_CL, V_SL_DR)``. Pick via
            ``solver_calibrate.full_jacobian``; the framework uses
            step-ratio plateau detection + a relative residual safety
            guard (chip-parameter-free).
    """

    n_newton: int

    def validate(self) -> None:
        super().validate()
        self._require_pos(self.n_newton, "n_newton")


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


@Solver1T1R.register_key(FullJacobianSolver1T1RConfig)
class FullJacobianSolver1T1R(Solver1T1R):
    """Full-coupled-Jacobian DC solver for a 1T1R tile.

    Each Newton iteration assembles the global Jacobian for **all**
    circuit unknowns at once, then solves the resulting block-tridiagonal
    linear system (with 3×3 row blocks for the array + 2 Schur-eliminated
    boundary scalars). Compared with the nested block-Gauss-Seidel solver
    this captures every coupling simultaneously — no per-axis
    decomposition, no Schur elimination of ``V_X``.

    Class-level numerical constant:

      * ``MAX_STEP__V``: Per-iteration ``|Δu|`` cap [V] applied
        component-wise to every unknown class (V_BL, V_SL, V_X,
        V_BL_CL, V_SL_DR). Newton damping heuristic; method-intrinsic,
        not chip-tuned.
    """

    MAX_STEP__V: float = 0.05

    def __init__(
        self,
        *,
        config: FullJacobianSolver1T1RConfig,
        rram: RRAM,
        nmos: NMOS,
        bl_driver: TIA,
        sl_driver: Driver,
    ) -> None:
        self.config = config
        self.rram = rram
        self.nmos = nmos
        self.bl_driver = bl_driver
        self.sl_driver = sl_driver

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
        del bl_segment_r__MOhm, sl_segment_r__MOhm  # full Jacobian does not need R form

        # --- Wire Jacobian templates ---

        # Shape: [num_row]
        bl_wire_diag_tmpl = bl_segment_g__uS + F.pad(bl_segment_g__uS[1:], (0, 1))
        sl_wire_diag_tmpl = sl_segment_g__uS + F.pad(sl_segment_g__uS[1:], (0, 1))
        # Shape: [num_row-1]
        bl_wire_offdiag = -bl_segment_g__uS[1:]
        sl_wire_offdiag = -sl_segment_g__uS[1:]
        # Shape: []
        bl_driver_segment_g = bl_segment_g__uS[0]
        sl_driver_segment_g = sl_segment_g__uS[0]

        # --- Shapes ---

        rram_state_g_snapshot = rram_snapshot.g__uS
        *batch, num_col, num_row = rram_state_g_snapshot.shape
        if not (num_col > 1):
            raise ValueError(f"require: num_col ({num_col}) > 1")
        if not (num_row > 1):
            raise ValueError(f"require: num_row ({num_row}) > 1")
        # Shape: [..., 1, num_row] -> [..., num_col, num_row]
        v_wl_drive_grid__V = v_wl_drive__V.expand(*batch, num_col, num_row)

        # --- Initialization ---

        # Clamps at ideal references.
        device = rram_state_g_snapshot.device
        dtype = rram_state_g_snapshot.dtype
        v_bl_clamp__V = torch.full(
            (*batch, num_col),
            self.bl_driver.v_ref__V,
            device=device,
            dtype=dtype,
        )
        v_sl_drive__V = torch.full(
            (*batch, num_col),
            self.sl_driver.v_ref__V,
            device=device,
            dtype=dtype,
        )
        # Wire nodes at the clamp voltage (zero IR drop seed).
        v_bl_node = v_bl_clamp__V.unsqueeze(-1).expand_as(rram_state_g_snapshot).contiguous()
        v_sl_node = v_sl_drive__V.unsqueeze(-1).expand_as(rram_state_g_snapshot).contiguous()
        # V_X seeded via the same Padé current-divider as nested's warm start.
        dc_nmos_seed = self.nmos.solve_dc(v_wl_drive_grid__V, v_bl_node, v_sl_node, nmos_snapshot)
        v_cell_bl_to_sl = v_bl_node - v_sl_node
        v_rram_drop_init = (
            dc_nmos_seed.did_dvd__uS * v_cell_bl_to_sl / (dc_nmos_seed.did_dvd__uS + rram_state_g_snapshot)
        )
        v_x_node = v_bl_node - v_rram_drop_init

        # --- Newton loop ---

        max_step__V = self.MAX_STEP__V
        # Broadcast helpers for stamping wire diag templates onto per-row blocks.
        shape_broadcast_row = [1] * v_bl_node.ndim
        shape_broadcast_row[-1] = num_row

        for _ in range(self.config.n_newton):
            # === Residuals ============================================
            # Cell DC at current (V_BL, V_SL, V_X).
            nmos_dc = self.nmos.solve_dc(v_wl_drive_grid__V, v_x_node, v_sl_node, nmos_snapshot)
            rram_dc = self.rram.solve_dc(v_bl_node - v_x_node, rram_snapshot)
            # Shape: [..., num_col, num_row]
            i_n = nmos_dc.ids__uA
            i_r = rram_dc.i__uA
            f_x = i_n - i_r

            # Wire KCL residuals — F_BL uses I_R (RRAM injects into BL),
            # F_SL uses I_N (NMOS draws from SL).
            # Shape: [..., num_col, 1]
            v_bl_clamp_grid__V = v_bl_clamp__V.unsqueeze(-1)
            v_sl_drive_grid__V = v_sl_drive__V.unsqueeze(-1)
            f_bl = col_wire_kcl_residual(v_bl_node, v_bl_clamp_grid__V, bl_segment_g__uS, i_r)
            f_sl = col_wire_kcl_residual(v_sl_node, v_sl_drive_grid__V, sl_segment_g__uS, -i_n)

            # Boundary clamp residuals + small-signal slopes.
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
            f_cl_bl = v_bl_clamp__V - v_bl_target__V
            f_cl_sl = v_sl_drive__V - v_sl_target__V

            # === Jacobian (per-row 3×3 diagonal block, sparse off-diag) =====
            # Per-cell local derivatives.
            g_r = rram_dc.di_dv__uS  # ≥ 0
            g_nd = nmos_dc.did_dvd__uS  # ≥ 0  (∂I_N/∂V_D = ∂I_N/∂V_X)
            g_ns = nmos_dc.did_dvs__uS  # ≤ 0  (∂I_N/∂V_S = ∂I_N/∂V_SL)

            # Wire diagonal contributions broadcast over (col, row).
            # Shape: [..., num_col, num_row]
            bl_wire_diag = bl_wire_diag_tmpl.view(shape_broadcast_row).expand_as(v_bl_node)
            sl_wire_diag = sl_wire_diag_tmpl.view(shape_broadcast_row).expand_as(v_sl_node)

            # Row 0 = BL: [bl_wire_diag + g_r,   0,                -g_r]
            # Row 1 = SL: [0,                    sl_wire_diag − g_ns, −g_nd]
            # Row 2 = X : [-g_r,                 g_ns,              g_nd + g_r]
            zero_node = torch.zeros_like(g_r)
            diag_row_bl = torch.stack([bl_wire_diag + g_r, zero_node, -g_r], dim=-1)
            diag_row_sl = torch.stack([zero_node, sl_wire_diag - g_ns, -g_nd], dim=-1)
            diag_row_x = torch.stack([-g_r, g_ns, g_nd + g_r], dim=-1)
            # Shape: [..., num_col, num_row, 3, 3]
            diag_blocks = torch.stack([diag_row_bl, diag_row_sl, diag_row_x], dim=-2)

            # Off-diagonal blocks: only BL-BL and SL-SL wire couplings.
            # Pad to length num_row; sub[0] and sup[N-1] are unused placeholders.
            # Shape: [num_row]
            sub_bl_pad = F.pad(bl_wire_offdiag, (1, 0)).view(shape_broadcast_row)
            sub_sl_pad = F.pad(sl_wire_offdiag, (1, 0)).view(shape_broadcast_row)
            sup_bl_pad = F.pad(bl_wire_offdiag, (0, 1)).view(shape_broadcast_row)
            sup_sl_pad = F.pad(sl_wire_offdiag, (0, 1)).view(shape_broadcast_row)
            sub_bl_full = sub_bl_pad.expand_as(v_bl_node)
            sub_sl_full = sub_sl_pad.expand_as(v_sl_node)
            sup_bl_full = sup_bl_pad.expand_as(v_bl_node)
            sup_sl_full = sup_sl_pad.expand_as(v_sl_node)
            zero_full = torch.zeros_like(sub_bl_full)
            # 3×3 sub block: diag(sub_bl, sub_sl, 0)
            sub_row_bl = torch.stack([sub_bl_full, zero_full, zero_full], dim=-1)
            sub_row_sl = torch.stack([zero_full, sub_sl_full, zero_full], dim=-1)
            sub_row_x = torch.stack([zero_full, zero_full, zero_full], dim=-1)
            sub_blocks = torch.stack([sub_row_bl, sub_row_sl, sub_row_x], dim=-2)
            sup_row_bl = torch.stack([sup_bl_full, zero_full, zero_full], dim=-1)
            sup_row_sl = torch.stack([zero_full, sup_sl_full, zero_full], dim=-1)
            sup_row_x = torch.stack([zero_full, zero_full, zero_full], dim=-1)
            sup_blocks = torch.stack([sup_row_bl, sup_row_sl, sup_row_x], dim=-2)

            # === Schur eliminate V_BL_CL, V_SL_DR =====================
            # F_CL_BL = V_BL_CL − TIA(I_PORT)
            # I_PORT  = g_seg[0]·(V_BL_CL − V_BL[0])
            # ∂F_CL_BL/∂V_BL_CL = 1 − r·g
            # ∂F_CL_BL/∂V_BL[0] = +r·g
            # Linearise: Δv_clamp = α + β·Δv_bl[0]
            #     α = −F_CL_BL / (1 − r·g)
            #     β = −r·g / (1 − r·g)
            # Substituting into F_BL[0] (which carries −g_seg[0]·Δv_clamp) gives:
            #     diag_blocks[0_row, 0, 0] += −g_seg[0]·β   (Δv_bl[0] self-term)
            #     rhs[0_row, 0]            +=  g_seg[0]·α   (constant injection)
            rg_bl = r_bl_driver__MOhm * bl_driver_segment_g
            rg_sl = r_sl_driver__MOhm * sl_driver_segment_g
            denom_bl = 1.0 - rg_bl
            denom_sl = 1.0 - rg_sl
            beta_bl = -rg_bl / denom_bl
            beta_sl = -rg_sl / denom_sl
            alpha_bl = -f_cl_bl / denom_bl
            alpha_sl = -f_cl_sl / denom_sl

            # Shape: [..., col]
            mod_bl = -bl_driver_segment_g * beta_bl
            # Shape: [..., col]
            mod_sl = -sl_driver_segment_g * beta_sl
            zero_col = torch.zeros_like(mod_bl)
            mod_row0_3x3_row_bl = torch.stack([mod_bl, zero_col, zero_col], dim=-1)
            mod_row0_3x3_row_sl = torch.stack([zero_col, mod_sl, zero_col], dim=-1)
            mod_row0_3x3_row_x = torch.stack([zero_col, zero_col, zero_col], dim=-1)
            # Row-0-only modification with only [0, 0] and [1, 1] non-zero.
            # Shape: [..., col, 3, 3]
            mod_row0_3x3 = torch.stack(
                [mod_row0_3x3_row_bl, mod_row0_3x3_row_sl, mod_row0_3x3_row_x],
                dim=-2,
            )
            # Embed at wire-row 0 of the diag-block stack via F.pad: unsqueeze
            # a wire-row dim then pad with zeros on the right. F.pad's
            # last-4-dims spec runs
            # (col1_last_low, col1_last_high,  col2_last_low, col2_last_high,
            #  row_low, row_high,  wire_row_low, wire_row_high).
            # Shape: [..., col, num_row, 3, 3]
            mod_diag_full = F.pad(
                mod_row0_3x3.unsqueeze(-3),
                (0, 0, 0, 0, 0, num_row - 1),
            )
            diag_blocks_modified = diag_blocks + mod_diag_full

            # === Build RHS = −F and add row-0 Schur correction =========
            # rhs[wire_row=0, BL=0] += +g_seg[0]·α_bl
            # rhs[wire_row=0, SL=1] += +g_seg[0]·α_sl   (uses sl_driver_segment_g)
            rhs = torch.stack([-f_bl, -f_sl, -f_x], dim=-1)  # [..., col, num_row, 3]
            rhs_mod_bl = bl_driver_segment_g * alpha_bl  # [..., col]
            rhs_mod_sl = sl_driver_segment_g * alpha_sl
            mod_row0_vec = torch.stack(
                [rhs_mod_bl, rhs_mod_sl, zero_col],
                dim=-1,
            )  # [..., col, 3]
            mod_rhs_full = F.pad(
                mod_row0_vec.unsqueeze(-2),
                (0, 0, 0, num_row - 1),
            )  # [..., col, num_row, 3]
            rhs_modified = rhs + mod_rhs_full

            # === Block-tridiagonal solve ==============================
            # diag/sub/sup: [..., col, num_row, 3, 3]; rhs: [..., col, num_row, 3]
            delta = solve_block_tridiagonal(sub_blocks, diag_blocks_modified, sup_blocks, rhs_modified)
            # Unpack the 3 unknown classes per (col, row).
            dv_bl = delta[..., 0]
            dv_sl = delta[..., 1]
            dv_x = delta[..., 2]

            # Recover the eliminated clamp deltas.
            dv_bl_at_row0 = dv_bl.select(-1, 0)
            dv_sl_at_row0 = dv_sl.select(-1, 0)
            delta_v_bl_clamp = alpha_bl + beta_bl * dv_bl_at_row0
            delta_v_sl_drive = alpha_sl + beta_sl * dv_sl_at_row0

            # === Damped update ========================================
            v_bl_node = v_bl_node + dv_bl.clamp(min=-max_step__V, max=max_step__V)
            v_sl_node = v_sl_node + dv_sl.clamp(min=-max_step__V, max=max_step__V)
            v_x_node = v_x_node + dv_x.clamp(min=-max_step__V, max=max_step__V)
            v_bl_clamp__V = v_bl_clamp__V + delta_v_bl_clamp.clamp(min=-max_step__V, max=max_step__V)
            v_sl_drive__V = v_sl_drive__V + delta_v_sl_drive.clamp(min=-max_step__V, max=max_step__V)

        # === Exit-state finalization ====================================
        # Final cell + boundary eval at the converged state. ``i_cell``
        # reported in the DCOP is the RRAM current (BL-side reading); at
        # convergence ``I_R == I_N``.
        v_bl_clamp_grid__V = v_bl_clamp__V.unsqueeze(-1)
        v_sl_drive_grid__V = v_sl_drive__V.unsqueeze(-1)
        nmos_dc_final = self.nmos.solve_dc(v_wl_drive_grid__V, v_x_node, v_sl_node, nmos_snapshot)
        rram_dc_final = self.rram.solve_dc(v_bl_node - v_x_node, rram_snapshot)
        i_n_final = nmos_dc_final.ids__uA
        i_r_final = rram_dc_final.i__uA
        i_cell = i_r_final

        i_bl_driver = col_driver_current(v_bl_node, v_bl_clamp_grid__V, bl_segment_g__uS)
        i_sl_driver = col_driver_current(v_sl_node, v_sl_drive_grid__V, sl_segment_g__uS)

        # === Optional residual diagnostics ==============================
        residuals: Solver1T1RResiduals | None
        if compute_residuals:
            cell_res = (i_n_final - i_r_final).abs()
            # BL wire KCL uses I_R; SL wire KCL uses I_N — matches the
            # per-rail current actually flowing on each ladder.
            wire_bl_res = col_wire_kcl_residual(
                v_bl_node,
                v_bl_clamp_grid__V,
                bl_segment_g__uS,
                i_r_final,
            ).abs()
            wire_sl_res = col_wire_kcl_residual(
                v_sl_node,
                v_sl_drive_grid__V,
                sl_segment_g__uS,
                -i_n_final,
            ).abs()
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
