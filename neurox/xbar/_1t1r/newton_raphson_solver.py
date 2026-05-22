"""DC solver for a 1T1R crossbar array.

See also:
    docs/dev/modules/xbar/_1t1r/newton_raphson_solver.md
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.analog.clamp_driver import ClampDriver
from neurox.device import RRAM, RRAMSnapshot
from neurox.device.nmos import NMOS, NMOSSnapshot
from neurox.xbar.solver import (
    col_driver_current,
    col_wire_kcl_residual,
    solve_tridiagonal,
)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Solver1T1RDCOP:
    """Complete steady-state solution of one DC solve.

    Attributes:
        i_bl_driver: BL driver current [uA]. Shape: [..., num_col].
        i_sl_driver: SL driver current [uA]. Shape: [..., num_col].
        v_bl_node: BL node voltages [V]. Shape: [..., num_col, num_row].
        v_sl_node: SL node voltages [V]. Shape: [..., num_col, num_row].
        v_x_node: Internal access-transistor drain voltages [V]. Shape:
            [..., num_col, num_row].
        i_cell: Cell currents [uA]. Shape: [..., num_col, num_row].
        v_bl_clamp: BL clamp voltages [V]. Shape: [..., num_col].
        v_sl_drive: SL drive voltages [V]. Shape: [..., num_col].
    """

    i_bl_driver: Tensor
    i_sl_driver: Tensor
    v_bl_node: Tensor
    v_sl_node: Tensor
    v_x_node: Tensor
    i_cell: Tensor
    v_bl_clamp: Tensor
    v_sl_drive: Tensor


class NewtonRaphsonSolver1T1R:
    """Plain stateless 1T1R DC solver.

    Holds long-lived references to RRAM / NMOS / BL / SL drivers via ``__init__``;
    never owns buffers, parameters, snapshots, runtime caches, or any other
    PyTorch-registered objects, and is deliberately not an ``nn.Module``.
    New instance state must not be added in future revisions — all per-call
    inputs (wire tensors, snapshots) flow in through ``solve_dc`` kwargs.
    """

    N_UNROLL_OUTER: int = 5
    I_ATOL__uA: float = 1e-3

    def __init__(
        self,
        *,
        rram: RRAM,
        nmos: NMOS,
        bl_driver: ClampDriver,
        sl_driver: ClampDriver,
    ) -> None:
        """Bind the solver to its 1T1R device and boundary-driver instances.

        Args:
            rram: Programmed RRAM device model.
            nmos: Fabricated access-NMOS model.
            bl_driver: BL clamp driver.
            sl_driver: SL clamp driver.
        """
        self.rram = rram
        self.nmos = nmos
        self.bl_driver = bl_driver
        self.sl_driver = sl_driver

    # ---------------------------------------------------------------
    # Public entry point
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
        bl_driver_snapshot: object,
        sl_driver_snapshot: object,
    ) -> Solver1T1RDCOP:
        """Solve the fabricated 1T1R tile for one WL-drive tensor.

        Args:
            v_wl_drive__V: WL drive voltage tensor [V]. Shape: [..., 1, num_row].
            bl_segment_r__MOhm: 1-D BL segment resistances [MOhm]; index 0 is
                driver-to-first.
            sl_segment_r__MOhm: 1-D SL segment resistances [MOhm]; index 0 is
                driver-to-first.
            bl_segment_g__uS: BL segment conductances [uS], reciprocal of
                ``bl_segment_r__MOhm``.
            sl_segment_g__uS: SL segment conductances [uS], reciprocal of
                ``sl_segment_r__MOhm``.
            rram_snapshot: Per-solve RRAM snapshot.
            nmos_snapshot: Per-solve NMOS snapshot.
            bl_driver_snapshot: Per-solve BL driver snapshot.
            sl_driver_snapshot: Per-solve SL driver snapshot.

        Returns:
            Complete steady-state solution for the current VMM.
        """

        # --- Per-solve wire Jacobian templates ---

        bl_segment_r = bl_segment_r__MOhm
        sl_segment_r = sl_segment_r__MOhm
        bl_segment_g = bl_segment_g__uS
        sl_segment_g = sl_segment_g__uS

        bl_wire_diag = bl_segment_g + F.pad(bl_segment_g[1:], (0, 1))
        bl_wire_offdiag = -bl_segment_g[1:]
        sl_wire_diag = sl_segment_g + F.pad(sl_segment_g[1:], (0, 1))
        sl_wire_offdiag = -sl_segment_g[1:]
        bl_driver_segment_g__uS = bl_segment_g[0]
        sl_driver_segment_g__uS = sl_segment_g[0]

        rram_state_g_snapshot = rram_snapshot.g__uS
        *batch, num_col, num_row = rram_state_g_snapshot.shape
        # Use the programmed conductance for the warm start and the
        # runtime snapshot inside the Newton loop.
        rram_state_g_static = self.rram.g__uS.expand_as(rram_state_g_snapshot)
        if not (num_col > 1):
            raise ValueError(f"require: num_col ({num_col}) > 1")
        if not (num_row > 1):
            raise ValueError(f"require: num_row ({num_row}) > 1")
        # Shape: [..., 1, num_row] -> [..., num_col, num_row]
        v_wl_drive_grid__V = v_wl_drive__V.expand(*batch, num_col, num_row)

        # --- Warm start phase 1: Padé cell seed ---

        v_bl_clamp_ref__V = self.bl_driver.v_ref__V
        v_sl_drive_ref__V = self.sl_driver.v_ref__V
        v_bl_node_seed = torch.full_like(rram_state_g_snapshot, v_bl_clamp_ref__V)
        v_sl_node_seed = torch.full_like(rram_state_g_snapshot, v_sl_drive_ref__V)
        i_cell_init, _, v_x_node_init = self._solve_cell_pade_warm_start(
            v_bl_node_seed,
            v_sl_node_seed,
            v_wl_drive_grid__V,
            rram_state_g_static,
            rram_snapshot,
            nmos_snapshot,
        )

        # --- First clamp-driver evaluations ---

        # Warm-up driver calls use the cell-side sums as port-current
        # seeds before the boundary KCL is available.
        # Shape: [..., num_col, num_row] -> [..., num_col]
        i_bl_driver_in__uA = i_cell_init.sum(dim=-1)
        i_sl_driver_in__uA = -i_cell_init.sum(dim=-1)
        v_bl_clamp__V, _ = self.bl_driver.solve_clamp(i_bl_driver_in__uA, bl_driver_snapshot, v_clamp_init__V=None)
        v_sl_drive__V, _ = self.sl_driver.solve_clamp(i_sl_driver_in__uA, sl_driver_snapshot, v_clamp_init__V=None)
        # Shape: [..., num_col] -> [..., num_col, 1]
        v_bl_clamp_grid__V = v_bl_clamp__V.unsqueeze(-1)
        v_sl_drive_grid__V = v_sl_drive__V.unsqueeze(-1)

        # --- Warm start phase 2: first-order wire voltages ---

        # BL wire runs along dim=-1, driver at index 0, cell currents +i_cell.
        # First-order IR-drop seed on the BL ladder.
        bl_shape_broadcast = [1] * rram_state_g_snapshot.ndim
        bl_shape_broadcast[-1] = num_row
        bl_segment_r_broadcast = bl_segment_r.view(bl_shape_broadcast)
        i_bl_downstream = torch.flip(torch.cumsum(torch.flip(i_cell_init, [-1]), -1), [-1])
        # Shape: [..., num_col, num_row]
        v_bl_node = v_bl_clamp_grid__V - torch.cumsum(i_bl_downstream * bl_segment_r_broadcast, dim=-1)

        # SL wire runs along dim=-1, driver at index 0, cell currents
        # ``i_inject = -i_cell`` (cell sources +i_cell into SL).
        # Matching seed on the SL ladder.
        sl_shape_broadcast = [1] * rram_state_g_snapshot.ndim
        sl_shape_broadcast[-1] = num_row
        sl_segment_r_broadcast = sl_segment_r.view(sl_shape_broadcast)
        i_sl_inject = -i_cell_init
        i_sl_downstream = torch.flip(torch.cumsum(torch.flip(i_sl_inject, [-1]), -1), [-1])
        # Shape: [..., num_col, num_row]
        v_sl_node = v_sl_drive_grid__V - torch.cumsum(i_sl_downstream * sl_segment_r_broadcast, dim=-1)

        # --- Outer Newton loop ---

        v_x_node = v_x_node_init
        i_cell = i_cell_init
        g_cell_eff = torch.zeros_like(v_x_node)  # placeholder overwritten below
        for _ in range(self.N_UNROLL_OUTER):
            i_cell, g_cell_eff, v_x_node = self._solve_cell_newton_warm_start(
                v_bl_node,
                v_sl_node,
                v_wl_drive_grid__V,
                v_x_node,
                rram_snapshot,
                nmos_snapshot,
            )

            # Refresh the clamp boundaries from the latest port current.
            i_bl_driver_in__uA = (v_bl_clamp__V - v_bl_node.select(-1, 0)) * bl_driver_segment_g__uS
            i_sl_driver_in__uA = (v_sl_drive__V - v_sl_node.select(-1, 0)) * sl_driver_segment_g__uS
            v_bl_clamp__V, r_bl_driver_in__MOhm = self.bl_driver.solve_clamp(
                i_bl_driver_in__uA,
                bl_driver_snapshot,
                v_clamp_init__V=v_bl_clamp__V,
            )
            v_sl_drive__V, r_sl_driver_in__MOhm = self.sl_driver.solve_clamp(
                i_sl_driver_in__uA,
                sl_driver_snapshot,
                v_clamp_init__V=v_sl_drive__V,
            )
            # Shape: [..., num_col] -> [..., num_col, 1]
            v_bl_clamp_grid__V = v_bl_clamp__V.unsqueeze(-1)
            v_sl_drive_grid__V = v_sl_drive__V.unsqueeze(-1)

            f_bl_kcl = col_wire_kcl_residual(v_bl_node, v_bl_clamp_grid__V, bl_segment_g, i_cell)
            f_sl_kcl = col_wire_kcl_residual(v_sl_node, v_sl_drive_grid__V, sl_segment_g, -i_cell)
            dv_bl_node = self._solve_driver_newton_rank1(
                v_bl_node,
                f_bl_kcl,
                g_cell_eff,
                r_bl_driver_in__MOhm,
                driver_segment_g__uS=bl_driver_segment_g__uS,
                wire_diag=bl_wire_diag,
                wire_offdiag=bl_wire_offdiag,
                dim=-1,
            )
            dv_sl_node = self._solve_driver_newton_rank1(
                v_sl_node,
                f_sl_kcl,
                g_cell_eff,
                r_sl_driver_in__MOhm,
                driver_segment_g__uS=sl_driver_segment_g__uS,
                wire_diag=sl_wire_diag,
                wire_offdiag=sl_wire_offdiag,
                dim=-1,
            )
            v_bl_node = v_bl_node + dv_bl_node
            v_sl_node = v_sl_node + dv_sl_node

        # --- Gather boundary outputs ---

        i_bl_driver = col_driver_current(v_bl_node, v_bl_clamp_grid__V, bl_segment_g)
        i_sl_driver = col_driver_current(v_sl_node, v_sl_drive_grid__V, sl_segment_g)
        return Solver1T1RDCOP(
            i_bl_driver=i_bl_driver,
            i_sl_driver=i_sl_driver,
            v_bl_node=v_bl_node,
            v_sl_node=v_sl_node,
            v_x_node=v_x_node,
            i_cell=i_cell,
            v_bl_clamp=v_bl_clamp__V,
            v_sl_drive=v_sl_drive__V,
        )

    # ---------------------------------------------------------------
    # Per-cell V_X / cell-current helpers
    # ---------------------------------------------------------------

    def _solve_cell_pade_warm_start(
        self,
        v_bl_node: Tensor,
        v_sl_node: Tensor,
        v_wl_drive_grid: Tensor,
        rram_state_g: Tensor,
        rram_snapshot: RRAMSnapshot,
        nmos_snapshot: NMOSSnapshot,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Warm-start the per-cell solve from a divider-style seed.

        Args:
            v_bl_node: BL node voltages [V]. Shape: [..., num_col, num_row].
            v_sl_node: SL node voltages [V]. Shape: [..., num_col, num_row].
            v_wl_drive_grid: Broadcast WL drive tensor [V]. Shape:
                [..., num_col, num_row].
            rram_state_g: Programmed RRAM conductance [uS]. Shape:
                [..., num_col, num_row].
            rram_snapshot: Per-solve RRAM snapshot.
            nmos_snapshot: Per-solve NMOS snapshot.

        Returns:
            Tuple `(i_cell, g_cell_eff, v_x_node)` with shape
            [..., num_col, num_row] for each tensor.
        """
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

    def _solve_cell_newton_warm_start(
        self,
        v_bl_node: Tensor,
        v_sl_node: Tensor,
        v_wl_drive_grid__V: Tensor,
        v_x_node_init: Tensor,
        rram_snapshot: RRAMSnapshot,
        nmos_snapshot: NMOSSnapshot,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Warm-restart the per-cell solve from a previous `v_x_node`.

        Args:
            v_bl_node: BL node voltages [V]. Shape: [..., num_col, num_row].
            v_sl_node: SL node voltages [V]. Shape: [..., num_col, num_row].
            v_wl_drive_grid__V: Broadcast WL drive tensor [V]. Shape:
                [..., num_col, num_row].
            v_x_node_init: Warm-start internal node voltage [V]. Shape:
                [..., num_col, num_row].
            rram_snapshot: Per-solve RRAM snapshot.
            nmos_snapshot: Per-solve NMOS snapshot.

        Returns:
            Tuple `(i_cell, g_cell_eff, v_x_node)` with shape
            [..., num_col, num_row] for each tensor.
        """
        return self._solve_cell_newton_once(
            v_bl_node,
            v_sl_node,
            v_wl_drive_grid__V,
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
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run one Newton step on the cell KCL.

        Args:
            v_bl_node: BL node voltages [V]. Shape: [..., num_col, num_row].
            v_sl_node: SL node voltages [V]. Shape: [..., num_col, num_row].
            v_wl_drive_grid__V: Broadcast WL drive tensor [V]. Shape:
                [..., num_col, num_row].
            v_x_node_init: Warm-start internal node voltage [V]. Shape:
                [..., num_col, num_row].
            rram_snapshot: Per-solve RRAM snapshot.
            nmos_snapshot: Per-solve NMOS snapshot.

        Returns:
            Tuple `(i_cell, g_cell_eff, v_x_node)` with shape
            [..., num_col, num_row] for each tensor.
        """
        dc_nmos_at_init = self.nmos.solve_dc(v_wl_drive_grid__V, v_x_node_init, v_sl_node, nmos_snapshot)
        dc_rram_at_init = self.rram.solve_dc(v_bl_node - v_x_node_init, rram_snapshot)
        f_cell_residual = dc_nmos_at_init.ids__uA - dc_rram_at_init.i__uA
        dFcell_dVx = dc_nmos_at_init.did_dvd__uS + dc_rram_at_init.di_dv__uS
        v_x_node = v_x_node_init - f_cell_residual / dFcell_dVx

        dc_nmos_final = self.nmos.solve_dc(v_wl_drive_grid__V, v_x_node, v_sl_node, nmos_snapshot)
        dc_rram_final = self.rram.solve_dc(v_bl_node - v_x_node, rram_snapshot)
        i_cell = dc_nmos_final.ids__uA
        g_cell_eff = (
            dc_nmos_final.did_dvd__uS * dc_rram_final.di_dv__uS / (dc_nmos_final.did_dvd__uS + dc_rram_final.di_dv__uS)
        )
        return i_cell, g_cell_eff, v_x_node

    # ---------------------------------------------------------------
    # Wire Newton helper
    # ---------------------------------------------------------------

    def _solve_driver_newton_rank1(
        self,
        v_node: Tensor,
        f_kcl: Tensor,
        g_cell_eff: Tensor,
        r_driver_in__MOhm: Tensor,
        *,
        driver_segment_g__uS: Tensor,
        wire_diag: Tensor,
        wire_offdiag: Tensor,
        dim: int,
    ) -> Tensor:
        """Solve one wire Newton step with clamp-boundary coupling.

        Args:
            v_node: Current wire-node voltage tensor. Shape: [..., num_axis, ...].
            f_kcl: Wire KCL residual. Shape: [..., num_axis, ...].
            g_cell_eff: Effective per-cell conductance [uS]. Shape:
                [..., num_col, num_row].
            r_driver_in__MOhm: Driver small-signal resistance [MOhm].
            driver_segment_g__uS: Conductance of the driver-to-first-wire
                segment [uS].
            wire_diag: Tridiagonal main diagonal for one wire axis [uS].
            wire_offdiag: Tridiagonal off-diagonal for one wire axis [uS].
            dim: Active wire axis.

        Returns:
            Newton correction with the same shape as `v_node`.
        """
        num_axis = v_node.shape[dim]
        shape_broadcast = [1] * v_node.ndim
        shape_broadcast[dim] = num_axis

        # --- Broadcast the tridiagonal templates ---

        # Broadcast the tridiagonal templates along the active wire axis.
        # Shape: [num_axis] -> [..., num_axis, ...]
        g_jac_diag = wire_diag.view(shape_broadcast) + g_cell_eff
        # Shape: [num_axis - 1] -> [..., num_axis, ...]
        sub_pattern = F.pad(wire_offdiag, (1, 0)).view(shape_broadcast)
        # Shape: [num_axis - 1] -> [..., num_axis, ...]
        sup_pattern = F.pad(wire_offdiag, (0, 1)).view(shape_broadcast)
        # Shape: [..., num_axis, ...] -> [..., num_axis, ...]
        sub = sub_pattern.expand_as(v_node)
        # Shape: [..., num_axis, ...] -> [..., num_axis, ...]
        sup = sup_pattern.expand_as(v_node)

        # Shape: [..., num_axis, ...]
        dv_base = solve_tridiagonal(sub, g_jac_diag, sup, -f_kcl, dim=dim)

        # --- Apply the rank-1 clamp correction ---

        # Fold the clamp boundary into the tridiagonal solve with a
        # Sherman-Morrison rank-1 correction.
        g_rank1_left_coeff = (-driver_segment_g__uS) * r_driver_in__MOhm
        # Shape: [...] -> [..., num_axis, ...]
        pad_spec = (0, 0) * (-dim - 1) + (0, num_axis - 1)
        v_rank1_rhs = F.pad(g_rank1_left_coeff.unsqueeze(dim), pad_spec)

        # Shape: [..., num_axis, ...]
        v_rank1_response = solve_tridiagonal(sub, g_jac_diag, sup, v_rank1_rhs, dim=dim)
        sm_numerator = (g_cell_eff * dv_base).sum(dim=dim, keepdim=True)
        sm_denominator = 1.0 + (g_cell_eff * v_rank1_response).sum(dim=dim, keepdim=True)
        return dv_base - v_rank1_response * (sm_numerator / sm_denominator)
