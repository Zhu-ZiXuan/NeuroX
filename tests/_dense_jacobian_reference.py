"""Dense + finite-difference reference helper for 1T1R solver tests.

Not collected by pytest (underscore prefix). Provides:

  * :func:`build_residual_fn` — closure that, given a flattened unknown
    vector ``u = [V_BL, V_SL, V_X, V_BL_CL, V_SL_DR]`` for a single
    (batch, col) instance, evaluates the 5-class residual vector using
    the SAME device/driver objects as the production solver.

  * :func:`numerical_jacobian` — finite-difference Jacobian, used to
    cross-check :class:`FullJacobianSolver1T1R`'s analytical stamping.

Intended use is for very small problems (R ≤ 8, single col) where a
dense ``(3R+2) × (3R+2)`` matrix and ``3R+2`` extra residual evaluations
per Jacobian column are tractable.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from neurox.xbar.solver import col_wire_kcl_residual


def build_residual_fn(
    *,
    v_wl_drive_grid__V: Tensor,  # shape [num_row]
    bl_segment_g__uS: Tensor,  # shape [num_row]
    sl_segment_g__uS: Tensor,  # shape [num_row]
    rram_snapshot,
    nmos_snapshot,
    bl_driver,
    bl_driver_snapshot,
    sl_driver,
    sl_driver_snapshot,
    rram,
    nmos,
    num_row: int,
) -> Callable[[Tensor], Tensor]:
    """Return a residual function ``F(u) -> [3R+2]`` for a single column.

    ``u`` layout: ``[V_BL[0:R], V_SL[0:R], V_X[0:R], V_BL_CL, V_SL_DR]``.
    Snapshots / device objects must already be sliced to a single
    ``(batch, col) = ()`` instance — caller's responsibility.
    """
    bl_driver_segment_g = bl_segment_g__uS[0]
    sl_driver_segment_g = sl_segment_g__uS[0]
    r = num_row

    def residual(u: Tensor) -> Tensor:
        v_bl = u[:r]
        v_sl = u[r : 2 * r]
        v_x = u[2 * r : 3 * r]
        v_bl_cl = u[3 * r]
        v_sl_dr = u[3 * r + 1]

        # Cell DC.
        nmos_dc = nmos.solve_dc(v_wl_drive_grid__V, v_x, v_sl, nmos_snapshot)
        rram_dc = rram.solve_dc(v_bl - v_x, rram_snapshot)
        i_n = nmos_dc.ids__uA
        i_r = rram_dc.i__uA
        f_x = i_n - i_r  # [r]

        # Wire residuals — col_wire_kcl_residual expects [..., col, row]
        # so reshape inputs to add the singleton col dim.
        v_bl_2d = v_bl.unsqueeze(0)  # [1, r]
        v_sl_2d = v_sl.unsqueeze(0)
        v_bl_cl_grid = v_bl_cl.reshape(1, 1)
        v_sl_dr_grid = v_sl_dr.reshape(1, 1)
        i_r_2d = i_r.unsqueeze(0)
        i_n_2d = i_n.unsqueeze(0)

        f_bl = col_wire_kcl_residual(v_bl_2d, v_bl_cl_grid, bl_segment_g__uS, i_r_2d).squeeze(0)
        f_sl = col_wire_kcl_residual(v_sl_2d, v_sl_dr_grid, sl_segment_g__uS, -i_n_2d).squeeze(0)

        # Boundary residuals.
        i_bl_port = (v_bl_cl - v_bl[0]) * bl_driver_segment_g
        i_sl_port = (v_sl_dr - v_sl[0]) * sl_driver_segment_g
        v_bl_target, _ = bl_driver.solve_clamp(
            i_bl_port.reshape(1),
            bl_driver_snapshot,
            v_clamp_init__V=v_bl_cl.reshape(1),
        )
        v_sl_target, _ = sl_driver.solve_clamp(
            i_sl_port.reshape(1),
            sl_driver_snapshot,
            v_clamp_init__V=v_sl_dr.reshape(1),
        )
        f_cl_bl = v_bl_cl - v_bl_target.squeeze(0)
        f_cl_sl = v_sl_dr - v_sl_target.squeeze(0)

        return torch.cat([f_bl, f_sl, f_x, f_cl_bl.reshape(1), f_cl_sl.reshape(1)])

    return residual


def numerical_jacobian(residual_fn: Callable[[Tensor], Tensor], u: Tensor, eps: float = 1e-6) -> Tensor:
    """Build the Jacobian ``∂F/∂u`` by symmetric finite differences.

    Symmetric (central) FD: ``(F(u+eps·e_i) − F(u−eps·e_i)) / (2 eps)``.
    Order-eps² accurate, way better than forward-difference at fp64.

    Returns Jacobian shape ``[len(F), len(u)]``.
    """
    n = u.numel()
    f0 = residual_fn(u)
    m = f0.numel()
    j = torch.zeros(m, n, dtype=u.dtype, device=u.device)
    for i in range(n):
        u_plus = u.clone()
        u_minus = u.clone()
        u_plus[i] += eps
        u_minus[i] -= eps
        j[:, i] = (residual_fn(u_plus) - residual_fn(u_minus)) / (2 * eps)
    return j
