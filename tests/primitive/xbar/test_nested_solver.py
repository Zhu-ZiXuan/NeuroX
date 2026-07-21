"""Unit tests for :class:`NestedParallelRailSolver`.

The subject is the resistor-network IR-drop solve: the harness
(:func:`tests.utils.standalone_solver_fixture.build_solver_harness`)
builds a fully LINEAR tiny tile — table-driven linear cells and ideal
``r_out = 0`` rail clamps — so the exact DCOP is the solution of a dense
KCL conductance system with Dirichlet boundaries at the reference taps.

Covers:
  * dense-oracle correctness: node voltages and driver currents match a
    ``torch.linalg.solve`` float64 assembly of the same network.
  * wire-residual decay: the returned DCOP's BL / SL wire KCL residuals
    sit at fp64 round-off.
  * inner-only entry point: ``solve_array_fixed_clamp`` is a direct
    solve for the linear network (the coupled block-2x2 wire Newton's
    Jacobian is exact, so one full step lands on the solution).
  * ``compute_residuals=False`` elides the residual algebra.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
import torch._dynamo
from torch import Tensor

from neurox.primitive.xbar.solver import NestedParallelRailSolver, NestedParallelRailSolverConfig
from tests.utils.standalone_solver_fixture import SolverHarness, build_solver_harness


@pytest.fixture(autouse=True)
def _eager_solver() -> Iterator[None]:
    """Run the solver eagerly for these tests.

    ``solve_dc`` is ``@torch.compile(dynamic=False)``; fully unrolling it
    would spend minutes compiling for no benefit to what is asserted.
    """
    with torch._dynamo.config.patch(disable=True):
        yield


def _dense_kcl_solution(harness: SolverHarness) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Solve the harness network as one dense KCL system per column.

    Assembly derives from the solver's own KCL convention
    (``_wire_kcl.col_wire_kcl_residual``): at wire node ``k`` the
    residual is ``i_inject + (v[k] - v[k-1]) g[k] + (v[k] - v[k+1])
    g[k+1]`` with the driver at ``v[-1]``; the linear cell injects
    ``i = g_cell (v_bl - v_sl)`` drained from BL and pushed into SL; the
    ideal drivers pin the rail boundaries at the reference taps
    (Dirichlet). Unknowns per column are ``[v_bl(0..R-1), v_sl(0..R-1)]``.

    Returns ``(v_bl_node, v_sl_node, i_bl_driver, i_sl_driver)`` with
    grid shapes ``(col, row)`` and line shapes ``(col,)``.
    """
    cfg = harness.cell_config
    idx = harness.w_state_idx
    dtype = harness.v_wl_drive__V.dtype
    device = harness.v_wl_drive__V.device

    # WL-switched per-cell chord conductance from the config tables
    # (batch dim is 1: drop it for the per-column assembly).
    g_on = torch.tensor(cfg.g_cell_on_table__uS, device=device, dtype=dtype)[idx]
    g_off = torch.tensor(cfg.g_cell_off_table__uS, device=device, dtype=dtype)[idx]
    on = harness.v_wl_drive__V[0] > cfg.v_wl_on_threshold__V
    g_cell = torch.where(on, g_on, g_off)  # (col, row)

    g_bl = harness.bl_segment_g__uS
    g_sl = harness.sl_segment_g__uS
    v_bl_ref = harness.bl_v_ref__V.to(device=device, dtype=dtype)
    v_sl_ref = harness.sl_v_ref__V.to(device=device, dtype=dtype)

    col_num, row_num = g_cell.shape
    n = row_num
    lhs = torch.zeros(col_num, 2 * n, 2 * n, device=device, dtype=dtype)
    rhs = torch.zeros(col_num, 2 * n, device=device, dtype=dtype)
    for k in range(n):
        g_right_bl = g_bl[k + 1] if k + 1 < n else torch.zeros((), device=device, dtype=dtype)
        g_right_sl = g_sl[k + 1] if k + 1 < n else torch.zeros((), device=device, dtype=dtype)
        # BL node k: +i_cell drained.
        lhs[:, k, k] = g_bl[k] + g_right_bl + g_cell[:, k]
        lhs[:, k, n + k] = -g_cell[:, k]
        if k > 0:
            lhs[:, k, k - 1] = -g_bl[k]
        if k + 1 < n:
            lhs[:, k, k + 1] = -g_bl[k + 1]
        # SL node k: -i_cell injected.
        lhs[:, n + k, n + k] = g_sl[k] + g_right_sl + g_cell[:, k]
        lhs[:, n + k, k] = -g_cell[:, k]
        if k > 0:
            lhs[:, n + k, n + k - 1] = -g_sl[k]
        if k + 1 < n:
            lhs[:, n + k, n + k + 1] = -g_sl[k + 1]
    rhs[:, 0] = g_bl[0] * v_bl_ref
    rhs[:, n] = g_sl[0] * v_sl_ref

    x = torch.linalg.solve(lhs, rhs)
    v_bl_node = x[:, :n]
    v_sl_node = x[:, n:]
    i_bl_driver = (v_bl_ref - v_bl_node[:, 0]) * g_bl[0]
    i_sl_driver = (v_sl_ref - v_sl_node[:, 0]) * g_sl[0]
    return v_bl_node, v_sl_node, i_bl_driver, i_sl_driver


def test_dcop_matches_dense_kcl_oracle(device: torch.device) -> None:
    """Nested solver reproduces the dense KCL solve of the linear network."""
    harness = build_solver_harness(
        solver_config=NestedParallelRailSolverConfig(n_outer=3, n_inner=3),
        device=device,
    )
    dcop = harness.solver.solve_dc(**harness.solver_kwargs(), compute_residuals=False)
    v_bl_exp, v_sl_exp, i_bl_exp, i_sl_exp = _dense_kcl_solution(harness)

    tol = {"rtol": 0.0, "atol": 1e-9}
    # Ideal drivers: the clamp boundaries settle exactly at the taps.
    torch.testing.assert_close(dcop.v_bl_clamp, harness.bl_v_ref__V.expand_as(dcop.v_bl_clamp), **tol)
    torch.testing.assert_close(dcop.v_sl_drive, harness.sl_v_ref__V.expand_as(dcop.v_sl_drive), **tol)
    # Node-voltage profiles and driver currents (batch dim is 1).
    torch.testing.assert_close(dcop.v_bl_node[0], v_bl_exp, **tol)
    torch.testing.assert_close(dcop.v_sl_node[0], v_sl_exp, **tol)
    torch.testing.assert_close(dcop.i_bl_driver[0], i_bl_exp, **tol)
    torch.testing.assert_close(dcop.i_sl_driver[0], i_sl_exp, **tol)


def test_wire_residuals_at_machine_precision(device: torch.device) -> None:
    """Nested solver drives both wire KCL residuals to fp64 noise."""
    harness = build_solver_harness(
        solver_config=NestedParallelRailSolverConfig(n_outer=3, n_inner=3),
        device=device,
    )
    dcop = harness.solver.solve_dc(**harness.solver_kwargs(), compute_residuals=True)
    residuals = dcop.residuals
    assert residuals is not None
    assert residuals.wire_bl__uA.max().item() < 1e-9
    assert residuals.wire_sl__uA.max().item() < 1e-9


def test_inner_direct_solve_exactness(device: torch.device) -> None:
    """``solve_array_fixed_clamp`` solves the linear network directly.

    With a linear cell the inner coupled block-2x2 wire Newton has an
    exact Jacobian, so it is a direct solve: the pinned-clamp solution
    matches the dense KCL oracle (the pinned values equal the reference
    taps, i.e. the same Dirichlet boundary) and the wire residuals sit
    at fp64 noise.
    """
    harness = build_solver_harness(
        solver_config=NestedParallelRailSolverConfig(n_outer=1, n_inner=3),
        device=device,
    )
    solver = harness.solver
    assert isinstance(solver, NestedParallelRailSolver)
    v_wl = harness.v_wl_drive__V
    *batch, col_num, _row = v_wl.shape
    dtype = v_wl.dtype
    v_bl_clamp = harness.bl_v_ref__V.to(device=device, dtype=dtype).expand(*batch, col_num)
    v_sl_drive = harness.sl_v_ref__V.to(device=device, dtype=dtype).expand(*batch, col_num)
    dcop = solver.solve_array_fixed_clamp(
        v_bl_clamp__V=v_bl_clamp,
        v_sl_drive__V=v_sl_drive,
        bl_segment_r__MOhm=harness.bl_segment_r__MOhm,
        sl_segment_r__MOhm=harness.sl_segment_r__MOhm,
        bl_segment_g__uS=harness.bl_segment_g__uS,
        sl_segment_g__uS=harness.sl_segment_g__uS,
        cell=harness.cell,
        cell_snap=harness.cell_snapshot(),
        compute_residuals=True,
    )
    assert dcop.residuals is not None
    assert dcop.residuals.wire_bl__uA.max().item() < 1e-9
    assert dcop.residuals.wire_sl__uA.max().item() < 1e-9

    v_bl_exp, v_sl_exp, _i_bl_exp, _i_sl_exp = _dense_kcl_solution(harness)
    tol = {"rtol": 0.0, "atol": 1e-9}
    torch.testing.assert_close(dcop.v_bl_node[0], v_bl_exp, **tol)
    torch.testing.assert_close(dcop.v_sl_node[0], v_sl_exp, **tol)


def test_residuals_none_on_hot_path(device: torch.device) -> None:
    """``compute_residuals=False`` elides the residual algebra."""
    harness = build_solver_harness(
        solver_config=NestedParallelRailSolverConfig(n_outer=2, n_inner=2),
        device=device,
    )
    dcop = harness.solver.solve_dc(**harness.solver_kwargs(), compute_residuals=False)
    assert dcop.residuals is None
