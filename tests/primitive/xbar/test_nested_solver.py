"""Unit tests for :class:`NestedParallelRailSolver`.

The subject is the resistor-network IR-drop solve: the harness
(:func:`tests.utils.standalone_solver_fixture.build_solver_harness`)
builds a fully LINEAR tiny tile — table-driven linear cells and ideal
``r_out = 0`` rail clamps — so the exact DCOP is the solution of a dense
KCL conductance system with Dirichlet boundaries at the reference taps.

Covers:
  * dense-oracle correctness: node voltages and driver currents match a
    ``torch.linalg.solve`` float64 assembly of the same network.
  * wire-residual decay: the :class:`SolverObservation` submitted to
    :class:`SolverProber` carries BL / SL wire KCL residuals at fp64
    round-off, alongside the converged DCOP.
  * inner-only entry point: ``solve_array_fixed_clamp`` is a direct
    solve for the linear network (the coupled block-2x2 wire Newton's
    Jacobian is exact, so one full step lands on the solution).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
import torch._dynamo
from torch import Tensor

from neurox.primitive.xbar.cell import XbarCellDcop
from neurox.primitive.xbar.solver import (
    NestedParallelRailSolver,
    NestedParallelRailSolverConfig,
    SolverObservation,
    SolverProber,
)
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
    grid shapes ``(col, row)`` and per-column shapes ``(col,)``.
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
    dcop = harness.solver.solve_dc(**harness.solver_kwargs())
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


def _assert_fully_detached(observation: SolverObservation[XbarCellDcop]) -> None:
    """Every tensor on the observation — including inside its DCOP — is detached."""
    for tensor in (
        observation.wire_bl__uA,
        observation.wire_sl__uA,
        observation.clamp_bl__V,
        observation.clamp_sl__V,
        observation.dcop.i_bl_driver,
        observation.dcop.i_sl_driver,
        observation.dcop.v_bl_node,
        observation.dcop.v_sl_node,
        observation.dcop.v_bl_clamp,
        observation.dcop.v_sl_drive,
        observation.dcop.cell.i__uA,
        observation.dcop.cell.di_dvbl__uS,
        observation.dcop.cell.di_dvsl__uS,
    ):
        assert tensor.grad_fn is None
        assert tensor.requires_grad is False


def test_observation_carries_dcop_and_residuals(device: torch.device) -> None:
    """A probed solve emits one detached observation: DCOP + wire residuals.

    Nested solver drives both wire KCL residuals to fp64 noise; the payload
    also carries the converged DCOP (finite tensors, expected leading shape),
    all tensors fully detached including inside ``dcop`` and ``dcop.cell``.
    """
    harness = build_solver_harness(
        solver_config=NestedParallelRailSolverConfig(n_outer=3, n_inner=3),
        device=device,
    )
    with SolverProber() as prober:
        dcop = harness.solver.solve_dc(**harness.solver_kwargs())
    records = prober.records
    # One public solve_dc invocation emits exactly one observation record.
    assert len(records) == 1
    observation = records[0]

    assert observation.wire_bl__uA.max().item() < 1e-9
    assert observation.wire_sl__uA.max().item() < 1e-9

    # The observation carries the converged DCOP with the same leading shape.
    assert observation.dcop.v_bl_node.shape == dcop.v_bl_node.shape
    assert torch.isfinite(observation.dcop.v_bl_node).all()
    assert torch.isfinite(observation.dcop.cell.i__uA).all()
    _assert_fully_detached(observation)


def test_solve_output_bit_identical_probed_vs_unprobed(device: torch.device) -> None:
    """The demand gate never perturbs the solve: DCOP is bit-identical."""
    harness = build_solver_harness(
        solver_config=NestedParallelRailSolverConfig(n_outer=3, n_inner=3),
        device=device,
    )
    unprobed = harness.solver.solve_dc(**harness.solver_kwargs())
    with SolverProber():
        probed = harness.solver.solve_dc(**harness.solver_kwargs())
    for field in ("v_bl_node", "v_sl_node", "v_bl_clamp", "v_sl_drive", "i_bl_driver", "i_sl_driver"):
        assert torch.equal(getattr(unprobed, field), getattr(probed, field))
    assert torch.equal(unprobed.cell.i__uA, probed.cell.i__uA)


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
    with SolverProber() as prober:
        dcop = solver.solve_array_fixed_clamp(
            v_bl_clamp__V=v_bl_clamp,
            v_sl_drive__V=v_sl_drive,
            bl_segment_r__MOhm=harness.bl_segment_r__MOhm,
            sl_segment_r__MOhm=harness.sl_segment_r__MOhm,
            bl_segment_g__uS=harness.bl_segment_g__uS,
            sl_segment_g__uS=harness.sl_segment_g__uS,
            cell=harness.cell,
            cell_snap=harness.cell_snapshot(),
        )
    records = prober.records
    assert len(records) == 1
    observation = records[0]
    assert observation.wire_bl__uA.max().item() < 1e-9
    assert observation.wire_sl__uA.max().item() < 1e-9
    # Pinned clamps → clamp residual identically zero.
    assert torch.equal(observation.clamp_bl__V, torch.zeros_like(observation.clamp_bl__V))
    assert torch.equal(observation.clamp_sl__V, torch.zeros_like(observation.clamp_sl__V))

    v_bl_exp, v_sl_exp, _i_bl_exp, _i_sl_exp = _dense_kcl_solution(harness)
    tol = {"rtol": 0.0, "atol": 1e-9}
    torch.testing.assert_close(dcop.v_bl_node[0], v_bl_exp, **tol)
    torch.testing.assert_close(dcop.v_sl_node[0], v_sl_exp, **tol)
