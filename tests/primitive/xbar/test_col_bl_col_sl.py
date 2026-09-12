"""Focused tests for the adaptive parallel-rail solver."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any, NamedTuple
from unittest.mock import patch

import pytest
import torch
from torch import Tensor

from neurox.common.module import DcopBase
from neurox.primitive.analog import VoltageDriver, VoltageDriverConfig, VoltageDriverPolicy
from neurox.primitive.xbar.solver import col_bl_col_sl as solver
from neurox.primitive.xbar.solver._wire_kcl import f_kcl__uA, f_kcl_roundoff__uA
from tests.utils.standalone_solver_fixture import (
    BL_V_REF__V,
    SL_V_REF__V,
    BL_SEGMENT_R__MOhm,
    SL_SEGMENT_R__MOhm,
    SolverDcop,
    SolverHarness,
    build_solver_harness,
    solve_dcop,
)


def _kwargs(device: torch.device, *, col_num: int = 3, row_num: int = 4) -> dict[str, Any]:
    return build_solver_harness(device=device, col_num=col_num, row_num=row_num).solve_kwargs()


def _solver(kwargs: dict[str, Any]) -> solver.ColBlColSlArraySolver[Any, Any, Any, Any]:
    return solver.ColBlColSlArraySolver(
        bl_segment_r__MOhm=kwargs["bl_segment_r__MOhm"],
        sl_segment_r__MOhm=kwargs["sl_segment_r__MOhm"],
        cell=kwargs["cell"],
        bl_driver=kwargs["bl_driver"],
        sl_driver=kwargs["sl_driver"],
        dtype=kwargs["bl_driver_snap"].v_ref__V.dtype,
    )


def _solve(
    kwargs: dict[str, Any],
    *,
    record_trace: bool,
) -> tuple[SolverDcop, solver.ColBlColSlArrayTrace | None]:
    dcop, trace = solve_dcop(
        _solver(kwargs),
        cell_snap=kwargs["cell_snap"],
        bl_driver_snap=kwargs["bl_driver_snap"],
        sl_driver_snap=kwargs["sl_driver_snap"],
        record_trace=record_trace,
    )

    return dcop, trace


def _used_iterations(residual: Tensor, *, history_ndim: int = 1) -> Tensor:
    return (~residual.isnan()).any(dim=tuple(range(residual.ndim - history_ndim)))


def _assert_terminal_converged(residual: Tensor, threshold: Tensor, *, dim: int = -1) -> None:
    residual = residual.movedim(dim, -1)
    threshold = threshold.movedim(dim, -1)
    valid = ~residual.isnan()
    assert valid.any()
    assert not ((~valid[..., :-1]) & valid[..., 1:]).any()
    terminal = (valid.sum(dim=-1) - 1).clamp_min(0).unsqueeze(-1)
    final_residual = residual.gather(-1, terminal).squeeze(-1)
    final_threshold = threshold.gather(-1, terminal).squeeze(-1)
    assert (final_residual[valid.any(dim=-1)] <= final_threshold[valid.any(dim=-1)]).all()


def _assert_converged(trace: solver.ColBlColSlArrayTrace) -> None:
    assert trace.node_trace is not None
    _assert_terminal_converged(trace.residual__V, trace.threshold__V)
    _assert_terminal_converged(trace.node_trace.residual__uA, trace.node_trace.threshold__uA, dim=-2)
    port_used = _used_iterations(trace.residual__V)
    node_used = _used_iterations(trace.node_trace.residual__uA, history_ndim=2)
    assert torch.equal(node_used.any(dim=0), port_used)


def test_loop_states_carry_only_next_iteration_values() -> None:
    assert {field.name for field in fields(solver._NodeState)} == {
        "is_active",
        "v_bl_node__V",
        "v_sl_node__V",
    }
    assert {field.name for field in fields(solver.ColBlColSlArrayState)} == {
        "is_active",
        "v_bl_node__V",
        "v_bl_port__V",
        "v_sl_node__V",
        "v_sl_port__V",
    }


@pytest.mark.parametrize("record_trace", [False, True])
def test_settled_nodes_still_require_port_updates(device: torch.device, record_trace: bool) -> None:
    kwargs = _nonideal_kwargs(device)
    port_solver = _solver(kwargs)
    bl = kwargs["bl_driver_snap"].v_ref__V
    sl = kwargs["sl_driver_snap"].v_ref__V
    shape = kwargs["cell_snap"].v_wl__V.shape
    is_active = torch.ones_like(bl, dtype=torch.bool)

    node_state, node_trace = port_solver.node_solver.solve(
        bl.expand(shape).clone(),
        sl.expand(shape).clone(),
        bl,
        sl,
        cell_snap=kwargs["cell_snap"],
        is_active=is_active,
        final_fn=lambda state: state,
        record_trace=record_trace,
        trace_mask=None,
    )
    port_state, port_trace = port_solver._evaluate_port(
        node_state,
        node_trace=node_trace,
        v_bl_port__V=bl,
        v_sl_port__V=sl,
        is_active=is_active,
        cell_snap=kwargs["cell_snap"],
        bl_driver_snap=kwargs["bl_driver_snap"],
        sl_driver_snap=kwargs["sl_driver_snap"],
    )
    # Settled nodes still require a port correction under resistive drivers.
    assert not node_state.is_active.any()
    assert port_state.is_active.all()
    assert (port_state.v_bl_port__V < bl).all()
    assert (port_state.v_sl_port__V > sl).all()
    assert is_active.all()
    assert port_trace.node_trace is node_trace
    if record_trace:
        assert node_trace is not None
        _assert_terminal_converged(node_trace.residual__uA, node_trace.threshold__uA)
    else:
        assert node_trace is None


def test_solver_rejects_nonpositive_iteration_capacity(device: torch.device) -> None:
    kwargs = _kwargs(device)
    with (
        patch.object(solver.ColBlColSlArraySolver, "MAX_ITER", 0),
        pytest.raises(ValueError, match="max_iter must be positive"),
    ):
        _solve(kwargs, record_trace=True)


class _AliasingCellInput(NamedTuple):
    """Standalone cell input with no module snapshot base."""

    g_cell__uS: Tensor


class _AliasingCellDcop(DcopBase):
    i__uA: Tensor
    di_dvbl__uS: Tensor
    di_dvsl__uS: Tensor


class _AliasingCell:
    inst_shape = (4, 2)

    def solve_dc(
        self,
        v_bl__V: Tensor,
        v_sl__V: Tensor,
        snap: _AliasingCellInput,
    ) -> _AliasingCellDcop:
        i_cell__uA = (v_bl__V - v_sl__V) * snap.g_cell__uS
        return _AliasingCellDcop(
            i__uA=i_cell__uA,
            di_dvbl__uS=snap.g_cell__uS,
            di_dvsl__uS=-snap.g_cell__uS,
        )


def _dense_kcl_solution(
    harness: SolverHarness, *, bl_r: float = 0.0, sl_r: float = 0.0
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Solve the same linear network as one dense system per column."""
    cfg = harness.cell_config
    idx = harness.w_state_idx
    dtype = harness.v_wl_drive__V.dtype
    device = harness.v_wl_drive__V.device
    g_on = torch.tensor(cfg.g_cell_on_table__uS, device=device, dtype=dtype)[idx]
    g_off = torch.tensor(cfg.g_cell_off_table__uS, device=device, dtype=dtype)[idx]
    g_cell = torch.where(harness.v_wl_drive__V[0].unsqueeze(-1) > cfg.v_wl_on_threshold__V, g_on, g_off)
    g_bl = 1.0 / harness.bl_segment_r__MOhm
    g_sl = 1.0 / harness.sl_segment_r__MOhm
    v_bl_ref = harness.bl_v_ref__V.to(device=device, dtype=dtype)
    v_sl_ref = harness.sl_v_ref__V.to(device=device, dtype=dtype)

    row_num, col_num = g_cell.shape
    lhs = torch.zeros(col_num, 2 * row_num, 2 * row_num, device=device, dtype=dtype)
    rhs = torch.zeros(col_num, 2 * row_num, device=device, dtype=dtype)
    for row in range(row_num):
        g_right = 0.0 if row + 1 == row_num else 1.0
        lhs[:, row, row] = g_bl * (1.0 + g_right) + g_cell[row, :]
        lhs[:, row, row_num + row] = -g_cell[row, :]
        lhs[:, row_num + row, row_num + row] = g_sl * (1.0 + g_right) + g_cell[row, :]
        lhs[:, row_num + row, row] = -g_cell[row, :]
        if row > 0:
            lhs[:, row, row - 1] = -g_bl
            lhs[:, row_num + row, row_num + row - 1] = -g_sl
        if row + 1 < row_num:
            lhs[:, row, row + 1] = -g_bl
            lhs[:, row_num + row, row_num + row + 1] = -g_sl
    drive_bl_g = 1.0 / (harness.bl_segment_r__MOhm + bl_r)
    drive_sl_g = 1.0 / (harness.sl_segment_r__MOhm + sl_r)
    lhs[:, 0, 0] += drive_bl_g - g_bl
    lhs[:, row_num, row_num] += drive_sl_g - g_sl
    rhs[:, 0] = drive_bl_g * v_bl_ref
    rhs[:, row_num] = drive_sl_g * v_sl_ref

    solution = torch.linalg.solve(lhs, rhs)
    v_bl_node__V = solution[:, :row_num]
    v_sl_node__V = solution[:, row_num:]
    i_bl_port__uA = (v_bl_ref - v_bl_node__V[:, 0]) * drive_bl_g
    i_sl_port__uA = (v_sl_ref - v_sl_node__V[:, 0]) * drive_sl_g
    return v_bl_node__V.T, v_sl_node__V.T, i_bl_port__uA, i_sl_port__uA


def _nonideal_driver(
    *,
    reference: torch.Tensor,
    inst_shape: tuple[int, ...],
    r_out__MOhm: float,
) -> tuple[VoltageDriver, object]:
    driver = VoltageDriver(
        config=VoltageDriverConfig(
            r_out__MOhm=r_out__MOhm,
            offset_sigma__V=0.0,
            thermal_sigma__V=0.0,
            energy_per_op__fJ=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        policy=VoltageDriverPolicy(offset=False, thermal=False),
        inst_shape=inst_shape,
        dtype=reference.dtype,
        T__K=300.0,
    )
    driver.to(reference.device)
    driver.eval()
    driver.fabricate()
    return driver, driver.snapshot(v_ref__V=reference, shape=tuple(reference.shape))


def _nonideal_kwargs(device: torch.device) -> dict[str, Any]:
    kwargs = _kwargs(device)
    old_bl = kwargs["bl_driver"]
    old_sl = kwargs["sl_driver"]
    old_bl_snap = kwargs["bl_driver_snap"]
    old_sl_snap = kwargs["sl_driver_snap"]
    bl_driver, bl_snap = _nonideal_driver(
        reference=old_bl_snap.v_ref__V,
        inst_shape=old_bl.inst_shape,
        r_out__MOhm=2e-3,
    )
    sl_driver, sl_snap = _nonideal_driver(
        reference=old_sl_snap.v_ref__V,
        inst_shape=old_sl.inst_shape,
        r_out__MOhm=3e-3,
    )
    kwargs.update(
        bl_driver=bl_driver,
        bl_driver_snap=bl_snap,
        sl_driver=sl_driver,
        sl_driver_snap=sl_snap,
    )
    return kwargs


@pytest.mark.parametrize("nonideal", [False, True])
def test_dcop_matches_dense_kcl_oracle(device: torch.device, nonideal: bool) -> None:
    harness = build_solver_harness(device=device)
    kwargs = _nonideal_kwargs(device) if nonideal else harness.solve_kwargs()
    bl_r = 2e-3 if nonideal else 0.0
    sl_r = 3e-3 if nonideal else 0.0
    dcop, trace = _solve(kwargs, record_trace=True)
    expected = _dense_kcl_solution(harness, bl_r=bl_r, sl_r=sl_r)
    actual = dcop.v_bl_node__V[0], dcop.v_sl_node__V[0], dcop.i_bl_port__uA[0, 0], dcop.i_sl_port__uA[0, 0]
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=0.0, atol=1e-9)
    assert trace is not None
    _assert_converged(trace)


def test_terminal_electrical_state_and_trace_report_convergence(device: torch.device) -> None:
    kwargs = _nonideal_kwargs(device)
    dcop, trace = _solve(kwargs, record_trace=True)
    assert trace is not None
    f_bl = f_kcl__uA(
        dcop.v_bl_node__V,
        dcop.v_bl_port__V,
        1.0 / kwargs["bl_segment_r__MOhm"],
        dcop.cell_dcop.i__uA,
        dim=-2,
    )
    f_sl = f_kcl__uA(
        dcop.v_sl_node__V,
        dcop.v_sl_port__V,
        1.0 / kwargs["sl_segment_r__MOhm"],
        -dcop.cell_dcop.i__uA,
        dim=-2,
    )
    array_solver = _solver(kwargs)
    roundoff__uA = torch.maximum(
        f_kcl_roundoff__uA(
            dcop.v_bl_node__V,
            dcop.v_bl_port__V,
            array_solver.bl_g__uS,
            dim=-2,
        ),
        f_kcl_roundoff__uA(
            dcop.v_sl_node__V,
            dcop.v_sl_port__V,
            array_solver.sl_g__uS,
            dim=-2,
        ),
    )
    node_threshold__uA = (
        array_solver.node_solver.atol + array_solver.node_solver.rtol * dcop.cell_dcop.i__uA.abs() + roundoff__uA
    )
    assert (torch.maximum(f_bl.abs(), f_sl.abs()) <= node_threshold__uA).all()

    bl = kwargs["bl_driver"].solve_dc(dcop.i_bl_port__uA, kwargs["bl_driver_snap"], v_port_init__V=dcop.v_bl_port__V)
    sl = kwargs["sl_driver"].solve_dc(dcop.i_sl_port__uA, kwargs["sl_driver_snap"], v_port_init__V=dcop.v_sl_port__V)
    f_bl_port__V = bl.v_port__V - dcop.v_bl_port__V
    f_sl_port__V = sl.v_port__V - dcop.v_sl_port__V
    v_bl_scale__V = torch.maximum(dcop.v_bl_port__V.abs(), dcop.v_bl_node__V[..., :1, :].abs())
    v_sl_scale__V = torch.maximum(dcop.v_sl_port__V.abs(), dcop.v_sl_node__V[..., :1, :].abs())
    v_port_scale__V = torch.maximum(
        torch.maximum(v_bl_scale__V, v_sl_scale__V),
        torch.maximum(bl.v_port__V.abs(), sl.v_port__V.abs()),
    )
    threshold__V = (
        array_solver.atol
        + array_solver.rtol * v_port_scale__V
        + array_solver.roundoff
        * torch.maximum(
            (bl.dvport_di__MOhm * array_solver.bl_g__uS).abs() * v_bl_scale__V,
            (sl.dvport_di__MOhm * array_solver.sl_g__uS).abs() * v_sl_scale__V,
        )
    )
    assert (torch.maximum(f_bl_port__V.abs(), f_sl_port__V.abs()) <= threshold__V).all()
    _assert_converged(trace)


@pytest.mark.parametrize(("col_num", "row_num"), [(1, 4), (3, 1), (1, 1)])
def test_degenerate_array_axes_converge(device: torch.device, col_num: int, row_num: int) -> None:
    dcop, trace = _solve(_kwargs(device, col_num=col_num, row_num=row_num), record_trace=True)
    assert dcop.v_bl_node__V.shape[-2:] == (row_num, col_num)
    assert trace is not None
    _assert_converged(trace)


def test_complete_runtime_snap_leading_is_solved_as_one_population(device: torch.device) -> None:
    kwargs = _kwargs(device, col_num=2, row_num=4)
    cell = kwargs["cell"]
    control = torch.tensor(
        [
            [[0.9, 0.0, 0.9, 0.0], [0.0, 0.9, 0.0, 0.9]],
            [[0.0, 0.9, 0.0, 0.9], [0.9, 0.0, 0.9, 0.0]],
        ],
        dtype=torch.float64,
        device=device,
    )
    control = control.movedim(-1, -2).contiguous()
    kwargs["cell_snap"] = cell.snapshot(control=control, shape=control.shape)
    port_shape = (*control.shape[:-2], 1, control.shape[-1])
    kwargs["bl_driver_snap"] = kwargs["bl_driver"].snapshot(
        v_ref__V=kwargs["bl_driver_snap"].v_ref__V.expand(port_shape),
        shape=port_shape,
    )
    kwargs["sl_driver_snap"] = kwargs["sl_driver"].snapshot(
        v_ref__V=kwargs["sl_driver_snap"].v_ref__V.expand(port_shape),
        shape=port_shape,
    )
    _, trace = _solve(kwargs, record_trace=True)

    assert trace is not None
    initial_node_trace = trace.select(0).node_trace
    assert initial_node_trace is not None
    assert (~initial_node_trace.residual__uA[..., 0].isnan()).any(dim=-2).sum() == 4
    _assert_terminal_converged(initial_node_trace.residual__uA, initial_node_trace.threshold__uA)
    _assert_converged(trace)


def test_single_row_matches_closed_form(device: torch.device) -> None:
    """One wire node per rail reduces to the exact series-link circuit."""
    kwargs = _kwargs(device, row_num=1)
    dcop, _ = _solve(kwargs, record_trace=False)
    cell_snap = kwargs["cell_snap"]
    g_cell__uS = cell_snap.g_cell_on__uS
    dv__V = BL_V_REF__V - SL_V_REF__V
    expected_i__uA = g_cell__uS * dv__V / (1.0 + g_cell__uS * (BL_SEGMENT_R__MOhm + SL_SEGMENT_R__MOhm))
    expected_v_bl__V = BL_V_REF__V - expected_i__uA * BL_SEGMENT_R__MOhm
    expected_v_sl__V = SL_V_REF__V + expected_i__uA * SL_SEGMENT_R__MOhm

    tol = {"rtol": 1e-11, "atol": 1e-12}
    torch.testing.assert_close(dcop.v_bl_node__V, expected_v_bl__V, **tol)
    torch.testing.assert_close(dcop.v_sl_node__V, expected_v_sl__V, **tol)


def test_trace_switch_preserves_solution_and_records_only_executed_samples(device: torch.device) -> None:
    kwargs = _nonideal_kwargs(device)
    plain, absent = _solve(kwargs, record_trace=False)
    dcop, trace = _solve(kwargs, record_trace=True)
    assert absent is None
    assert trace is not None
    torch.testing.assert_close(plain.v_bl_node__V, dcop.v_bl_node__V)
    torch.testing.assert_close(plain.v_sl_node__V, dcop.v_sl_node__V)
    node_trace = trace.node_trace
    assert node_trace is not None
    port_capacity = solver.ColBlColSlArraySolver.MAX_ITER
    node_capacity = solver._NodeSolver.MAX_ITER
    assert trace.residual__V.shape == (*dcop.v_bl_port__V.shape, port_capacity)
    assert node_trace.residual__uA.shape == (*dcop.v_bl_node__V.shape, node_capacity, port_capacity)
    assert trace.limited.dtype is torch.bool
    assert node_trace.limited.dtype is torch.bool
    _assert_converged(trace)
    port_used = _used_iterations(trace.residual__V)
    port_iterations = port_used.sum() - 1
    assert 0 < port_iterations < port_capacity
    port_valid = ~trace.residual__V.isnan()
    assert not trace.limited[~port_valid].any()
    for value in (trace.threshold__V, trace.dv_bl_port_abs__V, trace.dv_sl_port_abs__V):
        assert torch.equal(value.isnan(), ~port_valid)
    node_valid = ~node_trace.residual__uA.isnan()
    assert not node_trace.limited[~node_valid.any(dim=-4, keepdim=True)].any()
    for value in (node_trace.threshold__uA, node_trace.dv_bl_node_abs__V, node_trace.dv_sl_node_abs__V):
        assert torch.equal(value.isnan(), ~node_valid)


@pytest.mark.parametrize("record_trace", [False, True])
@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_aliasing_cell_is_public_caller_fullgraph_safe(device: torch.device, record_trace: bool, backend: str) -> None:
    kwargs = _kwargs(device, col_num=2, row_num=4)
    kwargs["cell"] = _AliasingCell()
    kwargs["cell_snap"] = _AliasingCellInput(g_cell__uS=torch.full((1, 4, 2), 50.0, dtype=torch.float64, device=device))
    array_solver: solver.ColBlColSlArraySolver[_AliasingCellInput, _AliasingCellDcop, Any, Any] = _solver(kwargs)

    def solve() -> tuple[Tensor, ...]:
        dcop, trace = solve_dcop(
            array_solver,
            cell_snap=kwargs["cell_snap"],
            bl_driver_snap=kwargs["bl_driver_snap"],
            sl_driver_snap=kwargs["sl_driver_snap"],
            record_trace=record_trace,
        )
        if trace is None:
            return dcop.cell_dcop.i__uA, dcop.v_bl_node__V
        assert trace.node_trace is not None
        return (
            dcop.cell_dcop.i__uA,
            trace.residual__V,
            trace.threshold__V,
            trace.dv_bl_port_abs__V,
            trace.node_trace.residual__uA,
            trace.node_trace.threshold__uA,
        )

    expected = solve()
    actual = torch.compile(solve, backend=backend, fullgraph=True)()
    torch.testing.assert_close(actual, expected, equal_nan=True)


@pytest.mark.parametrize("record_trace", [False, True])
def test_fullgraph_reuses_graph_with_fresh_solver_snapshots(device: torch.device, record_trace: bool) -> None:
    kwargs = _kwargs(device, col_num=2, row_num=4)
    cell = _AliasingCell()
    kwargs["cell"] = cell
    array_solver: solver.ColBlColSlArraySolver[_AliasingCellInput, _AliasingCellDcop, Any, Any] = _solver(kwargs)
    compiled_graphs = []

    def backend(graph: torch.fx.GraphModule, _inputs: list[Tensor]) -> object:
        compiled_graphs.append(graph)
        return graph.forward

    def solve(snap: _AliasingCellInput) -> tuple[Tensor, Tensor]:
        dcop, trace = solve_dcop(
            array_solver,
            cell_snap=snap,
            bl_driver_snap=kwargs["bl_driver_snap"],
            sl_driver_snap=kwargs["sl_driver_snap"],
            record_trace=record_trace,
        )
        if trace is not None:
            assert trace.node_trace is not None
            return dcop.cell_dcop.i__uA, trace.node_trace.residual__uA
        return dcop.cell_dcop.i__uA, dcop.v_bl_node__V

    snaps = [
        _AliasingCellInput(g_cell__uS=torch.full((1, 4, 2), g_cell__uS, dtype=torch.float64, device=device))
        for g_cell__uS in (30.0, 50.0, 80.0, 30.0)
    ]
    expected = [solve(snap) for snap in snaps]
    with torch._dynamo.config.patch(disable=False):
        compiled = torch.compile(solve, backend=backend, fullgraph=True, dynamic=False)
        for snap, reference in zip(snaps, expected, strict=True):
            actual = compiled(snap)
            torch.testing.assert_close(actual[0], reference[0])
            torch.testing.assert_close(actual[1], reference[1], equal_nan=True)
    assert len(compiled_graphs) == 1


def test_trace_off_never_calls_trace_allocator(device: torch.device) -> None:
    with (
        patch.object(solver, "run_solving_trace_scan", side_effect=AssertionError("scan reached")),
        patch.object(solver._PortTrace, "empty", side_effect=AssertionError("port history allocated")),
        patch.object(solver._NodeTrace, "empty", side_effect=AssertionError("node history allocated")),
    ):
        _, trace = _solve(_kwargs(device), record_trace=False)
    assert trace is None


def test_capped_node_solve_aborts_the_regular_solve(device: torch.device) -> None:
    kwargs = _nonideal_kwargs(device)
    snap = kwargs["cell_snap"]
    conductance = snap.g_cell_on__uS.clone()
    conductance[..., :, 0] = 10000.0
    kwargs["cell_snap"] = replace(snap, g_cell_on__uS=conductance)

    with (
        patch.object(solver._NodeSolver, "MAX_ITER", 1),
        pytest.raises(RuntimeError, match="did not converge"),
    ):
        _solve(kwargs, record_trace=False)


def test_capped_node_solve_returns_the_requested_trace(device: torch.device) -> None:
    kwargs = _nonideal_kwargs(device)
    snap = kwargs["cell_snap"]
    conductance = snap.g_cell_on__uS.clone()
    conductance[..., :, 0] = 10000.0
    kwargs["cell_snap"] = replace(snap, g_cell_on__uS=conductance)

    with (
        patch.object(solver._NodeSolver, "MAX_ITER", 1),
        patch.object(solver.ColBlColSlArraySolver, "MAX_ITER", 2),
    ):
        dcop, trace = _solve(kwargs, record_trace=True)

    assert trace is not None
    assert trace.node_trace is not None
    assert (trace.node_trace.residual__uA[..., 0, 0] > trace.node_trace.threshold__uA[..., 0, 0]).any()
    # A capped node solve still supplies voltages for the port residual check.
    port_used = _used_iterations(trace.residual__V)
    assert port_used[0]
    valid = ~trace.residual__V.isnan()
    assert (trace.residual__V / trace.threshold__V)[valid].isfinite().all()
    assert dcop.v_bl_node__V.isfinite().all()
    assert dcop.v_sl_node__V.isfinite().all()


def test_wire_newton_correction_obeys_step_limit(device: torch.device) -> None:
    kwargs = _kwargs(device, row_num=16)
    kwargs["bl_segment_r__MOhm"] = 2e-3
    kwargs["sl_segment_r__MOhm"] = 4e-3
    node_solver = _solver(kwargs).node_solver
    v_bl_port__V = kwargs["bl_driver_snap"].v_ref__V
    v_sl_port__V = kwargs["sl_driver_snap"].v_ref__V
    node_shape = kwargs["cell_snap"].v_wl__V.shape
    v_bl_node__V = v_bl_port__V.expand(node_shape)
    v_sl_node__V = v_sl_port__V.expand(node_shape)
    state, trace = node_solver._evaluate_node(
        v_bl_port__V,
        v_sl_port__V,
        v_bl_node__V,
        v_sl_node__V,
        is_active=torch.ones_like(v_bl_port__V, dtype=torch.bool),
        cell_snap=kwargs["cell_snap"],
    )
    assert trace is not None
    assert state.is_active.all()
    applied_dv_bl_node__V = state.v_bl_node__V - v_bl_node__V
    applied_dv_sl_node__V = state.v_sl_node__V - v_sl_node__V
    torch.testing.assert_close(applied_dv_bl_node__V.abs().amax(), applied_dv_bl_node__V.new_tensor(0.05))
    assert (applied_dv_sl_node__V.abs() <= 0.05 + 1e-14).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("wire_r", [5e-7, 5e-6, 2e-4])
def test_internal_threshold_handles_wire_roundoff(device: torch.device, dtype: torch.dtype, wire_r: float) -> None:
    harness = build_solver_harness(device=device, dtype=dtype, col_num=2, row_num=16)
    kwargs = harness.solve_kwargs()
    kwargs.update(bl_segment_r__MOhm=wire_r, sl_segment_r__MOhm=wire_r)
    dcop, trace = _solve(kwargs, record_trace=True)
    assert trace is not None
    _assert_converged(trace)
    initial_node_trace = trace.select(0).node_trace
    assert initial_node_trace is not None
    assert _used_iterations(initial_node_trace.residual__uA).sum() - 1 < solver._NodeSolver.MAX_ITER
    oracle = replace(
        harness, bl_segment_r__MOhm=wire_r, sl_segment_r__MOhm=wire_r, v_wl_drive__V=harness.v_wl_drive__V.double()
    )
    v_bl, v_sl, i_bl, i_sl = _dense_kcl_solution(oracle)
    eps = torch.finfo(dtype).eps
    for actual, expected in ((dcop.v_bl_node__V[0], v_bl), (dcop.v_sl_node__V[0], v_sl)):
        torch.testing.assert_close(
            actual, expected, rtol=1e-5 if dtype == torch.float32 else 1e-10, atol=16 * eps, check_dtype=False
        )
    for actual, expected in ((dcop.i_bl_port__uA[0, 0], i_bl), (dcop.i_sl_port__uA[0, 0], i_sl)):
        torch.testing.assert_close(
            actual, expected, rtol=1e-4 if dtype == torch.float32 else 1e-10, atol=4 * eps / wire_r, check_dtype=False
        )


def test_strong_cell_does_not_hide_another_nodes_kcl_error(device: torch.device) -> None:
    kwargs = build_solver_harness(device=device, dtype=torch.float32, col_num=1, row_num=5).solve_kwargs()
    # A unit far-end current and a tiny first-row current have an analytic
    # ladder equilibrium. Powers of two keep the seed voltages representable.
    rows = torch.arange(1, 6, device=device, dtype=torch.float32).reshape(1, 5, 1)
    v_bl = 16.0 - rows - 2**-18
    v_sl = rows + 2**-18
    current = torch.zeros_like(rows)
    current[..., 0, :] = 2**-18
    current[..., -1, :] = 1.0
    kwargs.update(
        bl_segment_r__MOhm=1.0,
        sl_segment_r__MOhm=1.0,
        cell=_AliasingCell(),
        cell_snap=_AliasingCellInput(g_cell__uS=current / (v_bl - v_sl)),
    )
    node_solver = _solver(kwargs).node_solver
    v_bl[..., :1, :] += 2**-17
    v_bl_port = v_bl.new_full((1, 1, 1), 16.0)
    v_sl_port = torch.zeros_like(v_bl_port)
    state, trace = node_solver._evaluate_node(
        v_bl_port,
        v_sl_port,
        v_bl,
        v_sl,
        is_active=torch.ones_like(v_bl_port, dtype=torch.bool),
        cell_snap=kwargs["cell_snap"],
    )
    assert trace is not None
    cell_dcop = node_solver.cell.solve_dc(v_bl, v_sl, kwargs["cell_snap"])
    f_bl = f_kcl__uA(v_bl, v_bl_port, node_solver.bl_g__uS, cell_dcop.i__uA, dim=-2)
    f_sl = f_kcl__uA(v_sl, v_sl_port, node_solver.sl_g__uS, -cell_dcop.i__uA, dim=-2)
    torch.testing.assert_close(trace.residual__uA, torch.maximum(f_bl.abs(), f_sl.abs()))
    torch.testing.assert_close(state.is_active, (trace.residual__uA > trace.threshold__uA).any(dim=-2, keepdim=True))
    # Componentwise normalization catches the first-row violation even though
    # pooling residual and threshold maxima separately would accept the column.
    assert (trace.residual__uA / trace.threshold__uA).amax() > 1
    assert state.is_active.all()

    settled, history = node_solver.solve(
        v_bl,
        v_sl,
        v_bl_port,
        v_sl_port,
        cell_snap=kwargs["cell_snap"],
        record_trace=True,
        trace_mask=None,
        is_active=torch.ones_like(v_bl_port, dtype=torch.bool),
        final_fn=lambda state: state,
    )
    assert history is not None
    assert not settled.is_active.any()
    assert history is not None
    assert _used_iterations(history.residual__uA).sum() > 1
    assert (history.residual__uA[..., 0] / history.threshold__uA[..., 0]).amax() > 1


@pytest.mark.parametrize("field", ["di_dvbl__uS", "di_dvsl__uS"])
@pytest.mark.parametrize("bad_value", [torch.nan, torch.inf])
def test_nonfinite_derivative_with_finite_current_fails_fast(field: str, bad_value: float) -> None:
    kwargs = _kwargs(torch.device("cpu"), col_num=1, row_num=2)
    node_solver = _solver(kwargs).node_solver
    v_bl = torch.ones(1, 2, 1, dtype=torch.float64)
    v_sl = torch.zeros_like(v_bl)
    cell_dcop = kwargs["cell"].solve_dc(v_bl, v_sl, kwargs["cell_snap"])
    assert cell_dcop.i__uA.isfinite().all()
    bad_dcop = replace(cell_dcop, **{field: torch.full_like(v_bl, bad_value)})
    with (
        patch.object(kwargs["cell"], "solve_dc", return_value=bad_dcop),
        pytest.raises(RuntimeError, match="non-finite state"),
    ):
        node_solver._evaluate_node(
            v_bl[..., :1, :],
            v_sl[..., :1, :],
            v_bl,
            v_sl,
            is_active=torch.ones_like(v_bl[..., :1, :], dtype=torch.bool),
            cell_snap=kwargs["cell_snap"],
        )


@pytest.mark.parametrize("rail", ["bl", "sl"])
@pytest.mark.parametrize("bad_value", [torch.nan, torch.inf])
def test_nonfinite_node_voltage_fails_through_kcl(rail: str, bad_value: float) -> None:
    kwargs = _kwargs(torch.device("cpu"), col_num=1, row_num=2)
    node_solver = _solver(kwargs).node_solver
    v_bl = torch.ones(1, 2, 1, dtype=torch.float64)
    v_sl = torch.zeros_like(v_bl)
    (v_bl if rail == "bl" else v_sl)[..., -1, :] = bad_value
    # Keep branch outputs finite to isolate propagation through the wire KCL.
    dcop = _AliasingCellDcop(
        i__uA=torch.zeros_like(v_bl), di_dvbl__uS=torch.ones_like(v_bl), di_dvsl__uS=-torch.ones_like(v_bl)
    )
    with (
        patch.object(kwargs["cell"], "solve_dc", return_value=dcop),
        pytest.raises(RuntimeError, match="non-finite state"),
    ):
        node_solver._evaluate_node(
            v_bl[..., :1, :],
            v_sl[..., :1, :],
            v_bl,
            v_sl,
            is_active=torch.ones_like(v_bl[..., :1, :], dtype=torch.bool),
            cell_snap=kwargs["cell_snap"],
        )


def test_nonfinite_correction_is_rejected_before_step_clipping() -> None:
    kwargs = _kwargs(torch.device("cpu"), col_num=1, row_num=2)
    node_solver = _solver(kwargs).node_solver
    v_bl = torch.ones(1, 2, 1, dtype=torch.float64)
    v_sl = torch.zeros_like(v_bl)
    with (
        patch.object(
            solver,
            "solve_block_tridiagonal_2x2",
            return_value=(torch.full_like(v_bl, torch.inf), torch.full_like(v_sl, torch.inf)),
        ),
        pytest.raises(RuntimeError, match="non-finite state"),
    ):
        node_solver._evaluate_node(
            v_bl[..., :1, :],
            v_sl[..., :1, :],
            v_bl,
            v_sl,
            is_active=torch.ones_like(v_bl[..., :1, :], dtype=torch.bool),
            cell_snap=kwargs["cell_snap"],
        )


def test_nonfinite_branch_dcop_fails_fast(device: torch.device) -> None:
    """Finite residual arithmetic cannot conceal a non-finite branch DCOP."""
    kwargs = _kwargs(device)
    cell_snap = kwargs["cell_snap"]
    kwargs["cell_snap"] = replace(
        cell_snap,
        g_cell_on__uS=torch.full_like(cell_snap.g_cell_on__uS, torch.nan),
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        _solve(kwargs, record_trace=False)


def test_solver_entry_is_inference_only(device: torch.device) -> None:
    """Caller gradients never enter the structured higher-order solve."""
    kwargs = _kwargs(device, col_num=1, row_num=1)
    bl_snap = kwargs["bl_driver_snap"]
    kwargs["bl_driver_snap"] = replace(
        bl_snap,
        v_ref__V=bl_snap.v_ref__V.clone().requires_grad_(),
    )
    dcop, _ = _solve(kwargs, record_trace=False)

    assert not dcop.v_bl_node__V.requires_grad
    assert dcop.v_bl_node__V.grad_fn is None


def test_solver_graph_size_is_independent_of_iteration_caps(device: torch.device) -> None:
    kwargs = _kwargs(device, col_num=1, row_num=2)
    array_solver = _solver(kwargs)
    graph_sizes: list[int] = []

    def backend(graph: torch.fx.GraphModule, _inputs: list[Tensor]) -> object:
        graph_sizes.append(
            sum(len(module.graph.nodes) for module in graph.modules() if isinstance(module, torch.fx.GraphModule))
        )
        return graph.forward

    def solve() -> Tensor:
        dcop, _ = solve_dcop(
            array_solver,
            record_trace=False,
            cell_snap=kwargs["cell_snap"],
            bl_driver_snap=kwargs["bl_driver_snap"],
            sl_driver_snap=kwargs["sl_driver_snap"],
        )
        return dcop.cell_dcop.i__uA

    with torch._dynamo.config.patch(disable=False):
        compiled = torch.compile(solve, backend=backend, fullgraph=True, dynamic=False)
        for cap in (20, 32, 64):
            with (
                patch.object(solver._NodeSolver, "MAX_ITER", cap),
                patch.object(solver.ColBlColSlArraySolver, "MAX_ITER", cap),
            ):
                compiled()
    assert len(graph_sizes) == 3
    assert len(set(graph_sizes)) == 1


def test_column_trace_mask_preserves_row_grid_and_solution(device: torch.device) -> None:
    kwargs = _nonideal_kwargs(device)
    array_solver = _solver(kwargs)
    expected, _ = _solve(kwargs, record_trace=True)
    selected = torch.tensor([[[True, False, True]]], device=device)
    actual, trace = solve_dcop(
        array_solver,
        cell_snap=kwargs["cell_snap"],
        bl_driver_snap=kwargs["bl_driver_snap"],
        sl_driver_snap=kwargs["sl_driver_snap"],
        record_trace=True,
        trace_mask=selected,
    )
    torch.testing.assert_close(actual.v_bl_node__V, expected.v_bl_node__V)
    torch.testing.assert_close(actual.v_sl_node__V, expected.v_sl_node__V)
    assert trace is not None
    assert trace.node_trace is not None
    assert trace.residual__V[..., 1, :].isnan().all()
    assert trace.node_trace.residual__uA[..., 1, :, :].isnan().all()
    assert not trace.node_trace.limited[..., 1, :, :].any()
    assert trace.node_trace.residual__uA.shape[:3] == (1, 4, 3)
    assert trace.node_trace.residual__uA[..., 0, 0, 0].isfinite().all()
