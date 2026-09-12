"""Adaptive detailed-cell convergence, observation, and replacement laws."""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Any
from unittest.mock import patch

import pytest
import torch

from neurox.common.torch_compat import torch_while_loop
from neurox.primitive.device.mosfet import MosfetConfig, MosfetPolicy
from neurox.primitive.device.rram import RramConfig, RramPolicy
from neurox.primitive.nonideality import StateDependentGammaConfig, StuckAtFaultConfig, TelegraphConfig
from neurox.primitive.xbar.cell.x1t1r_detail import (
    XbarCell1t1rDetail,
    XbarCell1t1rDetailConfig,
    XbarCell1t1rDetailPolicy,
    XbarCell1t1rDetailSnap,
    XbarCell1t1rDetailTrace,
    _Solver,
    _State,
)
from neurox.primitive.xbar.solver.col_bl_col_sl import ColBlColSlArraySolver
from tests.utils.standalone_solver_fixture import build_solver_harness, solve_dcop

_CPU = torch.device("cpu")


def _rram_config() -> RramConfig:
    return RramConfig(
        g_min__uS=10.0,
        nonlinearity_alpha=0.5,
        drift_decay_rate=0.03,
        drift_t0=1.0,
        read_thermal__uS=0.002,
        prog_gamma=StateDependentGammaConfig(
            k_slope=0.0,
            k_intercept=100.0,
            theta=1.0,
            min_val=10.0,
            max_val=100.0,
        ),
        read_telegraph=TelegraphConfig(
            amplitude_mean=0.005,
            amplitude_std=0.001,
            p_high_state=0.01,
        ),
        stuck_at=StuckAtFaultConfig(p_at_min=0.001, p_at_max=0.001),
    )


def _mosfet_config() -> MosfetConfig:
    return MosfetConfig(
        T_nom__K=300.0,
        c_ox__fF_per_um2=31.4,
        mu0__cm2_per_V_s=200.0,
        ute=1.5,
        vth0__V=0.4,
        kt1__V=-0.002,
        n_factor=1.25,
        A_vt__mV_um=3.0,
        A_beta_relative__um=0.002,
    )


def _rram_policy() -> RramPolicy:
    return RramPolicy(
        prog_gamma=False,
        drift=False,
        stuck_at=False,
        read_telegraph=False,
        read_thermal=False,
    )


def _mosfet_policy() -> MosfetPolicy:
    return MosfetPolicy(A_vt_mismatch=False, A_beta_mismatch=False)


def _build_cell(
    *,
    inst_shape: tuple[int, ...] = (2, 2),
    dtype: torch.dtype = torch.float64,
    device: torch.device = _CPU,
) -> XbarCell1t1rDetail:
    config = XbarCell1t1rDetailConfig(
        rram_config=_rram_config(),
        nmos_config=_mosfet_config(),
        state_to_g_map__uS=(10.0, 100.0),
        access_nmos_W__um=0.1,
        access_nmos_L__um=0.05,
        rram_g_max__uS=100.0,
    )
    cell = XbarCell1t1rDetail(
        config=config,
        policy=XbarCell1t1rDetailPolicy(
            rram_policy=_rram_policy(),
            nmos_policy=_mosfet_policy(),
        ),
        inst_shape=inst_shape,
        dtype=dtype,
        T__K=300.0,
    )
    cell.to(device)
    cell.eval()
    cell.fabricate()
    program = torch.arange(cell.inst_count, dtype=torch.long, device=device).remainder(2).reshape(inst_shape)
    cell.program(program)
    return cell


def _snap(
    cell: XbarCell1t1rDetail,
    *,
    v_wl__V: float,
    shape: tuple[int, ...] = (2, 2),
) -> XbarCell1t1rDetailSnap:
    control = cell.rram.snapshot(shape=shape).g__uS.new_full(shape, v_wl__V)
    return cell.snapshot(control=control, shape=shape)


def _trace_iterations(trace: XbarCell1t1rDetailTrace) -> torch.Tensor:
    selected = ~trace.residual__uA.isnan()
    return selected.any(dim=tuple(range(selected.ndim - 1))).sum() - 1


def _terminal_residual_ratio(trace: XbarCell1t1rDetailTrace) -> torch.Tensor:
    selected = ~trace.residual__uA.isnan()
    terminal = (selected.sum(dim=-1) - 1).clamp_min(0).unsqueeze(-1)
    ratio = (trace.residual__uA / trace.threshold__uA).gather(-1, terminal).squeeze(-1)
    return ratio[selected.any(dim=-1)].amax()


@pytest.mark.parametrize(("v_wl__V", "v_bl__V", "v_sl__V"), [(0.0, 0.3, 0.0), (0.9, 0.3, 0.0), (0.9, 0.0, 0.3)])
def test_adaptive_cell_converges_for_wl_and_bias_cases(v_wl__V: float, v_bl__V: float, v_sl__V: float) -> None:
    cell = _build_cell()
    dcop, trace = cell.solve_dc_trace(
        torch.full((2, 2), v_bl__V, dtype=torch.float64),
        torch.full((2, 2), v_sl__V, dtype=torch.float64),
        _snap(cell, v_wl__V=v_wl__V),
    )

    assert _terminal_residual_ratio(trace) <= 1
    assert bool((dcop.di_dvbl__uS >= 0.0).all())
    assert bool((dcop.di_dvsl__uS <= 0.0).all())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1, 1), (3, 4, 17)], ids=["one-cell", "leading-grid"])
def test_adaptive_cell_supports_dtypes_and_leading_grids(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    inst_shape = shape[-2:]
    cell = _build_cell(inst_shape=inst_shape, dtype=dtype)
    dcop, trace = cell.solve_dc_trace(
        torch.full(shape, 0.3, dtype=dtype),
        torch.zeros(shape, dtype=dtype),
        _snap(cell, v_wl__V=0.9, shape=shape),
    )

    assert dcop.i__uA.dtype is dtype
    assert _terminal_residual_ratio(trace) <= 1
    assert trace.residual__uA.shape == (*shape, _Solver.MAX_ITER)
    assert (~trace.residual__uA[..., 0].isnan()).sum() == math.prod(shape)


def test_solve_dc_never_allocates_a_trajectory() -> None:
    cell = _build_cell()
    snap = _snap(cell, v_wl__V=0.9)
    v_bl__V = torch.full((2, 2), 0.3, dtype=torch.float64)
    v_sl__V = torch.zeros((2, 2), dtype=torch.float64)

    with patch(
        "neurox.primitive.xbar.cell.x1t1r_detail.run_solving_trace_scan",
        side_effect=AssertionError("trace allocation reached"),
    ):
        cell.solve_dc(v_bl__V, v_sl__V, snap)


@pytest.mark.parametrize("record_trace", [False, True], ids=["plain", "trace"])
def test_solver_returns_the_caller_projection(record_trace: bool) -> None:
    cell = _build_cell()
    snap = _snap(cell, v_wl__V=0.9)
    v_bl = torch.full((2, 2), 0.3, dtype=torch.float64)
    v_sl = torch.zeros_like(v_bl)
    reference = cell.solve_dc(v_bl, v_sl, snap).v_x__V.sum(dim=-1)
    terminal_states: list[_State] = []

    def final_fn(state: _State) -> torch.Tensor:
        terminal_states.append(state)
        return state.v_x__V.sum(dim=-1)

    result, trace = cell.solver.solve(
        v_bl,
        v_sl,
        v_wl__V=snap.v_wl__V,
        nmos_snap=snap.nmos_snap,
        rram_snap=snap.rram_snap,
        final_fn=final_fn,
        record_trace=record_trace,
        trace_mask=None,
    )
    if record_trace:
        assert trace is not None
        assert _terminal_residual_ratio(trace) <= 1
    else:
        assert trace is None

    assert len(terminal_states) == 1
    assert not terminal_states[0].is_active.any()
    torch.testing.assert_close(result, reference)


def test_trace_keeps_the_raw_population_with_fixed_capacity() -> None:
    cell = _build_cell()
    _dcop, trace = cell.solve_dc_trace(
        torch.full((2, 2), 0.3, dtype=torch.float64),
        torch.zeros((2, 2), dtype=torch.float64),
        _snap(cell, v_wl__V=0.9),
    )

    assert trace.residual__uA.shape == (2, 2, _Solver.MAX_ITER)
    assert (~trace.residual__uA[..., 0].isnan()).sum() == 4
    unused = torch.arange(_Solver.MAX_ITER) > _trace_iterations(trace)
    for value in (trace.residual__uA, trace.threshold__uA, trace.dv_x_abs__V):
        assert value.shape == trace.residual__uA.shape
        assert value[..., unused].isnan().all()
    assert _terminal_residual_ratio(trace) <= 1


def test_trace_selection_does_not_change_the_solution() -> None:
    cell = _build_cell()
    snap = _snap(cell, v_wl__V=0.9)
    v_bl__V = torch.full((2, 2), 0.3, dtype=torch.float64)
    v_sl__V = torch.zeros((2, 2), dtype=torch.float64)
    trace_mask = torch.tensor([[True, False], [False, True]])

    plain_dcop = cell.solve_dc(v_bl__V, v_sl__V, snap)
    traced_dcop, trace = cell.solve_dc_trace(v_bl__V, v_sl__V, snap, trace_mask=trace_mask)
    valid = ~trace.residual__uA.isnan()
    assert not valid[~trace_mask].any()
    assert valid[trace_mask, 0].all()
    assert not ((~valid[..., :-1]) & valid[..., 1:]).any()
    terminal = (valid.sum(dim=-1) - 1).clamp_min(0).unsqueeze(-1)
    terminal_residual = trace.residual__uA.gather(-1, terminal).squeeze(-1)
    terminal_threshold = trace.threshold__uA.gather(-1, terminal).squeeze(-1)
    assert (terminal_residual[trace_mask] <= terminal_threshold[trace_mask]).all()
    assert torch.equal(trace.threshold__uA.isnan(), ~valid)
    assert torch.equal(trace.dv_x_abs__V.isnan(), ~valid)

    torch.testing.assert_close(plain_dcop.i__uA, traced_dcop.i__uA)
    torch.testing.assert_close(plain_dcop.v_x__V, traced_dcop.v_x__V)
    assert valid[..., 0].sum() == trace_mask.sum()
    assert (valid.sum(dim=(0, 1)) <= trace_mask.sum()).all()


def test_access_node_while_carries_only_state_and_counter() -> None:
    cell = _build_cell()
    captured = []

    def inspect_carry(cond_fn, body_fn, carried_state):
        step, state = carried_state
        assert isinstance(step, torch.Tensor)
        assert step.shape == ()
        assert step.dtype == torch.int64
        assert isinstance(state, _State)
        assert {field.name for field in fields(state)} == {"is_active", "v_x__V"}
        captured.append(carried_state)
        return torch_while_loop(cond_fn, body_fn, carried_state)

    with patch("neurox.common.loop.torch_while_loop", side_effect=inspect_carry):
        cell.solve_dc(
            torch.full((2, 2), 0.3, dtype=torch.float64),
            torch.zeros((2, 2), dtype=torch.float64),
            _snap(cell, v_wl__V=0.9),
        )
    assert len(captured) == 1


@pytest.mark.parametrize("record_trace", [False, True], ids=["trace-off", "trace-on"])
@pytest.mark.parametrize("bad_v_bl__V", [torch.nan, torch.inf], ids=["nan", "inf"])
def test_nonfinite_newton_state_aborts_the_cell_solve(bad_v_bl__V: float, record_trace: bool) -> None:
    cell = _build_cell()
    v_bl__V = torch.full((2, 2), 0.3, dtype=torch.float64)
    v_bl__V[0, 0] = bad_v_bl__V
    args = (v_bl__V, torch.zeros((2, 2), dtype=torch.float64), _snap(cell, v_wl__V=0.9))
    solve = cell.solve_dc_trace if record_trace else cell.solve_dc

    with pytest.raises(RuntimeError, match="Newton solve produced a non-finite state"):
        solve(*args)


def test_iteration_cap_aborts_the_regular_cell_solve() -> None:
    cell = _build_cell()
    v_bl__V = torch.full((2, 2), 0.3, dtype=torch.float64)
    v_sl__V = torch.zeros((2, 2), dtype=torch.float64)
    snap = _snap(cell, v_wl__V=0.9)

    with (
        patch.object(_Solver, "MAX_ITER", 1),
        patch.object(cell, "_dcop_from_state") as final_fn,
        pytest.raises(RuntimeError, match="did not converge"),
    ):
        cell.solve_dc(v_bl__V, v_sl__V, snap)
    final_fn.assert_not_called()


def test_iteration_cap_returns_the_requested_cell_trace() -> None:
    cell = _build_cell()
    v_bl__V = torch.full((2, 2), 0.3, dtype=torch.float64)
    v_sl__V = torch.zeros((2, 2), dtype=torch.float64)
    snap = _snap(cell, v_wl__V=0.9)

    with patch.object(_Solver, "MAX_ITER", 1):
        dcop, trace = cell.solve_dc_trace(v_bl__V, v_sl__V, snap)

    assert (trace.residual__uA[..., 0] > trace.threshold__uA[..., 0]).any()
    assert dcop.i__uA.isfinite().all()
    assert dcop.v_x__V.isfinite().all()


def test_public_solve_is_inference_only() -> None:
    cell = _build_cell()
    v_bl__V = torch.full((2, 2), 0.3, dtype=torch.float64, requires_grad=True)
    dcop, trace = cell.solve_dc_trace(
        v_bl__V,
        torch.zeros((2, 2), dtype=torch.float64, requires_grad=True),
        _snap(cell, v_wl__V=0.9),
    )

    assert not dcop.i__uA.requires_grad
    assert not trace.residual__uA.requires_grad


@pytest.mark.parametrize("record_trace", [False, True], ids=["trace-off", "trace-on"])
def test_caller_compilation_uses_the_same_numerical_method(record_trace: bool, device: torch.device) -> None:
    cell = _build_cell(device=device)
    snap = _snap(cell, v_wl__V=0.9)
    v_bl__V = torch.full((2, 2), 0.3, dtype=torch.float64, device=device)
    v_sl__V = torch.zeros((2, 2), dtype=torch.float64, device=device)

    def solve(v_bl: torch.Tensor, v_sl: torch.Tensor) -> torch.Tensor:
        if record_trace:
            dcop, _trace = cell.solve_dc_trace(v_bl, v_sl, snap)
        else:
            dcop = cell.solve_dc(v_bl, v_sl, snap)
        return dcop.i__uA

    expected = solve(v_bl__V, v_sl__V)
    with torch._dynamo.config.patch(disable=False):
        actual = torch.compile(solve, backend="inductor", fullgraph=True)(v_bl__V, v_sl__V)
    torch.testing.assert_close(actual, expected, rtol=1.0e-7, atol=1.0e-9)


def _detail_array_solve_kwargs(
    cell: XbarCell1t1rDetail,
    *,
    device: torch.device,
) -> dict[str, Any]:
    harness = build_solver_harness(
        device=device,
        col_num=2,
        row_num=2,
    )
    shape = (1, 2, 2)
    cell_snap = cell.snapshot(
        control=harness.v_wl_drive__V.unsqueeze(-2).expand(shape),
        shape=shape,
    )
    kwargs = harness.solve_kwargs()
    kwargs.update(
        cell=cell,
        cell_snap=cell_snap,
    )
    return kwargs


def _array_solve_with_detail_cell(
    cell: XbarCell1t1rDetail,
    *,
    record_trace: bool,
    device: torch.device,
) -> object:
    kwargs = _detail_array_solve_kwargs(cell, device=device)
    array_solver = ColBlColSlArraySolver(
        bl_segment_r__MOhm=kwargs["bl_segment_r__MOhm"],
        sl_segment_r__MOhm=kwargs["sl_segment_r__MOhm"],
        cell=cell,
        bl_driver=kwargs["bl_driver"],
        sl_driver=kwargs["sl_driver"],
        dtype=kwargs["bl_driver_snap"].v_ref__V.dtype,
    )
    return solve_dcop(
        array_solver,
        cell_snap=kwargs["cell_snap"],
        bl_driver_snap=kwargs["bl_driver_snap"],
        sl_driver_snap=kwargs["sl_driver_snap"],
        record_trace=record_trace,
    )


@pytest.mark.parametrize("record_trace", [False, True], ids=["trace-off", "trace-on"])
def test_detail_array_solver_is_public_caller_fullgraph_safe(record_trace: bool, device: torch.device) -> None:
    cell = _build_cell(inst_shape=(2, 2), device=device)
    kwargs = _detail_array_solve_kwargs(cell, device=device)
    array_solver = ColBlColSlArraySolver(
        bl_segment_r__MOhm=kwargs["bl_segment_r__MOhm"],
        sl_segment_r__MOhm=kwargs["sl_segment_r__MOhm"],
        cell=cell,
        bl_driver=kwargs["bl_driver"],
        sl_driver=kwargs["sl_driver"],
        dtype=kwargs["bl_driver_snap"].v_ref__V.dtype,
    )

    def solve() -> torch.Tensor:
        dcop, _trace = solve_dcop(
            array_solver,
            cell_snap=kwargs["cell_snap"],
            bl_driver_snap=kwargs["bl_driver_snap"],
            sl_driver_snap=kwargs["sl_driver_snap"],
            record_trace=record_trace,
        )
        return dcop.cell_dcop.i__uA

    expected = solve()
    actual = torch.compile(solve, backend="eager", fullgraph=True)()
    torch.testing.assert_close(actual, expected)


def test_array_solver_uses_plain_cell_interface_without_requesting_cell_trace(device: torch.device) -> None:
    cell = _build_cell(inst_shape=(2, 2), device=device)
    with patch(
        "neurox.primitive.xbar.cell.x1t1r_detail.run_solving_trace_scan",
        side_effect=AssertionError("cell trace allocation reached"),
    ):
        plain = _array_solve_with_detail_cell(cell, record_trace=False, device=device)
        traced = _array_solve_with_detail_cell(cell, record_trace=True, device=device)

    assert plain[1] is None
    assert traced[1] is not None
    torch.testing.assert_close(plain[0].cell_dcop.i__uA, traced[0].cell_dcop.i__uA)


def test_evaluate_returns_raw_observations_for_external_masking(device: torch.device) -> None:
    cell = _build_cell(device=device)
    snap = _snap(cell, v_wl__V=0.9)
    evaluator = _Solver(rram=cell.rram, nmos=cell.nmos, dtype=torch.float64)
    v_bl = torch.full((2, 2), 0.3, dtype=torch.float64, device=device)
    v_sl = torch.zeros_like(v_bl)
    v_x = torch.full_like(v_bl, 0.1)
    active = torch.tensor([[True, False], [True, True]], device=device)
    selected = torch.tensor([[True, True], [False, True]], device=device)
    state, trace = evaluator._evaluate_vx(
        v_bl,
        v_sl,
        v_x,
        v_wl__V=snap.v_wl__V,
        nmos_snap=snap.nmos_snap,
        rram_snap=snap.rram_snap,
        is_active=active,
    )
    nmos = cell.nmos.solve_dc(snap.v_wl__V, v_x, v_sl, snap.nmos_snap)
    rram = cell.rram.solve_dc(v_bl - v_x, snap.rram_snap)
    expected = torch.where(active & selected, (nmos.ids__uA - rram.i__uA).abs(), torch.nan)
    assert trace is not None
    trace = trace.mask_invalid(active & selected)
    torch.testing.assert_close(trace.residual__uA, expected, equal_nan=True)
    assert torch.equal(state.is_active & selected, trace.residual__uA > trace.threshold__uA)
