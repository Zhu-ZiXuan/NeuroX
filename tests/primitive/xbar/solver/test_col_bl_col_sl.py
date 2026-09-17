"""Focused tests for the adaptive parallel-rail solver."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, NamedTuple
from unittest.mock import patch

import pytest
import torch
from torch import Tensor

from neurox.common.module import DcopBase
from neurox.primitive.analog import VoltageDriver, VoltageDriverConfig, VoltageDriverPolicy, VoltageDriverSnap
from neurox.primitive.xbar.cell import (
    XbarCell1t1rLinear,
    XbarCell1t1rLinearConfig,
    XbarCell1t1rLinearPolicy,
    XbarCell1t1rLinearSnap,
)
from neurox.primitive.xbar.solver import col_bl_col_sl as solver
from neurox.primitive.xbar.solver._wire_kcl import f_kcl__uA
from neurox.primitive.xbar.solver.resistive_cell import ResistiveCellDcop


class _SolverDcop(DcopBase):
    """Caller-owned full electrical result for standalone solver assertions."""

    cell_dcop: ResistiveCellDcop
    i_bl_port__uA: Tensor
    i_sl_port__uA: Tensor
    v_bl_node__V: Tensor
    v_sl_node__V: Tensor
    v_bl_port__V: Tensor
    v_sl_port__V: Tensor


# --- Hand-written harness constants (arbitrary small witnesses) ---

COL_NUM = 3
ROW_NUM = 4
X_BATCH = 1

# Two weight states: WL-off leakage ~5 uS, WL-on chord ~50/100 uS.
G_CELL_OFF_TABLE__uS = (4.0, 5.0)
G_CELL_ON_TABLE__uS = (50.0, 100.0)
VX_RATIO_OFF_TABLE = (0.5, 0.5)
VX_RATIO_ON_TABLE = (0.4, 0.6)
V_WL_ON_THRESHOLD__V = 0.5

# Resistance of one rail link [MOhm] — the lattice is uniform, the clamp
# driver's own link to node 0 included. BL and SL values differ so a rail
# swap cannot cancel; both are large enough against the cell chords above
# that the IR drop along the ladder stays plainly visible.
BL_SEGMENT_R__MOhm = 2e-4
SL_SEGMENT_R__MOhm = 4e-4

# Fixed clamp boundary voltages [V].
BL_V_REF__V = 0.3
SL_V_REF__V = 0.1


def _linear_cell_config() -> XbarCell1t1rLinearConfig:
    return XbarCell1t1rLinearConfig(
        g_cell_off_table__uS=G_CELL_OFF_TABLE__uS,
        g_cell_on_table__uS=G_CELL_ON_TABLE__uS,
        vx_ratio_off_table=VX_RATIO_OFF_TABLE,
        vx_ratio_on_table=VX_RATIO_ON_TABLE,
        v_wl_on_threshold__V=V_WL_ON_THRESHOLD__V,
    )


def _ideal_driver_config() -> VoltageDriverConfig:
    return VoltageDriverConfig(
        r_out__MOhm=0.0,
        offset_sigma__V=0.0,
        thermal_sigma__V=0.0,
        energy_per_op__fJ=0.0,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
    )


@dataclass(frozen=True)
class _SolverHarness:
    """All inputs required to construct and call `solver.ColBlColSlArraySolver` directly.

    Also carries the dense-oracle inputs: the hand-written linear cell
    config, the programmed state-index grid, and the resolved rail
    reference taps (exact clamp targets, since both drivers are ideal).
    """

    cell: XbarCell1t1rLinear
    cell_config: XbarCell1t1rLinearConfig
    w_state_idx: Tensor
    bl_driver: VoltageDriver
    sl_driver: VoltageDriver
    bl_driver_snap: VoltageDriverSnap
    sl_driver_snap: VoltageDriverSnap
    bl_segment_r__MOhm: float
    sl_segment_r__MOhm: float
    v_wl_drive__V: Tensor
    bl_v_ref__V: Tensor
    sl_v_ref__V: Tensor

    def cell_snapshot(self) -> XbarCell1t1rLinearSnap:
        """Build the per-call cell snap at the harness WL drive.

        The cell takes its own gate voltage per cell, so the per-row drive is
        expanded onto the cell grid exactly as an owning array does.
        """
        shape = (*self.v_wl_drive__V.shape[:-1], *self.cell.inst_shape)
        return self.cell.snapshot(
            control=self.v_wl_drive__V.unsqueeze(-1).expand(shape),
            shape=shape,
        )

    def solve_kwargs(self) -> dict[str, Any]:
        """Return construction arguments and per-call snapshots for the solver."""
        return {
            "bl_segment_r__MOhm": self.bl_segment_r__MOhm,
            "sl_segment_r__MOhm": self.sl_segment_r__MOhm,
            "cell": self.cell,
            "cell_snap": self.cell_snapshot(),
            "bl_driver": self.bl_driver,
            "bl_driver_snap": self.bl_driver_snap,
            "sl_driver": self.sl_driver,
            "sl_driver_snap": self.sl_driver_snap,
        }


def _build_solver_harness(
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    v_wl_drive__V: float = 0.9,
    col_num: int = COL_NUM,
    row_num: int = ROW_NUM,
) -> _SolverHarness:
    """Build a linear network with alternating cell states and ideal rail clamps."""
    cell_config = _linear_cell_config()
    grid_shape = (row_num, col_num)

    # --- Cell + ideal boundary drivers (all policies empty / all-off) ---

    cell = XbarCell1t1rLinear(
        config=cell_config,
        policy=XbarCell1t1rLinearPolicy(),
        inst_shape=grid_shape,
        dtype=dtype,
    )
    driver_config = _ideal_driver_config()
    driver_policy = VoltageDriverPolicy(offset=False, thermal=False)
    bl_driver = VoltageDriver(
        config=driver_config,
        policy=driver_policy,
        inst_shape=(1, col_num),
        dtype=dtype,
    )
    sl_driver = VoltageDriver(
        config=driver_config,
        policy=driver_policy,
        inst_shape=(1, col_num),
        dtype=dtype,
    )

    for m in (cell, bl_driver, sl_driver):
        m.to(device)
        m.eval()
        m.fabricate()

    # --- Cell programming: alternate the two table states over the grid ---

    w_state_idx = (torch.arange(col_num * row_num, device=device) % 2).reshape(grid_shape)
    cell.program(w_state_idx)

    # --- v_wl_drive — uniform per-row WL control for the cell snap ---

    v_wl_drive = torch.full((X_BATCH, row_num), v_wl_drive__V, device=device, dtype=dtype)

    # --- Fixed boundary voltages and driver snapshots ---

    bl_ref_full = torch.full((X_BATCH, 1, col_num), BL_V_REF__V, device=device, dtype=dtype)
    sl_ref_full = torch.full((X_BATCH, 1, col_num), SL_V_REF__V, device=device, dtype=dtype)
    bl_drv_snap = bl_driver.snapshot(v_ref__V=bl_ref_full, shape=bl_ref_full.shape)
    sl_drv_snap = sl_driver.snapshot(v_ref__V=sl_ref_full, shape=sl_ref_full.shape)

    # --- Scalar rail-reference taps for the dense oracle ---

    # The tap is uniform over the clamp bank: the dense oracle takes it
    # as one scalar Dirichlet boundary value.
    # Shape: [batch, col] -> []
    bl_v_ref = bl_ref_full[0, 0, 0]
    sl_v_ref = sl_ref_full[0, 0, 0]

    return _SolverHarness(
        cell=cell,
        cell_config=cell_config,
        w_state_idx=w_state_idx,
        bl_driver=bl_driver,
        sl_driver=sl_driver,
        bl_driver_snap=bl_drv_snap,
        sl_driver_snap=sl_drv_snap,
        bl_segment_r__MOhm=BL_SEGMENT_R__MOhm,
        sl_segment_r__MOhm=SL_SEGMENT_R__MOhm,
        v_wl_drive__V=v_wl_drive,
        bl_v_ref__V=bl_v_ref,
        sl_v_ref__V=sl_v_ref,
    )


def _solve_dcop(
    array_solver: solver.ColBlColSlArraySolver[Any, Any, Any, Any],
    *,
    cell_snap: Any,
    bl_driver_snap: Any,
    sl_driver_snap: Any,
    record_trace: bool,
    trace_mask: Tensor | None = None,
) -> tuple[_SolverDcop, solver.ColBlColSlArrayTrace | None]:
    """Request a full terminal DCOP for electrical solver assertions."""

    def final_fn(state: solver.ColBlColSlArrayState) -> _SolverDcop:
        return _SolverDcop(
            cell_dcop=array_solver.cell.solve_dc(state.v_bl_node__V, state.v_sl_node__V, cell_snap),
            i_bl_port__uA=(state.v_bl_port__V - state.v_bl_node__V.narrow(array_solver.row_dim, 0, 1))
            * array_solver.bl_g__uS,
            i_sl_port__uA=(state.v_sl_port__V - state.v_sl_node__V.narrow(array_solver.row_dim, 0, 1))
            * array_solver.sl_g__uS,
            v_bl_node__V=state.v_bl_node__V,
            v_sl_node__V=state.v_sl_node__V,
            v_bl_port__V=state.v_bl_port__V,
            v_sl_port__V=state.v_sl_port__V,
        )

    return array_solver.solve(
        cell_snap=cell_snap,
        bl_driver_snap=bl_driver_snap,
        sl_driver_snap=sl_driver_snap,
        final_fn=final_fn,
        record_trace=record_trace,
        trace_mask=trace_mask,
    )


def _kwargs(device: torch.device, *, col_num: int = 3, row_num: int = 4) -> dict[str, Any]:
    return _build_solver_harness(device=device, col_num=col_num, row_num=row_num).solve_kwargs()


def _solver(kwargs: dict[str, Any]) -> solver.ColBlColSlArraySolver[Any, Any, Any, Any]:
    return solver.ColBlColSlArraySolver(
        bl_segment_r__MOhm=kwargs["bl_segment_r__MOhm"],
        sl_segment_r__MOhm=kwargs["sl_segment_r__MOhm"],
        cell=kwargs["cell"],
        bl_driver=kwargs["bl_driver"],
        sl_driver=kwargs["sl_driver"],
        dtype=kwargs["bl_driver_snap"].v_open__V.dtype,
    )


def _solve(
    kwargs: dict[str, Any],
    *,
    record_trace: bool,
) -> tuple[_SolverDcop, solver.ColBlColSlArrayTrace | None]:
    return _solve_dcop(
        _solver(kwargs),
        cell_snap=kwargs["cell_snap"],
        bl_driver_snap=kwargs["bl_driver_snap"],
        sl_driver_snap=kwargs["sl_driver_snap"],
        record_trace=record_trace,
    )


def _used_iterations(residual: Tensor, *, history_ndim: int = 1) -> Tensor:
    return (~residual.isnan()).any(dim=tuple(range(residual.ndim - history_ndim)))


@pytest.mark.parametrize("record_trace", [False, True])
def test_settled_nodes_still_require_port_updates(device: torch.device, record_trace: bool) -> None:
    kwargs = _nonideal_kwargs(device)
    port_solver = _solver(kwargs)
    bl = kwargs["bl_driver_snap"].v_open__V
    sl = kwargs["sl_driver_snap"].v_open__V
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
    assert port_state.is_active.all()
    assert (port_state.v_bl_port__V < bl).all()
    assert (port_state.v_sl_port__V > sl).all()
    assert is_active.all()
    if node_trace is not None:
        assert port_trace.node_trace is not None
        torch.testing.assert_close(port_trace.node_trace.residual__uA, node_trace.residual__uA, equal_nan=True)
    else:
        assert port_trace.node_trace is None
    if record_trace:
        assert node_trace is not None
    else:
        assert node_trace is None


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
    harness: _SolverHarness, *, bl_r: float = 0.0, sl_r: float = 0.0
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
        reference=old_bl_snap.v_open__V,
        inst_shape=old_bl.inst_shape,
        r_out__MOhm=2e-3,
    )
    sl_driver, sl_snap = _nonideal_driver(
        reference=old_sl_snap.v_open__V,
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
    harness = _build_solver_harness(device=device)
    kwargs = _nonideal_kwargs(device) if nonideal else harness.solve_kwargs()
    bl_r = 2e-3 if nonideal else 0.0
    sl_r = 3e-3 if nonideal else 0.0
    dcop, trace = _solve(kwargs, record_trace=True)
    expected = _dense_kcl_solution(harness, bl_r=bl_r, sl_r=sl_r)
    actual = dcop.v_bl_node__V[0], dcop.v_sl_node__V[0], dcop.i_bl_port__uA[0, 0], dcop.i_sl_port__uA[0, 0]
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=0.0, atol=1e-9)
    assert trace is not None


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
        v_ref__V=kwargs["bl_driver_snap"].v_open__V.expand(port_shape),
        shape=port_shape,
    )
    kwargs["sl_driver_snap"] = kwargs["sl_driver"].snapshot(
        v_ref__V=kwargs["sl_driver_snap"].v_open__V.expand(port_shape),
        shape=port_shape,
    )
    _, trace = _solve(kwargs, record_trace=True)

    assert trace is not None
    initial_node_trace = trace.select(0).node_trace
    assert initial_node_trace is not None
    assert (~initial_node_trace.residual__uA[..., 0].isnan()).any(dim=-2).sum() == 4


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
    assert torch.equal(
        _used_iterations(node_trace.residual__uA, history_ndim=2).any(dim=0),
        _used_iterations(trace.residual__V),
    )
    port_valid = ~trace.residual__V.isnan()
    assert not trace.limited[~port_valid].any()
    for value in (trace.threshold__V, trace.dv_bl_port_abs__V, trace.dv_sl_port_abs__V):
        assert torch.equal(value.isnan(), ~port_valid)
    node_valid = ~node_trace.residual__uA.isnan()
    assert not node_trace.limited[~node_valid.any(dim=-4, keepdim=True)].any()
    for value in (node_trace.threshold__uA, node_trace.dv_bl_node_abs__V, node_trace.dv_sl_node_abs__V):
        assert torch.equal(value.isnan(), ~node_valid)


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
        _dcop, trace = _solve(kwargs, record_trace=True)

    assert trace is not None
    assert trace.node_trace is not None
    assert (trace.node_trace.residual__uA[..., 0, 0] > trace.node_trace.threshold__uA[..., 0, 0]).any()
    # A capped node solve still supplies voltages for the port residual check.
    port_used = _used_iterations(trace.residual__V)
    assert port_used[0]
    valid = ~trace.residual__V.isnan()
    assert (trace.residual__V / trace.threshold__V)[valid].isfinite().all()


def test_wire_newton_correction_obeys_step_limit(device: torch.device) -> None:
    kwargs = _kwargs(device, row_num=16)
    kwargs["bl_segment_r__MOhm"] = 2e-3
    kwargs["sl_segment_r__MOhm"] = 4e-3
    node_solver = _solver(kwargs).node_solver
    v_bl_port__V = kwargs["bl_driver_snap"].v_open__V
    v_sl_port__V = kwargs["sl_driver_snap"].v_open__V
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


def test_strong_cell_does_not_hide_another_nodes_kcl_error(device: torch.device) -> None:
    kwargs = _build_solver_harness(device=device, dtype=torch.float32, col_num=1, row_num=5).solve_kwargs()
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


def test_column_trace_mask_preserves_row_grid_and_solution(device: torch.device) -> None:
    kwargs = _nonideal_kwargs(device)
    array_solver = _solver(kwargs)
    expected, _ = _solve(kwargs, record_trace=True)
    selected = torch.tensor([[[True, False, True]]], device=device)
    actual, trace = _solve_dcop(
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
