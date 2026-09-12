"""Capacitive accounting of one held-boundary 1T1R array operation."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor

from neurox import Profiler
from neurox.primitive.analog import VoltageDriver, VoltageDriverConfig, VoltageDriverPolicy, VoltageDriverSnap
from neurox.primitive.xbar.array import XbarArray1t1r, XbarArray1t1rConfig, XbarArray1t1rPolicy
from neurox.primitive.xbar.cell import XbarCell1t1rLinearConfig, XbarCell1t1rLinearPolicy

_DTYPE = torch.float64
_COL_NUM = 3
_ROW_NUM = 2
_VDD__V = 0.7
_BL_V_REF__V = 0.30
_SL_V_REF__V = 0.05

_BL_NODE_C__fF = 0.31
_X_NODE_C__fF = 0.37
_SL_NODE_C__fF = 0.41
_WL_NODE_C__fF = 0.43

_VX_RATIO_ON_TABLE = (0.25, 0.75)
_VX_RATIO_OFF_TABLE = (0.10, 0.40)
_VX_RATIO_OFF_AT_REST = (0.0, 0.0)
_V_WL_ON_THRESHOLD__V = 0.5

type _Array = XbarArray1t1r[VoltageDriverSnap, VoltageDriverSnap]


def _states() -> Tensor:
    return torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.long)


def _phases() -> Tensor:
    return torch.tensor([[0.90, 0.10], [0.10, 0.90]], dtype=_DTYPE)


def _cell_config(
    *,
    g_cell_on__uS: float,
    vx_ratio_off_table: tuple[float, ...],
) -> XbarCell1t1rLinearConfig:
    return XbarCell1t1rLinearConfig(
        g_cell_off_table__uS=(0.0, 0.0),
        g_cell_on_table__uS=(g_cell_on__uS, g_cell_on__uS),
        vx_ratio_on_table=_VX_RATIO_ON_TABLE,
        vx_ratio_off_table=vx_ratio_off_table,
        v_wl_on_threshold__V=_V_WL_ON_THRESHOLD__V,
    )


def _build_array(
    *,
    g_cell_on__uS: float = 0.0,
    bl_node_c__fF: float = _BL_NODE_C__fF,
    sl_node_c__fF: float = _SL_NODE_C__fF,
    vx_ratio_off_table: tuple[float, ...] = _VX_RATIO_OFF_TABLE,
) -> _Array:
    array = XbarArray1t1r(
        bl_driver=_ideal_driver(),
        sl_driver=_ideal_driver(),
        config=XbarArray1t1rConfig(
            row_cell_space__um=1.0,
            col_cell_space__um=1.0,
            bl_segment_r__MOhm=1.0e-3,
            sl_segment_r__MOhm=2.0e-3,
            bl_node_c__fF=bl_node_c__fF,
            x_node_c__fF=_X_NODE_C__fF,
            sl_node_c__fF=sl_node_c__fF,
            wl_node_c__fF=_WL_NODE_C__fF,
            cell_config=_cell_config(
                g_cell_on__uS=g_cell_on__uS,
                vx_ratio_off_table=vx_ratio_off_table,
            ),
        ),
        policy=XbarArray1t1rPolicy(
            cell_policy=XbarCell1t1rLinearPolicy(),
            solve_chunk_size=0,
        ),
        inst_shape=(),
        row_num=_ROW_NUM,
        col_num=_COL_NUM,
        vdd__V=_VDD__V,
        dtype=_DTYPE,
        T__K=300.0,
    )
    array.eval()
    array.fabricate()
    array.program(_states())
    return array


def _ideal_driver() -> VoltageDriver:
    driver = VoltageDriver(
        config=VoltageDriverConfig(
            r_out__MOhm=0.0,
            offset_sigma__V=0.0,
            thermal_sigma__V=0.0,
            energy_per_op__fJ=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        policy=VoltageDriverPolicy(offset=False, thermal=False),
        inst_shape=(1, _COL_NUM),
        dtype=_DTYPE,
        T__K=300.0,
    )
    driver.eval()
    driver.fabricate()
    return driver


def _solve(
    array: _Array,
    v_wl__V: Tensor,
    monkeypatch: pytest.MonkeyPatch,
    *,
    wl_phase_dims: tuple[int, ...],
    bl_ref__V: float = _BL_V_REF__V,
    sl_ref__V: float = _SL_V_REF__V,
) -> tuple[Tensor, Tensor]:
    leading_shape = tuple(v_wl__V.shape[:-1])
    bl_driver = array.solver.bl_driver
    sl_driver = array.solver.sl_driver
    bl_ref__V = torch.full((*leading_shape, 1, _COL_NUM), bl_ref__V, dtype=_DTYPE)
    sl_ref__V = torch.full((*leading_shape, 1, _COL_NUM), sl_ref__V, dtype=_DTYPE)
    billed: list[Tensor] = []
    monkeypatch.setattr(array, "_record_dynamic_energy", billed.append)
    with Profiler(), torch.no_grad():
        dcop = array.solve_dc(
            v_wl__V=v_wl__V.unsqueeze(array.col_dim),
            wl_phase_dims=wl_phase_dims,
            bl_driver_snap=bl_driver.snapshot(v_ref__V=bl_ref__V, shape=bl_ref__V.shape),
            sl_driver_snap=sl_driver.snapshot(v_ref__V=sl_ref__V, shape=sl_ref__V.shape),
        )
    [energy__fJ] = billed
    return energy__fJ, dcop.i_bl_port__uA


def _v_x(col: int, row: int, v_wl__V: Tensor) -> float:
    state = int(_states()[row, col])
    table = _VX_RATIO_ON_TABLE if float(v_wl__V[row]) > _V_WL_ON_THRESHOLD__V else _VX_RATIO_OFF_TABLE
    return _BL_V_REF__V - table[state] * (_BL_V_REF__V - _SL_V_REF__V)


def _rest_energy__fJ() -> float:
    per_cell = (
        _BL_NODE_C__fF * abs(_BL_V_REF__V) + _X_NODE_C__fF * abs(_BL_V_REF__V) + _SL_NODE_C__fF * abs(_SL_V_REF__V)
    )
    return _VDD__V * _COL_NUM * _ROW_NUM * per_cell


def _expected_energy__fJ(phases: Tensor) -> float:
    total = _rest_energy__fJ()
    for phase in phases:
        for col in range(_COL_NUM):
            for row in range(_ROW_NUM):
                total += _VDD__V * _X_NODE_C__fF * abs(_v_x(col, row, phase) - _BL_V_REF__V)
                total += _VDD__V * _WL_NODE_C__fF * abs(float(phase[row]))
    return total


def test_operation_bills_one_rest_establishment_plus_every_wl_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    phases = _phases()
    billed, _current = _solve(_build_array(), phases, monkeypatch, wl_phase_dims=(0,))

    assert float(billed) == pytest.approx(_expected_energy__fJ(phases), rel=1e-12)


@pytest.mark.parametrize("phase_num", [1, 3, 7])
def test_idle_phase_count_does_not_repeat_rest_establishment(
    phase_num: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idle = torch.zeros((phase_num, _ROW_NUM), dtype=_DTYPE)
    array = _build_array(vx_ratio_off_table=_VX_RATIO_OFF_AT_REST)

    billed, _current = _solve(array, idle, monkeypatch, wl_phase_dims=(0,))

    assert float(billed) == pytest.approx(_rest_energy__fJ(), rel=1e-12)


def test_wl_phase_dim_can_move_without_changing_the_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    phase_first = _phases().unsqueeze(1).expand(-1, 3, -1)
    batch_first = phase_first.movedim(0, 1)

    first_energy, first_current = _solve(_build_array(), phase_first, monkeypatch, wl_phase_dims=(0,))
    second_energy, second_current = _solve(_build_array(), batch_first, monkeypatch, wl_phase_dims=(1,))

    torch.testing.assert_close(first_energy, second_energy)
    torch.testing.assert_close(first_current.movedim(0, 1), second_current)


def test_multiple_wl_phase_dims_share_one_rest_establishment(monkeypatch: pytest.MonkeyPatch) -> None:
    phase_grid = _phases().unsqueeze(1).expand(-1, 3, -1)
    flat_phases = phase_grid.flatten(0, 1)

    grid_energy, grid_current = _solve(_build_array(), phase_grid, monkeypatch, wl_phase_dims=(0, 1))
    flat_energy, flat_current = _solve(_build_array(), flat_phases, monkeypatch, wl_phase_dims=(0,))

    torch.testing.assert_close(grid_energy, flat_energy)
    torch.testing.assert_close(grid_current.flatten(0, 1), flat_current)


def test_access_node_rests_at_the_ideal_bl_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    phases = _phases()
    billed, _current = _solve(_build_array(), phases, monkeypatch, wl_phase_dims=(0,))

    off_divider_total = _rest_energy__fJ()
    for phase in phases:
        for col in range(_COL_NUM):
            for row in range(_ROW_NUM):
                state = int(_states()[row, col])
                off_rest = _BL_V_REF__V - _VX_RATIO_OFF_TABLE[state] * (_BL_V_REF__V - _SL_V_REF__V)
                off_divider_total += _VDD__V * _X_NODE_C__fF * abs(_v_x(col, row, phase) - off_rest)
                off_divider_total += _VDD__V * _WL_NODE_C__fF * abs(float(phase[row]))

    assert float(billed) == pytest.approx(_expected_energy__fJ(phases), rel=1e-12)
    assert float(billed) != pytest.approx(off_divider_total, rel=1e-12)


def test_grounded_rest_leaves_only_phase_excursions(monkeypatch: pytest.MonkeyPatch) -> None:
    phases = _phases()
    billed, _current = _solve(
        _build_array(),
        phases,
        monkeypatch,
        wl_phase_dims=(0,),
        bl_ref__V=0.0,
        sl_ref__V=0.0,
    )

    expected = 0.0
    for phase in phases:
        for _col in range(_COL_NUM):
            for row in range(_ROW_NUM):
                expected += _VDD__V * _WL_NODE_C__fF * abs(float(phase[row]))
    assert float(billed) == pytest.approx(expected, rel=1e-12)


def test_bl_node_cap_uses_each_solved_node_displacement(monkeypatch: pytest.MonkeyPatch) -> None:
    delta__fF = 0.5
    phases = _phases()
    base, _current = _solve(_build_array(g_cell_on__uS=80.0), phases, monkeypatch, wl_phase_dims=(0,))
    raised_array = _build_array(g_cell_on__uS=80.0, bl_node_c__fF=_BL_NODE_C__fF + delta__fF)
    states: list[Any] = []
    original = raised_array._energy_from_state

    def capture_state(
        state: Any,
        *,
        cell_snap: Any,
        bl_driver_snap: Any,
        sl_driver_snap: Any,
        cell_dcop: Any,
    ) -> Tensor:
        states.append(state)
        return original(
            state,
            cell_snap=cell_snap,
            bl_driver_snap=bl_driver_snap,
            sl_driver_snap=sl_driver_snap,
            cell_dcop=cell_dcop,
        )

    monkeypatch.setattr(raised_array, "_energy_from_state", capture_state)
    raised, _current = _solve(raised_array, phases, monkeypatch, wl_phase_dims=(0,))
    [state] = states

    v_rest__V = torch.full_like(state.v_bl_node__V, _BL_V_REF__V)
    phase_delta = (state.v_bl_node__V - v_rest__V).abs().sum()
    rest_delta = _ROW_NUM * _COL_NUM * abs(_BL_V_REF__V)
    expected_slope__fJ = _VDD__V * delta__fF * float(phase_delta + rest_delta)

    assert float(raised - base) == pytest.approx(expected_slope__fJ, rel=1e-10)


def test_sl_node_cap_uses_each_solved_node_displacement(monkeypatch: pytest.MonkeyPatch) -> None:
    delta__fF = 0.5
    phases = _phases()
    base, _current = _solve(_build_array(g_cell_on__uS=80.0), phases, monkeypatch, wl_phase_dims=(0,))
    raised_array = _build_array(g_cell_on__uS=80.0, sl_node_c__fF=_SL_NODE_C__fF + delta__fF)
    states: list[Any] = []
    original = raised_array._energy_from_state

    def capture_state(
        state: Any,
        *,
        cell_snap: Any,
        bl_driver_snap: Any,
        sl_driver_snap: Any,
        cell_dcop: Any,
    ) -> Tensor:
        states.append(state)
        return original(
            state,
            cell_snap=cell_snap,
            bl_driver_snap=bl_driver_snap,
            sl_driver_snap=sl_driver_snap,
            cell_dcop=cell_dcop,
        )

    monkeypatch.setattr(raised_array, "_energy_from_state", capture_state)
    raised, _current = _solve(raised_array, phases, monkeypatch, wl_phase_dims=(0,))
    [state] = states

    v_rest__V = torch.full_like(state.v_sl_node__V, _SL_V_REF__V)
    phase_delta = (state.v_sl_node__V - v_rest__V).abs().sum()
    rest_delta = _ROW_NUM * _COL_NUM * abs(_SL_V_REF__V)
    expected_slope__fJ = _VDD__V * delta__fF * float(phase_delta + rest_delta)

    assert float(raised - base) == pytest.approx(expected_slope__fJ, rel=1e-10)
