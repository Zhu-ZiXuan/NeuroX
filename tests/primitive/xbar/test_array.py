"""Array integration for the residual-driven structured solver."""

from __future__ import annotations

import torch
from torch import Tensor

from neurox import stamp_names
from neurox.primitive.analog import VoltageDriver, VoltageDriverConfig, VoltageDriverPolicy
from neurox.primitive.xbar.array import (
    XbarArray1t1r,
    XbarArray1t1rConfig,
    XbarArray1t1rPolicy,
)
from neurox.primitive.xbar.cell import XbarCell1t1rLinearConfig, XbarCell1t1rLinearPolicy

_ROW_NUM = 4
_COL_NUM = 3
_DTYPE = torch.float64


def _cell_config() -> XbarCell1t1rLinearConfig:
    return XbarCell1t1rLinearConfig(
        g_cell_off_table__uS=(4.0, 5.0),
        g_cell_on_table__uS=(50.0, 100.0),
        vx_ratio_off_table=(0.5, 0.5),
        vx_ratio_on_table=(0.4, 0.6),
        v_wl_on_threshold__V=0.5,
    )


def _array_config() -> XbarArray1t1rConfig:
    return XbarArray1t1rConfig(
        row_cell_space__um=1.0,
        col_cell_space__um=1.0,
        bl_segment_r__MOhm=2e-4,
        sl_segment_r__MOhm=4e-4,
        bl_node_c__fF=0.1,
        x_node_c__fF=0.1,
        sl_node_c__fF=0.1,
        wl_node_c__fF=0.1,
        cell_config=_cell_config(),
    )


def _build_array(*, device: torch.device) -> XbarArray1t1r:
    array = XbarArray1t1r(
        config=_array_config(),
        bl_driver=_driver(device=device),
        sl_driver=_driver(device=device),
        policy=XbarArray1t1rPolicy(
            cell_policy=XbarCell1t1rLinearPolicy(),
            solve_chunk_size=0,
        ),
        inst_shape=(),
        row_num=_ROW_NUM,
        col_num=_COL_NUM,
        vdd__V=0.9,
        dtype=_DTYPE,
    )
    array.to(device)
    array.eval()
    array.fabricate()
    state = (torch.arange(_COL_NUM * _ROW_NUM, device=device) % 2).reshape(_ROW_NUM, _COL_NUM)
    array.program(state)
    stamp_names(array)
    return array


def _driver(*, device: torch.device) -> VoltageDriver:
    driver = VoltageDriver(
        config=VoltageDriverConfig(
            r_out__MOhm=2e-3,
            offset_sigma__V=0.0,
            thermal_sigma__V=0.0,
            energy_per_op__fJ=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        policy=VoltageDriverPolicy(offset=False, thermal=False),
        inst_shape=(1, _COL_NUM),
        dtype=_DTYPE,
    )
    driver.to(device)
    driver.eval()
    driver.fabricate()
    stamp_names(driver)
    return driver


def _inputs(array: XbarArray1t1r) -> tuple[Tensor, object, object]:
    device = _array_device(array)
    v_wl__V = torch.tensor(
        [
            [[0.9, 0.0, 0.9, 0.0]],
            [[0.0, 0.9, 0.0, 0.9]],
            [[0.9, 0.9, 0.0, 0.0]],
            [[0.0, 0.0, 0.9, 0.9]],
            [[0.9, 0.9, 0.9, 0.9]],
        ],
        dtype=_DTYPE,
        device=device,
    )
    leading = tuple(v_wl__V.shape[:-1])
    bl_driver = array.solver.bl_driver
    sl_driver = array.solver.sl_driver
    bl_ref = torch.full((), 0.3, dtype=_DTYPE, device=device).expand(*leading, _COL_NUM).unsqueeze(array.row_dim)
    sl_ref = torch.full((), 0.1, dtype=_DTYPE, device=device).expand(*leading, _COL_NUM).unsqueeze(array.row_dim)
    return (
        v_wl__V.unsqueeze(array.col_dim),
        bl_driver.snapshot(v_ref__V=bl_ref, shape=bl_ref.shape),
        sl_driver.snapshot(v_ref__V=sl_ref, shape=sl_ref.shape),
    )


def _solve_trace(array: XbarArray1t1r):
    v_wl__V, bl_snap, sl_snap = _inputs(array)
    return array.solve_dc_trace(
        v_wl__V=v_wl__V,
        wl_phase_dims=(-3,),
        bl_driver_snap=bl_snap,
        sl_driver_snap=sl_snap,
    )


def _array_device(array: XbarArray1t1r) -> torch.device:
    return next(array.buffers()).device


def test_trace_preserves_array_observation_layout(device: torch.device) -> None:
    array = _build_array(device=device)
    traced = _solve_trace(array)

    trace = traced[1]
    assert trace is not None
    assert traced[0].i_bl_port__uA.shape == (5, 1, 1, _COL_NUM)
    assert trace.residual__V.shape[:-1] == traced[0].v_bl_port__V.shape
    assert trace.node_trace is not None
    assert trace.node_trace.residual__uA.shape[:-2] == traced[0].v_bl_node__V.shape
    assert trace.node_trace.residual__uA.shape[-1] == trace.residual__V.shape[-1]
    assert trace.node_trace.limited.shape == (
        *traced[0].v_bl_port__V.shape,
        *trace.node_trace.residual__uA.shape[-2:],
    )
    for residual, iteration_dim in (
        (trace.residual__V, -1),
        (trace.node_trace.residual__uA, -2),
    ):
        residual = residual.movedim(iteration_dim, -1)
        valid = ~residual.isnan()
        assert valid.any()
        assert not ((~valid[..., :-1]) & valid[..., 1:]).any()
