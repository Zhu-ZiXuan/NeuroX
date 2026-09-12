"""Array integration for the residual-driven structured solver."""

from __future__ import annotations

import pytest
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


def _build_array(
    *,
    device: torch.device,
    expected_chunk_size: int,
) -> XbarArray1t1r:
    array = XbarArray1t1r(
        config=_array_config(),
        bl_driver=_driver(device=device),
        sl_driver=_driver(device=device),
        policy=XbarArray1t1rPolicy(
            cell_policy=XbarCell1t1rLinearPolicy(),
            solve_chunk_size=expected_chunk_size,
        ),
        inst_shape=(),
        row_num=_ROW_NUM,
        col_num=_COL_NUM,
        vdd__V=0.9,
        dtype=_DTYPE,
        T__K=300.0,
    )
    array.to(device)
    array.eval()
    array.fabricate()
    array.program((torch.arange(_COL_NUM * _ROW_NUM, device=device) % 2).reshape(_COL_NUM, _ROW_NUM))
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
        inst_shape=(_COL_NUM,),
        dtype=_DTYPE,
        T__K=300.0,
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
    bl_ref = torch.full((), 0.3, dtype=_DTYPE, device=device).expand(*leading, _COL_NUM)
    sl_ref = torch.full((), 0.1, dtype=_DTYPE, device=device).expand(*leading, _COL_NUM)
    return (
        v_wl__V,
        bl_driver.snapshot(v_ref__V=bl_ref, shape=bl_ref.shape),
        sl_driver.snapshot(v_ref__V=sl_ref, shape=sl_ref.shape),
    )


def _solve(array: XbarArray1t1r, *, record_trace: bool):
    v_wl__V, bl_snap, sl_snap = _inputs(array)
    kwargs = {
        "v_wl__V": v_wl__V,
        "wl_phase_dims": (-2,),
        "bl_driver_snap": bl_snap,
        "sl_driver_snap": sl_snap,
    }
    if record_trace:
        return array.solve_dc_trace(**kwargs)
    return array.solve_dc(**kwargs), None


def _array_device(array: XbarArray1t1r) -> torch.device:
    return next(array.buffers()).device


@pytest.mark.parametrize("expected_chunk_size", [0, 2, 3, 20])
def test_chunked_solve_matches_unchunked_solve(device: torch.device, expected_chunk_size: int) -> None:
    reference = _solve(_build_array(device=device, expected_chunk_size=0), record_trace=False)
    actual = _solve(_build_array(device=device, expected_chunk_size=expected_chunk_size), record_trace=False)

    torch.testing.assert_close(actual[0].i_bl_port__uA, reference[0].i_bl_port__uA)
    torch.testing.assert_close(actual[0].i_sl_port__uA, reference[0].i_sl_port__uA)
    assert actual[1] is None


@pytest.mark.parametrize("expected_chunk_size", [2, 3])
def test_trace_is_explicit_and_reassembled(device: torch.device, expected_chunk_size: int) -> None:
    array = _build_array(device=device, expected_chunk_size=expected_chunk_size)
    reference_trace = _solve(_build_array(device=device, expected_chunk_size=0), record_trace=True)[1]

    untraced = _solve(array, record_trace=False)
    traced = _solve(array, record_trace=True)

    assert untraced[1] is None
    trace = traced[1]
    assert trace is not None
    assert reference_trace is not None
    actual_raw, actual_raw_spec = torch.utils._pytree.tree_flatten(trace)
    reference_raw, reference_raw_spec = torch.utils._pytree.tree_flatten(reference_trace)
    assert actual_raw_spec == reference_raw_spec
    torch.testing.assert_close(actual_raw, reference_raw, equal_nan=True)
    assert trace.residual__V.shape[-1] == 20
    assert trace.node_trace is not None
    assert trace.node_trace.residual__uA.shape[-2:] == (20, 20)
    for residual, threshold, iteration_dim in (
        (trace.residual__V, trace.threshold__V, -1),
        (trace.node_trace.residual__uA, trace.node_trace.threshold__uA, -2),
    ):
        residual = residual.movedim(iteration_dim, -1)
        threshold = threshold.movedim(iteration_dim, -1)
        valid = ~residual.isnan()
        assert valid.any()
        assert not ((~valid[..., :-1]) & valid[..., 1:]).any()
        terminal = (valid.sum(dim=-1) - 1).clamp_min(0).unsqueeze(-1)
        terminal_residual = residual.gather(-1, terminal).squeeze(-1)
        terminal_threshold = threshold.gather(-1, terminal).squeeze(-1)
        assert (terminal_residual[valid.any(dim=-1)] <= terminal_threshold[valid.any(dim=-1)]).all()
    torch.testing.assert_close(
        traced[0].i_bl_port__uA,
        untraced[0].i_bl_port__uA,
        rtol=1e-10,
        atol=1e-12,
    )


def test_solve_dc_keeps_the_dcop_surface(device: torch.device) -> None:
    array = _build_array(device=device, expected_chunk_size=3)
    v_wl__V, bl_snap, sl_snap = _inputs(array)
    kwargs = {
        "v_wl__V": v_wl__V,
        "wl_phase_dims": (-2,),
        "bl_driver_snap": bl_snap,
        "sl_driver_snap": sl_snap,
    }

    dcop = array.solve_dc(**kwargs)
    traced, trace = array.solve_dc_trace(**kwargs)

    torch.testing.assert_close(dcop.i_bl_port__uA, traced.i_bl_port__uA)
    assert trace is not None
