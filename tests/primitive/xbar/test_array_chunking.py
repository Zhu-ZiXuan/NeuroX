"""Chunked array execution preserves port state and energy while bounding CUDA memory."""

from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch._dynamo
from torch import Tensor

from neurox import Profiler, stamp_names
from neurox.primitive.analog import VoltageDriver, VoltageDriverConfig, VoltageDriverPolicy, VoltageDriverSnap
from neurox.primitive.xbar.array import (
    XbarArray1t1r,
    XbarArray1t1rConfig,
    XbarArray1t1rDcop,
    XbarArray1t1rPolicy,
)
from neurox.primitive.xbar.cell import XbarCell1t1rLinearConfig, XbarCell1t1rLinearPolicy

_ARRAY_COL = 8
_ARRAY_LEADING = 32
_ARRAY_CHUNK = 8
_ARRAY_ROW_TALL = 128
_VALUE_ROW = 6
_VALUE_LEADING = 7
_BL_V_REF__V = 0.3
_SL_V_REF__V = 0.1
_VDD__V = 0.9
_DTYPE = torch.float64
type _Array = XbarArray1t1r[VoltageDriverSnap, VoltageDriverSnap]


def _array(*, row_num: int, expected_chunk_size: int, device: torch.device, wire_r__MOhm: float = 1e-4) -> _Array:
    """Build a tiny 1T1R array with a fully linear cell and every policy off."""
    cell_config = XbarCell1t1rLinearConfig(
        g_cell_off_table__uS=(4.0, 5.0),
        g_cell_on_table__uS=(50.0, 100.0),
        vx_ratio_off_table=(0.5, 0.5),
        vx_ratio_on_table=(0.4, 0.6),
        v_wl_on_threshold__V=0.5,
    )
    array = XbarArray1t1r(
        bl_driver=_ideal_driver(device),
        sl_driver=_ideal_driver(device),
        config=XbarArray1t1rConfig(
            row_cell_space__um=1.0,
            col_cell_space__um=1.0,
            bl_segment_r__MOhm=wire_r__MOhm,
            sl_segment_r__MOhm=2 * wire_r__MOhm,
            bl_node_c__fF=0.1,
            x_node_c__fF=0.1,
            sl_node_c__fF=0.1,
            wl_node_c__fF=0.1,
            cell_config=cell_config,
        ),
        policy=XbarArray1t1rPolicy(
            cell_policy=XbarCell1t1rLinearPolicy(),
            solve_chunk_size=expected_chunk_size,
        ),
        inst_shape=(),
        row_num=row_num,
        col_num=_ARRAY_COL,
        vdd__V=_VDD__V,
        dtype=_DTYPE,
        T__K=300.0,
    )
    array.to(device)
    array.eval()
    array.fabricate()
    array.program((torch.arange(_ARRAY_COL * row_num, device=device) % 2).reshape(row_num, _ARRAY_COL))
    stamp_names(array)
    return array


def _ideal_driver(device: torch.device) -> VoltageDriver:
    """Build a boundary clamp with zero output resistance and nonideality."""
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
        inst_shape=(1, _ARRAY_COL),
        dtype=_DTYPE,
        T__K=300.0,
    )
    driver.to(device)
    driver.eval()
    driver.fabricate()
    stamp_names(driver)
    return driver


def _solve_array(array: _Array, v_wl: Tensor) -> XbarArray1t1rDcop:
    """Settle one array against its bound ideal clamps at fixed references."""
    leading = tuple(v_wl.shape[:-1])
    bl_driver = array.solver.bl_driver
    sl_driver = array.solver.sl_driver
    bl_ref = torch.full((*leading, 1, _ARRAY_COL), _BL_V_REF__V, dtype=_DTYPE, device=v_wl.device)
    sl_ref = torch.full((*leading, 1, _ARRAY_COL), _SL_V_REF__V, dtype=_DTYPE, device=v_wl.device)
    state = array.solve_dc(
        v_wl__V=v_wl.unsqueeze(array.col_dim),
        wl_phase_dims=(-3,),
        bl_driver_snap=bl_driver.snapshot(v_ref__V=bl_ref, shape=bl_ref.shape),
        sl_driver_snap=sl_driver.snapshot(v_ref__V=sl_ref, shape=sl_ref.shape),
    )
    bl_driver.drive(i_port__uA=state.i_bl_port__uA)
    sl_driver.drive(i_port__uA=state.i_sl_port__uA)
    return state


def test_chunk_size_moves_neither_the_port_state_nor_the_energy(
    monkeypatch: pytest.MonkeyPatch, device: torch.device
) -> None:
    """Chunk size is a memory knob: every chunking agrees numerically."""
    v_wl = torch.rand(_VALUE_LEADING, 1, _VALUE_ROW, dtype=_DTYPE, device=device) * 1.2
    folded: dict[int, tuple[Tensor, Tensor, Tensor]] = {}

    for expected_chunk_size in (0, 2, 3, 5, 100):
        array = _array(row_num=_VALUE_ROW, expected_chunk_size=expected_chunk_size, device=device)
        billed: list[Tensor] = []
        monkeypatch.setattr(array, "_record_dynamic_energy", billed.append)
        with Profiler(leading_rank=1):
            state = _solve_array(array, v_wl)
        [energy__fJ] = billed
        folded[expected_chunk_size] = (state.i_bl_port__uA, state.v_bl_port__V, energy__fJ)

    assert folded[0][2].shape == (_VALUE_LEADING,)
    whole = folded[0]
    for expected_chunk_size, (i_bl_port__uA, v_bl_port__V, energy__fJ) in folded.items():
        message = str(expected_chunk_size)
        torch.testing.assert_close(i_bl_port__uA, whole[0], rtol=1.0e-10, atol=1.0e-12, msg=message)
        torch.testing.assert_close(v_bl_port__V, whole[1], rtol=1.0e-10, atol=1.0e-12, msg=message)
        torch.testing.assert_close(energy__fJ, whole[2], rtol=1.0e-10, atol=1.0e-12, msg=message)


@pytest.mark.parametrize(
    ("expected_chunk_size", "leading_size"),
    [(0, _VALUE_LEADING), (8, 1)],
)
def test_unpartitioned_solve_uses_one_flat_chunk(
    expected_chunk_size: int,
    leading_size: int,
    device: torch.device,
) -> None:
    """A one-chunk call needs no full result allocation."""
    array = _array(row_num=_VALUE_ROW, expected_chunk_size=expected_chunk_size, device=device)
    v_wl = torch.rand(leading_size, 1, _VALUE_ROW, dtype=_DTYPE, device=device) * 1.2
    flat_leading_shapes: list[tuple[int, ...]] = []
    solve = array.solver.solve

    @torch.compiler.disable(recursive=False)
    def probing(**kwargs: Any) -> Any:
        flat_leading_shapes.append(tuple(kwargs["cell_snap"].v_wl__V.shape[:-2]))
        return solve(**kwargs)

    # Inspect callback inputs outside the full-graph entry; the memory test
    # below exercises the compiled array path.
    eager_chunking = array._solve_dc_chunking.__wrapped__.__get__(array, type(array))
    with (
        patch.object(array, "_solve_dc_chunking", eager_chunking),
        patch.object(array.solver, "solve", probing),
    ):
        _solve_array(array, v_wl)

    assert flat_leading_shapes == [(leading_size,)]


def test_chunking_reduces_compiled_cuda_peak_memory(device: torch.device) -> None:
    if device.type != "cuda":
        pytest.skip("CUDA allocation statistics require a CUDA device")

    v_wl = torch.rand(_ARRAY_LEADING, 1, _ARRAY_ROW_TALL, dtype=_DTYPE, device=device) * 1.2
    peaks = {}
    for chunk_size in (0, _ARRAY_CHUNK):
        array = _array(row_num=_ARRAY_ROW_TALL, expected_chunk_size=chunk_size, device=device, wire_r__MOhm=1e-6)
        with torch.no_grad():
            warmup = _solve_array(array, v_wl)
            torch.cuda.synchronize(device)
            del warmup
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            result = _solve_array(array, v_wl)
            torch.cuda.synchronize(device)
            peaks[chunk_size] = torch.cuda.max_memory_allocated(device) - baseline
            del result

    assert peaks[_ARRAY_CHUNK] < peaks[0], peaks
