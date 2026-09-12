"""Array solver diagnostics and driver ownership in the Xue and Ye assemblies."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
import torch

from neurox import Profiler, Reporter, check_unique_binding, stamp_names
from neurox.primitive.xbar.solver import ColBlColSlArrayState
from tests.works.macro.cim.xue2020jssc._utils import build_config as build_xue_config
from tests.works.macro.cim.xue2020jssc._utils import build_macro as build_xue_macro
from tests.works.macro.cim.ye2023jssc._utils import INPUT_NUM, OUTPUT_NUM
from tests.works.macro.cim.ye2023jssc._utils import build_macro as build_ye_macro


def test_ye_array_chunked_trace_preserves_dcop_and_projection(device: torch.device) -> None:
    unchunked = build_ye_macro(device=device, lane_num=2, scan_num=2, solve_chunk_size=0)
    chunked = build_ye_macro(device=device, lane_num=2, scan_num=2, solve_chunk_size=3)
    weight = (torch.arange(INPUT_NUM * OUTPUT_NUM, device=device) % 8).reshape(INPUT_NUM, OUTPUT_NUM)
    results = []
    for macro in (unchunked, chunked):
        macro.program(weight)
        port_shape = (3, macro.scan_num, macro.col_num)
        v_bl__V = (torch.arange(3 * macro.col_num, device=device) % 2).reshape(3, 1, macro.col_num) * macro._v_bl__V
        v_wl__V = macro._v_wl_scan__V.expand(3, macro.scan_num, macro.row_num)
        bl_snap = macro.bl_driver.snapshot(v_ref__V=v_bl__V, shape=port_shape)
        sl_snap = macro.sl_driver.snapshot(v_ref__V=torch.zeros_like(v_bl__V), shape=port_shape)
        dcop, trace = macro.array.solve_dc_trace(
            v_wl__V=v_wl__V,
            wl_phase_dims=(-2,),
            bl_driver_snap=bl_snap,
            sl_driver_snap=sl_snap,
        )
        plain_dcop = macro.array.solve_dc(
            v_wl__V=v_wl__V,
            wl_phase_dims=(-2,),
            bl_driver_snap=bl_snap,
            sl_driver_snap=sl_snap,
        )
        torch.testing.assert_close(
            torch.utils._pytree.tree_leaves(dcop),
            torch.utils._pytree.tree_leaves(plain_dcop),
        )
        results.append((dcop, trace))

    (reference_dcop, reference_trace), (actual_dcop, actual_trace) = results
    for field in (
        "i_bl_port__uA",
        "v_bl_port__V",
        "i_sl_port__uA",
        "v_sl_port__V",
        "i_tbl_by_row__uA",
    ):
        torch.testing.assert_close(getattr(actual_dcop, field), getattr(reference_dcop, field))
    assert actual_trace is not None
    assert reference_trace is not None
    assert actual_trace.node_trace is not None
    assert actual_trace.residual__V.shape[-1] == 20
    assert actual_trace.node_trace.residual__uA.shape[-2:] == (20, 20)
    actual_leaves, actual_spec = torch.utils._pytree.tree_flatten(actual_trace)
    reference_leaves, reference_spec = torch.utils._pytree.tree_flatten(reference_trace)
    assert actual_spec == reference_spec
    torch.testing.assert_close(actual_leaves, reference_leaves, equal_nan=True)


@pytest.mark.parametrize("family", ["xue", "ye"])
@pytest.mark.parametrize("record_energy", [False, True])
def test_terminal_outputs_share_one_cell_evaluation(
    family: str,
    record_energy: bool,
    device: torch.device,
) -> None:
    macro = build_xue_macro(build_xue_config(), device=device) if family == "xue" else build_ye_macro(device=device)
    array = macro.array
    array.program(torch.zeros(array.cell.inst_shape, dtype=torch.long, device=device))
    grid = torch.full((1, *array.cell.inst_shape), 0.2, dtype=torch.float64, device=device)
    cell_snap = array.cell.snapshot(control=torch.full_like(grid, 0.9), shape=tuple(grid.shape))
    port = grid[..., 0]
    state = ColBlColSlArrayState(
        is_active=torch.zeros_like(port, dtype=torch.bool),
        v_bl_node__V=grid,
        v_sl_node__V=torch.zeros_like(grid),
        v_bl_port__V=port,
        v_sl_port__V=torch.zeros_like(port),
    )
    bl_driver = macro.cablc if family == "xue" else macro.bl_driver
    bl_snap = bl_driver.snapshot(v_ref__V=port, shape=tuple(port.shape))
    sl_snap = macro.sl_driver.snapshot(v_ref__V=torch.zeros_like(port), shape=tuple(port.shape))

    def solve_terminal(**kwargs: Any) -> tuple[Any, None]:
        chunk_state = torch.utils._pytree.tree_map(lambda tensor: tensor.flatten(0, grid.ndim - 3), state)
        return kwargs["final_fn"](chunk_state), None

    with (
        patch.object(array.solver, "solve", solve_terminal),
        patch.object(array.cell, "solve_dc", wraps=array.cell.solve_dc) as solve,
        patch.object(array, "_dcop_from_state", wraps=array._dcop_from_state) as dcop_fn,
        patch.object(array, "_energy_from_state", wraps=array._energy_from_state) as energy_fn,
    ):
        solve_chunking = array._solve_dc_chunking.__wrapped__.__get__(array, type(array))
        dcop, energy, _ = solve_chunking(
            cell_snap=cell_snap,
            bl_driver_snap=bl_snap,
            sl_driver_snap=sl_snap,
            leading_shape=tuple(grid.shape[:-2]),
            wl_phase_dims=(0,),
            record_energy=record_energy,
            record_trace=False,
        )
    assert solve.call_count == 1
    dcop_fn.assert_called_once()
    assert dcop_fn.call_args.kwargs["cell_dcop"] is not None
    torch.testing.assert_close(dcop.v_bl_port__V, state.v_bl_port__V)
    assert (energy is not None) == record_energy
    if record_energy:
        energy_fn.assert_called_once()
        assert energy_fn.call_args.kwargs["cell_dcop"] is dcop_fn.call_args.kwargs["cell_dcop"]
    else:
        energy_fn.assert_not_called()


@pytest.mark.parametrize("family", ["xue", "ye"])
def test_bound_drivers_keep_one_owner_for_lifecycle_and_accounting(family: str, device: torch.device) -> None:
    macro = build_xue_macro(build_xue_config(), device=device) if family == "xue" else build_ye_macro(device=device)
    bl_name = "cablc" if family == "xue" else "bl_driver"
    bl_driver = getattr(macro, bl_name)
    sl_driver = macro.sl_driver
    assert macro.array.solver.bl_driver is bl_driver
    assert macro.array.solver.sl_driver is sl_driver
    check_unique_binding(macro)
    stamp_names(macro)
    reporter = Reporter(macro)

    for name, driver in ((bl_name, bl_driver), ("sl_driver", sl_driver)):
        assert [path for path, module in macro.named_modules(remove_duplicate=False) if module is driver] == [name]
        [entry] = [entry for entry in reporter.static_entries if entry.qualified_name == name]
        assert entry.area__um2 == driver.area__um2
        assert entry.leakage__uW == driver.leakage__uW

    with (
        patch.object(
            bl_driver, "_sample_fabrication_variation", wraps=bl_driver._sample_fabrication_variation
        ) as bl_sample,
        patch.object(
            sl_driver, "_sample_fabrication_variation", wraps=sl_driver._sample_fabrication_variation
        ) as sl_sample,
    ):
        macro.fabricate()
    bl_sample.assert_called_once()
    sl_sample.assert_called_once()

    with Profiler() as profiler:
        bl_driver.drive(i_port__uA=torch.zeros(bl_driver.inst_shape, device=device))
        sl_driver.drive(i_port__uA=torch.zeros(sl_driver.inst_shape, device=device))
    assert [record.qualified_name for record in profiler.records] == [bl_name, "sl_driver"]
    assert set(reporter.by_name(profiler)) == {bl_name, "sl_driver"}
