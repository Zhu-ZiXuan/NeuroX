"""Xue2020 flattened-array serialization tests."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from neurox import Profiler, Reporter
from neurox.primitive.xbar.array import XbarArray1t1rDcop
from neurox.primitive.xbar.cell import XbarCell1t1rLinear

from ._utils import (
    MAG_MAX,
    QUANTIZATION_MODE,
    TINY_ADC_BITS,
    TINY_INPUT_NUM,
    TINY_OUTPUT_NUM,
    Xue2020JsscCimMacro,
    build_config,
    build_macro,
    midpoint_refs,
    probe_i_sub_grid,
    with_ref_levels,
)

_POLARITY_NUM = 2
_FLOAT_TOLERANCE = {"rtol": 1.0e-10, "atol": 1.0e-12}


def _mixed_weight(input_num: int, output_num: int) -> Tensor:
    """Return a non-degenerate mixed-sign weight witness."""
    gen = torch.Generator().manual_seed(11)
    return torch.randint(-3, 4, (input_num, output_num), generator=gen, dtype=torch.long)


def _twin_pair(
    device: torch.device, *, scan_num: int = 2, output_num: int = TINY_OUTPUT_NUM
) -> tuple[Xue2020JsscCimMacro, list[Xue2020JsscCimMacro], Tensor]:
    """Build one serialized macro and its per-scan twins."""
    lane_num = output_num // scan_num
    config = build_config(lane_num=lane_num, scan_num=scan_num)
    grid = probe_i_sub_grid(build_macro(config, device=device), m_max=MAG_MAX)
    levels = midpoint_refs(grid, adc_bits=TINY_ADC_BITS)

    big = build_macro(with_ref_levels(config, levels), device=device)
    twin_config = with_ref_levels(build_config(lane_num=lane_num, scan_num=1), levels)
    twins = [build_macro(twin_config, device=device) for _ in range(scan_num)]

    w = _mixed_weight(TINY_INPUT_NUM, output_num)
    big.program(w.to(device))
    for scan, twin in enumerate(twins):
        twin.program(w[:, scan::scan_num].contiguous().to(device))
    return big, twins, w


def _solve_dcop(macro: Xue2020JsscCimMacro, x: Tensor) -> XbarArray1t1rDcop:
    """Solve the programmed array under the serialized WL and driver inputs."""
    v_wl__V = macro._x_transcoder.encode(x.long(), dim=-2) * macro._v_wl_on__V
    seat_shape = (*v_wl__V.shape[:-1], 1, macro.lane_num, macro.scan_num, _POLARITY_NUM, macro.config.w_digit_num)
    bl_v_ref__V = macro.cablc_vref.values()
    bl_v_ref__V = bl_v_ref__V.view(*bl_v_ref__V.shape, 1, 1, 1, 1, 1, 1)
    bl_snap = macro.cablc.snapshot(v_ref__V=bl_v_ref__V, shape=seat_shape).flatten_axes(-4, -1)
    sl_snap = macro.sl_driver.snapshot(v_ref__V=macro._sl_v_ref__V, shape=seat_shape).flatten_axes(-4, -1)
    return macro.array.solve_dc(
        v_wl__V=v_wl__V.unsqueeze(macro.array.col_dim),
        wl_phase_dims=(-3,),
        bl_driver_snap=bl_snap,
        sl_driver_snap=sl_snap,
    )


def test_serialization_commutes_with_the_flattened_solve(device: torch.device) -> None:
    scan_num = 2
    big, twins, _w = _twin_pair(device, scan_num=scan_num)
    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0]], dtype=torch.long, device=device)  # batch (2,)

    big_dcop = _solve_dcop(big, x)
    lane_num = big.lane_num
    seat_axes = (lane_num, scan_num, _POLARITY_NUM, big.config.w_digit_num)

    for scan, twin in enumerate(twins):
        twin_dcop = _solve_dcop(twin, x)
        for field in ("i_bl_port__uA", "v_bl_port__V", "i_sl_port__uA", "v_sl_port__V"):
            # Shape: [..., act, phys_col] -> [..., act, twin_phys_col]
            big_value = getattr(big_dcop, field).unflatten(-1, seat_axes).select(-3, scan).flatten(-3)
            torch.testing.assert_close(
                big_value,
                getattr(twin_dcop, field),
                **_FLOAT_TOLERANCE,
            )

    # Reject a witness that cannot expose column permutations.
    i_bl = big_dcop.i_bl_port__uA
    assert float(i_bl.max() - i_bl.min()) > 0.0, "witness columns are indistinguishable; the twin proves nothing"


def test_vec_mat_mul_commutes_with_serialization(device: torch.device) -> None:
    scan_num = 2
    big, twins, _w = _twin_pair(device, scan_num=scan_num)
    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0], [0, 1, 2, 3]], dtype=torch.long, device=device)

    with torch.no_grad():
        out = big.vec_mat_mul(x, quantization_mode=QUANTIZATION_MODE, adc_active_bits=TINY_ADC_BITS).cpu()
        for scan, twin in enumerate(twins):
            twin_out = twin.vec_mat_mul(x, quantization_mode=QUANTIZATION_MODE, adc_active_bits=TINY_ADC_BITS).cpu()
            assert torch.equal(out[..., scan::scan_num], twin_out), (
                f"scan {scan} decode differs:\n{out[..., scan::scan_num].tolist()}\nvs\n{twin_out.tolist()}"
            )
    # Reject a witness that cannot expose column permutations.
    assert int(out.min()) < 0 < int(out.max()), f"witness decode is degenerate: {out.tolist()}"


def test_program_writes_lsb_first_digits_at_the_documented_columns(device: torch.device) -> None:
    w_digit_num = 2
    scan_num = 2
    radix = 2
    config = build_config(
        w_digit_num=w_digit_num,
        lane_num=TINY_OUTPUT_NUM // scan_num,
        scan_num=scan_num,
        w_digit_radix=radix,
    )
    macro = build_macro(config, device=device)
    lane_num = macro.output_num // scan_num

    w = torch.tensor([[1, -1, 2, -2], [2, -2, 1, -1], [3, 0, -3, 0], [0, 3, 0, -3]], dtype=torch.long)
    assert tuple(w.shape) == (macro.input_num, macro.output_num)

    # Shape: [row, phys_col]
    want_state = torch.zeros(macro.row_num, macro.col_num, dtype=torch.long)
    for row in range(macro.input_num):
        for col in range(macro.output_num):
            value = int(w[row, col])
            lane, scan = divmod(col, scan_num)
            for w_digit in range(w_digit_num):
                magnitude = abs(value) // radix**w_digit % radix
                for pol in range(_POLARITY_NUM):  # 0 = PWG (positive), 1 = NWG (negative)
                    phys_col = ((lane * scan_num + scan) * _POLARITY_NUM + pol) * w_digit_num + w_digit
                    carries = (value >= 0) if pol == 0 else (value < 0)
                    want_state[row, phys_col] = magnitude if carries else 0

    # Both digit order and polarity must be observable.
    seat_state = want_state.movedim(-1, 0).unflatten(
        0,
        (lane_num, scan_num, _POLARITY_NUM, w_digit_num),
    )
    assert not torch.equal(seat_state, seat_state.flip(-1)), "witness is blind to an MSB-first digit order"
    assert not torch.equal(seat_state, seat_state.flip(-2)), "witness is blind to a PWG/NWG swap"

    macro.program(w.to(device))
    cell = macro.array.cell
    assert isinstance(cell, XbarCell1t1rLinear)
    table__uS = cell._g_cell_on_table__uS
    assert len(torch.unique(table__uS)) == table__uS.numel(), "witness conductance table is degenerate"
    torch.testing.assert_close(
        cell._g_cell_on__uS,
        table__uS[want_state.to(device)].unsqueeze(0),
        **_FLOAT_TOLERANCE,
    )


def test_array_cap_energy_independent_of_scan_num(device: torch.device) -> None:
    w = _mixed_weight(TINY_INPUT_NUM, TINY_OUTPUT_NUM)
    x = torch.tensor([1, 2, 1, 3], dtype=torch.long, device=device)

    def array_row(scan_num: int) -> float:
        macro = build_macro(
            build_config(lane_num=TINY_OUTPUT_NUM // scan_num, scan_num=scan_num),
            device=device,
        )
        macro.program(w.to(device))
        with Profiler() as prof, torch.no_grad():
            macro.vec_mat_mul(x, quantization_mode=QUANTIZATION_MODE, adc_active_bits=TINY_ADC_BITS)
        return Reporter(macro).by_name(prof)["array"]

    e_mux2 = array_row(2)
    e_mux4 = array_row(4)
    assert e_mux2 > 0.0
    assert e_mux4 == pytest.approx(e_mux2, rel=1e-12), (
        f"array cap energy moved with the MUX factoring: {e_mux2} vs {e_mux4} "
        "(the WL ladder must bill once per plane, independent of scan_num)"
    )


def test_true_shape_law(device: torch.device) -> None:
    for w_digit_num, scan_num in ((2, 2), (3, 2)):
        config = build_config(
            w_digit_num=w_digit_num,
            lane_num=TINY_OUTPUT_NUM // scan_num,
            scan_num=scan_num,
        )
        for inst in ((), (2,)):
            macro = build_macro(config, device=device, inst_shape=inst)
            lane_num = macro.lane_num
            for name, trailing in (
                ("cablc", (1, 1, lane_num, 1, _POLARITY_NUM, config.w_digit_num)),
                ("sl_driver", (1, 1, lane_num, 1, _POLARITY_NUM, config.w_digit_num)),
                ("tmcsa", (lane_num, 1)),
                ("control", (1,)),
            ):
                module = getattr(macro, name)
                want = (*inst, *trailing)
                assert module.inst_shape == want, f"{name}.inst_shape {module.inst_shape} != {want}"
                assert module.inst_count == math.prod(inst) * math.prod(trailing)
            assert macro.array.cell.inst_shape == (
                *inst,
                1,
                macro.row_num,
                macro.scan_num * lane_num * _POLARITY_NUM * config.w_digit_num,
            )
