"""Xue2020 flattened-array serialization tests."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from neurox import Profiler
from tests.works.macro.cim.xue2020jssc.macro._utils import (
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


def _mixed_weight(input_num: int, output_num: int) -> Tensor:
    """Return a non-degenerate mixed-sign weight witness."""
    gen = torch.Generator().manual_seed(11)
    return torch.randint(-3, 4, (input_num, output_num), generator=gen, dtype=torch.int32)


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
        twin.program(w[:, scan * lane_num : (scan + 1) * lane_num].contiguous().to(device))
    return big, twins, w


def test_vec_mat_mul_commutes_with_serialization(device: torch.device) -> None:
    scan_num = 2
    # Unequal scan and lane counts expose swapped readout axes.
    big, twins, _w = _twin_pair(device, scan_num=scan_num, output_num=6)
    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0], [0, 1, 2, 3]], dtype=torch.int32, device=device)

    with torch.no_grad():
        out = big.vec_mat_mul(x, quantization_mode=QUANTIZATION_MODE, adc_active_bits=TINY_ADC_BITS).cpu()
        for scan, twin in enumerate(twins):
            twin_out = twin.vec_mat_mul(x, quantization_mode=QUANTIZATION_MODE, adc_active_bits=TINY_ADC_BITS).cpu()
            block = out[..., scan * big.lane_num : (scan + 1) * big.lane_num]
            torch.testing.assert_close(block, twin_out)
    # Reject a witness that cannot expose column permutations.
    assert int(out.min()) < 0 < int(out.max()), f"witness decode is degenerate: {out.tolist()}"


def test_array_cap_energy_independent_of_scan_num(device: torch.device) -> None:
    w = _mixed_weight(TINY_INPUT_NUM, TINY_OUTPUT_NUM)
    x = torch.tensor([1, 2, 1, 3], dtype=torch.int32, device=device)

    def array_row(scan_num: int) -> float:
        macro = build_macro(
            build_config(lane_num=TINY_OUTPUT_NUM // scan_num, scan_num=scan_num),
            device=device,
        )
        macro.program(w.to(device))
        with Profiler(concat_dim=0) as prof, torch.no_grad():
            macro.vec_mat_mul(x, quantization_mode=QUANTIZATION_MODE, adc_active_bits=TINY_ADC_BITS)
        energy__fJ = prof.result["array"].dynamic_energy__fJ
        assert energy__fJ is not None
        return energy__fJ.item()

    e_mux2 = array_row(2)
    e_mux4 = array_row(4)
    assert e_mux2 > 0.0
    assert e_mux4 == pytest.approx(e_mux2, rel=1e-12), (
        f"array cap energy moved with the MUX factoring: {e_mux2} vs {e_mux4} "
        "(the WL ladder must bill once per plane, independent of scan_num)"
    )
