"""WH-2T1R macro integration."""

import dataclasses

import pytest
import torch

from neurox import Profiler, Reporter, stamp_names
from neurox.common.encoding import Encoding
from neurox.primitive.macro.cim import CimMacro

from ._utils import ADC_BITS, INPUT_NUM, OUTPUT_NUM, build_config, build_macro, build_policy


def test_macro_owns_the_logical_to_physical_mapping(device: torch.device) -> None:
    config = build_config()
    macro = build_macro(device=device)

    assert config.w_digit_n == config.w_digit_num
    assert config.w_digit_r == config.w_digit_radix
    assert config.w_enc is Encoding.UNSIGNED
    assert config.x_digit_n == 1
    assert config.x_digit_r == 2
    assert config.x_enc is Encoding.UNSIGNED
    assert macro.row_num == macro.output_num
    assert macro.col_num == macro.input_num * (config.w_digit_num + 1)
    assert macro.lane_num == 1
    assert macro.scan_num == OUTPUT_NUM


def test_weight_radix_cannot_exceed_the_cell_state_count() -> None:
    config = dataclasses.replace(build_config(), w_digit_radix=3)
    with pytest.raises(ValueError, match=r"array\.w_state_num \(2\) >= w_digit_r \(3\)"):
        CimMacro.from_config(
            config=config,
            policy=build_policy(),
            inst_shape=(),
            dtype=torch.float64,
            T__K=300.0,
        )


def test_each_scan_activates_one_row_per_lane(device: torch.device) -> None:
    lane_num = 2
    scan_num = 3
    macro = build_macro(device=device, lane_num=lane_num, scan_num=scan_num)

    wl_on = macro._v_wl_scan__V > 0
    grouped = wl_on.unflatten(-1, (lane_num, scan_num))

    assert wl_on.shape == (scan_num, lane_num * scan_num)
    lane_activation_count = grouped.sum(dim=-1)
    row_activation_count = wl_on.sum(dim=0)
    assert torch.equal(lane_activation_count, torch.ones_like(lane_activation_count))
    assert torch.equal(row_activation_count, torch.ones_like(row_activation_count))


def test_macro_matches_unsigned_mac_and_uses_configured_schedule(device: torch.device) -> None:
    macro = build_macro(device=device)
    weight = torch.tensor([[1, 3, 7, 0], [2, 4, 1, 6]], device=device)
    x = torch.tensor([[1, 0], [0, 1], [1, 1]], device=device)
    macro.program(weight)

    code = macro.vec_mat_mul(x, quantization_mode=0, adc_active_bits=ADC_BITS)
    highest_precision_code = macro.vec_mat_mul(x, quantization_mode=0, adc_active_bits=None)

    expected = (x.unsqueeze(-1) * weight).sum(dim=-2).clamp(max=15)
    assert torch.equal(code, expected)
    assert torch.equal(highest_precision_code, code)
    assert macro.latency__ns(adc_active_bits=ADC_BITS) == 9.0 * OUTPUT_NUM
    assert macro.latency__ns(adc_active_bits=None) == macro.latency__ns(adc_active_bits=ADC_BITS)


def test_batch_and_instance_axes_preserve_the_logical_vmm(device: torch.device) -> None:
    macro = build_macro(device=device, inst_shape=(2,))
    weight = torch.tensor(
        [
            [[1, 3, 7, 0], [2, 4, 1, 6]],
            [[7, 0, 2, 1], [1, 5, 3, 4]],
        ],
        device=device,
    )
    x = torch.tensor(
        [
            [[1, 0], [0, 1]],
            [[0, 1], [1, 1]],
            [[1, 1], [1, 0]],
        ],
        device=device,
    )
    macro.program(weight)

    code = macro.vec_mat_mul(x, quantization_mode=0, adc_active_bits=ADC_BITS)
    expected = (x.unsqueeze(-1) * weight).sum(dim=-2).clamp(max=15)

    assert code.shape == (3, 2, OUTPUT_NUM)
    assert torch.equal(code, expected)


@pytest.mark.parametrize("solve_chunk_size", [0, 3])
def test_parallel_readout_lanes_preserve_output_order_and_reduce_latency(
    device: torch.device,
    solve_chunk_size: int,
) -> None:
    macro = build_macro(device=device, lane_num=2, scan_num=2, solve_chunk_size=solve_chunk_size)
    weight = torch.tensor([[1, 3, 7, 0], [2, 4, 1, 6]], device=device)
    x = torch.tensor([[1, 0], [0, 1], [1, 1]], device=device)
    macro.program(weight)

    code = macro.vec_mat_mul(x, quantization_mode=0, adc_active_bits=ADC_BITS)

    expected = (x.unsqueeze(-1) * weight).sum(dim=-2).clamp(max=15)
    assert torch.equal(code, expected)
    assert macro.lane_num == 2
    assert macro.scan_num == 2
    assert macro.rscsa.inst_shape[-2:] == (2, 1)
    assert macro.latency__ns(adc_active_bits=ADC_BITS) == 2.0 * 9.0


def test_macro_energy_rows_follow_physical_owners(device: torch.device) -> None:
    macro = build_macro(device=device)
    stamp_names(macro)
    reporter = Reporter(macro)
    macro.program(torch.ones((INPUT_NUM, OUTPUT_NUM), dtype=torch.long, device=device))

    with Profiler() as profiler:
        macro.vec_mat_mul(
            torch.ones(INPUT_NUM, dtype=torch.long, device=device), quantization_mode=0, adc_active_bits=4
        )

    rows = reporter.by_name(profiler)
    assert ".bl_conduction" in rows
    assert ".tbl_conduction" in rows
    assert "rscsa" in rows
