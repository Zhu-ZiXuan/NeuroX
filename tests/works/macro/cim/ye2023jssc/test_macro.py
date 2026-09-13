"""WH-2T1R macro integration."""

import pytest
import torch

from neurox import Profiler, Reporter, stamp_names

from ._utils import ADC_BITS, INPUT_NUM, OUTPUT_NUM, build_macro


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
    assert macro.latency__ns(adc_active_bits=ADC_BITS) == 2.0 * 9.0


def test_macro_energy_rows_follow_physical_owners(device: torch.device) -> None:
    macro = build_macro(device=device)
    stamp_names(macro)
    reporter = Reporter(macro)
    macro.program(torch.ones((INPUT_NUM, OUTPUT_NUM), dtype=torch.int32, device=device))

    with Profiler() as profiler:
        macro.vec_mat_mul(
            torch.ones(INPUT_NUM, dtype=torch.int32, device=device), quantization_mode=0, adc_active_bits=4
        )

    rows = reporter.by_name(profiler)
    assert ".bl_conduction" in rows
    assert ".tbl_conduction" in rows
    assert "rscsa" in rows
