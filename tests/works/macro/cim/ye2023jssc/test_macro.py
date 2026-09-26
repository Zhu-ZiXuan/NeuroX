"""WH-2T1R macro integration."""

from __future__ import annotations

import torch

from neurox import Profiler, stamp_names

from ._utils import ADC_BITS, INPUT_NUM, OUTPUT_NUM, build_macro


def test_readout_order_preserves_vmm_across_batch_and_instance_axes(device: torch.device) -> None:
    # Unequal lane and scan counts expose transposed readout axes.
    macro = build_macro(device=device, input_num=3, inst_shape=(2,), lane_num=2, scan_num=3)
    weight = torch.tensor(
        [
            [[1, 3, 7, 0, 2, 5], [2, 4, 1, 6, 3, 2], [0, 0, 7, 0, 0, 0]],
            [[7, 0, 2, 1, 0, 3], [1, 5, 3, 4, 2, 1], [7, 0, 0, 0, 0, 0]],
        ],
        device=device,
    )
    x = torch.tensor(
        [
            [[1, 0, 0], [0, 1, 0]],
            [[0, 1, 0], [1, 1, 1]],
            [[1, 1, 1], [1, 0, 0]],
        ],
        device=device,
    )
    macro.program(weight)

    code = macro.vec_mat_mul(x, quantization_mode=0, adc_active_bits=ADC_BITS)
    expected = (x.unsqueeze(-1) * weight).sum(dim=-2).clamp(max=15)

    assert code.shape == (3, 2, 6)
    assert torch.equal(code, expected)
    assert code.min() == 0
    assert code.max() == 15


def test_partial_scan_selects_drive_events_and_latency(device: torch.device) -> None:
    macro = build_macro(device=device, inst_shape=(2,), lane_num=2, scan_num=2)
    macro.program(torch.ones((2, INPUT_NUM, OUTPUT_NUM), dtype=torch.int32, device=device))
    x = torch.ones((3, 1, INPUT_NUM), dtype=torch.int32, device=device)
    counts = torch.tensor([[0, 0], [1, 3], [2, 4]], device=device)
    x[0] = 0
    stamp_names(macro)
    macro.set_profile_leading_rank(1)
    with Profiler(concat_dim=0) as profiler:
        output = macro.vec_mat_mul(x, quantization_mode=0, adc_active_bits=ADC_BITS, effective_output_num=counts)
    assert torch.equal(
        output,
        torch.full_like(output, INPUT_NUM).where(torch.arange(OUTPUT_NUM, device=device) < counts.unsqueeze(-1), 0),
    )
    energy__fJ = {name: item.dynamic_energy__fJ for name, item in profiler.result.items()}
    assert energy__fJ["rscsa"][0] == 0
    assert energy__fJ["bl_conduction"][0] == 0
    assert energy__fJ["bl_conduction"][1] > 0
    assert energy__fJ["bl_conduction"][2] > 0
    assert torch.equal(
        macro.latency__ns(adc_active_bits=ADC_BITS, effective_output_num=counts),
        torch.tensor([[0, 0], [9, 18], [9, 18]], device=device),
    )
