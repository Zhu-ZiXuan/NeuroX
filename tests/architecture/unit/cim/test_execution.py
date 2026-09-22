"""Shared CIM mapping preserves VMMs and accounts for enabled digital operations."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from neurox import Profiler
from neurox.encoding import Encoding
from neurox.primitive.digital import AccumulatorConfig, AdderConfig, RadixAccumulatorConfig, RadixSummatorConfig

from ._utils import build_unit, unit_config


@pytest.mark.parametrize("differential", [False, True])
@pytest.mark.parametrize(
    ("matrix_shape", "w_slices", "x_slices", "merge", "weight_circuit", "lanes", "scans"),
    [
        ((3, 5), 1, 1, False, False, 1, 4),
        ((9, 3), 2, 1, True, False, 1, 4),
        ((9, 17), 1, 3, False, False, 1, 4),
        ((9, 3), 2, 3, True, True, 1, 4),
        ((5, 17), 2, 3, False, True, 1, 4),
        ((5, 7), 2, 3, False, True, 1, 5),
        ((5, 3), 2, 3, True, True, 3, 1),
        ((5, 3), 2, 3, True, True, 3, 2),
    ],
)
def test_vmm_preserves_products_across_slicing_tiling_and_merge(
    matrix_shape: tuple[int, int],
    w_slices: int,
    x_slices: int,
    merge: bool,
    weight_circuit: bool,
    lanes: int,
    scans: int,
    differential: bool,
    device: torch.device,
) -> None:
    config = unit_config()
    macro = config.cim_macro_config
    config = replace(
        config,
        cim_macro_config=replace(
            macro,
            lane_num=lanes,
            scan_num=scans,
            x_digit_radix=3,
            x_value_range=(0, 2),
            w_encoding=Encoding.UNSIGNED if differential else macro.w_enc,
            w_signed=not differential,
            w_value_range=(0, 3) if differential else macro.w_value_range,
        ),
        w_slice_num=w_slices,
        x_slice_num=x_slices,
        merge=merge,
        w_radix_summator_config=config.w_radix_summator_config if weight_circuit else None,
    )
    unit = build_unit(config, matrix_shape=matrix_shape).to(device)
    output_num, input_num = matrix_shape
    w_lo, w_hi = unit.w_value_range
    x_lo, x_hi = unit.x_value_range
    weight = torch.arange(output_num * input_num, dtype=torch.int32, device=device).reshape(matrix_shape)
    weight = weight % (w_hi - w_lo + 1) + w_lo
    # A noncontiguous caller layout exposes accidental assumptions about leading axes.
    x = torch.arange(6 * input_num, dtype=torch.int32, device=device).reshape(2, input_num, 3).transpose(-1, -2)
    x = x % (x_hi - x_lo + 1) + x_lo
    unit._program_matrix(weight.unsqueeze(0))
    actual = unit._vmm(x.unsqueeze(-2), quantization_mode=0, adc_active_bits=None).squeeze(-2)
    expected = (x.cpu().long() @ weight.cpu().long().T).to(device)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("physical_outputs", [4, 5])
@pytest.mark.parametrize("merge", [False, True])
def test_programming_keeps_each_polarity_pair_adjacent_inside_its_macro_and_input_slot(
    physical_outputs: int, merge: bool, monkeypatch: pytest.MonkeyPatch, device: torch.device
) -> None:
    config = unit_config()
    config = replace(
        config,
        cim_macro_config=replace(
            config.cim_macro_config,
            input_num=6,
            scan_num=physical_outputs,
            w_encoding=Encoding.TRUE_FORM,
            # The declared hardware capability controls pairing independently of encoding.
            w_signed=False,
            w_value_range=(0, 3),
        ),
        w_slice_num=1,
        merge=merge,
    )
    unit = build_unit(config, matrix_shape=(5, 3)).to(device)
    programmed: list[torch.Tensor] = []
    counts: list[torch.Tensor] = []
    original_program = unit.cim_macro.program
    original_vmm = unit.cim_macro.vec_mat_mul

    def program(weight: torch.Tensor) -> None:
        programmed.append(weight.detach().clone())
        original_program(weight)

    def vmm(
        x: torch.Tensor, *, quantization_mode: int, adc_active_bits: int | None, effective_output_num: torch.Tensor
    ) -> torch.Tensor:
        counts.append(effective_output_num.detach().clone())
        return original_vmm(
            x,
            quantization_mode=quantization_mode,
            adc_active_bits=adc_active_bits,
            effective_output_num=effective_output_num,
        )

    monkeypatch.setattr(unit.cim_macro, "program", program)
    monkeypatch.setattr(unit.cim_macro, "vec_mat_mul", vmm)
    weight = torch.tensor([[-1, 2, -3], [3, -2, 1], [2, 0, -1], [-3, -1, 2], [1, -3, 3]], device=device)
    unit._program_matrix(weight.unsqueeze(0))
    # Five signed outputs fill three two-pair tiles. Sharing puts the last
    # tile in the second input slot of the first macro, without splitting a pair.
    macro_num = 2 if merge else 3
    expected = torch.zeros((macro_num, 6, physical_outputs), dtype=weight.dtype, device=device)
    expected[0, :3, :4] = torch.tensor([[0, 1, 3, 0], [2, 0, 0, 2], [0, 3, 1, 0]], device=device)
    expected[1, :3, :4] = torch.tensor([[2, 0, 0, 3], [0, 0, 0, 1], [0, 1, 2, 0]], device=device)
    if merge:
        expected[0, 3:, :2] = torch.tensor([[1, 0], [0, 3], [3, 0]], device=device)
    else:
        expected[2, :3, :2] = torch.tensor([[1, 0], [0, 3], [3, 0]], device=device)
    torch.testing.assert_close(programmed[-1].reshape_as(expected), expected)

    x = torch.tensor([[1, 2, 3], [0, 1, 1]], device=device)
    actual = unit._vmm(x.unsqueeze(-2), quantization_mode=0, adc_active_bits=None).squeeze(-2)
    torch.testing.assert_close(actual, (x.cpu() @ weight.cpu().T).to(device))
    expected_counts = [[4, 4], [2, 0]] if merge else [[4, 4, 2]]
    torch.testing.assert_close(counts[-1], torch.tensor(expected_counts, device=device))


@pytest.mark.parametrize(
    ("encoding", "slice_num", "expected_range"),
    [
        (None, 1, (0, 3)),
        (Encoding.UNSIGNED, 2, (0, 15)),
        (Encoding.TRUE_FORM, 1, (-3, 3)),
        (Encoding.TRUE_FORM, 2, (-15, 15)),
        (Encoding.CANONICAL, 2, (-12, 12)),
    ],
)
def test_nonnegative_macros_support_requested_logical_weight_encoding(
    encoding: Encoding | None,
    slice_num: int,
    expected_range: tuple[int, int],
    device: torch.device,
) -> None:
    config = unit_config()
    config = replace(
        config,
        cim_macro_config=replace(
            config.cim_macro_config, w_encoding=Encoding.UNSIGNED, w_signed=False, w_value_range=(0, 3)
        ),
        w_slice_num=slice_num,
        w_slice_encoding=encoding,
        x_slice_num=1,
    )
    # Enumerate the complete accepted interval, including both signed extremes.
    weight = torch.arange(expected_range[0], expected_range[1] + 1, dtype=torch.int32, device=device).unsqueeze(-1)
    unit = build_unit(config, matrix_shape=tuple(weight.shape)).to(device)
    assert unit.w_value_range == expected_range
    unit._program_matrix(weight.unsqueeze(0))
    actual = unit._vmm(
        torch.ones((1, 1), dtype=torch.int32, device=device).unsqueeze(-2), quantization_mode=0, adc_active_bits=None
    ).squeeze(-2)
    torch.testing.assert_close(actual, weight.T.long())


@pytest.mark.parametrize("radix", [2, 3, 8])
@pytest.mark.parametrize("slices", [2, 3])
def test_transcoder_metadata_controls_pairing_and_both_operand_signs(
    radix: int, slices: int, device: torch.device, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = unit_config()
    config = replace(
        config,
        cim_macro_config=replace(
            config.cim_macro_config,
            # The actual carrier can be narrower or wider than encoding metadata.
            w_encoding=Encoding.UNSIGNED,
            w_signed=False,
            w_value_range=(0, radix - 1),
            x_value_range=(-1, 1),
        ),
        w_slice_num=slices,
        w_slice_encoding=Encoding.TRUE_FORM,
        x_slice_num=2,
        x_slice_encoding=Encoding.TRUE_FORM,
        w_polarity_adder_config=AdderConfig(
            bit_width=32, energy_per_op__fJ=0.0, area_per_inst__um2=0.0, leakage_per_inst__uW=0.0
        ),
    )
    unit = build_unit(config, matrix_shape=(4, 5)).to(device)
    assert unit.cim_macro.inst_count == slices * 2
    hi = radix**slices - 1
    lo = -hi
    weight = torch.tensor([lo, -1, 0, hi], dtype=torch.int64, device=device).repeat(5).reshape(4, 5)
    x = torch.tensor([[-2, -1, 0, 1, -2], [1, 0, -1, -2, 1]], dtype=torch.int64, device=device)
    original_program = unit.cim_macro.program

    def program(physical_weight: torch.Tensor) -> None:
        assert physical_weight.min() >= 0
        assert physical_weight.max() < radix
        original_program(physical_weight)

    monkeypatch.setattr(unit.cim_macro, "program", program)
    unit._program_matrix(weight.unsqueeze(0))
    actual = unit._vmm(x.unsqueeze(-2), quantization_mode=0, adc_active_bits=None).squeeze(-2)
    torch.testing.assert_close(actual, (x.cpu() @ weight.cpu().T).to(device))


@pytest.mark.parametrize("modeled", [False, True])
def test_single_signed_slice_uses_local_polarity_recovery(modeled: bool, device: torch.device) -> None:
    config = unit_config()
    config = replace(
        config,
        cim_macro_config=replace(
            config.cim_macro_config, w_encoding=Encoding.UNSIGNED, w_signed=False, w_value_range=(0, 1)
        ),
        w_slice_num=1,
        w_slice_encoding=Encoding.TRUE_FORM,
        x_slice_num=1,
        w_polarity_adder_config=AdderConfig(
            bit_width=32, energy_per_op__fJ=2.0, area_per_inst__um2=0.0, leakage_per_inst__uW=0.0
        )
        if modeled
        else None,
    )
    unit = build_unit(config, matrix_shape=(1, 1)).to(device)
    unit._program_matrix(torch.tensor([[-1]], device=device).unsqueeze(0))
    unit.set_profile_leading_rank(1)
    profiler = Profiler(concat_dim=0, sync_device=device)
    profiler.collect_static_data(unit)
    with profiler:
        actual = unit._vmm(
            torch.tensor([[1], [0]], device=device).unsqueeze(-2), quantization_mode=0, adc_active_bits=None
        ).squeeze(-2)
    torch.testing.assert_close(actual, torch.tensor([[-1], [0]], device=device))
    assert unit.cim_macro.inst_count == 1
    assert unit._vmm_global_latency__ns() == 2.0
    if modeled:
        item = profiler.result[unit.w_polarity_adder.qualified_name]
        torch.testing.assert_close(item.dynamic_energy__fJ, torch.full((2,), 2.0, device=device), check_dtype=False)


def test_polarity_codes_are_quantized_before_difference_and_local_recovery(device: torch.device) -> None:
    config = unit_config()
    config = replace(
        config,
        cim_macro_config=replace(
            config.cim_macro_config,
            w_encoding=Encoding.UNSIGNED,
            w_signed=False,
            w_value_range=(0, 3),
            rescale_factors=(2.0,),
            adc_bits=6,
        ),
        w_slice_num=1,
        x_slice_num=2,
    )
    unit = build_unit(config, matrix_shape=(1, 4)).to(device)
    unit._program_matrix(torch.tensor([[2, -1, 2, -1]], dtype=torch.int32, device=device).unsqueeze(0))
    x = torch.full((1, 4), 3, dtype=torch.int32, device=device)
    # Each phase/bit produces polarity codes 1 and 0, reconstructing to 6.
    actual = unit._vmm(x.unsqueeze(-2), quantization_mode=0, adc_active_bits=6).squeeze(-2)
    torch.testing.assert_close(actual, torch.tensor([[6]], device=device))


def test_polarity_register_wrap_precedes_global_tile_accumulation(device: torch.device) -> None:
    config = unit_config()
    config = replace(
        config,
        cim_macro_config=replace(
            config.cim_macro_config,
            input_num=3,
            max_active_num=3,
            w_encoding=Encoding.UNSIGNED,
            w_signed=False,
            w_value_range=(0, 3),
        ),
        w_slice_num=1,
        x_slice_num=1,
        w_polarity_adder_config=AdderConfig(
            bit_width=3, energy_per_op__fJ=0.0, area_per_inst__um2=0.0, leakage_per_inst__uW=0.0
        ),
    )
    unit = build_unit(config, matrix_shape=(1, 6)).to(device)
    unit._program_matrix(torch.tensor([[3, 3, -1, 3, 3, -1]], dtype=torch.int32, device=device).unsqueeze(0))
    actual = unit._vmm(
        torch.ones((1, 6), dtype=torch.int32, device=device).unsqueeze(-2), quantization_mode=0, adc_active_bits=None
    ).squeeze(-2)
    # Each tile's difference of five wraps to -3 before the wide global sum.
    torch.testing.assert_close(actual, torch.tensor([[-6]], device=device))


@pytest.mark.parametrize(
    ("lane_num", "scan_num", "macro_num", "local_circuits"), [(1, 4, 5, 5), (2, 2, 5, 5), (3, 2, 3, 6), (5, 1, 5, 10)]
)
def test_adjacent_polarity_mapping_bills_macro_capacity_and_local_pair_throughput(
    lane_num: int, scan_num: int, macro_num: int, local_circuits: int, device: torch.device
) -> None:
    config = unit_config()
    config = replace(
        config,
        cim_macro_config=replace(
            config.cim_macro_config,
            w_encoding=Encoding.UNSIGNED,
            w_signed=False,
            w_value_range=(0, 3),
            lane_num=lane_num,
            scan_num=scan_num,
            area_per_inst__um2=11.0,
            leakage_per_inst__uW=13.0,
        ),
        w_polarity_adder_config=AdderConfig(
            bit_width=32, energy_per_op__fJ=4.0, area_per_inst__um2=5.0, leakage_per_inst__uW=7.0
        ),
        local_accumulator_config=RadixAccumulatorConfig(
            bit_width=32, energy_per_op__fJ=3.0, area_per_inst__um2=7.0, leakage_per_inst__uW=2.0
        ),
    )
    unit = build_unit(config, matrix_shape=(9, 3)).to(device)
    unit._program_matrix(torch.ones((9, 3), dtype=torch.int32, device=device).unsqueeze(0))
    x = torch.ones((2, 3, 3), dtype=torch.int32, device=device)
    x[1] = 0
    unit.set_profile_leading_rank(2)
    profiler = Profiler(concat_dim=0, sync_device=device)
    profiler.collect_static_data(unit)
    with profiler:
        actual = unit._vmm(x.unsqueeze(-2), quantization_mode=0, adc_active_bits=None).squeeze(-2)
    torch.testing.assert_close(actual, x.long().sum(-1, keepdim=True).expand(2, 3, 9))
    items = profiler.result
    difference = items[unit.w_polarity_adder.qualified_name]
    local = items[unit.local_accumulator.qualified_name]
    # Eighteen slice outputs, six accesses, one operation per difference.
    # Both the all-zero negative terms and zero input sample remain enabled; padding does not.
    torch.testing.assert_close(
        difference.dynamic_energy__fJ, torch.full((2, 3), 432.0, device=device), check_dtype=False
    )
    torch.testing.assert_close(local.dynamic_energy__fJ, torch.full((2, 3), 486.0, device=device), check_dtype=False)
    assert (difference.area__um2, difference.leakage__uW) == (5.0 * local_circuits, 7.0 * local_circuits)
    assert (local.area__um2, local.leakage__uW) == (7.0 * local_circuits, 2.0 * local_circuits)
    # Complete polarity pairs determine macro capacity, including an odd unused port.
    macro_item = items[unit.cim_macro.qualified_name]
    assert (macro_item.area__um2, macro_item.leakage__uW) == (11.0 * macro_num, 13.0 * macro_num)


def test_phase_results_are_quantized_before_recovery(device: torch.device) -> None:
    config = unit_config()
    config = replace(
        config,
        cim_macro_config=replace(config.cim_macro_config, max_active_num=1, rescale_factors=(2.0,), adc_bits=6),
        w_slice_num=1,
        x_slice_num=1,
    )
    unit = build_unit(config, matrix_shape=(2, 3)).to(device)
    weight = torch.ones((2, 3), dtype=torch.int32, device=device)
    weight[1] = -1
    unit._program_matrix(weight.unsqueeze(0))
    x = torch.ones((1, 3), dtype=torch.int32, device=device)
    # Each access supplies +1 or -1, quantized to 0 or -1 before summation.
    actual = unit._vmm(x.unsqueeze(-2), quantization_mode=0, adc_active_bits=6).squeeze(-2)
    torch.testing.assert_close(actual, torch.tensor([[0, -3]], device=device))


def test_tile_recovery_applies_register_width_and_gates_padding(device: torch.device) -> None:
    config = replace(
        unit_config(),
        w_slice_num=1,
        x_slice_num=1,
        tile_accumulator_config=AccumulatorConfig(
            bit_width=4, energy_per_op__fJ=2.0, area_per_inst__um2=5.0, leakage_per_inst__uW=3.0
        ),
    )
    unit = build_unit(config, matrix_shape=(9, 17)).to(device)
    unit._program_matrix(torch.ones((9, 17), dtype=torch.int32, device=device).unsqueeze(0))
    x = torch.ones((2, 5, 17), dtype=torch.int32, device=device)
    x[1] = 0
    unit.set_profile_leading_rank(2)
    profiler = Profiler(concat_dim=0, sync_device=device)
    profiler.collect_static_data(unit)
    with profiler:
        actual = unit._vmm(x.unsqueeze(-2), quantization_mode=0, adc_active_bits=None).squeeze(-2)
    # Seventeen wraps to one at the four-bit tile accumulator's output.
    expected = torch.ones((2, 5, 9), dtype=torch.int64, device=device)
    expected[1] = 0
    torch.testing.assert_close(actual, expected)
    item = profiler.result[unit.tile_accumulator.qualified_name]
    # Three input tiles each supply nine enabled outputs, including zero values.
    # The three padding ports occupy hardware but incur no evaluation energy.
    torch.testing.assert_close(item.dynamic_energy__fJ, torch.full((2, 5), 54.0, device=device), check_dtype=False)
    assert (item.area__um2, item.leakage__uW) == (60.0, 36.0)


def test_global_circuits_bill_each_tile_update_and_one_weight_recovery(device: torch.device) -> None:
    config = unit_config()
    config = replace(
        config,
        cim_macro_config=replace(config.cim_macro_config, input_num=1, max_active_num=1, x_value_range=(0, 3)),
        x_slice_num=1,
        w_radix_summator_config=RadixSummatorConfig(
            bit_width=32, energy_per_op__fJ=2.0, area_per_inst__um2=3.0, leakage_per_inst__uW=5.0
        ),
        tile_accumulator_config=AccumulatorConfig(
            bit_width=32, energy_per_op__fJ=3.0, area_per_inst__um2=7.0, leakage_per_inst__uW=11.0
        ),
    )
    unit = build_unit(config, matrix_shape=(3, 2)).to(device)
    unit._program_matrix(torch.full((3, 2), 10, dtype=torch.int32, device=device).unsqueeze(0))
    x = torch.full((2, 5, 2), 2, dtype=torch.int32, device=device)
    unit.set_profile_leading_rank(2)
    profiler = Profiler(concat_dim=0, sync_device=device)
    profiler.collect_static_data(unit)
    with profiler:
        actual = unit._vmm(x.unsqueeze(-2), quantization_mode=0, adc_active_bits=None).squeeze(-2)
    torch.testing.assert_close(actual, torch.full((2, 5, 3), 40, dtype=torch.int64, device=device))
    weight_item = profiler.result[unit.w_radix_summator.qualified_name]
    tile_item = profiler.result[unit.tile_accumulator.qualified_name]
    # Six slice outputs receive two tile updates; the final reconstruction
    # processes their six digits once. Two padded ports remain disabled.
    torch.testing.assert_close(
        weight_item.dynamic_energy__fJ, torch.full((2, 5), 12.0, device=device), check_dtype=False
    )
    torch.testing.assert_close(tile_item.dynamic_energy__fJ, torch.full((2, 5), 36.0, device=device), check_dtype=False)
    assert (weight_item.area__um2, weight_item.leakage__uW) == (9.0, 15.0)
    assert (tile_item.area__um2, tile_item.leakage__uW) == (56.0, 88.0)


def test_local_recovery_bills_valid_updates_and_allocates_one_circuit_per_lane(device: torch.device) -> None:
    config = unit_config()
    config = replace(
        config,
        cim_macro_config=replace(config.cim_macro_config, lane_num=2, scan_num=2),
        local_accumulator_config=RadixAccumulatorConfig(
            bit_width=32, energy_per_op__fJ=3.0, area_per_inst__um2=7.0, leakage_per_inst__uW=2.0
        ),
    )
    unit = build_unit(config, matrix_shape=(9, 3)).to(device)
    unit._program_matrix(torch.ones((9, 3), dtype=torch.int32, device=device).unsqueeze(0))
    unit.set_profile_leading_rank(2)
    profiler = Profiler(concat_dim=0, sync_device=device)
    profiler.collect_static_data(unit)
    with profiler:
        actual = unit._vmm(
            torch.ones((2, 3, 3), dtype=torch.int32, device=device).unsqueeze(-2),
            quantization_mode=0,
            adc_active_bits=None,
        ).squeeze(-2)
    torch.testing.assert_close(actual, torch.full((2, 3, 9), 3, dtype=torch.int64, device=device))
    # Each of eighteen slice outputs has six phase updates and three weighted
    # updates. Output padding and empty reuse slots cause no updates.
    item = profiler.result[unit.local_accumulator.qualified_name]
    torch.testing.assert_close(item.dynamic_energy__fJ, torch.full((2, 3), 486.0, device=device), check_dtype=False)
    # Three macros each have two lanes; serial phases and input slices add no silicon.
    assert (item.area__um2, item.leakage__uW) == (42.0, 12.0)


@pytest.mark.parametrize("encoding", [None, *Encoding])
def test_direct_slices_ignore_encoding_recovery_costs_and_register_width(
    encoding: Encoding | None, device: torch.device
) -> None:
    config = replace(
        unit_config(),
        w_slice_num=1,
        w_slice_encoding=encoding,
        x_slice_num=1,
        x_slice_encoding=encoding,
        w_radix_summator_config=RadixSummatorConfig(
            bit_width=4, energy_per_op__fJ=3.0, area_per_inst__um2=5.0, leakage_per_inst__uW=7.0
        ),
    )
    unit = build_unit(config, matrix_shape=(5, 4)).to(device)
    assert unit.w_value_range == config.cim_macro_config.w_value_range
    assert unit.x_value_range == config.cim_macro_config.x_value_range
    unit._program_matrix(torch.full((5, 4), 3, dtype=torch.int32, device=device).unsqueeze(0))
    unit.set_profile_leading_rank(1)
    profiler = Profiler(concat_dim=0, sync_device=device)
    profiler.collect_static_data(unit)
    with profiler:
        actual = unit._vmm(
            torch.ones((2, 4), dtype=torch.int32, device=device).unsqueeze(-2),
            quantization_mode=0,
            adc_active_bits=None,
        ).squeeze(-2)
    torch.testing.assert_close(actual, torch.full((2, 5), 12, dtype=torch.int64, device=device))
    assert sum(item.area__um2 or 0.0 for item in profiler.result.values()) == 0.0
    assert sum(item.leakage__uW or 0.0 for item in profiler.result.values()) == 0.0
    assert all(
        item.dynamic_energy__fJ is None or not item.dynamic_energy__fJ.any() for item in profiler.result.values()
    )
