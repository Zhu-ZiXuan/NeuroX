"""Tests for unit-level input-axis block packing and phase accumulation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest
import torch

from neurox import Profiler, stamp_names
from neurox.architecture.mapping import (
    TilingMode,
)
from neurox.architecture.unit.conv2d import Conv2dCimUnit, Conv2dCimUnitConfig, Conv2dCimUnitPolicy
from neurox.encoding import Encoding
from neurox.primitive.digital import AccumulatorConfig, ShiftAdderConfig
from neurox.primitive.macro.cim import (
    CimMacroQuantizationScheme,
    IdealCimMacroConfig,
    IdealCimMacroPolicy,
)

# All units are built on IdealCimMacroConfig, so the embedded macro policy is
# the empty marker. `adc_active_bits = None` bypasses its virtual ADC.
_IDEAL_MACRO_POLICY = IdealCimMacroPolicy()
_QUANTIZATION_MODE = 0
_ADC_BITS = None


def _ideal_macro_config(
    *,
    input_num: int = 8,
    output_num: int = 8,
    max_active_num: int = 2,
) -> IdealCimMacroConfig:
    return IdealCimMacroConfig(
        input_num=input_num,
        rescale_factors=(1.0,),
        max_active_num=max_active_num,
        lane_num=1,
        scan_num=output_num,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
        w_digit_num=2,
        w_digit_radix=2,
        w_encoding=Encoding.TRUE_FORM,
        x_digit_num=1,
        x_digit_radix=2,
        x_encoding=Encoding.UNSIGNED,
        x_value_range=(0, 1),
        w_value_range=(-3, 3),
        adc_bits=8,
        quantization_scheme=CimMacroQuantizationScheme.ZERO_POINT,
    )


def _zero_ppa() -> dict[str, float]:
    return {
        "energy_per_op__fJ": 0.0,
        "latency_per_op__ns": 0.0,
        "leakage_per_inst__uW": 0.0,
        "area_per_inst__um2": 0.0,
    }


def _accumulator_config() -> AccumulatorConfig:
    return AccumulatorConfig(bit_width=32, **_zero_ppa())


def _shift_adder_config() -> ShiftAdderConfig:
    return ShiftAdderConfig(bit_width=32, **_zero_ppa())


def _unit_kwargs(w_logical_shape: tuple[int, ...], policy: Conv2dCimUnitPolicy) -> dict[str, Any]:
    return {
        "policy": policy,
        "w_logical_shape": (*w_logical_shape, 1, 1),
        "dtype": torch.float32,
    }


def _unit_policy() -> Conv2dCimUnitPolicy:
    return Conv2dCimUnitPolicy(
        cim_macro_policy=_IDEAL_MACRO_POLICY,
    )


def _build_direct(
    *,
    w_logical_shape: tuple[int, ...],
    input_num: int = 8,
    max_active_num: int = 2,
) -> Conv2dCimUnit:
    config = Conv2dCimUnitConfig(
        merge=True,
        cim_macro_config=_ideal_macro_config(input_num=input_num, max_active_num=max_active_num),
        phase_accumulator_config=_accumulator_config(),
        w_slice_num=1,
        w_slice_encoding=None,
        w_shift_adder_config=None,
        tiling=TilingMode.SLICE_PLANES,
        x_slice_num=1,
        x_slice_encoding=None,
        x_shift_adder_config=None,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
    )
    unit = Conv2dCimUnit(
        config=config,
        **_unit_kwargs(
            w_logical_shape,
            _unit_policy(),
        ),
    )
    unit.eval()
    return unit


def _sliced_unit_config(
    *,
    tiling: TilingMode,
    input_num: int,
    max_active_num: int,
) -> Conv2dCimUnitConfig:
    return Conv2dCimUnitConfig(
        merge=True,
        cim_macro_config=_ideal_macro_config(input_num=input_num, max_active_num=max_active_num),
        phase_accumulator_config=_accumulator_config(),
        w_slice_num=2,
        w_slice_encoding=Encoding.TRUE_FORM,
        tiling=tiling,
        w_shift_adder_config=_shift_adder_config(),
        x_slice_num=2,
        x_slice_encoding=Encoding.UNSIGNED,
        x_shift_adder_config=_shift_adder_config(),
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
    )


def _build_inter(
    *,
    w_logical_shape: tuple[int, ...],
    input_num: int = 8,
    max_active_num: int = 2,
) -> Conv2dCimUnit:
    config = _sliced_unit_config(
        input_num=input_num,
        max_active_num=max_active_num,
        tiling=TilingMode.SLICE_PLANES,
    )
    unit = Conv2dCimUnit(
        config=config,
        **_unit_kwargs(
            w_logical_shape,
            _unit_policy(),
        ),
    )
    unit.eval()
    return unit


def _build_intra(
    *,
    w_logical_shape: tuple[int, ...],
    input_num: int = 8,
    max_active_num: int = 2,
) -> Conv2dCimUnit:
    config = _sliced_unit_config(
        input_num=input_num,
        max_active_num=max_active_num,
        tiling=TilingMode.SLICE_OUTPUTS,
    )
    unit = Conv2dCimUnit(
        config=config,
        **_unit_kwargs(
            w_logical_shape,
            _unit_policy(),
        ),
    )
    unit.eval()
    return unit


def _randint_in_range(value_range: tuple[int, int], shape: tuple[int, ...]) -> torch.Tensor:
    lo, hi = value_range
    return torch.randint(lo, hi + 1, shape, dtype=torch.int32)


def _assert_unit_matches_torch(unit: Conv2dCimUnit, weight: torch.Tensor, activation: torch.Tensor) -> None:
    unit.program(weight[..., None, None])
    actual = unit._matmul(activation, quantization_mode=_QUANTIZATION_MODE, adc_active_bits=_ADC_BITS)
    expected = torch.matmul(activation.long(), weight.transpose(-1, -2).long())
    assert actual.shape == expected.shape
    assert torch.equal(actual.long(), expected)


@pytest.mark.parametrize("tiling", [TilingMode.SLICE_PLANES, TilingMode.SLICE_OUTPUTS])
@pytest.mark.parametrize("w_sliced", [False, True])
@pytest.mark.parametrize("x_sliced", [False, True])
@pytest.mark.parametrize("hardware_recovery", [False, True])
def test_weight_and_input_slicers_compose_independently(
    tiling: TilingMode,
    w_sliced: bool,
    x_sliced: bool,
    hardware_recovery: bool,
) -> None:
    torch.manual_seed(2)
    n = 17
    k = 19
    m = 5
    config = Conv2dCimUnitConfig(
        merge=True,
        cim_macro_config=_ideal_macro_config(max_active_num=3),
        phase_accumulator_config=_accumulator_config(),
        w_slice_num=2 if w_sliced else 1,
        w_slice_encoding=Encoding.TRUE_FORM if w_sliced else None,
        x_slice_num=2 if x_sliced else 1,
        x_slice_encoding=Encoding.UNSIGNED if x_sliced else None,
        tiling=tiling,
        w_shift_adder_config=_shift_adder_config() if w_sliced and hardware_recovery else None,
        x_shift_adder_config=_shift_adder_config() if x_sliced and hardware_recovery else None,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
    )
    unit = Conv2dCimUnit(
        config=config,
        policy=_unit_policy(),
        w_logical_shape=(n, k, 1, 1),
        dtype=torch.float32,
    )
    weight = _randint_in_range(unit.w_value_range, (n, k))
    activation = _randint_in_range(unit.x_value_range, (m, k))
    _assert_unit_matches_torch(unit, weight, activation)


def _macro_inputs(unit: Conv2dCimUnit, x: torch.Tensor) -> torch.Tensor:
    unit.program(torch.zeros(unit._w_logical_shape, dtype=torch.int32))
    stamp_names(unit)
    captured = []
    original = unit.cim_macro.vec_mat_mul

    def capture(value, **kwargs):
        captured.append(value)
        return original(value, **kwargs)

    with patch.object(unit.cim_macro, "vec_mat_mul", side_effect=capture):
        unit._matmul(x, quantization_mode=_QUANTIZATION_MODE, adc_active_bits=_ADC_BITS)
    return captured[-1]


def test_input_activation_and_block_routing_layout() -> None:
    """D and P land left of the instance-aligned block and partition inputs."""
    unit = _build_direct(w_logical_shape=(4, 8), input_num=8, max_active_num=2)
    p = unit.input_activation.input_phase_num
    values = torch.arange(1, 17, dtype=torch.int32).reshape(2, 8)
    planes = _macro_inputs(unit, values)
    x = values.view(2, 1, 1, 1, 1, 8)
    assert planes.shape == (1, p, 2, 1, 1, 1, 1, 8)
    # Masks partition the row axis: summing P restores the full plane.
    assert torch.equal(planes[0].sum(dim=0), x)
    # Zero-fill outside each phase's selected-input window.
    for phase in range(p):
        assert torch.equal(
            planes[0, phase],
            torch.where(torch.arange(8) // 2 == phase, x, torch.zeros_like(x)),
        )


def test_split_input_stages_keep_batch_axes_left_of_d_and_p() -> None:
    unit = _build_direct(w_logical_shape=(4, 8), input_num=8, max_active_num=2)
    values = torch.randint(0, 2, (3, 2, 8), dtype=torch.int32)
    planes = _macro_inputs(unit, values)
    x = values.view(3, 2, 1, 1, 1, 1, 8)
    assert planes.shape == (3, 1, 4, 2, 1, 1, 1, 1, 8)
    assert torch.equal(planes[:, 0].sum(dim=1), x)


def test_input_phase_non_divisible_ceil_covers_all_inputs() -> None:
    """input_num=10 and max_active_num=3 cover all inputs in four phases."""
    unit = _build_direct(w_logical_shape=(4, 10), input_num=10, max_active_num=3)
    assert unit.input_activation.input_phase_num == 4
    inputs = torch.arange(1, 11, dtype=torch.int32)
    phases = unit.input_activation.map_x(inputs)
    assert phases.shape == (4, 10)
    # Each input belongs to exactly one phase.
    assert torch.equal((phases != 0).sum(dim=0), torch.ones(10, dtype=torch.int64))
    assert torch.equal(phases.sum(dim=0), inputs)
    # The short final block owns only its single real row (row 9).
    assert torch.equal(phases[3], torch.tensor([0] * 9 + [10]))


def test_degenerate_input_phase_axis_size_one() -> None:
    """max_active_num == input_num retains a size-one phase axis."""
    unit = _build_direct(w_logical_shape=(4, 8), input_num=8, max_active_num=8)
    assert unit.input_activation.input_phase_num == 1
    values = torch.randint(0, 2, (2, 8), dtype=torch.int32)
    planes = _macro_inputs(unit, values)
    x = values.view(2, 1, 1, 1, 1, 8)
    assert planes.shape == (1, 1, 2, 1, 1, 1, 1, 8)
    assert torch.equal(planes[0, 0], x)


@pytest.mark.parametrize("build", [_build_direct, _build_inter, _build_intra])
def test_input_phase_count_skips_padding_only_blocks(build: Callable[..., Conv2dCimUnit]) -> None:
    """K < input_num omits phases holding only macro-input padding.

    input_num=8, max_active_num=2, K=3 gives two, not ceil(8/2)=4.
    """
    torch.manual_seed(3)
    n = 4
    k = 3  # k < input_num: one short block leaves input positions unused
    m = 3
    unit = build(w_logical_shape=(n, k), input_num=8, max_active_num=2)
    assert unit.input_activation.input_phase_num == 2
    weight = _randint_in_range(unit.w_value_range, (n, k))
    activation = _randint_in_range(unit.x_value_range, (m, k))
    _assert_unit_matches_torch(unit, weight, activation)


@pytest.mark.parametrize("build", [_build_direct, _build_inter, _build_intra])
def test_conv_matmul_parity_with_input_phases(build: Callable[..., Conv2dCimUnit]) -> None:
    torch.manual_seed(7)
    n = 5
    k = 10  # k > input_num exercises Tc tiling alongside P
    m = 3
    unit = build(w_logical_shape=(n, k), input_num=8, max_active_num=2)
    assert unit.input_activation.input_phase_num == 4
    weight = _randint_in_range(unit.w_value_range, (n, k))
    activation = _randint_in_range(unit.x_value_range, (m, k))
    _assert_unit_matches_torch(unit, weight, activation)


@pytest.mark.parametrize("build", [_build_direct, _build_inter, _build_intra])
def test_conv_matmul_parity_non_divisible(build: Callable[..., Conv2dCimUnit]) -> None:
    """Non-divisible geometry still equals the oracle: P from a ceiling division covers every input, row 9 included."""
    torch.manual_seed(9)
    n = 5
    k = 10  # k == input_num: all 10 positions carry real weight
    m = 3
    unit = build(w_logical_shape=(n, k), input_num=10, max_active_num=3)
    assert unit.input_activation.input_phase_num == 4
    weight = _randint_in_range(unit.w_value_range, (n, k))
    activation = _randint_in_range(unit.x_value_range, (m, k))
    _assert_unit_matches_torch(unit, weight, activation)


# --- Balanced input-axis block packing ---


@pytest.mark.parametrize("build", [_build_direct, _build_inter, _build_intra])
def test_balanced_block_packing_matches_torch(build: Callable[..., Conv2dCimUnit]) -> None:
    """Five or ten logical output blocks are balanced over physical macros."""
    torch.manual_seed(13)
    n = 40
    k = 3
    m = 5
    unit = build(w_logical_shape=(n, k), input_num=8, max_active_num=2)
    assert unit.merge.input_slot_capacity == 2
    assert unit.merge.merge_step_num == 2
    weight = _randint_in_range(unit.w_value_range, (n, k))
    activation = _randint_in_range(unit.x_value_range, (m, k))
    _assert_unit_matches_torch(unit, weight, activation)


def test_direct_block_placement_is_balanced_and_zero_padded() -> None:
    """B=5 and C=2 use G=3 macros with a 2+2+1 balanced assignment."""
    unit = _build_direct(w_logical_shape=(40, 3), input_num=8, max_active_num=2)
    assert unit.merge.out_tile_num == 5
    assert unit.merge.input_slot_capacity == 2
    assert unit.merge.macro_group_num == 3
    assert unit.merge.merge_step_num == 2

    weight = torch.arange(1, 121, dtype=torch.int32).reshape(40, 3)
    unit.program(weight[..., None, None])
    programmed = unit.cim_macro._w
    assert isinstance(programmed, torch.Tensor)
    assert programmed.shape == (1, 1, 1, 1, 3, 8, 8)

    # block_id = block_step * G + macro_index
    for block_step in range(2):
        for macro_index in range(3):
            block_id = block_step * 3 + macro_index
            rows = slice(block_step * 3, (block_step + 1) * 3)
            actual = programmed[0, 0, 0, 0, macro_index, rows]
            if block_id < 5:
                expected = weight[block_id * 8 : (block_id + 1) * 8].transpose(0, 1)
                assert torch.equal(actual, expected)
            else:
                assert torch.count_nonzero(actual) == 0


def test_block_schedule_routes_input_to_each_slot() -> None:
    unit = _build_direct(w_logical_shape=(40, 3), input_num=8, max_active_num=2)
    routed = _macro_inputs(unit, torch.tensor([[2, 3, 5]], dtype=torch.int32))
    assert routed.shape == (2, 2, 1, 1, 1, 1, 3, 8)

    # Phase reduction restores x in each block slot.
    restored = routed.sum(dim=1)
    assert torch.equal(restored[0, 0, 0, 0, 0, 0], torch.tensor([2, 3, 5, 0, 0, 0, 0, 0]))
    assert torch.equal(restored[1, 0, 0, 0, 0, 0], torch.tensor([0, 0, 0, 2, 3, 5, 0, 0]))
    assert torch.equal(restored[1, 0, 0, 0, 0, 1], torch.tensor([0, 0, 0, 2, 3, 5, 0, 0]))
    assert torch.count_nonzero(restored[1, 0, 0, 0, 0, 2]) == 0


def test_large_balanced_case_uses_seventeen_plus_sixteen() -> None:
    unit = Conv2dCimUnit(
        config=Conv2dCimUnitConfig(
            merge=True,
            cim_macro_config=_ideal_macro_config(input_num=32, output_num=1, max_active_num=1),
            phase_accumulator_config=_accumulator_config(),
            w_slice_num=1,
            w_slice_encoding=None,
            w_shift_adder_config=None,
            tiling=TilingMode.SLICE_PLANES,
            x_slice_num=1,
            x_slice_encoding=None,
            x_shift_adder_config=None,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
        ),
        policy=_unit_policy(),
        w_logical_shape=(33, 1, 1, 1),
        dtype=torch.float32,
    )
    assert unit.merge.input_slot_capacity == 32
    assert unit.merge.macro_group_num == 2
    assert unit.merge.merge_step_num == 17
    assert unit.cim_macro.inst_shape == (1, 1, 1, 1, 2)


# --- Caller prefix under leading-resolved profiling ---

_ACTIVATION_BATCH = (3, 1)  # a two-axis caller prefix
_M = 3


def test_caller_prefix_stays_leftmost() -> None:
    torch.manual_seed(11)
    n = 40
    k = 3
    unit = _build_direct(w_logical_shape=(n, k), input_num=8, max_active_num=2)
    unit.program(_randint_in_range(unit.w_value_range, (n, k))[..., None, None])
    activation = _randint_in_range(unit.x_value_range, (*_ACTIVATION_BATCH, _M, k))
    with Profiler(leading_rank=len(_ACTIVATION_BATCH)):
        routed = _macro_inputs(unit, activation)
    d = unit.merge.merge_step_num
    p = unit.input_activation.input_phase_num
    # The caller prefix stays the leftmost contiguous block.
    # Shape: [3, 1, D, P, ...]
    assert routed.shape[:4] == (*_ACTIVATION_BATCH, d, p)
