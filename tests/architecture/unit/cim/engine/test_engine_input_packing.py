"""Tests for engine-level input-axis block packing and phase accumulation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from neurox import Profiler
from neurox.architecture.unit.cim.engine import (
    CimEngine,
    CimEngineConfig,
    CimEnginePolicy,
    DirectWeightSliceStageConfig,
    DirectWeightSliceStagePolicy,
    DirectXSliceStageConfig,
    DirectXSliceStagePolicy,
    InputActivationStageConfig,
    InputActivationStagePolicy,
    InterWeightSliceStageConfig,
    InterWeightSliceStagePolicy,
    IntraWeightSliceStageConfig,
    IntraWeightSliceStagePolicy,
    PlacementStageConfig,
    PlacementStagePolicy,
    SerialXSliceStageConfig,
    SerialXSliceStagePolicy,
    WeightSliceStageConfig,
    WeightSliceStagePolicy,
    XSliceStagePolicy,
)
from neurox.encoding import Encoding
from neurox.primitive.digital import AccumulatorConfig, ShiftAdderConfig
from neurox.primitive.macro.cim import (
    CimMacroQuantizationScheme,
    IdealCimMacroConfig,
    IdealCimMacroPolicy,
)

# All engines are built on IdealCimMacroConfig, so the embedded macro policy is
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


def _engine_kwargs(w_logical_shape: tuple[int, ...], policy: CimEnginePolicy) -> dict[str, Any]:
    return {
        "policy": policy,
        "w_logical_shape": w_logical_shape,
        "dtype": torch.float32,
        "ideal_macro": False,
    }


def _placement_config() -> PlacementStageConfig:
    return PlacementStageConfig(
        contraction_accumulator_config=_accumulator_config(),
    )


def _input_activation_config() -> InputActivationStageConfig:
    return InputActivationStageConfig(
        phase_accumulator_config=_accumulator_config(),
    )


def _engine_policy(
    *,
    weight_slice: WeightSliceStagePolicy,
    x_slice: XSliceStagePolicy,
) -> CimEnginePolicy:
    return CimEnginePolicy(
        cim_macro_policy=_IDEAL_MACRO_POLICY,
        placement=PlacementStagePolicy(),
        input_activation=InputActivationStagePolicy(),
        weight_slice=weight_slice,
        x_slice=x_slice,
    )


def _build_direct(
    *,
    w_logical_shape: tuple[int, ...],
    input_num: int = 8,
    max_active_num: int = 2,
) -> CimEngine:
    config = CimEngineConfig(
        cim_macro_config=_ideal_macro_config(input_num=input_num, max_active_num=max_active_num),
        placement=_placement_config(),
        input_activation=_input_activation_config(),
        weight_slice=DirectWeightSliceStageConfig(),
        x_slice=DirectXSliceStageConfig(),
    )
    engine = CimEngine(
        config=config,
        **_engine_kwargs(
            w_logical_shape,
            _engine_policy(
                weight_slice=DirectWeightSliceStagePolicy(),
                x_slice=DirectXSliceStagePolicy(),
            ),
        ),
    )
    engine.eval()
    return engine


def _sliced_engine_config(
    *,
    input_num: int,
    max_active_num: int,
    weight_slice: WeightSliceStageConfig,
) -> CimEngineConfig:
    return CimEngineConfig(
        cim_macro_config=_ideal_macro_config(input_num=input_num, max_active_num=max_active_num),
        placement=_placement_config(),
        input_activation=_input_activation_config(),
        weight_slice=weight_slice,
        x_slice=SerialXSliceStageConfig(
            x_slice_num=2,
            shift_adder_config=_shift_adder_config(),
        ),
    )


def _inter_weight_slice_config() -> InterWeightSliceStageConfig:
    return InterWeightSliceStageConfig(
        w_slice_num=2,
        w_encoding=Encoding.TRUE_FORM,
        shift_adder_config=_shift_adder_config(),
    )


def _intra_weight_slice_config() -> IntraWeightSliceStageConfig:
    return IntraWeightSliceStageConfig(
        w_slice_num=2,
        w_encoding=Encoding.TRUE_FORM,
        shift_adder_config=_shift_adder_config(),
    )


def _build_inter(
    *,
    w_logical_shape: tuple[int, ...],
    input_num: int = 8,
    max_active_num: int = 2,
) -> CimEngine:
    config = _sliced_engine_config(
        input_num=input_num,
        max_active_num=max_active_num,
        weight_slice=_inter_weight_slice_config(),
    )
    engine = CimEngine(
        config=config,
        **_engine_kwargs(
            w_logical_shape,
            _engine_policy(
                weight_slice=InterWeightSliceStagePolicy(),
                x_slice=SerialXSliceStagePolicy(),
            ),
        ),
    )
    engine.eval()
    return engine


def _build_intra(
    *,
    w_logical_shape: tuple[int, ...],
    input_num: int = 8,
    max_active_num: int = 2,
) -> CimEngine:
    config = _sliced_engine_config(
        input_num=input_num,
        max_active_num=max_active_num,
        weight_slice=_intra_weight_slice_config(),
    )
    engine = CimEngine(
        config=config,
        **_engine_kwargs(
            w_logical_shape,
            _engine_policy(
                weight_slice=IntraWeightSliceStagePolicy(),
                x_slice=SerialXSliceStagePolicy(),
            ),
        ),
    )
    engine.eval()
    return engine


def _randint_in_range(value_range: tuple[int, int], shape: tuple[int, ...]) -> torch.Tensor:
    lo, hi = value_range
    return torch.randint(lo, hi + 1, shape, dtype=torch.int32)


def _assert_engine_matches_torch(engine: CimEngine, weight: torch.Tensor, activation: torch.Tensor) -> None:
    engine.program(weight)
    actual = engine.matmul(activation, quantization_mode=_QUANTIZATION_MODE, adc_active_bits=_ADC_BITS)
    expected = torch.matmul(activation.long(), weight.transpose(-1, -2).long())
    assert actual.shape == expected.shape
    assert torch.equal(actual.long(), expected)


@pytest.mark.parametrize(
    ("weight_slice", "weight_policy"),
    [
        (DirectWeightSliceStageConfig(), DirectWeightSliceStagePolicy()),
        (_inter_weight_slice_config(), InterWeightSliceStagePolicy()),
        (_intra_weight_slice_config(), IntraWeightSliceStagePolicy()),
    ],
)
@pytest.mark.parametrize(
    ("x_slice", "x_policy"),
    [
        (DirectXSliceStageConfig(), DirectXSliceStagePolicy()),
        (
            SerialXSliceStageConfig(
                x_slice_num=2,
                shift_adder_config=_shift_adder_config(),
            ),
            SerialXSliceStagePolicy(),
        ),
    ],
)
def test_weight_and_input_slice_stages_compose_independently(
    weight_slice: WeightSliceStageConfig,
    weight_policy: WeightSliceStagePolicy,
    x_slice: DirectXSliceStageConfig | SerialXSliceStageConfig,
    x_policy: XSliceStagePolicy,
) -> None:
    torch.manual_seed(2)
    n = 17
    k = 19
    m = 5
    config = CimEngineConfig(
        cim_macro_config=_ideal_macro_config(max_active_num=3),
        placement=_placement_config(),
        input_activation=_input_activation_config(),
        weight_slice=weight_slice,
        x_slice=x_slice,
    )
    engine = CimEngine(
        config=config,
        policy=_engine_policy(weight_slice=weight_policy, x_slice=x_policy),
        w_logical_shape=(n, k),
        dtype=torch.float32,
        ideal_macro=False,
    )
    weight = _randint_in_range(engine.w_value_range, (n, k))
    activation = _randint_in_range(engine.x_value_range, (m, k))
    _assert_engine_matches_torch(engine, weight, activation)


def test_inter_plane_contraction_accumulator_tracks_physical_weight_planes() -> None:
    engine = _build_inter(w_logical_shape=(17, 19), input_num=8, max_active_num=3)
    plan = engine.placement.plan
    assert engine.placement.contraction_accumulator.inst_shape == (
        2,
        plan.block_group_num,
    )


def test_input_activation_and_block_routing_layout() -> None:
    """D and P land left of the instance-aligned block and partition inputs."""
    engine = _build_direct(w_logical_shape=(4, 8), input_num=8, max_active_num=2)
    p = engine.input_activation.input_phase_num
    x = engine._organize_x(torch.arange(1, 17, dtype=torch.int32).reshape(2, 8))
    phased = engine.input_activation.unroll_input_phases(x)
    planes = engine.placement.unroll_block_steps(phased)
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
    engine = _build_direct(w_logical_shape=(4, 8), input_num=8, max_active_num=2)
    x = engine._organize_x(torch.randint(0, 2, (3, 2, 8), dtype=torch.int32))
    phased = engine.input_activation.unroll_input_phases(x)
    planes = engine.placement.unroll_block_steps(phased)
    assert planes.shape == (3, 1, 4, 2, 1, 1, 1, 1, 8)
    assert torch.equal(planes[:, 0].sum(dim=1), x)


def test_input_phase_non_divisible_ceil_covers_all_inputs() -> None:
    """input_num=10 and max_active_num=3 cover all inputs in four phases."""
    engine = _build_direct(w_logical_shape=(4, 10), input_num=10, max_active_num=3)
    assert engine.input_activation.input_phase_num == 4
    inputs = torch.arange(1, 11, dtype=torch.int32)
    phases = engine.input_activation.unroll_input_phases(inputs)
    assert phases.shape == (4, 10)
    # Each input belongs to exactly one phase.
    assert torch.equal((phases != 0).sum(dim=0), torch.ones(10, dtype=torch.int64))
    assert torch.equal(phases.sum(dim=0), inputs)
    # The short final block owns only its single real row (row 9).
    assert torch.equal(phases[3], torch.tensor([0] * 9 + [10]))


def test_degenerate_input_phase_axis_size_one() -> None:
    """max_active_num == input_num retains a size-one phase axis."""
    engine = _build_direct(w_logical_shape=(4, 8), input_num=8, max_active_num=8)
    assert engine.input_activation.input_phase_num == 1
    x = engine._organize_x(torch.randint(0, 2, (2, 8), dtype=torch.int32))
    phased = engine.input_activation.unroll_input_phases(x)
    planes = engine.placement.unroll_block_steps(phased)
    assert planes.shape == (1, 1, 2, 1, 1, 1, 1, 8)
    assert torch.equal(planes[0, 0], x)


@pytest.mark.parametrize("build", [_build_direct, _build_inter, _build_intra])
def test_input_phase_count_skips_padding_only_blocks(build: Callable[..., CimEngine]) -> None:
    """K < input_num omits phases holding only macro-input padding.

    input_num=8, max_active_num=2, K=3 gives two, not ceil(8/2)=4.
    """
    torch.manual_seed(3)
    n = 4
    k = 3  # k < input_num: one short block leaves input positions unused
    m = 3
    engine = build(w_logical_shape=(n, k), input_num=8, max_active_num=2)
    assert engine.input_activation.input_phase_num == 2
    weight = _randint_in_range(engine.w_value_range, (n, k))
    activation = _randint_in_range(engine.x_value_range, (m, k))
    _assert_engine_matches_torch(engine, weight, activation)


@pytest.mark.parametrize("build", [_build_direct, _build_inter, _build_intra])
def test_engine_matmul_parity_with_input_phases(build: Callable[..., CimEngine]) -> None:
    torch.manual_seed(7)
    n = 5
    k = 10  # k > input_num exercises Tc tiling alongside P
    m = 3
    engine = build(w_logical_shape=(n, k), input_num=8, max_active_num=2)
    assert engine.input_activation.input_phase_num == 4
    weight = _randint_in_range(engine.w_value_range, (n, k))
    activation = _randint_in_range(engine.x_value_range, (m, k))
    _assert_engine_matches_torch(engine, weight, activation)


@pytest.mark.parametrize("build", [_build_direct, _build_inter, _build_intra])
def test_engine_matmul_parity_non_divisible(build: Callable[..., CimEngine]) -> None:
    """Non-divisible geometry still equals the oracle: P from a ceiling division covers every input, row 9 included."""
    torch.manual_seed(9)
    n = 5
    k = 10  # k == input_num: all 10 positions carry real weight
    m = 3
    engine = build(w_logical_shape=(n, k), input_num=10, max_active_num=3)
    assert engine.input_activation.input_phase_num == 4
    weight = _randint_in_range(engine.w_value_range, (n, k))
    activation = _randint_in_range(engine.x_value_range, (m, k))
    _assert_engine_matches_torch(engine, weight, activation)


# --- Balanced input-axis block packing ---


@pytest.mark.parametrize("build", [_build_direct, _build_inter, _build_intra])
def test_balanced_block_packing_matches_torch(build: Callable[..., CimEngine]) -> None:
    """Five or ten logical output blocks are balanced over physical macros."""
    torch.manual_seed(13)
    n = 40
    k = 3
    m = 5
    engine = build(w_logical_shape=(n, k), input_num=8, max_active_num=2)
    assert engine.placement.plan.block_group_capacity == 2
    assert engine.placement.plan.block_slot_num == 2
    weight = _randint_in_range(engine.w_value_range, (n, k))
    activation = _randint_in_range(engine.x_value_range, (m, k))
    _assert_engine_matches_torch(engine, weight, activation)


def test_direct_block_placement_is_balanced_and_zero_padded() -> None:
    """B=5 and C=2 use G=3 macros with a 2+2+1 balanced assignment."""
    engine = _build_direct(w_logical_shape=(40, 3), input_num=8, max_active_num=2)
    assert engine.placement.plan.output_block_num == 5
    assert engine.placement.plan.block_group_capacity == 2
    assert engine.placement.plan.block_group_num == 3
    assert engine.placement.plan.block_slot_num == 2

    weight = torch.arange(1, 121, dtype=torch.int32).reshape(40, 3)
    engine.program(weight)
    programmed = engine.cim_macro._w
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
    engine = _build_direct(w_logical_shape=(40, 3), input_num=8, max_active_num=2)
    x = engine._organize_x(torch.tensor([[2, 3, 5]], dtype=torch.int32))
    phased = engine.input_activation.unroll_input_phases(x)
    routed = engine.placement.unroll_block_steps(phased)
    assert routed.shape == (2, 2, 1, 1, 1, 1, 1, 8)

    # Phase reduction restores x in each block slot.
    restored = routed.sum(dim=1)
    assert torch.equal(restored[0, 0, 0, 0, 0, 0], torch.tensor([2, 3, 5, 0, 0, 0, 0, 0]))
    assert torch.equal(restored[1, 0, 0, 0, 0, 0], torch.tensor([0, 0, 0, 2, 3, 5, 0, 0]))


def test_large_balanced_case_uses_seventeen_plus_sixteen() -> None:
    engine = CimEngine(
        config=CimEngineConfig(
            cim_macro_config=_ideal_macro_config(input_num=32, output_num=1, max_active_num=1),
            placement=_placement_config(),
            input_activation=_input_activation_config(),
            weight_slice=DirectWeightSliceStageConfig(),
            x_slice=DirectXSliceStageConfig(),
        ),
        policy=_engine_policy(
            weight_slice=DirectWeightSliceStagePolicy(),
            x_slice=DirectXSliceStagePolicy(),
        ),
        w_logical_shape=(33, 1),
        dtype=torch.float32,
        ideal_macro=False,
    )
    assert engine.placement.plan.block_group_capacity == 32
    assert engine.placement.plan.block_group_num == 2
    assert engine.placement.plan.block_slot_num == 17
    assert engine.cim_macro.inst_shape == (1, 1, 1, 1, 2)


# --- Caller prefix under leading-resolved profiling ---

_ACTIVATION_BATCH = (3, 1)  # a two-axis caller prefix
_M = 3


def _phased(engine: CimEngine, activation: torch.Tensor) -> torch.Tensor:
    """Run the two stages that precede `unroll_block_steps`."""
    return engine.input_activation.unroll_input_phases(engine._organize_x(activation))


def test_caller_prefix_stays_leftmost() -> None:
    torch.manual_seed(11)
    n = 40
    k = 3
    engine = _build_direct(w_logical_shape=(n, k), input_num=8, max_active_num=2)
    engine.program(_randint_in_range(engine.w_value_range, (n, k)))
    activation = _randint_in_range(engine.x_value_range, (*_ACTIVATION_BATCH, _M, k))
    phased = _phased(engine, activation)
    with Profiler(leading_rank=len(_ACTIVATION_BATCH)):
        routed = engine.placement.unroll_block_steps(phased)
    d = engine.placement.plan.block_slot_num
    p = engine.input_activation.input_phase_num
    # The caller prefix stays the leftmost contiguous block.
    # Shape: [3, 1, D, P, ...]
    assert routed.shape[:4] == (*_ACTIVATION_BATCH, d, p)
