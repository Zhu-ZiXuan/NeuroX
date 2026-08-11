"""Tests for engine-level input-axis block packing and phase accumulation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

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
from neurox.common.encoding import Encoding
from neurox.common.profiler import NeuroxProfiler
from neurox.primitive.digital import AccumulatorConfig, ShiftAdderConfig
from neurox.primitive.macro.cim import IdealCimMacroConfig, IdealCimMacroPolicy

# All engines are built on IdealCimMacroConfig, so the embedded macro policy is
# the empty marker. ``adc_bits is None`` selects the lossless oracle: no ADC
# quantization, so engine outputs equal ``torch.matmul`` exactly.
_IDEAL_MACRO_POLICY = IdealCimMacroPolicy()
_QUANTIZATION_MODE = 0
_ADC_BITS: int | None = None


def _ideal_macro_config(
    *,
    max_active_num: int = 2,
) -> IdealCimMacroConfig:
    return IdealCimMacroConfig(
        max_active_num=max_active_num,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
        x_value_range=(0, 1),
        w_value_range=(-3, 3),
        # Only the lossless oracle is exercised here; the declared window and
        # width just have to be legal.
        quantization_input_ranges=((-256, 255),),
        adc_max_bits=8,
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
        "T__K": 300.0,
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
        input_num=input_num,
        output_num=8,
        cim_macro_config=_ideal_macro_config(max_active_num=max_active_num),
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
        input_num=input_num,
        output_num=8,
        cim_macro_config=_ideal_macro_config(max_active_num=max_active_num),
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
    actual = engine.matmul(activation, quantization_mode=_QUANTIZATION_MODE, adc_bits=_ADC_BITS)
    expected = torch.matmul(activation.to(torch.int64), weight.transpose(-1, -2).to(torch.int64))
    assert actual.shape == expected.shape
    assert torch.equal(actual.to(torch.int64), expected)


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
    n, k, m = 17, 19, 5
    config = CimEngineConfig(
        input_num=8,
        output_num=8,
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
        T__K=300.0,
        ideal_macro=False,
    )
    weight = _randint_in_range(engine.w_value_range, (n, k))
    activation = _randint_in_range(engine.x_value_range, (m, k))
    _assert_engine_matches_torch(engine, weight, activation)


def test_inter_plane_contraction_accumulator_tracks_physical_weight_planes() -> None:
    engine = _build_inter(w_logical_shape=(17, 19), input_num=8, max_active_num=3)
    plan = engine.placement.plan
    assert engine.placement.contraction_accumulator.inst_shape == (
        1,
        2,
        plan.block_group_num,
    )


def test_block_slot_and_activation_masks_are_independent() -> None:
    """One block slot and four phases independently cover local inputs."""
    engine = _build_direct(w_logical_shape=(4, 8), input_num=8, max_active_num=2)
    assert engine.placement.plan.block_slot_num == 1
    assert engine.input_activation._input_phase_num == 4

    activation_mask = engine.input_activation._active_input_mask
    assert activation_mask.shape == (4, 8)
    rows = torch.arange(8)
    phases = torch.arange(4).unsqueeze(-1)
    assert torch.equal(activation_mask, rows // 2 == phases)

    slot_mask = engine.placement._block_slot_mask
    assert slot_mask.shape == (1, 8)
    assert torch.all(slot_mask)


def test_input_activation_and_block_routing_layout() -> None:
    """D and P land left of the instance-aligned block and partition inputs."""
    engine = _build_direct(w_logical_shape=(4, 8), input_num=8, max_active_num=2)
    p = engine.input_activation._input_phase_num
    x = engine._organize_x(torch.arange(1, 17, dtype=torch.int64).reshape(2, 8))
    phased = engine.input_activation.unroll_input_phases(x)
    planes = engine.placement.unroll_block_steps(phased)
    assert planes.shape == (1, p, 2, 1, 1, 1, 1, 8)
    # Masks partition the row axis: summing P restores the full plane.
    assert torch.equal(planes[0].sum(dim=0), x)
    # Zero-fill outside each phase's selected-input window.
    for phase in range(p):
        assert torch.equal(
            planes[0, phase],
            torch.where(engine.input_activation._active_input_mask[phase], x, torch.zeros_like(x)),
        )


def test_split_input_stages_keep_batch_axes_left_of_d_and_p() -> None:
    """Unaligned caller batch axes remain left of the execution schedule."""
    engine = _build_direct(w_logical_shape=(4, 8), input_num=8, max_active_num=2)
    x = engine._organize_x(torch.randint(0, 2, (3, 2, 8), dtype=torch.int64))
    phased = engine.input_activation.unroll_input_phases(x)
    planes = engine.placement.unroll_block_steps(phased)
    assert planes.shape == (3, 1, 4, 2, 1, 1, 1, 1, 8)
    assert torch.equal(planes[:, 0].sum(dim=1), x)


def test_input_phase_non_divisible_ceil_covers_all_inputs() -> None:
    """input_num=10 and max_active_num=3 cover all inputs in four phases."""
    engine = _build_direct(w_logical_shape=(4, 10), input_num=10, max_active_num=3)
    assert engine.input_activation._input_phase_num == 4
    mask = engine.input_activation._active_input_mask
    assert mask.shape == (4, 10)
    # Each input belongs to exactly one phase.
    assert torch.equal(mask.sum(dim=0), torch.ones(10, dtype=mask.dtype))
    # The short final block owns only its single real row (row 9).
    assert int(mask[3].sum()) == 1


def test_degenerate_input_phase_axis_size_one() -> None:
    """max_active_num == input_num retains a size-one phase axis."""
    engine = _build_direct(w_logical_shape=(4, 8), input_num=8, max_active_num=8)
    assert engine.input_activation._input_phase_num == 1
    x = engine._organize_x(torch.randint(0, 2, (2, 8), dtype=torch.int64))
    phased = engine.input_activation.unroll_input_phases(x)
    planes = engine.placement.unroll_block_steps(phased)
    assert planes.shape == (1, 1, 2, 1, 1, 1, 1, 8)
    assert torch.equal(planes[0, 0], x)


@pytest.mark.parametrize("build", [_build_direct, _build_inter, _build_intra])
def test_input_phase_count_skips_padding_only_blocks(build: Callable[..., CimEngine]) -> None:
    """K < input_num omits phases containing only tile padding.

    input_num=8, max_active_num=2, K=3 gives two phases, well
    below the full ceil(8/2) = 4. Correctness is still checked against the
    torch oracle: the skipped blocks contributed exactly 0."""
    torch.manual_seed(3)
    n, k, m = 4, 3, 3  # k < input_num: one short block leaves input positions unused
    engine = build(w_logical_shape=(n, k), input_num=8, max_active_num=2)
    assert engine.input_activation._input_phase_num == 2
    assert engine.input_activation._active_input_mask.shape == (2, 3)
    weight = _randint_in_range(engine.w_value_range, (n, k))
    activation = _randint_in_range(engine.x_value_range, (m, k))
    _assert_engine_matches_torch(engine, weight, activation)


@pytest.mark.parametrize("build", [_build_direct, _build_inter, _build_intra])
def test_engine_matmul_parity_with_input_phases(build: Callable[..., CimEngine]) -> None:
    """Lossless multi-phase matmul equals the torch.matmul oracle."""
    torch.manual_seed(7)
    n, k, m = 5, 10, 3  # k > input_num exercises Tc tiling alongside P
    engine = build(w_logical_shape=(n, k), input_num=8, max_active_num=2)
    assert engine.input_activation._input_phase_num == 4
    weight = _randint_in_range(engine.w_value_range, (n, k))
    activation = _randint_in_range(engine.x_value_range, (m, k))
    _assert_engine_matches_torch(engine, weight, activation)


@pytest.mark.parametrize("build", [_build_direct, _build_inter, _build_intra])
def test_engine_matmul_parity_non_divisible(build: Callable[..., CimEngine]) -> None:
    """Non-divisible geometry still equals the torch.matmul oracle: P via ceil
    covers every input, so k == input_num == 10 populated positions all contribute. The
    old floor division (P = 3) would drop row 9 and mismatch."""
    torch.manual_seed(9)
    n, k, m = 5, 10, 3  # k == input_num: all 10 positions carry real weight
    engine = build(w_logical_shape=(n, k), input_num=10, max_active_num=3)
    assert engine.input_activation._input_phase_num == 4
    weight = _randint_in_range(engine.w_value_range, (n, k))
    activation = _randint_in_range(engine.x_value_range, (m, k))
    _assert_engine_matches_torch(engine, weight, activation)


@pytest.mark.parametrize("build", [_build_direct, _build_inter, _build_intra])
@pytest.mark.parametrize("activation_batch", [(), (3, 1)])
def test_engine_weight_batch_parity(
    build: Callable[..., CimEngine],
    activation_batch: tuple[int, ...],
) -> None:
    """Weight batches preserve block packing and right-aligned broadcasting."""
    torch.manual_seed(11)
    b, n, k, m = 2, 40, 3, 3
    engine = build(w_logical_shape=(b, n, k), input_num=8, max_active_num=2)
    weight = _randint_in_range(engine.w_value_range, (b, n, k))
    activation = _randint_in_range(engine.x_value_range, (*activation_batch, m, k))
    _assert_engine_matches_torch(engine, weight, activation)


# --- Balanced input-axis block packing ---


@pytest.mark.parametrize("build", [_build_direct, _build_inter, _build_intra])
def test_balanced_block_packing_matches_torch(build: Callable[..., CimEngine]) -> None:
    """Five or ten logical output blocks are balanced over physical macros."""
    torch.manual_seed(13)
    n, k, m = 40, 3, 5
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
            input_num=32,
            output_num=1,
            cim_macro_config=_ideal_macro_config(max_active_num=1),
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
        T__K=300.0,
        ideal_macro=False,
    )
    assert engine.placement.plan.block_group_capacity == 32
    assert engine.placement.plan.block_group_num == 2
    assert engine.placement.plan.block_slot_num == 17
    assert engine.cim_macro.inst_shape == (1, 1, 1, 1, 2)


# --- Weight batch versus leading-resolved profiling ---

_WEIGHT_BATCH_SHAPE = (2, 40, 3)  # (w_batch, N, K)
_ACTIVATION_BATCH = (3, 1)  # a caller batch of 3 plus the size-1 weight-batch slot
_CALLER_BATCH = _ACTIVATION_BATCH[:1]  # the prefix that stays left of D and P
_M = 3


def _phased(engine: CimEngine, activation: torch.Tensor) -> torch.Tensor:
    """Run the two stages that precede ``unroll_block_steps``."""
    return engine.input_activation.unroll_input_phases(engine._organize_x(activation))


def _batched_engine_and_activation() -> tuple[CimEngine, torch.Tensor]:
    """Build a weight-batched engine and a caller-batched activation for it."""
    torch.manual_seed(11)
    engine = _build_direct(w_logical_shape=_WEIGHT_BATCH_SHAPE, input_num=8, max_active_num=2)
    engine.program(_randint_in_range(engine.w_value_range, _WEIGHT_BATCH_SHAPE))
    activation = _randint_in_range(engine.x_value_range, (*_ACTIVATION_BATCH, _M, _WEIGHT_BATCH_SHAPE[-1]))
    return engine, activation


def test_weight_batch_without_profiler_keeps_split_leading_layout() -> None:
    """Outside a profiler a weight batch still runs, D and P splitting the caller dims."""
    engine, activation = _batched_engine_and_activation()
    routed = engine.placement.unroll_block_steps(_phased(engine, activation))
    d = engine.placement.plan.block_slot_num
    p = engine.input_activation._input_phase_num
    # The caller prefix (3, 1) is split by the inserted pair.
    # Shape: [3, D, P, 1, ...]
    assert routed.shape[:4] == (_ACTIVATION_BATCH[0], d, p, _ACTIVATION_BATCH[1])


def test_weight_batch_leaves_the_declared_caller_axis_leftmost() -> None:
    """A caller may declare every leading dim that stays left of D and P.

    With ``leading_rank=1`` the profiler's leftmost-1-dim slice reads exactly
    the caller axis of ``[3, D, P, w_batch=1, ...]``, so a weight-batched engine
    stays profilable per caller unit operation.
    """
    engine, activation = _batched_engine_and_activation()
    with NeuroxProfiler(leading_rank=len(_CALLER_BATCH)):
        routed = engine.placement.unroll_block_steps(_phased(engine, activation))
    d = engine.placement.plan.block_slot_num
    p = engine.input_activation._input_phase_num
    assert routed.shape[:4] == (_CALLER_BATCH[0], d, p, _ACTIVATION_BATCH[1])


def test_weight_batch_energy_is_billed_against_the_caller_axis() -> None:
    """Under ``leading_rank=1`` each billed element is one caller unit operation.

    A payload laid out over the routed tensor is reduced by the profiler onto its
    leftmost dim. That dim must index the caller: element ``i`` has to carry
    exactly the work of running caller ``i`` on its own, which a mis-billing that
    read D as the caller axis could not reproduce.
    """
    engine, activation = _batched_engine_and_activation()
    stage = engine.placement
    # PlacementStage is not itself a profile target, so the stage's own
    # accumulator stands in as the emitting host for the routed payload.
    emitter = stage.contraction_accumulator
    with NeuroxProfiler(leading_rank=len(_CALLER_BATCH)) as profiler:
        routed = stage.unroll_block_steps(_phased(engine, activation))
        emitter._record_dynamic_energy(routed.to(torch.float64))
    (event,) = profiler.energy_events
    billed = event.dynamic_energy__fJ
    assert billed.shape == _CALLER_BATCH

    alone = torch.stack(
        [
            stage.unroll_block_steps(_phased(engine, activation[i : i + 1])).to(torch.float64).sum()
            for i in range(_CALLER_BATCH[0])
        ]
    )
    assert torch.equal(billed, alone)
    # Not a degenerate comparison: the caller elements do measurably different work.
    assert len(set(alone.tolist())) == _CALLER_BATCH[0]


def test_weight_batch_runs_under_a_rank_zero_profiler() -> None:
    """A profiler with no caller leading dims imposes no layout: everything is summed."""
    engine, activation = _batched_engine_and_activation()
    with NeuroxProfiler() as profiler:
        actual = engine.matmul(activation, quantization_mode=_QUANTIZATION_MODE, adc_bits=_ADC_BITS)
    assert actual.shape == (*_ACTIVATION_BATCH[:-1], _WEIGHT_BATCH_SHAPE[0], _M, _WEIGHT_BATCH_SHAPE[1])
    assert profiler.leading_rank == 0


def test_unbatched_weight_keeps_the_whole_caller_prefix_leftmost() -> None:
    """Without a weight batch every declared caller dim stays left of D and P."""
    torch.manual_seed(11)
    n, k = 40, 3
    engine = _build_direct(w_logical_shape=(n, k), input_num=8, max_active_num=2)
    engine.program(_randint_in_range(engine.w_value_range, (n, k)))
    activation = _randint_in_range(engine.x_value_range, (*_ACTIVATION_BATCH, _M, k))
    phased = _phased(engine, activation)
    with NeuroxProfiler(leading_rank=len(_ACTIVATION_BATCH)):
        routed = engine.placement.unroll_block_steps(phased)
    d = engine.placement.plan.block_slot_num
    p = engine.input_activation._input_phase_num
    # The caller prefix stays the leftmost contiguous block.
    # Shape: [3, 1, D, P, ...]
    assert routed.shape[:4] == (*_ACTIVATION_BATCH, d, p)
