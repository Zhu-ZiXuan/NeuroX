"""Which axis of the macro access schedule `(M, Sx, D, P)` multiplies which digital block's window.

In `CimEngine.latency__ns`.
"""

from __future__ import annotations

import pytest
import torch

from neurox import Profiler, stamp_names
from neurox.architecture.unit.cim import (
    Conv2dCimUnit,
    Conv2dCimUnitConfig,
    Conv2dCimUnitPolicy,
    LinearCimUnit,
    LinearCimUnitConfig,
    LinearCimUnitPolicy,
)
from neurox.architecture.unit.cim.engine import (
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
    XSliceStageConfig,
    XSliceStagePolicy,
)
from neurox.common.encoding import Encoding
from neurox.primitive.digital import AccumulatorConfig, ShiftAdderConfig
from neurox.primitive.macro.cim import (
    CimMacroQuantizationScheme,
    IdealCimMacroConfig,
    IdealCimMacroPolicy,
)

# The ideal macro's own latency is zero, so every term the engine adds is a
# digital one and stays separable.
_ADC_BITS = None
_QUANTIZATION_MODE = 0

_INPUT_NUM = 8
_OUTPUT_NUM = 4

# Hand-derived placement of `(N, K) = (8, 4)` on an 8-input macro whose weight
# block spans all 4 output ports: L = min(4, 8) = 4, so Tc = 1; two output
# blocks fit one group of capacity 2, so G = 1 and D = 2.
_W_SHAPE = (8, 4)
_BLOCK_STEP_NUM = 2
# The intra-port layout halves the block width to Q = 2, giving four output
# blocks over two groups: G = 2, D = 2.
_INTRA_AGGREGATED_PORT_NUM = 2


def _macro_config(*, max_active_num: int) -> IdealCimMacroConfig:
    return IdealCimMacroConfig(
        input_num=_INPUT_NUM,
        rescale_factors=(1.0,),
        max_active_num=max_active_num,
        lane_num=1,
        scan_num=_OUTPUT_NUM,
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


def _accumulator_config(*, latency_per_op__ns: float, energy_per_op__fJ: float = 0.0) -> AccumulatorConfig:
    return AccumulatorConfig(
        bit_width=32,
        energy_per_op__fJ=energy_per_op__fJ,
        latency_per_op__ns=latency_per_op__ns,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )


def _shift_adder_config(*, latency_per_op__ns: float) -> ShiftAdderConfig:
    return ShiftAdderConfig(
        bit_width=32,
        energy_per_op__fJ=0.0,
        latency_per_op__ns=latency_per_op__ns,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )


def _engine_config(
    *,
    weight_slice: WeightSliceStageConfig,
    x_slice: XSliceStageConfig,
    max_active_num: int = 4,
    phase__ns: float = 0.0,
    contraction__ns: float = 0.0,
    phase_energy__fJ: float = 0.0,
) -> CimEngineConfig:
    return CimEngineConfig(
        cim_macro_config=_macro_config(max_active_num=max_active_num),
        placement=PlacementStageConfig(
            contraction_accumulator_config=_accumulator_config(latency_per_op__ns=contraction__ns),
        ),
        input_activation=InputActivationStageConfig(
            phase_accumulator_config=_accumulator_config(
                latency_per_op__ns=phase__ns,
                energy_per_op__fJ=phase_energy__fJ,
            ),
        ),
        weight_slice=weight_slice,
        x_slice=x_slice,
    )


def _engine_policy(config: CimEngineConfig) -> CimEnginePolicy:
    weight_slice: WeightSliceStagePolicy
    if isinstance(config.weight_slice, InterWeightSliceStageConfig):
        weight_slice = InterWeightSliceStagePolicy()
    elif isinstance(config.weight_slice, IntraWeightSliceStageConfig):
        weight_slice = IntraWeightSliceStagePolicy()
    else:
        weight_slice = DirectWeightSliceStagePolicy()
    x_slice: XSliceStagePolicy
    if isinstance(config.x_slice, SerialXSliceStageConfig):
        x_slice = SerialXSliceStagePolicy()
    else:
        x_slice = DirectXSliceStagePolicy()
    return CimEnginePolicy(
        cim_macro_policy=IdealCimMacroPolicy(),
        placement=PlacementStagePolicy(),
        input_activation=InputActivationStagePolicy(),
        weight_slice=weight_slice,
        x_slice=x_slice,
    )


def _build_linear(engine_config: CimEngineConfig, *, w_logical_shape: tuple[int, ...] = _W_SHAPE) -> LinearCimUnit:
    config = LinearCimUnitConfig(
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        engine=engine_config,
    )
    unit = LinearCimUnit(
        config=config,
        policy=LinearCimUnitPolicy(engine=_engine_policy(engine_config)),
        w_logical_shape=w_logical_shape,
        dtype=torch.float32,
        T__K=300.0,
        ideal_macro=False,
    )
    unit.eval()
    return unit


def _build_conv2d(engine_config: CimEngineConfig, *, w_logical_shape: tuple[int, int, int, int]) -> Conv2dCimUnit:
    config = Conv2dCimUnitConfig(
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        engine=engine_config,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
    )
    unit = Conv2dCimUnit(
        config=config,
        policy=Conv2dCimUnitPolicy(engine=_engine_policy(engine_config)),
        w_logical_shape=w_logical_shape,
        dtype=torch.float32,
        T__K=300.0,
        ideal_macro=False,
    )
    unit.eval()
    return unit


# --- The schedule the engine multiplies ---


def test_direct_layout_times_the_two_accumulators() -> None:
    """D and the port count are the only multipliers a direct/direct layout has, neither slice stage owning a block."""
    phase__ns = 0.25
    contraction__ns = 0.5
    unit = _build_linear(
        _engine_config(
            weight_slice=DirectWeightSliceStageConfig(),
            x_slice=DirectXSliceStageConfig(),
            max_active_num=4,
            phase__ns=phase__ns,
            contraction__ns=contraction__ns,
        )
    )
    engine = unit.engine
    assert engine.placement.block_step_num == _BLOCK_STEP_NUM
    assert engine.input_activation.input_phase_num == 1
    assert engine.x_slice.slice_num == 1
    assert engine.weight_slice.shift_adder is None
    assert engine.x_slice.shift_adder is None

    expected__ns = _BLOCK_STEP_NUM * _OUTPUT_NUM * (phase__ns + contraction__ns)
    assert unit.latency__ns((_W_SHAPE[1],), adc_active_bits=_ADC_BITS) == pytest.approx(expected__ns)


def test_phase_accumulator_runs_once_per_arrival() -> None:
    """P multiplies the serial register's window and nothing else's.

    The contraction tree closes its axis per block step.
    """

    def _phase_only(max_active_num: int) -> float:
        unit = _build_linear(
            _engine_config(
                weight_slice=DirectWeightSliceStageConfig(),
                x_slice=DirectXSliceStageConfig(),
                max_active_num=max_active_num,
                phase__ns=1.0,
            )
        )
        return unit.latency__ns((_W_SHAPE[1],), adc_active_bits=_ADC_BITS)

    def _contraction_only(max_active_num: int) -> float:
        unit = _build_linear(
            _engine_config(
                weight_slice=DirectWeightSliceStageConfig(),
                x_slice=DirectXSliceStageConfig(),
                max_active_num=max_active_num,
                contraction__ns=1.0,
            )
        )
        return unit.latency__ns((_W_SHAPE[1],), adc_active_bits=_ADC_BITS)

    single = _build_linear(
        _engine_config(
            weight_slice=DirectWeightSliceStageConfig(),
            x_slice=DirectXSliceStageConfig(),
            max_active_num=4,
        )
    )
    doubled = _build_linear(
        _engine_config(
            weight_slice=DirectWeightSliceStageConfig(),
            x_slice=DirectXSliceStageConfig(),
            max_active_num=2,
        )
    )
    assert single.engine.input_activation.input_phase_num == 1
    assert doubled.engine.input_activation.input_phase_num == 2

    assert _phase_only(2) == pytest.approx(2.0 * _phase_only(4))
    assert _contraction_only(2) == pytest.approx(_contraction_only(4))


# --- The recombination blocks ---


def test_inter_layout_recombines_once_per_step_it_closes() -> None:
    """Both reconstructions are one-pass positional sums.

    The weight-slice block runs once per input slice, the input-slice block once per block step.
    """
    x_slice_num = 3
    phase__ns = 0.25
    contraction__ns = 0.5
    w_recombine__ns = 1.0
    x_recombine__ns = 2.0
    unit = _build_linear(
        _engine_config(
            weight_slice=InterWeightSliceStageConfig(
                w_slice_num=2,
                w_encoding=Encoding.TRUE_FORM,
                shift_adder_config=_shift_adder_config(latency_per_op__ns=w_recombine__ns),
            ),
            x_slice=SerialXSliceStageConfig(
                x_slice_num=x_slice_num,
                shift_adder_config=_shift_adder_config(latency_per_op__ns=x_recombine__ns),
            ),
            max_active_num=4,
            phase__ns=phase__ns,
            contraction__ns=contraction__ns,
        )
    )
    engine = unit.engine
    assert engine.x_slice.slice_num == x_slice_num
    assert engine.placement.block_step_num == _BLOCK_STEP_NUM
    assert engine.weight_slice.aggregated_output_num == _OUTPUT_NUM

    step_num = x_slice_num * _BLOCK_STEP_NUM
    expected__ns = step_num * _OUTPUT_NUM * (phase__ns + contraction__ns + w_recombine__ns)
    expected__ns += _BLOCK_STEP_NUM * _OUTPUT_NUM * x_recombine__ns
    assert unit.latency__ns((_W_SHAPE[1],), adc_active_bits=_ADC_BITS) == pytest.approx(expected__ns)


def test_input_slice_recombination_ignores_the_slice_count() -> None:
    """Sx is the digit axis of the one pass that removes it, so it never multiplies."""

    def _x_recombine_only(x_slice_num: int) -> float:
        unit = _build_linear(
            _engine_config(
                weight_slice=DirectWeightSliceStageConfig(),
                x_slice=SerialXSliceStageConfig(
                    x_slice_num=x_slice_num,
                    shift_adder_config=_shift_adder_config(latency_per_op__ns=1.0),
                ),
                max_active_num=4,
            )
        )
        return unit.latency__ns((_W_SHAPE[1],), adc_active_bits=_ADC_BITS)

    assert _x_recombine_only(5) == pytest.approx(_x_recombine_only(2))
    assert _x_recombine_only(2) == pytest.approx(_BLOCK_STEP_NUM * _OUTPUT_NUM)


def test_intra_layout_recombines_over_the_ports_one_aggregation_leaves() -> None:
    """An intra-port layout leaves `output_num // w_slice_num` logical outputs.

    The reconstruction runs over those, not the macro ports.
    """
    unit = _build_linear(
        _engine_config(
            weight_slice=IntraWeightSliceStageConfig(
                w_slice_num=2,
                w_encoding=Encoding.TRUE_FORM,
                shift_adder_config=_shift_adder_config(latency_per_op__ns=1.0),
            ),
            x_slice=DirectXSliceStageConfig(),
            max_active_num=4,
        )
    )
    engine = unit.engine
    assert engine.weight_slice.aggregated_output_num == _INTRA_AGGREGATED_PORT_NUM
    assert engine.placement.block_step_num == _BLOCK_STEP_NUM

    expected__ns = _BLOCK_STEP_NUM * _INTRA_AGGREGATED_PORT_NUM
    assert unit.latency__ns((_W_SHAPE[1],), adc_active_bits=_ADC_BITS) == pytest.approx(expected__ns)


# --- The one runtime extent ---


def test_output_planes_multiply_the_whole_schedule() -> None:
    """M = H_out · W_out enters at the conv unit and multiplies every engine term, while a caller batch does not."""
    engine_config = _engine_config(
        weight_slice=DirectWeightSliceStageConfig(),
        x_slice=DirectXSliceStageConfig(),
        max_active_num=4,
        phase__ns=0.25,
        contraction__ns=0.5,
    )
    # (C_out, C_in, kh, kw) contracting to K = 4, the linear fixtures' shape.
    unit = _build_conv2d(engine_config, w_logical_shape=(8, 1, 2, 2))
    one_plane__ns = unit.engine.latency__ns(output_plane_num=1, adc_active_bits=_ADC_BITS)

    assert unit.latency__ns((1, 4, 4), adc_active_bits=_ADC_BITS) == pytest.approx(9 * one_plane__ns)
    assert unit.latency__ns((1, 5, 4), adc_active_bits=_ADC_BITS) == pytest.approx(12 * one_plane__ns)
    assert unit.latency__ns((7, 1, 4, 4), adc_active_bits=_ADC_BITS) == pytest.approx(9 * one_plane__ns)


# --- The round count against a measured forward ---


def test_phase_accumulator_rounds_match_the_measured_forward(device: torch.device) -> None:
    """The round count multiplying latency equals the operands a measured forward folds.

    Per instance and per caller operation.
    """
    phase__ns = 0.75
    phase_energy__fJ = 3.0
    unit = _build_linear(
        _engine_config(
            weight_slice=InterWeightSliceStageConfig(
                w_slice_num=2,
                w_encoding=Encoding.TRUE_FORM,
                shift_adder_config=_shift_adder_config(latency_per_op__ns=0.0),
            ),
            x_slice=SerialXSliceStageConfig(
                x_slice_num=3,
                shift_adder_config=_shift_adder_config(latency_per_op__ns=0.0),
            ),
            max_active_num=2,
            phase__ns=phase__ns,
            phase_energy__fJ=phase_energy__fJ,
        )
    )
    accumulator = unit.engine.input_activation.phase_accumulator
    batch_num = 5
    torch.manual_seed(0)
    weight = torch.randint(-3, 4, _W_SHAPE, dtype=torch.int32)
    x = torch.randint(0, 2, (batch_num, _W_SHAPE[1]), dtype=torch.int32)
    unit.to(device)
    unit.program(weight.to(device))
    stamp_names(unit)
    accumulator_name = accumulator.qualified_name

    with Profiler(leading_rank=1) as profiler:
        unit.linear(x.to(device), quantization_mode=_QUANTIZATION_MODE, adc_active_bits=_ADC_BITS)
    folded_operand_num = (
        sum(
            float(record.dynamic_energy__fJ.sum())
            for record in profiler.records
            if record.qualified_name == accumulator_name
        )
        / phase_energy__fJ
    )
    round_num = folded_operand_num / (batch_num * accumulator.inst_count)

    assert unit.latency__ns((_W_SHAPE[1],), adc_active_bits=_ADC_BITS) == pytest.approx(round_num * phase__ns)
