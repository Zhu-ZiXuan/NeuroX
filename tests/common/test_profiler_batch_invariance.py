"""Tests that dynamic energy scales exactly with the batch a model is fed.

Feeding the same batch twice — ``torch.cat([x, x])`` — must double the dynamic
energy. Identical data makes that an exact factor rather than an approximate
one, so the assertions carry no tolerance.

The defect this names is a caller-block size-1 stand-in in an energy payload,
which under-counts that emitter by the batch product. A size-1 on a non-caller
axis stays invisible: nothing distinguishes it from a genuine extent of one.

The model is the smallest one that puts several emitters on the path at once:
a linear CIM unit over an ideal macro, sliced so that the placement, input
activation, weight-slice and x-slice stages each bill a digital block. Its
config is hand-written, with a distinct per-op energy per block so a
mis-attributed row is visible in the per-emitter view.
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.architecture.unit.cim import LinearCimUnit, LinearCimUnitConfig, LinearCimUnitPolicy
from neurox.architecture.unit.cim.engine import (
    CimEngineConfig,
    CimEnginePolicy,
    InputActivationStageConfig,
    InputActivationStagePolicy,
    IntraWeightSliceStageConfig,
    IntraWeightSliceStagePolicy,
    PlacementStageConfig,
    PlacementStagePolicy,
    SerialXSliceStageConfig,
    SerialXSliceStagePolicy,
)
from neurox.common.encoding import Encoding
from neurox.common.profiler import NeuroxProfiler
from neurox.primitive.digital import AccumulatorConfig, ShiftAdderConfig
from neurox.primitive.macro.cim import IdealCimMacroConfig, IdealCimMacroPolicy

# Logical problem, sized so every stage does real work: the contraction spans
# two tiles (D), one tile holds two weight slices (Sw), the input is serialized
# into two slices (Sa) and into two row phases (P).
_INPUT_NUM = 8  # macro input ports
_OUTPUT_NUM = 8  # macro output ports
_MAX_ACTIVE_NUM = 4  # rows per phase
_W_SLICE_NUM = 2
_X_SLICE_NUM = 2
_LOGICAL_OUT_NUM = 6
_LOGICAL_IN_NUM = 10
_BATCH_NUM = 3

# Pairwise-distinct per-op energies: every emitter is identifiable from its row.
_CONTRACTION_E__FJ = 1.0
_PHASE_E__FJ = 2.0
_W_SLICE_E__FJ = 3.0
_X_SLICE_E__FJ = 5.0

_EMITTER_NUM = 4  # the blocks the four stages bill

_QUANTIZATION_MODE = 0
_ADC_BITS = 4


def _accumulator_config(energy_per_op__fJ: float) -> AccumulatorConfig:
    return AccumulatorConfig(
        bit_width=32,
        energy_per_op__fJ=energy_per_op__fJ,
        latency_per_op__ns=0.0,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )


def _shift_adder_config(energy_per_op__fJ: float) -> ShiftAdderConfig:
    return ShiftAdderConfig(
        bit_width=32,
        energy_per_op__fJ=energy_per_op__fJ,
        latency_per_op__ns=0.0,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )


def _build_unit(device: torch.device) -> LinearCimUnit:
    """Build the hand-configured unit on ``device``, in eval mode."""
    config = LinearCimUnitConfig(
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        engine=CimEngineConfig(
            input_num=_INPUT_NUM,
            output_num=_OUTPUT_NUM,
            cim_macro_config=IdealCimMacroConfig(
                max_active_num=_MAX_ACTIVE_NUM,
                leakage_per_inst__uW=0.0,
                area_per_inst__um2=0.0,
                x_value_range=(0, 1),
                w_value_range=(-1, 1),
                quantization_input_ranges=((-256, 255),),
                adc_max_bits=8,
            ),
            placement=PlacementStageConfig(
                contraction_accumulator_config=_accumulator_config(_CONTRACTION_E__FJ),
            ),
            input_activation=InputActivationStageConfig(
                phase_accumulator_config=_accumulator_config(_PHASE_E__FJ),
            ),
            weight_slice=IntraWeightSliceStageConfig(
                w_slice_num=_W_SLICE_NUM,
                w_encoding=Encoding.TRUE_FORM,
                shift_adder_config=_shift_adder_config(_W_SLICE_E__FJ),
            ),
            x_slice=SerialXSliceStageConfig(
                x_slice_num=_X_SLICE_NUM,
                shift_adder_config=_shift_adder_config(_X_SLICE_E__FJ),
            ),
        ),
    )
    policy = LinearCimUnitPolicy(
        engine=CimEnginePolicy(
            cim_macro_policy=IdealCimMacroPolicy(),
            placement=PlacementStagePolicy(),
            input_activation=InputActivationStagePolicy(),
            weight_slice=IntraWeightSliceStagePolicy(),
            x_slice=SerialXSliceStagePolicy(),
        ),
    )
    unit = LinearCimUnit(
        config=config,
        policy=policy,
        w_logical_shape=(_LOGICAL_OUT_NUM, _LOGICAL_IN_NUM),
        dtype=torch.float32,
        T__K=300.0,
        ideal_macro=False,
    )
    unit.eval()
    unit.to(device)
    return unit


def _programmed_unit(seed: int, device: torch.device) -> LinearCimUnit:
    """Return a unit holding a random legal weight."""
    torch.manual_seed(seed)
    unit = _build_unit(device)
    lo, hi = unit.w_value_range
    weight = torch.randint(lo, hi + 1, (_LOGICAL_OUT_NUM, _LOGICAL_IN_NUM), dtype=torch.int32, device=device)
    unit.program(weight)
    return unit


def _random_input(unit: LinearCimUnit, device: torch.device) -> Tensor:
    lo, hi = unit.x_value_range
    return torch.randint(lo, hi + 1, (_BATCH_NUM, _LOGICAL_IN_NUM), dtype=torch.int32, device=device)


def _measure(unit: LinearCimUnit, x: Tensor, *, leading_rank: int = 0) -> NeuroxProfiler:
    with NeuroxProfiler(leading_rank=leading_rank) as profiler:
        unit.linear(x, quantization_mode=_QUANTIZATION_MODE, adc_bits=_ADC_BITS)
    return profiler


# === Batch-doubling invariance ===


def test_the_repeated_batch_doubles_the_total_dynamic_energy(device: torch.device) -> None:
    """The headline invariant: twice the operations, twice the energy, exactly."""
    unit = _programmed_unit(seed=800, device=device)
    x = _random_input(unit, device)
    single = _measure(unit, x).total_dynamic_energy__fJ
    double = _measure(unit, torch.cat([x, x])).total_dynamic_energy__fJ
    assert single > 0.0
    assert double == 2.0 * single


def test_every_emitter_doubles_with_the_batch(device: torch.device) -> None:
    """Per-emitter, so a row that fails to scale names the block that billed it."""
    unit = _programmed_unit(seed=810, device=device)
    x = _random_input(unit, device)
    single = _measure(unit, x).report(unit).energy_by_name
    double = _measure(unit, torch.cat([x, x])).report(unit).energy_by_name
    assert len(single) == _EMITTER_NUM
    assert min(single.values()) > 0.0
    assert double == {name: 2.0 * energy for name, energy in single.items()}


def test_the_repeated_batch_repeats_the_per_operation_rows(device: torch.device) -> None:
    """At ``leading_rank=1`` the doubled run is the single run's rows twice over.

    The finest form of the same statement: the second copy of the batch bills
    what the first did, operation by operation, on every emitter.
    """
    unit = _programmed_unit(seed=820, device=device)
    x = _random_input(unit, device)
    single = _measure(unit, x, leading_rank=1).energy_events
    double = _measure(unit, torch.cat([x, x]), leading_rank=1).energy_events
    assert len(single) == len(double) == _EMITTER_NUM
    for one, two in zip(single, double, strict=True):
        assert one.module is two.module
        assert one.dynamic_energy__fJ.shape == (_BATCH_NUM,)
        torch.testing.assert_close(two.dynamic_energy__fJ, torch.cat([one.dynamic_energy__fJ] * 2), rtol=0, atol=0)
