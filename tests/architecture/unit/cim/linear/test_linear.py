"""Linear CIM units preserve logical products, bias state, and phase accounting."""

from __future__ import annotations

import pytest
import torch

from neurox import Profiler, stamp_names
from neurox.architecture.unit.cim import (
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
    PlacementStageConfig,
    PlacementStagePolicy,
)
from neurox.encoding import Encoding
from neurox.primitive.digital import AccumulatorConfig
from neurox.primitive.macro.cim import (
    CimMacroQuantizationScheme,
    IdealCimMacroConfig,
    IdealCimMacroPolicy,
)

_UNIT_POLICY = LinearCimUnitPolicy(
    engine=CimEnginePolicy(
        cim_macro_policy=IdealCimMacroPolicy(),
        placement=PlacementStagePolicy(),
        input_activation=InputActivationStagePolicy(),
        weight_slice=DirectWeightSliceStagePolicy(),
        x_slice=DirectXSliceStagePolicy(),
    ),
)

# `adc_active_bits = None` bypasses the virtual ADC, so per-plane codes are the
# exact integer partial dots.
_QUANTIZATION_MODE = 0
_ADC_BITS = None


def _ideal_macro_config(
    *,
    input_num: int = 16,
    output_num: int = 16,
    max_active_num: int | None = None,
    x_value_range: tuple[int, int] = (0, 1),
    w_value_range: tuple[int, int] = (-3, 3),
) -> IdealCimMacroConfig:
    return IdealCimMacroConfig(
        input_num=input_num,
        rescale_factors=(1.0,),
        max_active_num=16 if max_active_num is None else max_active_num,
        lane_num=1,
        scan_num=output_num,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
        w_digit_num=2,
        w_digit_radix=2,
        w_encoding=Encoding.TRUE_FORM,
        x_digit_num=2,
        x_digit_radix=2,
        x_encoding=Encoding.UNSIGNED,
        x_value_range=x_value_range,
        w_value_range=w_value_range,
        adc_bits=8,
        quantization_scheme=CimMacroQuantizationScheme.ZERO_POINT,
    )


def _accumulator_config(*, energy_per_op__fJ: float = 0.0) -> AccumulatorConfig:
    return AccumulatorConfig(
        bit_width=32,
        energy_per_op__fJ=energy_per_op__fJ,
        latency_per_op__ns=0.0,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )


def _unit_config(
    *,
    cim_macro_config: IdealCimMacroConfig | None = None,
    phase_energy_per_op__fJ: float = 0.0,
    area_per_inst__um2: float = 0.0,
) -> LinearCimUnitConfig:
    return LinearCimUnitConfig(
        area_per_inst__um2=area_per_inst__um2,
        leakage_per_inst__uW=0.0,
        engine=CimEngineConfig(
            cim_macro_config=_ideal_macro_config() if cim_macro_config is None else cim_macro_config,
            placement=PlacementStageConfig(
                contraction_accumulator_config=_accumulator_config(),
            ),
            input_activation=InputActivationStageConfig(
                phase_accumulator_config=_accumulator_config(energy_per_op__fJ=phase_energy_per_op__fJ),
            ),
            weight_slice=DirectWeightSliceStageConfig(),
            x_slice=DirectXSliceStageConfig(),
        ),
    )


def _build_unit(
    config: LinearCimUnitConfig,
    *,
    w_logical_shape: tuple[int, ...],
) -> LinearCimUnit:
    unit = LinearCimUnit(
        config=config,
        policy=_UNIT_POLICY,
        w_logical_shape=w_logical_shape,
        dtype=torch.float32,
        ideal_macro=False,
    )
    unit.eval()
    return unit


def _random_weight(unit: LinearCimUnit, shape: tuple[int, ...]) -> torch.Tensor:
    lo, hi = unit.w_value_range
    return torch.randint(lo, hi + 1, shape, dtype=torch.int32)


def _random_binary(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randint(0, 2, shape, dtype=torch.int32)


def _cpu_int64_linear_oracle(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Bit-exact int64 reference on CPU: `x @ W^T` in F.linear shapes."""
    x64 = x.cpu().long()
    w64 = weight.cpu().long()
    return torch.matmul(x64.unsqueeze(-2), w64.transpose(-1, -2)).squeeze(-2)


_SHAPE_CASES = [
    (16, 16, ()),
    (13, 16, (8,)),
    (16, 20, (8,)),
    (13, 20, (8,)),
    (17, 19, (3, 5)),
    (5, 7, (2, 1, 4)),
    (33, 35, (6,)),
]


# --- int64 CPU-oracle parity ---


@pytest.mark.parametrize(("n", "k", "batch"), _SHAPE_CASES)
def test_linear_matches_int64_cpu_oracle_on_device(
    n: int, k: int, batch: tuple[int, ...], device: torch.device
) -> None:
    torch.manual_seed(200 + n * 13 + k * 7 + len(batch))
    unit = _build_unit(_unit_config(), w_logical_shape=(n, k))
    weight = _random_weight(unit, (n, k))
    x = _random_binary((*batch, k))
    unit.to(device)
    unit.program(weight.to(device))
    actual = unit.linear(x.to(device), quantization_mode=_QUANTIZATION_MODE, adc_active_bits=_ADC_BITS)
    assert actual.device.type == device.type
    assert actual.shape == (*batch, n)
    assert actual.dtype == torch.int64
    expected = _cpu_int64_linear_oracle(x, weight)
    assert torch.equal(actual.cpu().long(), expected)


def test_linear_multi_phase_exact_matches_oracle() -> None:
    """P = 4: exact per-phase partials reduce to the exact product."""
    torch.manual_seed(400)
    n = 13
    k = 20
    config = _unit_config(cim_macro_config=_ideal_macro_config(max_active_num=4))
    unit = _build_unit(config, w_logical_shape=(n, k))
    weight = _random_weight(unit, (n, k))
    x = _random_binary((8, k))
    unit.program(weight)
    actual = unit.linear(x, quantization_mode=_QUANTIZATION_MODE, adc_active_bits=_ADC_BITS)
    assert actual.shape == (8, n)
    assert torch.equal(actual.long(), _cpu_int64_linear_oracle(x, weight))


def test_linear_accepts_single_vector_input() -> None:
    torch.manual_seed(500)
    n = 13
    k = 20
    unit = _build_unit(_unit_config(), w_logical_shape=(n, k))
    weight = _random_weight(unit, (n, k))
    x = _random_binary((k,))
    unit.program(weight)
    actual = unit.linear(x, quantization_mode=_QUANTIZATION_MODE, adc_active_bits=_ADC_BITS)
    assert actual.shape == (n,)
    assert torch.equal(actual.long(), _cpu_int64_linear_oracle(x, weight))


# --- integer bias ---


def test_linear_program_with_integer_bias_adds_exactly() -> None:
    torch.manual_seed(650)
    n = 13
    k = 20
    m = 8
    unit = _build_unit(_unit_config(), w_logical_shape=(n, k))
    weight = _random_weight(unit, (n, k))
    bias = torch.randint(-7, 8, (n,), dtype=torch.int32)
    x = _random_binary((m, k))
    unit.program(weight, bias)
    actual = unit.linear(x, quantization_mode=_QUANTIZATION_MODE, adc_active_bits=_ADC_BITS)
    expected = _cpu_int64_linear_oracle(x, weight) + bias.long()
    assert torch.equal(actual.long(), expected)


def test_linear_reprogram_without_bias_clears_slot() -> None:
    torch.manual_seed(660)
    n = 13
    k = 20
    m = 8
    unit = _build_unit(_unit_config(), w_logical_shape=(n, k))
    weight = _random_weight(unit, (n, k))
    x = _random_binary((m, k))
    unit.program(weight, torch.randint(-7, 8, (n,), dtype=torch.int32))
    unit.program(weight)
    actual = unit.linear(x, quantization_mode=_QUANTIZATION_MODE, adc_active_bits=_ADC_BITS)
    assert torch.equal(actual.long(), _cpu_int64_linear_oracle(x, weight))


# --- P > 1 input-phase accounting ---


def test_linear_phase_accounting_scales_with_input_phase_num() -> None:
    """The phase accumulator bills one operation per arriving per-phase code: P=2 logs twice the energy of P=1."""
    torch.manual_seed(700)
    n = 8
    k = 16
    m = 5
    energies: dict[int, float] = {}
    for max_active_num in (16, 8):  # P = 1, P = 2
        config = _unit_config(
            cim_macro_config=_ideal_macro_config(max_active_num=max_active_num),
            phase_energy_per_op__fJ=1.0,
        )
        unit = _build_unit(config, w_logical_shape=(n, k))
        unit.program(_random_weight(unit, (n, k)))
        stamp_names(unit)
        accumulator_name = unit.engine.input_activation.phase_accumulator.qualified_name
        with Profiler() as p:
            unit.linear(_random_binary((m, k)), quantization_mode=_QUANTIZATION_MODE, adc_active_bits=_ADC_BITS)
        energies[unit.engine.input_activation.input_phase_num] = sum(
            r.dynamic_energy__fJ for r in p.records if r.qualified_name == accumulator_name
        )
    assert energies[1] > 0.0
    assert energies[2] == pytest.approx(2.0 * energies[1])


# --- config / construction validation ---


def test_linear_accepts_non_divisor_input_blocking() -> None:
    config = _unit_config(cim_macro_config=_ideal_macro_config(max_active_num=6))
    unit = _build_unit(config, w_logical_shape=(13, 20))
    weight = _random_weight(unit, (13, 20))
    x = _random_binary((5, 20))
    unit.program(weight)
    actual = unit.linear(x, quantization_mode=_QUANTIZATION_MODE, adc_active_bits=_ADC_BITS)
    assert torch.equal(actual.long(), _cpu_int64_linear_oracle(x, weight))
