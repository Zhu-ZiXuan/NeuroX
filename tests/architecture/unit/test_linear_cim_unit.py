"""Tests for the linear CIM unit and the abstract unit / engine bases."""

from __future__ import annotations

import inspect

import pytest
import torch

from neurox.architecture.unit import Conv2dUnit, LinearUnit, UnitBase
from neurox.architecture.unit.cim import (
    CimUnit,
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
from neurox.common.profiler import NeuroxProfiler
from neurox.primitive.digital import AccumulatorConfig, SerialAccumulator
from neurox.primitive.macro.cim import IdealCimMacroConfig, IdealCimMacroPolicy

_UNIT_POLICY = LinearCimUnitPolicy(
    engine=CimEnginePolicy(
        cim_macro_policy=IdealCimMacroPolicy(),
        placement=PlacementStagePolicy(),
        input_activation=InputActivationStagePolicy(),
        weight_slice=DirectWeightSliceStagePolicy(),
        x_slice=DirectXSliceStagePolicy(),
    ),
)

# ``adc_bits == 0`` is the IdealCimMacro lossless sentinel: per-plane codes
# are the exact integer partial dots, so the whole unit pipeline must match
# an int64 CPU matmul oracle bit-exactly.
_ADC_MODE = 0
_ADC_BITS = 0


def _ideal_macro_config(
    *,
    max_active_num: int | None = None,
    x_value_range: tuple[int, int] = (0, 1),
    w_value_range: tuple[int, int] = (-3, 3),
) -> IdealCimMacroConfig:
    return IdealCimMacroConfig(
        max_active_num=16 if max_active_num is None else max_active_num,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
        x_value_range=x_value_range,
        w_value_range=w_value_range,
        adc_mode_num=1,
        adc_max_bits=0,
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
            input_num=16,
            output_num=16,
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
        T__K=300.0,
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
    """Bit-exact int64 reference on CPU: ``x @ W^T`` in F.linear shapes."""
    x64 = x.cpu().to(torch.int64)
    w64 = weight.cpu().to(torch.int64)
    return torch.matmul(x64.unsqueeze(-2), w64.transpose(-1, -2)).squeeze(-2)


_SHAPE_CASES = [
    (16, 16, ()),
    (13, 20, (8,)),
    (17, 19, (3, 5)),
    (5, 7, (2, 1, 4)),
    (33, 35, (6,)),
]


# --- int64 CPU-oracle parity ---


@pytest.mark.parametrize(("n", "k", "batch"), _SHAPE_CASES)
def test_linear_matches_int64_cpu_oracle_on_cpu(n: int, k: int, batch: tuple[int, ...]) -> None:
    torch.manual_seed(100 + n * 13 + k * 7 + len(batch))
    unit = _build_unit(_unit_config(), w_logical_shape=(n, k))
    weight = _random_weight(unit, (n, k))
    x = _random_binary((*batch, k))
    unit.program(weight)
    actual = unit.linear(x, adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
    expected = _cpu_int64_linear_oracle(x, weight)
    assert actual.shape == (*batch, n)
    assert torch.equal(actual.to(torch.int64), expected)


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
    actual = unit.linear(x.to(device), adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
    assert actual.device.type == device.type
    expected = _cpu_int64_linear_oracle(x, weight)
    assert torch.equal(actual.cpu().to(torch.int64), expected)


def test_linear_lowering_matches_int64_cpu_oracle() -> None:
    """The protected lowering template (seam 2 -> engine matmul -> seam 3)
    is the bias-free integer product."""
    torch.manual_seed(300)
    n, k, m = 13, 20, 8
    unit = _build_unit(_unit_config(), w_logical_shape=(n, k))
    weight = _random_weight(unit, (n, k))
    x = _random_binary((m, k))
    unit.program(weight)
    actual = unit._lower_matmul(x, adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
    expected = torch.matmul(x.to(torch.int64), weight.to(torch.int64).transpose(-1, -2))
    assert torch.equal(actual.to(torch.int64), expected)


def test_linear_multi_phase_lossless_matches_oracle() -> None:
    """P = 4: lossless per-phase partials still reduce to the exact product."""
    torch.manual_seed(400)
    n, k = 13, 20
    config = _unit_config(cim_macro_config=_ideal_macro_config(max_active_num=4))
    unit = _build_unit(config, w_logical_shape=(n, k))
    weight = _random_weight(unit, (n, k))
    x = _random_binary((8, k))
    unit.program(weight)
    actual = unit.linear(x, adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
    assert torch.equal(actual.to(torch.int64), _cpu_int64_linear_oracle(x, weight))


def test_linear_accepts_single_vector_input() -> None:
    torch.manual_seed(500)
    n, k = 13, 20
    unit = _build_unit(_unit_config(), w_logical_shape=(n, k))
    weight = _random_weight(unit, (n, k))
    x = _random_binary((k,))
    unit.program(weight)
    actual = unit.linear(x, adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
    assert actual.shape == (n,)
    assert torch.equal(actual.to(torch.int64), _cpu_int64_linear_oracle(x, weight))


# --- leading time-axis transparency ---


def test_linear_leading_time_axis_transparency() -> None:
    """A caller-owned leading time axis is a pure broadcast dim: the batched
    call equals the per-plane calls stacked, plane by plane."""
    torch.manual_seed(600)
    n, k, t, b = 13, 20, 16, 3
    unit = _build_unit(_unit_config(), w_logical_shape=(n, k))
    weight = _random_weight(unit, (n, k))
    x = _random_binary((t, b, k))
    unit.program(weight)
    batched = unit.linear(x, adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
    per_plane = torch.stack(
        [unit.linear(x[i], adc_mode=_ADC_MODE, adc_bits=_ADC_BITS) for i in range(t)],
        dim=0,
    )
    assert batched.shape == (t, b, n)
    assert torch.equal(batched, per_plane)
    assert torch.equal(batched.to(torch.int64), _cpu_int64_linear_oracle(x, weight))


# --- integer bias ---


def test_linear_program_with_integer_bias_adds_exactly() -> None:
    torch.manual_seed(650)
    n, k, m = 13, 20, 8
    unit = _build_unit(_unit_config(), w_logical_shape=(n, k))
    weight = _random_weight(unit, (n, k))
    bias = torch.randint(-7, 8, (n,), dtype=torch.int32)
    x = _random_binary((m, k))
    unit.program(weight, bias)
    actual = unit.linear(x, adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
    expected = _cpu_int64_linear_oracle(x, weight) + bias.to(torch.int64)
    assert torch.equal(actual.to(torch.int64), expected)


def test_linear_reprogram_without_bias_clears_slot() -> None:
    torch.manual_seed(660)
    n, k, m = 13, 20, 8
    unit = _build_unit(_unit_config(), w_logical_shape=(n, k))
    weight = _random_weight(unit, (n, k))
    x = _random_binary((m, k))
    unit.program(weight, torch.randint(-7, 8, (n,), dtype=torch.int32))
    unit.program(weight)
    actual = unit.linear(x, adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
    assert torch.equal(actual.to(torch.int64), _cpu_int64_linear_oracle(x, weight))


def test_linear_lowering_never_includes_bias() -> None:
    """The bias is added only by ``linear`` after the lowering template; the
    protected lowering itself stays bias-free."""
    torch.manual_seed(670)
    n, k, m = 13, 20, 8
    unit = _build_unit(_unit_config(), w_logical_shape=(n, k))
    weight = _random_weight(unit, (n, k))
    x = _random_binary((m, k))
    unit.program(weight, torch.randint(-7, 8, (n,), dtype=torch.int32))
    actual = unit._lower_matmul(x, adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
    expected = torch.matmul(x.to(torch.int64), weight.to(torch.int64).transpose(-1, -2))
    assert torch.equal(actual.to(torch.int64), expected)


def test_linear_program_rejects_float_bias() -> None:
    n, k = 13, 20
    unit = _build_unit(_unit_config(), w_logical_shape=(n, k))
    weight = _random_weight(unit, (n, k))
    with pytest.raises(ValueError, match="integer bias dtype"):
        unit.program(weight, torch.zeros(n, dtype=torch.float32))


def test_linear_cim_rejects_float_weight() -> None:
    n, k = 13, 20
    unit = _build_unit(_unit_config(), w_logical_shape=(n, k))
    with pytest.raises(TypeError, match="integer weight tensor"):
        unit.program(torch.zeros((n, k), dtype=torch.float32))


def test_linear_cim_rejects_float_input() -> None:
    n, k = 13, 20
    unit = _build_unit(_unit_config(), w_logical_shape=(n, k))
    unit.program(_random_weight(unit, (n, k)))
    with pytest.raises(TypeError, match="integer input tensor"):
        unit.linear(torch.zeros((2, k), dtype=torch.float32), adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)


def test_linear_program_rejects_wrong_shape_bias() -> None:
    n, k = 13, 20
    unit = _build_unit(_unit_config(), w_logical_shape=(n, k))
    weight = _random_weight(unit, (n, k))
    with pytest.raises(ValueError, match=r"bias\.shape"):
        unit.program(weight, torch.zeros(n + 1, dtype=torch.int32))


# --- P > 1 input-phase accounting ---


def test_linear_phase_accounting_scales_with_input_phase_num() -> None:
    """The phase accumulator is a ``SerialAccumulator`` billed per arriving
    per-phase code: P=2 logs exactly twice the accumulate energy of P=1."""
    torch.manual_seed(700)
    n, k, m = 8, 16, 5
    energies: dict[int, float] = {}
    for max_active_num in (16, 8):  # P = 1, P = 2
        config = _unit_config(
            cim_macro_config=_ideal_macro_config(max_active_num=max_active_num),
            phase_energy_per_op__fJ=1.0,
        )
        unit = _build_unit(config, w_logical_shape=(n, k))
        assert isinstance(unit.engine.input_activation.phase_accumulator, SerialAccumulator)
        unit.program(_random_weight(unit, (n, k)))
        with NeuroxProfiler() as p:
            unit.linear(_random_binary((m, k)), adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
        energies[unit.engine.input_activation._input_phase_num] = sum(
            e.dynamic_energy__fJ for e in p.energy_events if e.module is unit.engine.input_activation.phase_accumulator
        )
    assert energies[1] > 0.0
    assert energies[2] == pytest.approx(2.0 * energies[1])


# --- config / construction validation ---


def test_linear_config_rejects_negative_ppa() -> None:
    with pytest.raises(ValueError, match="area_per_inst__um2"):
        _unit_config(area_per_inst__um2=-1.0)


def test_linear_accepts_non_divisor_input_blocking() -> None:
    config = _unit_config(cim_macro_config=_ideal_macro_config(max_active_num=6))
    unit = _build_unit(config, w_logical_shape=(13, 20))
    weight = _random_weight(unit, (13, 20))
    x = _random_binary((5, 20))
    unit.program(weight)
    actual = unit.linear(x, adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
    assert torch.equal(actual.to(torch.int64), _cpu_int64_linear_oracle(x, weight))


def test_direct_engine_leaves_numeric_path_selection_to_macro() -> None:
    config = _unit_config(
        cim_macro_config=_ideal_macro_config(
            max_active_num=16,
            w_value_range=(-4095, 4095),
        )
    )
    unit = _build_unit(config, w_logical_shape=(13, 20))
    assert isinstance(unit, LinearCimUnit)


def test_linear_x_value_range_follows_engine() -> None:
    unit = _build_unit(_unit_config(), w_logical_shape=(13, 20))
    assert unit.x_value_range == (0, 1)


def test_linear_registered_for_from_config_dispatch() -> None:
    unit = CimUnit.from_config(
        config=_unit_config(),
        policy=_UNIT_POLICY,
        w_logical_shape=(13, 20),
        dtype=torch.float32,
        T__K=300.0,
        ideal_macro=False,
    )
    assert isinstance(unit, LinearCimUnit)


# --- abstract unit bases ---


@pytest.mark.parametrize(
    "abstract_cls",
    [UnitBase, LinearUnit, Conv2dUnit, CimUnit],
)
def test_abstract_bases_reject_instantiation(abstract_cls: type) -> None:
    assert inspect.isabstract(abstract_cls)
    with pytest.raises(TypeError):
        abstract_cls()
