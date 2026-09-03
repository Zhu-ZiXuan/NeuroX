"""Tests for engine-backed CIM unit behavior."""

from __future__ import annotations

import pytest
import torch

from neurox import Profiler, stamp_names
from neurox.architecture.unit import LinearUnit
from neurox.architecture.unit.cim import (
    CimUnit,
    CimUnitConfig,
    CimUnitPolicy,
    LinearCimUnit,
    LinearCimUnitConfig,
    LinearCimUnitPolicy,
)
from neurox.architecture.unit.cim.engine import (
    CimEngine,
    CimEngineConfig,
    CimEnginePolicy,
    DirectWeightSliceStage,
    DirectWeightSliceStageConfig,
    DirectWeightSliceStagePolicy,
    DirectXSliceStage,
    DirectXSliceStageConfig,
    DirectXSliceStagePolicy,
    InputActivationStageConfig,
    InputActivationStagePolicy,
    InterWeightSliceStage,
    InterWeightSliceStageConfig,
    InterWeightSliceStagePolicy,
    IntraWeightSliceStage,
    IntraWeightSliceStageConfig,
    IntraWeightSliceStagePolicy,
    PlacementStageConfig,
    PlacementStagePolicy,
    SerialXSliceStage,
    SerialXSliceStageConfig,
    SerialXSliceStagePolicy,
    WeightSliceStage,
    WeightSliceStageConfig,
    WeightSliceStagePolicy,
    XSliceStage,
    XSliceStageConfig,
    XSliceStagePolicy,
)
from neurox.architecture.unit.cim.engine.placement import _chunk_pad_along
from neurox.architecture.unit.ideal import IdealLinearUnit, IdealLinearUnitConfig, IdealLinearUnitPolicy
from neurox.common.encoding import Encoding
from neurox.common.serialize import ConfigDict
from neurox.primitive.digital import AccumulatorConfig, SerialAccumulator, ShiftAdderConfig
from neurox.primitive.macro.cim import (
    CimMacroQuantizationScheme,
    IdealCimMacroConfig,
    IdealCimMacroPolicy,
)

# All tests use IdealCimMacroConfig as the embedded macro config, so its
# nonideality policy is the empty marker.
_IDEAL_MACRO_POLICY = IdealCimMacroPolicy()
_IDEAL_UNIT_POLICY = IdealLinearUnitPolicy()

# `adc_active_bits = None` bypasses the virtual ADC, so unit outputs equal
# `torch.matmul` exactly—the same behavior `IdealLinearUnit` provides natively.
_TEST_ADC_BITS = None
_TEST_QUANTIZATION_MODE = 0
_TEST_ADC_MAX_BITS = 8
type _CimUnitType = type[CimUnit[CimUnitConfig, CimUnitPolicy]]
type _WeightSliceStageType = type[WeightSliceStage[WeightSliceStageConfig, WeightSliceStagePolicy]]
type _XSliceStageType = type[XSliceStage[XSliceStageConfig, XSliceStagePolicy]]
_TEST_RESCALE_FACTORS: tuple[float, ...] = (1.0,)


def _ideal_macro_config(
    *,
    input_num: int = 16,
    output_num: int = 16,
    max_active_num: int | None = None,
    x_value_range: tuple[int, int] = (0, 1),
    w_value_range: tuple[int, int] = (-3, 3),
    rescale_factors: tuple[float, ...] = _TEST_RESCALE_FACTORS,
    adc_bits: int = _TEST_ADC_MAX_BITS,
) -> IdealCimMacroConfig:
    return IdealCimMacroConfig(
        input_num=input_num,
        rescale_factors=rescale_factors,
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
        adc_bits=adc_bits,
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


def _wrap_unit(engine_config: CimEngineConfig) -> LinearCimUnitConfig:
    return LinearCimUnitConfig(
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        engine=engine_config,
    )


def _engine_policy(config: CimEngineConfig) -> CimEnginePolicy:
    weight_slice_policies: dict[type[WeightSliceStageConfig], WeightSliceStagePolicy] = {
        DirectWeightSliceStageConfig: DirectWeightSliceStagePolicy(),
        InterWeightSliceStageConfig: InterWeightSliceStagePolicy(),
        IntraWeightSliceStageConfig: IntraWeightSliceStagePolicy(),
    }
    x_slice_policy: XSliceStagePolicy
    if isinstance(config.x_slice, DirectXSliceStageConfig):
        x_slice_policy = DirectXSliceStagePolicy()
    else:
        x_slice_policy = SerialXSliceStagePolicy()
    return CimEnginePolicy(
        cim_macro_policy=_IDEAL_MACRO_POLICY,
        placement=PlacementStagePolicy(),
        input_activation=InputActivationStagePolicy(),
        weight_slice=weight_slice_policies[type(config.weight_slice)],
        x_slice=x_slice_policy,
    )


def _linear_unit_policy(config: LinearCimUnitConfig) -> LinearCimUnitPolicy:
    return LinearCimUnitPolicy(engine=_engine_policy(config.engine))


def _direct_engine_config(
    *,
    x_value_range: tuple[int, int] = (0, 1),
    w_value_range: tuple[int, int] = (-3, 3),
    max_active_num: int | None = None,
    rescale_factors: tuple[float, ...] = _TEST_RESCALE_FACTORS,
    adc_bits: int = _TEST_ADC_MAX_BITS,
    phase_accumulator_config: AccumulatorConfig | None = None,
) -> CimEngineConfig:
    return CimEngineConfig(
        cim_macro_config=_ideal_macro_config(
            x_value_range=x_value_range,
            w_value_range=w_value_range,
            max_active_num=max_active_num,
            rescale_factors=rescale_factors,
            adc_bits=adc_bits,
        ),
        placement=PlacementStageConfig(
            contraction_accumulator_config=_accumulator_config(),
        ),
        input_activation=InputActivationStageConfig(
            phase_accumulator_config=(
                _accumulator_config() if phase_accumulator_config is None else phase_accumulator_config
            ),
        ),
        weight_slice=DirectWeightSliceStageConfig(),
        x_slice=DirectXSliceStageConfig(),
    )


def _direct_config(
    *,
    x_value_range: tuple[int, int] = (0, 1),
    w_value_range: tuple[int, int] = (-3, 3),
    max_active_num: int | None = None,
) -> LinearCimUnitConfig:
    return _wrap_unit(
        _direct_engine_config(
            x_value_range=x_value_range,
            w_value_range=w_value_range,
            max_active_num=max_active_num,
        )
    )


def _ideal_unit_config(
    *,
    x_value_range: tuple[int, int] = (0, 1),
    w_value_range: tuple[int, int] = (-3, 3),
) -> IdealLinearUnitConfig:
    return IdealLinearUnitConfig(
        x_value_range=x_value_range,
        w_value_range=w_value_range,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
    )


def _sliced_engine_config(
    *,
    w_slice_num: int,
    x_slice_num: int,
    weight_slice_type: type[InterWeightSliceStageConfig] | type[IntraWeightSliceStageConfig],
    x_value_range: tuple[int, int] = (0, 1),
    w_value_range: tuple[int, int] = (-3, 3),
    max_active_num: int | None = None,
) -> CimEngineConfig:
    return CimEngineConfig(
        cim_macro_config=_ideal_macro_config(
            x_value_range=x_value_range,
            w_value_range=w_value_range,
            max_active_num=max_active_num,
        ),
        placement=PlacementStageConfig(
            contraction_accumulator_config=_accumulator_config(),
        ),
        input_activation=InputActivationStageConfig(
            phase_accumulator_config=_accumulator_config(),
        ),
        weight_slice=weight_slice_type(
            w_slice_num=w_slice_num,
            w_encoding=Encoding.TRUE_FORM,
            shift_adder_config=_shift_adder_config(),
        ),
        x_slice=SerialXSliceStageConfig(
            x_slice_num=x_slice_num,
            shift_adder_config=_shift_adder_config(),
        ),
    )


def _inter_config(
    *,
    w_slice_num: int,
    x_slice_num: int,
    x_value_range: tuple[int, int] = (0, 1),
    w_value_range: tuple[int, int] = (-3, 3),
    max_active_num: int | None = None,
) -> LinearCimUnitConfig:
    return _wrap_unit(
        _sliced_engine_config(
            w_slice_num=w_slice_num,
            x_slice_num=x_slice_num,
            weight_slice_type=InterWeightSliceStageConfig,
            x_value_range=x_value_range,
            w_value_range=w_value_range,
            max_active_num=max_active_num,
        )
    )


def _intra_config(
    *,
    w_slice_num: int,
    x_slice_num: int,
    x_value_range: tuple[int, int] = (0, 1),
    w_value_range: tuple[int, int] = (-3, 3),
    max_active_num: int | None = None,
) -> LinearCimUnitConfig:
    return _wrap_unit(
        _sliced_engine_config(
            w_slice_num=w_slice_num,
            x_slice_num=x_slice_num,
            weight_slice_type=IntraWeightSliceStageConfig,
            x_value_range=x_value_range,
            w_value_range=w_value_range,
            max_active_num=max_active_num,
        )
    )


def _build_ideal(
    config: IdealLinearUnitConfig,
    *,
    w_logical_shape: tuple[int, ...],
) -> IdealLinearUnit:
    unit = CimUnit.from_config(
        config=config,
        policy=_IDEAL_UNIT_POLICY,
        w_logical_shape=w_logical_shape,
        dtype=torch.float32,
        T__K=300.0,
        ideal_macro=False,
    )
    assert isinstance(unit, IdealLinearUnit)
    unit.eval()
    return unit


def _build_linear(
    config: LinearCimUnitConfig,
    *,
    w_logical_shape: tuple[int, ...],
) -> LinearCimUnit:
    unit = LinearCimUnit(
        config=config,
        policy=_linear_unit_policy(config),
        w_logical_shape=w_logical_shape,
        dtype=torch.float32,
        T__K=300.0,
        ideal_macro=False,
    )
    unit.eval()
    return unit


def _assert_unit_matches_torch(unit: LinearUnit, weight: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
    unit.program(weight)
    actual = unit.linear(activation, quantization_mode=_TEST_QUANTIZATION_MODE, adc_active_bits=_TEST_ADC_BITS)
    # F.linear-form oracle: activation [..., K] against weight [N, K].
    # Shape: [..., K] -> [..., N]
    expected = torch.matmul(activation.to(torch.int64).unsqueeze(-2), weight.transpose(-1, -2).to(torch.int64)).squeeze(
        -2
    )
    assert actual.shape == expected.shape
    assert torch.equal(actual.to(torch.int64), expected)
    return actual


def _build_unit_for_kind(
    macro_kind: str,
    config: IdealLinearUnitConfig | LinearCimUnitConfig,
    *,
    w_logical_shape: tuple[int, ...],
) -> LinearUnit:
    if macro_kind == "ideal":
        assert isinstance(config, IdealLinearUnitConfig)
        return _build_ideal(config, w_logical_shape=w_logical_shape)
    assert isinstance(config, LinearCimUnitConfig)
    return _build_linear(config, w_logical_shape=w_logical_shape)


def _randint_in_range(value_range: tuple[int, int], shape: tuple[int, ...]) -> torch.Tensor:
    lo, hi = value_range
    return torch.randint(lo, hi + 1, shape, dtype=torch.int32)


_SHAPE_CASES = [
    (16, 16, 1),
    (13, 16, 8),
    (16, 20, 8),
    (13, 20, 8),
    (17, 19, 17),
    (5, 7, 3),
    (33, 35, 4),
]


def test_chunk_pad_along_without_padding() -> None:
    x = torch.arange(16)
    y = _chunk_pad_along(x, axis=0, chunk_size=4, pad_value=0)
    assert y.shape == (4, 4)
    assert torch.equal(y.flatten(), x)


def test_chunk_pad_along_with_padding() -> None:
    x = torch.arange(13)
    y = _chunk_pad_along(x, axis=0, chunk_size=16, pad_value=0)
    assert y.shape == (1, 16)
    assert torch.equal(y[0, :13], x)
    assert torch.equal(y[0, 13:], torch.zeros(3, dtype=x.dtype))


def test_chunk_pad_along_accepts_negative_axis() -> None:
    x = torch.arange(60).reshape(3, 4, 5)
    y = _chunk_pad_along(x, axis=-1, chunk_size=3, pad_value=0)
    assert y.shape == (3, 4, 2, 3)


@pytest.mark.parametrize(("n", "k", "m"), _SHAPE_CASES)
def test_ideal_unit_matches_torch_matmul_for_shape_cases(n: int, k: int, m: int) -> None:
    torch.manual_seed(900 + n * 13 + k * 7 + m)
    unit = _build_ideal(_ideal_unit_config(), w_logical_shape=(n, k))
    weight = _randint_in_range(unit.w_value_range, (n, k))
    activation = _randint_in_range(unit.x_value_range, (m, k))
    _assert_unit_matches_torch(unit, weight, activation)


def test_ideal_unit_uses_exact_integer_matmul() -> None:
    unit = _build_ideal(
        _ideal_unit_config(x_value_range=(-4095, 4095), w_value_range=(-4095, 4095)),
        w_logical_shape=(3, 257),
    )
    weight = torch.full((3, 257), 4095, dtype=torch.int32)
    activation = torch.full((2, 257), 4095, dtype=torch.int32)
    _assert_unit_matches_torch(unit, weight, activation)


@pytest.mark.parametrize(("n", "k", "m"), _SHAPE_CASES)
def test_direct_engine_unit_matches_torch_matmul_for_shape_cases(n: int, k: int, m: int) -> None:
    torch.manual_seed(1000 + n * 13 + k * 7 + m)
    unit = _build_linear(_direct_config(), w_logical_shape=(n, k))
    weight = _randint_in_range(unit.w_value_range, (n, k))
    activation = _randint_in_range(unit.x_value_range, (m, k))
    _assert_unit_matches_torch(unit, weight, activation)


def test_direct_engine_unit_handles_wide_macro_weight_range() -> None:
    torch.manual_seed(1)
    n = 13
    k = 20
    m = 8
    unit = _build_linear(_direct_config(w_value_range=(-15, 15)), w_logical_shape=(n, k))
    weight = torch.randint(-15, 16, (n, k), dtype=torch.int32)
    activation = torch.randint(0, 2, (m, k), dtype=torch.int32)
    _assert_unit_matches_torch(unit, weight, activation)


def test_direct_engine_passes_logical_weights_to_macro() -> None:
    """Direct mapping preserves asymmetric logical weight values."""
    config = _wrap_unit(_direct_engine_config(w_value_range=(-3, 3)))
    unit = _build_linear(config, w_logical_shape=(2, 2))
    weight = torch.tensor([[1, 2], [-1, -2]], dtype=torch.int32)
    activation = torch.tensor([[1, 1]], dtype=torch.int32)
    _assert_unit_matches_torch(unit, weight, activation)


@pytest.mark.parametrize(("n", "k", "m"), _SHAPE_CASES)
def test_inter_array_slice_engine_unit_matches_torch_matmul_for_shape_cases(n: int, k: int, m: int) -> None:
    torch.manual_seed(2000 + n * 13 + k * 7 + m)
    unit = _build_linear(_inter_config(w_slice_num=3, x_slice_num=4), w_logical_shape=(n, k))
    weight = _randint_in_range(unit.w_value_range, (n, k))
    activation = _randint_in_range(unit.x_value_range, (m, k))
    _assert_unit_matches_torch(unit, weight, activation)


@pytest.mark.parametrize(("n", "k", "m"), _SHAPE_CASES)
def test_intra_array_slice_engine_unit_matches_torch_matmul_for_shape_cases(n: int, k: int, m: int) -> None:
    torch.manual_seed(3000 + n * 13 + k * 7 + m)
    unit = _build_linear(_intra_config(w_slice_num=3, x_slice_num=4), w_logical_shape=(n, k))
    weight = _randint_in_range(unit.w_value_range, (n, k))
    activation = _randint_in_range(unit.x_value_range, (m, k))
    _assert_unit_matches_torch(unit, weight, activation)


@pytest.mark.parametrize(
    ("w_slice_num", "x_slice_num", "macro_w_value_range", "x_value_range"),
    [
        (1, 1, (-15, 15), (0, 1)),
        (2, 3, (-15, 15), (0, 1)),
        (3, 2, (-3, 3), (0, 3)),
    ],
)
def test_inter_array_slice_engine_unit_matches_torch_for_slice_range_cases(
    w_slice_num: int,
    x_slice_num: int,
    macro_w_value_range: tuple[int, int],
    x_value_range: tuple[int, int],
) -> None:
    torch.manual_seed(4000 + w_slice_num * 100 + x_slice_num * 10 + macro_w_value_range[1])
    n = 17
    k = 19
    m = 5
    config = _inter_config(
        w_slice_num=w_slice_num,
        x_slice_num=x_slice_num,
        w_value_range=macro_w_value_range,
        x_value_range=x_value_range,
    )
    unit = _build_linear(config, w_logical_shape=(n, k))
    weight = _randint_in_range(unit.w_value_range, (n, k))
    activation = _randint_in_range(unit.x_value_range, (m, k))
    _assert_unit_matches_torch(unit, weight, activation)


@pytest.mark.parametrize(
    ("w_slice_num", "x_slice_num", "macro_w_value_range", "x_value_range"),
    [
        (1, 1, (-15, 15), (0, 1)),
        (2, 3, (-15, 15), (0, 1)),
        (3, 2, (-3, 3), (0, 3)),
    ],
)
def test_intra_array_slice_engine_unit_matches_torch_for_slice_range_cases(
    w_slice_num: int,
    x_slice_num: int,
    macro_w_value_range: tuple[int, int],
    x_value_range: tuple[int, int],
) -> None:
    torch.manual_seed(5000 + w_slice_num * 100 + x_slice_num * 10 + macro_w_value_range[1])
    n = 17
    k = 19
    m = 5
    config = _intra_config(
        w_slice_num=w_slice_num,
        x_slice_num=x_slice_num,
        w_value_range=macro_w_value_range,
        x_value_range=x_value_range,
    )
    unit = _build_linear(config, w_logical_shape=(n, k))
    weight = _randint_in_range(unit.w_value_range, (n, k))
    activation = _randint_in_range(unit.x_value_range, (m, k))
    _assert_unit_matches_torch(unit, weight, activation)


def test_direct_and_inter_slice_one_agree() -> None:
    torch.manual_seed(4)
    n = 13
    k = 20
    m = 8
    direct = _build_linear(_direct_config(), w_logical_shape=(n, k))
    inter = _build_linear(_inter_config(w_slice_num=1, x_slice_num=1), w_logical_shape=(n, k))
    weight = torch.randint(-3, 4, (n, k), dtype=torch.int32)
    activation = torch.randint(0, 2, (m, k), dtype=torch.int32)

    direct.program(weight)
    inter.program(weight)
    assert torch.equal(
        direct.linear(activation, quantization_mode=_TEST_QUANTIZATION_MODE, adc_active_bits=_TEST_ADC_BITS),
        inter.linear(activation, quantization_mode=_TEST_QUANTIZATION_MODE, adc_active_bits=_TEST_ADC_BITS),
    )


def test_inter_and_intra_slice_engines_agree() -> None:
    torch.manual_seed(5)
    n = 13
    k = 20
    m = 8
    inter = _build_linear(_inter_config(w_slice_num=3, x_slice_num=4), w_logical_shape=(n, k))
    intra = _build_linear(_intra_config(w_slice_num=3, x_slice_num=4), w_logical_shape=(n, k))
    weight = torch.randint(-63, 64, (n, k), dtype=torch.int32)
    activation = torch.randint(0, 16, (m, k), dtype=torch.int32)

    inter.program(weight)
    intra.program(weight)
    assert torch.equal(
        inter.linear(activation, quantization_mode=_TEST_QUANTIZATION_MODE, adc_active_bits=_TEST_ADC_BITS),
        intra.linear(activation, quantization_mode=_TEST_QUANTIZATION_MODE, adc_active_bits=_TEST_ADC_BITS),
    )


@pytest.mark.parametrize(
    ("macro_kind", "config"),
    [
        ("ideal", _ideal_unit_config()),
        ("direct", _direct_config()),
        ("inter", _inter_config(w_slice_num=3, x_slice_num=4)),
        ("intra", _intra_config(w_slice_num=3, x_slice_num=4)),
    ],
)
def test_unit_program_replaces_owned_weight_state(
    macro_kind: str,
    config: IdealLinearUnitConfig | LinearCimUnitConfig,
) -> None:
    torch.manual_seed(6000)
    n = 13
    k = 20
    m = 8
    unit = _build_unit_for_kind(macro_kind, config, w_logical_shape=(n, k))
    activation = _randint_in_range(unit.x_value_range, (m, k))
    weight_a = _randint_in_range(unit.w_value_range, (n, k))
    weight_b = _randint_in_range(unit.w_value_range, (n, k))

    y_a = _assert_unit_matches_torch(unit, weight_a, activation)
    y_b = _assert_unit_matches_torch(unit, weight_b, activation)
    assert not torch.equal(y_a, y_b)


@pytest.mark.parametrize(
    ("macro_kind", "config", "weight_shape", "activation_shape"),
    [
        ("ideal", _ideal_unit_config(), (13, 20), (2, 20)),
        ("ideal", _ideal_unit_config(), (13, 20), (8, 2, 20)),
        ("direct", _direct_config(), (13, 20), (2, 20)),
        ("direct", _direct_config(), (13, 20), (8, 2, 20)),
        ("inter", _inter_config(w_slice_num=3, x_slice_num=4), (13, 20), (2, 20)),
        ("inter", _inter_config(w_slice_num=3, x_slice_num=4), (13, 20), (8, 2, 20)),
        ("intra", _intra_config(w_slice_num=3, x_slice_num=4), (13, 20), (2, 20)),
        ("intra", _intra_config(w_slice_num=3, x_slice_num=4), (13, 20), (8, 2, 20)),
        ("inter", _inter_config(w_slice_num=2, x_slice_num=3), (5, 7), (4, 2, 3, 7)),
        ("intra", _intra_config(w_slice_num=2, x_slice_num=3), (5, 7), (4, 2, 3, 7)),
    ],
)
def test_unit_supports_activation_batch_prefixes(
    macro_kind: str,
    config: IdealLinearUnitConfig | LinearCimUnitConfig,
    weight_shape: tuple[int, ...],
    activation_shape: tuple[int, ...],
) -> None:
    torch.manual_seed(7000 + len(weight_shape) * 100 + len(activation_shape))
    unit = _build_unit_for_kind(macro_kind, config, w_logical_shape=weight_shape)
    weight = _randint_in_range(unit.w_value_range, weight_shape)
    activation = _randint_in_range(unit.x_value_range, activation_shape)
    _assert_unit_matches_torch(unit, weight, activation)


@pytest.mark.parametrize(
    ("macro_kind", "config"),
    [
        ("direct", _direct_config(max_active_num=4)),
        ("inter", _inter_config(w_slice_num=3, x_slice_num=4, max_active_num=4)),
        ("intra", _intra_config(w_slice_num=3, x_slice_num=4, max_active_num=4)),
    ],
)
def test_multi_input_phase_exact_unit_matches_torch_matmul(
    macro_kind: str,
    config: LinearCimUnitConfig,
) -> None:
    """P=4 exact input phases still match `torch.matmul`."""
    torch.manual_seed(8000)
    n = 13
    k = 20
    m = 8
    unit = _build_unit_for_kind(macro_kind, config, w_logical_shape=(n, k))
    weight = _randint_in_range(unit.w_value_range, (n, k))
    activation = _randint_in_range(unit.x_value_range, (m, k))
    _assert_unit_matches_torch(unit, weight, activation)


def test_direct_engine_unit_multi_input_phase_quantized_end_to_end() -> None:
    """P=2 output equals per-phase quantized codes accumulated over P."""
    torch.manual_seed(8100)
    n = 8
    k = 16
    m = 5
    adc_bits = 6
    max_active_num = 8
    input_phase_num = 2
    # One code carries two MAC units, so the per-plane floor is lossy and
    # quantize-then-accumulate cannot coincide with the whole-dot reading.
    factor = 2.0
    config = _wrap_unit(
        _direct_engine_config(
            max_active_num=max_active_num,
            rescale_factors=(factor,),
            adc_bits=adc_bits,
        )
    )
    unit = _build_linear(config, w_logical_shape=(n, k))  # .eval() → deterministic floor
    weight = _randint_in_range(unit.w_value_range, (n, k))
    activation = _randint_in_range(unit.x_value_range, (m, k))
    unit.program(weight)
    actual = unit.linear(activation, quantization_mode=0, adc_active_bits=adc_bits)

    # Reference: per-phase partial dots quantized by the calibrated factor and then
    # accumulated over the input-phase axis (Tc = G = D = 1).
    def _convert(dot: torch.Tensor) -> torch.Tensor:
        zero_point = 1 << (adc_bits - 1)
        return torch.div(dot, int(factor), rounding_mode="floor").clamp(-zero_point, zero_point - 1)

    xp = activation.to(torch.int64).unflatten(-1, (input_phase_num, max_active_num))
    wp = weight.to(torch.int64).unflatten(-1, (input_phase_num, max_active_num))
    plane_dot = torch.einsum("mpa,npa->mpn", xp, wp)
    # Shape: [m, input_phase, n] -> [m, n]
    expected = _convert(plane_dot).sum(dim=-2)

    assert actual.shape == (m, n)
    assert torch.equal(actual.to(torch.int64), expected)
    # Quantize-then-accumulate must differ from accumulate-then-quantize on
    # this random draw — otherwise the case does not pin the input-phase
    # semantics.
    whole_dot = activation.to(torch.int64) @ weight.to(torch.int64).transpose(-1, -2)
    assert not torch.equal(expected, _convert(whole_dot))


def test_phase_accumulator_energy_scales_with_input_phase_num() -> None:
    """The phase accumulator bills one operation per arriving per-phase code: P=2 logs twice the energy of P=1."""
    torch.manual_seed(8200)
    n = 8
    k = 16
    m = 5
    energies: dict[int, float] = {}
    for max_active_num in (16, 8):  # P = 1, P = 2
        config = _wrap_unit(
            _direct_engine_config(
                max_active_num=max_active_num,
                phase_accumulator_config=AccumulatorConfig(
                    bit_width=32,
                    energy_per_op__fJ=1.0,
                    latency_per_op__ns=0.0,
                    leakage_per_inst__uW=0.0,
                    area_per_inst__um2=0.0,
                ),
            )
        )
        unit = _build_linear(config, w_logical_shape=(n, k))
        assert isinstance(unit.engine.input_activation.phase_accumulator, SerialAccumulator)
        weight = _randint_in_range(unit.w_value_range, (n, k))
        activation = _randint_in_range(unit.x_value_range, (m, k))
        unit.program(weight)
        stamp_names(unit)
        accumulator_name = unit.engine.input_activation.phase_accumulator.qualified_name
        with Profiler() as profiler:
            unit.linear(activation, quantization_mode=_TEST_QUANTIZATION_MODE, adc_active_bits=_TEST_ADC_BITS)
        energies[unit.engine.input_activation._input_phase_num] = sum(
            record.dynamic_energy__fJ for record in profiler.records if record.qualified_name == accumulator_name
        )
    assert energies[1] > 0.0
    assert energies[2] == pytest.approx(2.0 * energies[1])


def test_ideal_unit_public_properties() -> None:
    unit = _build_ideal(
        _ideal_unit_config(x_value_range=(-5, 7), w_value_range=(-11, 13)),
        w_logical_shape=(13, 20),
    )
    assert unit.w_value_range == (-11, 13)
    assert unit.x_value_range == (-5, 7)
    # An ideal unit never quantizes its output: no ADC width, identity rescale.
    assert unit.adc_bits is None
    assert unit.rescale_factor(quantization_mode=0, adc_active_bits=1) == 1.0


def test_direct_engine_unit_public_properties() -> None:
    unit = _build_linear(
        _direct_config(x_value_range=(0, 3), w_value_range=(-15, 15)),
        w_logical_shape=(13, 20),
    )
    assert unit.w_value_range == (-15, 15)
    assert unit.x_value_range == (0, 3)
    assert unit.adc_bits == _TEST_ADC_MAX_BITS
    # The exact ideal-macro path uses the identity rescale.
    assert unit.rescale_factor(quantization_mode=_TEST_QUANTIZATION_MODE, adc_active_bits=_TEST_ADC_BITS) == 1.0


def test_inter_array_slice_engine_unit_public_properties() -> None:
    config = _inter_config(
        w_slice_num=3,
        x_slice_num=2,
        x_value_range=(0, 3),
        w_value_range=(-15, 15),
    )
    unit = _build_linear(config, w_logical_shape=(13, 20))
    assert unit.w_value_range == (-4095, 4095)
    assert unit.x_value_range == (0, 15)
    # The exact ideal-macro path uses the identity rescale.
    assert unit.rescale_factor(quantization_mode=_TEST_QUANTIZATION_MODE, adc_active_bits=_TEST_ADC_BITS) == 1.0


def test_intra_array_slice_engine_unit_public_properties() -> None:
    config = _intra_config(
        w_slice_num=3,
        x_slice_num=2,
        x_value_range=(0, 3),
        w_value_range=(-15, 15),
    )
    unit = _build_linear(config, w_logical_shape=(13, 20))
    assert unit.w_value_range == (-4095, 4095)
    assert unit.x_value_range == (0, 15)
    # The exact ideal-macro path uses the identity rescale.
    assert unit.rescale_factor(quantization_mode=_TEST_QUANTIZATION_MODE, adc_active_bits=_TEST_ADC_BITS) == 1.0


@pytest.mark.parametrize(
    ("config", "expected_type"),
    [
        (_ideal_unit_config(), IdealLinearUnit),
        (_direct_config(), LinearCimUnit),
        (_inter_config(w_slice_num=2, x_slice_num=3), LinearCimUnit),
        (_intra_config(w_slice_num=2, x_slice_num=3), LinearCimUnit),
    ],
)
def test_unit_from_config_dispatches_to_registered_subclass(
    config: IdealLinearUnitConfig | LinearCimUnitConfig,
    expected_type: _CimUnitType,
) -> None:
    policy = _IDEAL_UNIT_POLICY if isinstance(config, IdealLinearUnitConfig) else _linear_unit_policy(config)
    unit = CimUnit.from_config(
        config=config,
        policy=policy,
        w_logical_shape=(13, 20),
        dtype=torch.float32,
        T__K=300.0,
        ideal_macro=False,
    )
    unit.eval()
    assert isinstance(unit, expected_type)


def test_unit_from_config_rejects_mismatched_policy_type() -> None:
    config = _direct_config()
    with pytest.raises(
        TypeError,
        match=r"no CimUnit module registered for config LinearCimUnitConfig and policy IdealLinearUnitPolicy",
    ):
        CimUnit.from_config(
            config=config,
            policy=_IDEAL_UNIT_POLICY,
            w_logical_shape=(13, 20),
            dtype=torch.float32,
            T__K=300.0,
            ideal_macro=False,
        )


def test_linear_cim_unit_rejects_non_2d_w_logical_shape() -> None:
    with pytest.raises(ValueError, match=r"w_logical_shape must be \(N, K\)"):
        _build_linear(_direct_config(), w_logical_shape=(2, 13, 20))


def test_ideal_linear_unit_rejects_non_2d_w_logical_shape() -> None:
    with pytest.raises(ValueError, match=r"w_logical_shape must be \(N, K\)"):
        _build_ideal(_ideal_unit_config(), w_logical_shape=(2, 13, 20))


def test_engine_rejects_non_2d_w_logical_shape() -> None:
    engine_config = _direct_engine_config()
    with pytest.raises(ValueError, match=r"w_logical_shape must be \(N, K\)"):
        CimEngine.from_config(
            config=engine_config,
            policy=_engine_policy(engine_config),
            w_logical_shape=(2, 13, 20),
            dtype=torch.float32,
            T__K=300.0,
            ideal_macro=False,
        )


@pytest.mark.parametrize(
    ("engine_config", "expected_weight_stage", "expected_x_stage"),
    [
        (_direct_engine_config(), DirectWeightSliceStage, DirectXSliceStage),
        (
            _sliced_engine_config(
                w_slice_num=2,
                x_slice_num=3,
                weight_slice_type=InterWeightSliceStageConfig,
            ),
            InterWeightSliceStage,
            SerialXSliceStage,
        ),
        (
            _sliced_engine_config(
                w_slice_num=2,
                x_slice_num=3,
                weight_slice_type=IntraWeightSliceStageConfig,
            ),
            IntraWeightSliceStage,
            SerialXSliceStage,
        ),
    ],
)
def test_engine_from_config_dispatches_composed_stages(
    engine_config: CimEngineConfig,
    expected_weight_stage: _WeightSliceStageType,
    expected_x_stage: _XSliceStageType,
) -> None:
    engine = CimEngine.from_config(
        config=engine_config,
        policy=_engine_policy(engine_config),
        w_logical_shape=(13, 20),
        dtype=torch.float32,
        T__K=300.0,
        ideal_macro=False,
    )
    assert type(engine) is CimEngine
    assert isinstance(engine.weight_slice, expected_weight_stage)
    assert isinstance(engine.x_slice, expected_x_stage)


def test_unit_config_nested_engine_deserialization() -> None:
    """Receiver-bounded deserialization resolves the unit, engine, and macro leaves from `_neurox_class`."""
    config_dict: ConfigDict = {
        "_neurox_class": "LinearCimUnitConfig",
        "area_per_inst__um2": 0.0,
        "leakage_per_inst__uW": 0.0,
        "engine": {
            "cim_macro_config": {
                "_neurox_class": "IdealCimMacroConfig",
                "input_num": 16,
                "max_active_num": 16,
                "lane_num": 1,
                "scan_num": 16,
                "leakage_per_inst__uW": 0.0,
                "area_per_inst__um2": 0.0,
                "w_digit_num": 2,
                "w_digit_radix": 2,
                "w_encoding": "true_form",
                "x_digit_num": 2,
                "x_digit_radix": 2,
                "x_encoding": "unsigned",
                "x_value_range": [0, 1],
                "w_value_range": [-3, 3],
                "rescale_factors": [1.0],
                "adc_bits": 8,
                "quantization_scheme": "zero_point",
            },
            "placement": {
                "contraction_accumulator_config": {"bit_width": 32, **_zero_ppa()},
            },
            "input_activation": {
                "phase_accumulator_config": {"bit_width": 32, **_zero_ppa()},
            },
            "weight_slice": {
                "_neurox_class": "DirectWeightSliceStageConfig",
            },
            "x_slice": {
                "_neurox_class": "DirectXSliceStageConfig",
            },
        },
    }
    config = CimUnitConfig.from_dict(config_dict)
    assert type(config) is LinearCimUnitConfig
    assert type(config.engine) is CimEngineConfig
    assert type(config.engine.cim_macro_config) is IdealCimMacroConfig
    assert type(config.engine.input_activation) is InputActivationStageConfig
    assert type(config.engine.weight_slice) is DirectWeightSliceStageConfig
    assert type(config.engine.x_slice) is DirectXSliceStageConfig


def test_unit_policy_nested_engine_deserialization() -> None:
    config_dict: ConfigDict = {
        "_neurox_class": "LinearCimUnitPolicy",
        "engine": {
            "cim_macro_policy": {
                "_neurox_class": "IdealCimMacroPolicy",
            },
            "placement": {},
            "input_activation": {},
            "weight_slice": {
                "_neurox_class": "DirectWeightSliceStagePolicy",
            },
            "x_slice": {
                "_neurox_class": "DirectXSliceStagePolicy",
            },
        },
    }
    policy = CimUnitPolicy.from_dict(config_dict)
    assert type(policy) is LinearCimUnitPolicy
    assert type(policy.engine) is CimEnginePolicy
    assert type(policy.engine.cim_macro_policy) is IdealCimMacroPolicy
    assert type(policy.engine.input_activation) is InputActivationStagePolicy
    assert type(policy.engine.weight_slice) is DirectWeightSliceStagePolicy
    assert type(policy.engine.x_slice) is DirectXSliceStagePolicy
