"""Tests for unit-backed CIM unit behavior."""

from __future__ import annotations

import pytest
import torch

from neurox.architecture.mapping import (
    TilingMode,
)
from neurox.architecture.unit.linear import LinearCimUnit, LinearCimUnitConfig, LinearCimUnitPolicy, LinearUnit
from neurox.encoding import Encoding
from neurox.primitive.digital import AccumulatorConfig, ShiftAdderConfig
from neurox.primitive.macro.cim import (
    CimMacroQuantizationScheme,
    IdealCimMacroConfig,
    IdealCimMacroPolicy,
)

# All tests use IdealCimMacroConfig as the embedded macro config, so its
# nonideality policy is the empty marker.
_IDEAL_MACRO_POLICY = IdealCimMacroPolicy()

# `adc_active_bits = None` bypasses the virtual ADC for exact integer products.
_TEST_ADC_BITS = None
_TEST_QUANTIZATION_MODE = 0
_TEST_ADC_MAX_BITS = 8
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


def _unit_policy() -> LinearCimUnitPolicy:
    return LinearCimUnitPolicy(
        cim_macro_policy=_IDEAL_MACRO_POLICY,
    )


def _linear_unit_policy() -> LinearCimUnitPolicy:
    return _unit_policy()


def _direct_unit_config(
    *,
    x_value_range: tuple[int, int] = (0, 1),
    w_value_range: tuple[int, int] = (-3, 3),
    max_active_num: int | None = None,
    rescale_factors: tuple[float, ...] = _TEST_RESCALE_FACTORS,
    adc_bits: int = _TEST_ADC_MAX_BITS,
    phase_accumulator_config: AccumulatorConfig | None = None,
) -> LinearCimUnitConfig:
    return LinearCimUnitConfig(
        cim_macro_config=_ideal_macro_config(
            x_value_range=x_value_range,
            w_value_range=w_value_range,
            max_active_num=max_active_num,
            rescale_factors=rescale_factors,
            adc_bits=adc_bits,
        ),
        phase_accumulator_config=_accumulator_config()
        if phase_accumulator_config is None
        else phase_accumulator_config,
        w_slice_num=1,
        w_slice_encoding=None,
        w_shift_adder_config=None,
        tiling=TilingMode.SLICE_PLANES,
        x_slice_num=1,
        x_slice_encoding=None,
        x_shift_adder_config=None,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
    )


def _direct_config(
    *,
    x_value_range: tuple[int, int] = (0, 1),
    w_value_range: tuple[int, int] = (-3, 3),
    max_active_num: int | None = None,
) -> LinearCimUnitConfig:
    return _direct_unit_config(x_value_range=x_value_range, w_value_range=w_value_range, max_active_num=max_active_num)


def _sliced_unit_config(
    *,
    w_slice_num: int,
    x_slice_num: int,
    tiling: TilingMode,
    x_value_range: tuple[int, int] = (0, 1),
    w_value_range: tuple[int, int] = (-3, 3),
    max_active_num: int | None = None,
) -> LinearCimUnitConfig:
    return LinearCimUnitConfig(
        cim_macro_config=_ideal_macro_config(
            x_value_range=x_value_range, w_value_range=w_value_range, max_active_num=max_active_num
        ),
        phase_accumulator_config=_accumulator_config(),
        w_slice_num=w_slice_num,
        w_slice_encoding=Encoding.TRUE_FORM,
        w_shift_adder_config=_shift_adder_config(),
        tiling=tiling,
        x_slice_num=x_slice_num,
        x_slice_encoding=Encoding.UNSIGNED,
        x_shift_adder_config=_shift_adder_config(),
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
    )


def _inter_config(
    *,
    w_slice_num: int,
    x_slice_num: int,
    x_value_range: tuple[int, int] = (0, 1),
    w_value_range: tuple[int, int] = (-3, 3),
    max_active_num: int | None = None,
) -> LinearCimUnitConfig:
    return _sliced_unit_config(
        tiling=TilingMode.SLICE_PLANES,
        w_slice_num=w_slice_num,
        x_slice_num=x_slice_num,
        x_value_range=x_value_range,
        w_value_range=w_value_range,
        max_active_num=max_active_num,
    )


def _intra_config(
    *,
    w_slice_num: int,
    x_slice_num: int,
    x_value_range: tuple[int, int] = (0, 1),
    w_value_range: tuple[int, int] = (-3, 3),
    max_active_num: int | None = None,
) -> LinearCimUnitConfig:
    return _sliced_unit_config(
        tiling=TilingMode.SLICE_OUTPUTS,
        w_slice_num=w_slice_num,
        x_slice_num=x_slice_num,
        x_value_range=x_value_range,
        w_value_range=w_value_range,
        max_active_num=max_active_num,
    )


def _build_linear(
    config: LinearCimUnitConfig,
    *,
    w_logical_shape: tuple[int, ...],
) -> LinearCimUnit:
    unit = LinearCimUnit(
        config=config,
        policy=_linear_unit_policy(),
        w_logical_shape=w_logical_shape,
        dtype=torch.float32,
    )
    unit.eval()
    return unit


def _assert_unit_matches_torch(unit: LinearUnit, weight: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
    unit.program(weight)
    actual = unit.linear(activation, quantization_mode=_TEST_QUANTIZATION_MODE, adc_active_bits=_TEST_ADC_BITS)
    # F.linear-form oracle: activation [..., K] against weight [N, K].
    # Shape: [..., K] -> [..., N]
    expected = torch.matmul(activation.long().unsqueeze(-2), weight.transpose(-1, -2).long()).squeeze(-2)
    assert actual.shape == expected.shape
    assert torch.equal(actual.long(), expected)
    return actual


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


def test_direct_unit_handles_wide_macro_weight_range() -> None:
    torch.manual_seed(1)
    n = 13
    k = 20
    m = 8
    unit = _build_linear(_direct_config(w_value_range=(-15, 15)), w_logical_shape=(n, k))
    weight = torch.randint(-15, 16, (n, k), dtype=torch.int32)
    activation = torch.randint(0, 2, (m, k), dtype=torch.int32)
    _assert_unit_matches_torch(unit, weight, activation)


def test_direct_unit_passes_logical_weights_to_macro() -> None:
    """Direct mapping preserves asymmetric logical weight values."""
    config = _direct_unit_config(w_value_range=(-3, 3))
    unit = _build_linear(config, w_logical_shape=(2, 2))
    weight = torch.tensor([[1, 2], [-1, -2]], dtype=torch.int32)
    activation = torch.tensor([[1, 1]], dtype=torch.int32)
    _assert_unit_matches_torch(unit, weight, activation)


@pytest.mark.parametrize(("n", "k", "m"), _SHAPE_CASES)
def test_inter_array_slice_unit_matches_torch_matmul_for_shape_cases(n: int, k: int, m: int) -> None:
    torch.manual_seed(2000 + n * 13 + k * 7 + m)
    unit = _build_linear(_inter_config(w_slice_num=3, x_slice_num=4), w_logical_shape=(n, k))
    weight = _randint_in_range(unit.w_value_range, (n, k))
    activation = _randint_in_range(unit.x_value_range, (m, k))
    _assert_unit_matches_torch(unit, weight, activation)


@pytest.mark.parametrize(("n", "k", "m"), _SHAPE_CASES)
def test_intra_array_slice_unit_matches_torch_matmul_for_shape_cases(n: int, k: int, m: int) -> None:
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
def test_inter_array_slice_unit_matches_torch_for_slice_range_cases(
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
def test_intra_array_slice_unit_matches_torch_for_slice_range_cases(
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


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_direct_config(), id="direct"),
        pytest.param(_inter_config(w_slice_num=3, x_slice_num=4), id="inter"),
        pytest.param(_intra_config(w_slice_num=3, x_slice_num=4), id="intra"),
    ],
)
def test_unit_program_replaces_owned_weight_state(
    config: LinearCimUnitConfig,
) -> None:
    torch.manual_seed(6000)
    n = 13
    k = 20
    m = 8
    unit = _build_linear(config, w_logical_shape=(n, k))
    activation = _randint_in_range(unit.x_value_range, (m, k))
    weight_a = _randint_in_range(unit.w_value_range, (n, k))
    weight_b = _randint_in_range(unit.w_value_range, (n, k))

    y_a = _assert_unit_matches_torch(unit, weight_a, activation)
    y_b = _assert_unit_matches_torch(unit, weight_b, activation)
    assert not torch.equal(y_a, y_b)


@pytest.mark.parametrize(
    ("config", "weight_shape", "activation_shape"),
    [
        pytest.param(_direct_config(), (13, 20), (2, 20), id="direct"),
        pytest.param(_direct_config(), (13, 20), (8, 2, 20), id="direct"),
        pytest.param(_inter_config(w_slice_num=3, x_slice_num=4), (13, 20), (2, 20), id="inter"),
        pytest.param(_inter_config(w_slice_num=3, x_slice_num=4), (13, 20), (8, 2, 20), id="inter"),
        pytest.param(_intra_config(w_slice_num=3, x_slice_num=4), (13, 20), (2, 20), id="intra"),
        pytest.param(_intra_config(w_slice_num=3, x_slice_num=4), (13, 20), (8, 2, 20), id="intra"),
        pytest.param(_inter_config(w_slice_num=2, x_slice_num=3), (5, 7), (4, 2, 3, 7), id="inter"),
        pytest.param(_intra_config(w_slice_num=2, x_slice_num=3), (5, 7), (4, 2, 3, 7), id="intra"),
    ],
)
def test_unit_supports_activation_batch_prefixes(
    config: LinearCimUnitConfig,
    weight_shape: tuple[int, ...],
    activation_shape: tuple[int, ...],
) -> None:
    torch.manual_seed(7000 + len(weight_shape) * 100 + len(activation_shape))
    unit = _build_linear(config, w_logical_shape=weight_shape)
    weight = _randint_in_range(unit.w_value_range, weight_shape)
    activation = _randint_in_range(unit.x_value_range, activation_shape)
    _assert_unit_matches_torch(unit, weight, activation)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_inter_config(w_slice_num=3, x_slice_num=4, max_active_num=4), id="inter"),
        pytest.param(_intra_config(w_slice_num=3, x_slice_num=4, max_active_num=4), id="intra"),
    ],
)
def test_sliced_multi_input_phase_exact_unit_matches_torch_matmul(
    config: LinearCimUnitConfig,
) -> None:
    """P=4 exact input phases still match `torch.matmul`."""
    torch.manual_seed(8000)
    n = 13
    k = 20
    m = 8
    unit = _build_linear(config, w_logical_shape=(n, k))
    weight = _randint_in_range(unit.w_value_range, (n, k))
    activation = _randint_in_range(unit.x_value_range, (m, k))
    _assert_unit_matches_torch(unit, weight, activation)


def test_direct_unit_multi_input_phase_quantized_end_to_end() -> None:
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
    config = _direct_unit_config(max_active_num=max_active_num, rescale_factors=(factor,), adc_bits=adc_bits)
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

    xp = activation.long().unflatten(-1, (input_phase_num, max_active_num))
    wp = weight.long().unflatten(-1, (input_phase_num, max_active_num))
    plane_dot = torch.einsum("mpa,npa->mpn", xp, wp)
    # Shape: [m, input_phase, n] -> [m, n]
    expected = _convert(plane_dot).sum(dim=-2)

    assert actual.shape == (m, n)
    assert torch.equal(actual.long(), expected)
    # Quantize-then-accumulate must differ from accumulate-then-quantize on
    # this random draw — otherwise the case does not pin the input-phase
    # semantics.
    whole_dot = activation.long() @ weight.long().transpose(-1, -2)
    assert not torch.equal(expected, _convert(whole_dot))
