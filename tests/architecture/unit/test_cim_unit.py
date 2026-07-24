"""Tests for engine-backed CIM unit behavior."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from neurox.architecture.unit import (
    IdealLinearUnit,
    IdealLinearUnitConfig,
    IdealLinearUnitPolicy,
    LinearUnit,
)
from neurox.architecture.unit.cim import (
    CimUnit,
    CimUnitConfig,
    LinearCimUnit,
    LinearCimUnitConfig,
    LinearCimUnitPolicy,
)
from neurox.architecture.unit.cim.engine import (
    CimEngine,
    CimEnginePolicy,
    DirectCimEngine,
    DirectCimEngineConfig,
    InterArraySliceCimEngine,
    InterArraySliceCimEngineConfig,
    IntraArraySliceCimEngine,
    IntraArraySliceCimEngineConfig,
)
from neurox.common.profiler import NeuroxProfiler
from neurox.primitive.digital import AccumulatorConfig, SerialAccumulator, ShiftAdderConfig
from neurox.primitive.macro.cim import IdealCimMacroConfig, IdealCimMacroPolicy

# All tests use IdealCimMacroConfig as the embedded xbar config, so its nonideality
# policy is the empty marker; the units surrounding it carry that empty
# policy in their `cim_macro_policy=` field.
_IDEAL_XBAR_POLICY = IdealCimMacroPolicy()
_IDEAL_UNIT_POLICY = IdealLinearUnitPolicy()
_LINEAR_UNIT_POLICY = LinearCimUnitPolicy(cim_macro_policy=_IDEAL_XBAR_POLICY)

# Test-only sentinel: ``adc_bits == 0`` instructs IdealCimMacro to skip ADC
# quantization and the signed clamp, so unit outputs equal ``torch.matmul``
# exactly — the same behaviour ``IdealLinearUnit`` provides natively.
_TEST_ADC_BITS = 0
_TEST_ADC_MODE = 0


def _ideal_xbar_config(
    *,
    col_num: int = 16,
    row_num: int = 16,
    active_row_num: int | None = None,
    x_range: tuple[int, int] = (0, 1),
    w_digit_count: int = 1,
    w_digit_radix: int = 4,
    w_digit_range: tuple[int, int] = (-3, 3),
    adc_max_bits: int = _TEST_ADC_BITS,
) -> IdealCimMacroConfig:
    return IdealCimMacroConfig(
        col_num=col_num,
        row_num=row_num,
        active_row_num=row_num if active_row_num is None else active_row_num,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
        x_range=x_range,
        w_digit_count=w_digit_count,
        w_digit_radix=w_digit_radix,
        w_digit_range=w_digit_range,
        adc_mode_num=1,
        adc_max_bits=adc_max_bits,
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


def _wrap_unit(
    engine_config: DirectCimEngineConfig | InterArraySliceCimEngineConfig | IntraArraySliceCimEngineConfig,
) -> LinearCimUnitConfig:
    return LinearCimUnitConfig(
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        engine=engine_config,
    )


def _direct_engine_config(
    *,
    x_range: tuple[int, int] = (0, 1),
    w_digit_count: int = 1,
    active_row_num: int | None = None,
) -> DirectCimEngineConfig:
    return DirectCimEngineConfig(
        cim_macro_config=_ideal_xbar_config(
            x_range=x_range, w_digit_count=w_digit_count, active_row_num=active_row_num
        ),
        w_encoding="true_form",
        col_accumulator_config=_accumulator_config(),
        phase_accumulator_config=_accumulator_config(),
    )


def _direct_config(
    *,
    x_range: tuple[int, int] = (0, 1),
    w_digit_count: int = 1,
    active_row_num: int | None = None,
) -> LinearCimUnitConfig:
    return _wrap_unit(
        _direct_engine_config(x_range=x_range, w_digit_count=w_digit_count, active_row_num=active_row_num)
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


def _slice_config(
    *,
    w_slice_num: int,
    x_slice_num: int,
    x_range: tuple[int, int] = (0, 1),
    w_digit_count: int = 1,
    active_row_num: int | None = None,
) -> dict[str, Any]:
    return {
        "cim_macro_config": _ideal_xbar_config(
            x_range=x_range, w_digit_count=w_digit_count, active_row_num=active_row_num
        ),
        "w_slice_num": w_slice_num,
        "x_slice_num": x_slice_num,
        "w_encoding": "true_form",
        "col_accumulator_config": _accumulator_config(),
        "phase_accumulator_config": _accumulator_config(),
        "sa_shift_adder_config": _shift_adder_config(),
        "sw_shift_adder_config": _shift_adder_config(),
    }


def _inter_config(**kwargs: Any) -> LinearCimUnitConfig:
    return _wrap_unit(InterArraySliceCimEngineConfig(**_slice_config(**kwargs)))


def _intra_config(**kwargs: Any) -> LinearCimUnitConfig:
    return _wrap_unit(IntraArraySliceCimEngineConfig(**_slice_config(**kwargs)))


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
        ideal_xbar=False,
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
        policy=_LINEAR_UNIT_POLICY,
        w_logical_shape=w_logical_shape,
        dtype=torch.float32,
        T__K=300.0,
        ideal_xbar=False,
    )
    unit.eval()
    return unit


def _assert_unit_matches_torch(unit: LinearUnit, weight: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
    unit.program(weight)
    actual = unit.linear(activation, adc_mode=_TEST_ADC_MODE, adc_bits=_TEST_ADC_BITS)
    # F.linear-form oracle: activation [..., K] against weight [*prefix, N, K];
    # the weight prefix broadcasts right-aligned over the activation batch dims.
    # Shape: [..., K] -> [..., 1, K] @ [*prefix, K, N] -> [..., 1, N] -> [..., N]
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
    y = CimEngine.chunk_pad_along(x, axis=0, chunk_size=4, pad_value=0)
    assert y.shape == (4, 4)
    assert torch.equal(y.flatten(), x)


def test_chunk_pad_along_with_padding() -> None:
    x = torch.arange(13)
    y = CimEngine.chunk_pad_along(x, axis=0, chunk_size=16, pad_value=0)
    assert y.shape == (1, 16)
    assert torch.equal(y[0, :13], x)
    assert torch.equal(y[0, 13:], torch.zeros(3, dtype=x.dtype))


def test_chunk_pad_along_accepts_negative_axis() -> None:
    x = torch.arange(60).reshape(3, 4, 5)
    y = CimEngine.chunk_pad_along(x, axis=-1, chunk_size=3, pad_value=0)
    assert y.shape == (3, 4, 2, 3)


@pytest.mark.parametrize(("n", "k", "m"), _SHAPE_CASES)
def test_ideal_unit_matches_torch_matmul_for_shape_cases(n: int, k: int, m: int) -> None:
    torch.manual_seed(900 + n * 13 + k * 7 + m)
    unit = _build_ideal(_ideal_unit_config(), w_logical_shape=(n, k))
    weight = _randint_in_range(unit.w_value_range, (n, k))
    activation = _randint_in_range(unit.x_value_range, (m, k))
    _assert_unit_matches_torch(unit, weight, activation)


def test_ideal_unit_uses_lossless_integer_matmul() -> None:
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


def test_direct_engine_unit_handles_multi_digit_xbar_words() -> None:
    torch.manual_seed(1)
    n, k, m = 13, 20, 8
    unit = _build_linear(_direct_config(w_digit_count=2), w_logical_shape=(n, k))
    weight = torch.randint(-15, 16, (n, k), dtype=torch.int32)
    activation = torch.randint(0, 2, (m, k), dtype=torch.int32)
    _assert_unit_matches_torch(unit, weight, activation)


def test_direct_engine_transcoder_matches_wired_cim_macro_place_values() -> None:
    """BI-1: the wired ``CimMacro``'s digit geometry must drive the engine's
    weight transcoder, so digit place-values line up through the
    ``_build_cim_macro`` boundary (transcoder → macro)."""
    unit = _build_linear(_direct_config(w_digit_count=2), w_logical_shape=(13, 20))
    assert unit.engine.w_transcoder.radix == unit.engine.xbar.w_digit_radix
    assert unit.engine.w_transcoder.digit_count == unit.engine.xbar.w_digit_count


def test_direct_engine_lsb_first_place_values_on_asymmetric_weights() -> None:
    """LSB-first regression on the asymmetric set ``{1, 2, -1, -2}``.

    With ``radix=2, digit_count=2`` the weight value ``1`` encodes to digits
    ``[1, 0]`` and ``2`` to ``[0, 1]`` (digit index 0 = LSB = radix⁰). A reversed
    digit convention would swap their place-values and mis-map ``1 ↔ 2``.
    Programming this asymmetric set and matching ``torch.matmul`` pins the
    LSB-first place-value wiring through the transcoder → macro path.
    """
    config = _wrap_unit(
        DirectCimEngineConfig(
            cim_macro_config=_ideal_xbar_config(w_digit_count=2, w_digit_radix=2, w_digit_range=(-1, 1)),
            w_encoding="true_form",
            col_accumulator_config=_accumulator_config(),
            phase_accumulator_config=_accumulator_config(),
        )
    )
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
    ("w_slice_num", "x_slice_num", "w_digit_count", "x_range"),
    [
        (1, 1, 2, (0, 1)),
        (2, 3, 2, (0, 1)),
        (3, 2, 2, (0, 3)),
    ],
)
def test_inter_array_slice_engine_unit_matches_torch_for_slice_digit_cases(
    w_slice_num: int,
    x_slice_num: int,
    w_digit_count: int,
    x_range: tuple[int, int],
) -> None:
    torch.manual_seed(4000 + w_slice_num * 100 + x_slice_num * 10 + w_digit_count)
    n, k, m = 17, 19, 5
    config = _inter_config(
        w_slice_num=w_slice_num,
        x_slice_num=x_slice_num,
        w_digit_count=w_digit_count,
        x_range=x_range,
    )
    unit = _build_linear(config, w_logical_shape=(n, k))
    weight = _randint_in_range(unit.w_value_range, (n, k))
    activation = _randint_in_range(unit.x_value_range, (m, k))
    _assert_unit_matches_torch(unit, weight, activation)


@pytest.mark.parametrize(
    ("w_slice_num", "x_slice_num", "w_digit_count", "x_range"),
    [
        (1, 1, 2, (0, 1)),
        (2, 3, 2, (0, 1)),
        (3, 2, 2, (0, 3)),
    ],
)
def test_intra_array_slice_engine_unit_matches_torch_for_slice_digit_cases(
    w_slice_num: int,
    x_slice_num: int,
    w_digit_count: int,
    x_range: tuple[int, int],
) -> None:
    torch.manual_seed(5000 + w_slice_num * 100 + x_slice_num * 10 + w_digit_count)
    n, k, m = 17, 19, 5
    config = _intra_config(
        w_slice_num=w_slice_num,
        x_slice_num=x_slice_num,
        w_digit_count=w_digit_count,
        x_range=x_range,
    )
    unit = _build_linear(config, w_logical_shape=(n, k))
    weight = _randint_in_range(unit.w_value_range, (n, k))
    activation = _randint_in_range(unit.x_value_range, (m, k))
    _assert_unit_matches_torch(unit, weight, activation)


def test_direct_and_inter_slice_one_agree() -> None:
    torch.manual_seed(4)
    n, k, m = 13, 20, 8
    direct = _build_linear(_direct_config(), w_logical_shape=(n, k))
    inter = _build_linear(_inter_config(w_slice_num=1, x_slice_num=1), w_logical_shape=(n, k))
    weight = torch.randint(-3, 4, (n, k), dtype=torch.int32)
    activation = torch.randint(0, 2, (m, k), dtype=torch.int32)

    direct.program(weight)
    inter.program(weight)
    assert torch.equal(
        direct.linear(activation, adc_mode=_TEST_ADC_MODE, adc_bits=_TEST_ADC_BITS),
        inter.linear(activation, adc_mode=_TEST_ADC_MODE, adc_bits=_TEST_ADC_BITS),
    )


def test_inter_and_intra_slice_engines_agree() -> None:
    torch.manual_seed(5)
    n, k, m = 13, 20, 8
    inter = _build_linear(_inter_config(w_slice_num=3, x_slice_num=4), w_logical_shape=(n, k))
    intra = _build_linear(_intra_config(w_slice_num=3, x_slice_num=4), w_logical_shape=(n, k))
    weight = torch.randint(-63, 64, (n, k), dtype=torch.int32)
    activation = torch.randint(0, 16, (m, k), dtype=torch.int32)

    inter.program(weight)
    intra.program(weight)
    assert torch.equal(
        inter.linear(activation, adc_mode=_TEST_ADC_MODE, adc_bits=_TEST_ADC_BITS),
        intra.linear(activation, adc_mode=_TEST_ADC_MODE, adc_bits=_TEST_ADC_BITS),
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
    n, k, m = 13, 20, 8
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
        ("ideal", _ideal_unit_config(), (2, 13, 20), (2, 20)),
        ("ideal", _ideal_unit_config(), (2, 13, 20), (8, 2, 20)),
        ("direct", _direct_config(), (2, 13, 20), (2, 20)),
        ("direct", _direct_config(), (2, 13, 20), (8, 2, 20)),
        ("inter", _inter_config(w_slice_num=3, x_slice_num=4), (2, 13, 20), (2, 20)),
        ("inter", _inter_config(w_slice_num=3, x_slice_num=4), (2, 13, 20), (8, 2, 20)),
        ("intra", _intra_config(w_slice_num=3, x_slice_num=4), (2, 13, 20), (2, 20)),
        ("intra", _intra_config(w_slice_num=3, x_slice_num=4), (2, 13, 20), (8, 2, 20)),
        ("inter", _inter_config(w_slice_num=2, x_slice_num=3), (2, 3, 5, 7), (4, 2, 3, 7)),
        ("intra", _intra_config(w_slice_num=2, x_slice_num=3), (2, 3, 5, 7), (4, 2, 3, 7)),
    ],
)
def test_unit_supports_weight_and_activation_batch_prefixes(
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
        ("direct", _direct_config(active_row_num=4)),
        ("inter", _inter_config(w_slice_num=3, x_slice_num=4, active_row_num=4)),
        ("intra", _intra_config(w_slice_num=3, x_slice_num=4, active_row_num=4)),
    ],
)
def test_multi_sub_phase_lossless_unit_matches_torch_matmul(
    macro_kind: str,
    config: LinearCimUnitConfig,
) -> None:
    """P=4 with the lossless adc_bits=0 sentinel: the phase accumulator sums
    exact per-sub-phase partials, so the unit still matches ``torch.matmul``."""
    torch.manual_seed(8000)
    n, k, m = 13, 20, 8
    unit = _build_unit_for_kind(macro_kind, config, w_logical_shape=(n, k))
    weight = _randint_in_range(unit.w_value_range, (n, k))
    activation = _randint_in_range(unit.x_value_range, (m, k))
    _assert_unit_matches_torch(unit, weight, activation)


def test_direct_engine_unit_multi_sub_phase_quantized_end_to_end() -> None:
    """P=2 with adc_bits>0: the unit output equals per-sub-phase quantized
    codes accumulated over the sub-phase axis, then the Tc/col pipeline."""
    torch.manual_seed(8100)
    n, k, m = 8, 16, 5
    adc_bits = 6
    active_row_num = 8
    sub_phase_num = 2
    config = _wrap_unit(
        DirectCimEngineConfig(
            cim_macro_config=_ideal_xbar_config(active_row_num=active_row_num, adc_max_bits=adc_bits),
            w_encoding="true_form",
            col_accumulator_config=_accumulator_config(),
            phase_accumulator_config=_accumulator_config(),
        )
    )
    unit = _build_linear(config, w_logical_shape=(n, k))  # .eval() → deterministic floor
    weight = _randint_in_range(unit.w_value_range, (n, k))
    activation = _randint_in_range(unit.x_value_range, (m, k))
    unit.program(weight)
    actual = unit.linear(activation, adc_mode=0, adc_bits=adc_bits)

    # Reference: per-sub-phase partial dots, quantized per plane against the
    # per-conversion range, then accumulated over the sub-phase axis
    # (Tc = Tr = 1).
    half_range = (1 << (adc_bits - 1)) - 1
    rescale = (active_row_num * 3 * 1) / half_range  # active_row_num · max|w| · max|x|
    xp = activation.to(torch.int64).unflatten(-1, (sub_phase_num, active_row_num))
    wp = weight.to(torch.int64).unflatten(-1, (sub_phase_num, active_row_num))
    plane_dot = torch.einsum("mpa,npa->mpn", xp, wp)
    bound = 1 << (adc_bits - 1)
    codes = torch.floor(plane_dot.to(torch.float32) * (1.0 / rescale)).to(torch.int64).clamp(-bound, bound - 1)
    expected = codes.sum(dim=-2)  # [m, n]

    assert actual.shape == (m, n)
    assert torch.equal(actual.to(torch.int64), expected)
    # Quantize-then-accumulate must differ from accumulate-then-quantize on
    # this random draw — otherwise the case does not pin the sub-phase
    # semantics.
    whole_dot = activation.to(torch.int64) @ weight.to(torch.int64).transpose(-1, -2)
    whole_code = torch.floor(whole_dot.to(torch.float32) * (1.0 / rescale)).to(torch.int64).clamp(-bound, bound - 1)
    assert not torch.equal(expected, whole_code)


def test_phase_accumulator_energy_scales_with_sub_phase_num() -> None:
    """The phase accumulator is a ``SerialAccumulator`` billed per arriving
    per-sub-phase code, so at fixed geometry its accumulate energy scales
    with the sub-phase count: P=2 logs exactly twice the energy of P=1."""
    torch.manual_seed(8200)
    n, k, m = 8, 16, 5
    energies: dict[int, float] = {}
    for active_row_num in (16, 8):  # P = 1, P = 2
        config = _wrap_unit(
            DirectCimEngineConfig(
                cim_macro_config=_ideal_xbar_config(active_row_num=active_row_num),
                w_encoding="true_form",
                col_accumulator_config=_accumulator_config(),
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
        assert isinstance(unit.engine.phase_accumulator, SerialAccumulator)
        weight = _randint_in_range(unit.w_value_range, (n, k))
        activation = _randint_in_range(unit.x_value_range, (m, k))
        unit.program(weight)
        with NeuroxProfiler() as p:
            unit.linear(activation, adc_mode=_TEST_ADC_MODE, adc_bits=_TEST_ADC_BITS)
        energies[unit.engine._sub_phase_num] = sum(
            e.dynamic_energy__fJ for e in p.energy_events if e.module is unit.engine.phase_accumulator
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
    assert unit.adc_mode_num == 1
    assert unit.adc_max_bits == 0
    assert unit.adc_rescale_factor(adc_mode=0, adc_bits=0) == 1.0


def test_direct_engine_unit_public_properties() -> None:
    unit = _build_linear(
        _direct_config(x_range=(0, 3), w_digit_count=2),
        w_logical_shape=(13, 20),
    )
    assert unit.w_value_range == (-15, 15)
    assert unit.x_value_range == (0, 3)
    assert unit.adc_mode_num == 1
    assert unit.adc_max_bits == _TEST_ADC_BITS
    # Ideal-backed units derive rescale from bit width alone: bits == 0
    # (the full-precision sentinel) → identity rescale of 1.0.
    assert unit.adc_rescale_factor(adc_mode=_TEST_ADC_MODE, adc_bits=_TEST_ADC_BITS) == 1.0


def test_inter_array_slice_engine_unit_public_properties() -> None:
    config = _inter_config(w_slice_num=3, x_slice_num=2, x_range=(0, 3), w_digit_count=2)
    unit = _build_linear(config, w_logical_shape=(13, 20))
    assert unit.w_value_range == (-4095, 4095)
    assert unit.x_value_range == (0, 15)
    # Ideal-backed → bits == 0 → identity rescale.
    assert unit.adc_rescale_factor(adc_mode=_TEST_ADC_MODE, adc_bits=_TEST_ADC_BITS) == 1.0


def test_intra_array_slice_engine_unit_public_properties() -> None:
    config = _intra_config(w_slice_num=3, x_slice_num=2, x_range=(0, 3), w_digit_count=2)
    unit = _build_linear(config, w_logical_shape=(13, 20))
    assert unit.w_value_range == (-4095, 4095)
    assert unit.x_value_range == (0, 15)
    # Ideal-backed → bits == 0 → identity rescale.
    assert unit.adc_rescale_factor(adc_mode=_TEST_ADC_MODE, adc_bits=_TEST_ADC_BITS) == 1.0


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
    expected_type: type[CimUnit],
) -> None:
    policy = _IDEAL_UNIT_POLICY if isinstance(config, IdealLinearUnitConfig) else _LINEAR_UNIT_POLICY
    unit = CimUnit.from_config(
        config=config,
        policy=policy,
        w_logical_shape=(13, 20),
        dtype=torch.float32,
        T__K=300.0,
        ideal_xbar=False,
    )
    unit.eval()
    assert isinstance(unit, expected_type)


@pytest.mark.parametrize(
    ("engine_config", "expected_type"),
    [
        (_direct_engine_config(), DirectCimEngine),
        (InterArraySliceCimEngineConfig(**_slice_config(w_slice_num=2, x_slice_num=3)), InterArraySliceCimEngine),
        (IntraArraySliceCimEngineConfig(**_slice_config(w_slice_num=2, x_slice_num=3)), IntraArraySliceCimEngine),
    ],
)
def test_engine_from_config_dispatches_to_registered_variant(
    engine_config: DirectCimEngineConfig | InterArraySliceCimEngineConfig | IntraArraySliceCimEngineConfig,
    expected_type: type[CimEngine],
) -> None:
    engine = CimEngine.from_config(
        config=engine_config,
        policy=CimEnginePolicy(cim_macro_policy=_IDEAL_XBAR_POLICY),
        w_logical_shape=(13, 20),
        dtype=torch.float32,
        T__K=300.0,
        ideal_xbar=False,
    )
    assert isinstance(engine, expected_type)


def test_unit_config_nested_engine_deserialization() -> None:
    """Receiver-bounded deserialization resolves the unit, engine, and macro
    leaves from their ``_neurox_class`` discriminators."""
    payload = {
        "_neurox_class": "LinearCimUnitConfig",
        "area_per_inst__um2": 0.0,
        "leakage_per_inst__uW": 0.0,
        "engine": {
            "_neurox_class": "DirectCimEngineConfig",
            "cim_macro_config": {
                "_neurox_class": "IdealCimMacroConfig",
                "col_num": 16,
                "row_num": 16,
                "active_row_num": 16,
                "leakage_per_inst__uW": 0.0,
                "area_per_inst__um2": 0.0,
                "x_range": [0, 1],
                "w_digit_count": 1,
                "w_digit_radix": 4,
                "w_digit_range": [-3, 3],
                "adc_mode_num": 1,
                "adc_max_bits": 0,
            },
            "w_encoding": "true_form",
            "phase_accumulator_config": {"bit_width": 32, **_zero_ppa()},
            "col_accumulator_config": {"bit_width": 32, **_zero_ppa()},
        },
    }
    config = CimUnitConfig.from_dict(payload)
    assert type(config) is LinearCimUnitConfig
    assert type(config.engine) is DirectCimEngineConfig
    assert type(config.engine.cim_macro_config) is IdealCimMacroConfig
