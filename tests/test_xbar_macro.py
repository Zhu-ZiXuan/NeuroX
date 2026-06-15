"""Tests for xbar-backed macro behavior."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from neurox.analog.adc import AdcOperationPoint
from neurox.digital import AccumulatorConfig, ShiftAdderConfig
from neurox.macro.xbar import (
    DirectXbarMacro,
    DirectXbarMacroConfig,
    DirectXbarMacroPolicy,
    IdealXbarMacro,
    IdealXbarMacroConfig,
    IdealXbarMacroPolicy,
    InterArraySliceXbarMacro,
    InterArraySliceXbarMacroConfig,
    InterArraySliceXbarMacroPolicy,
    IntraArraySliceXbarMacro,
    IntraArraySliceXbarMacroConfig,
    IntraArraySliceXbarMacroPolicy,
    XbarMacro,
)
from neurox.xbar import IdealXbarConfig, IdealXbarPolicy

# All tests use IdealXbarConfig as the embedded xbar config, so its nonideality
# policy is the empty marker; the macros surrounding it carry that empty
# policy in their `xbar=` field.
_IDEAL_XBAR_POLICY = IdealXbarPolicy()
_IDEAL_MACRO_POLICY = IdealXbarMacroPolicy()
_DIRECT_MACRO_POLICY = DirectXbarMacroPolicy(xbar=_IDEAL_XBAR_POLICY)
_INTER_MACRO_POLICY = InterArraySliceXbarMacroPolicy(xbar=_IDEAL_XBAR_POLICY)
_INTRA_MACRO_POLICY = IntraArraySliceXbarMacroPolicy(xbar=_IDEAL_XBAR_POLICY)

# Test-only sentinel: ``adc_bits == 0`` instructs IdealXbar to skip ADC
# quantization and the signed clamp, so macro outputs equal ``torch.matmul``
# exactly — the same behaviour ``IdealXbarMacro`` provides natively.
_TEST_ADC_BITS = 0
_TEST_ADC_OP = AdcOperationPoint(adc_mode=0, adc_bits=_TEST_ADC_BITS)


def _ideal_xbar_config(
    *,
    col_num: int = 16,
    row_num: int = 16,
    x_range: tuple[int, int] = (0, 1),
    w_digit_count: int = 1,
    w_digit_radix: int = 4,
    w_digit_range: tuple[int, int] = (-3, 3),
) -> IdealXbarConfig:
    return IdealXbarConfig(
        col_num=col_num,
        row_num=row_num,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
        x_range=x_range,
        w_digit_count=w_digit_count,
        w_digit_radix=w_digit_radix,
        w_digit_range=w_digit_range,
        adc_mode_num=1,
        adc_max_bits=_TEST_ADC_BITS,
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


def _direct_config(
    *,
    x_range: tuple[int, int] = (0, 1),
    w_digit_count: int = 1,
) -> DirectXbarMacroConfig:
    return DirectXbarMacroConfig(
        xbar_config=_ideal_xbar_config(x_range=x_range, w_digit_count=w_digit_count),
        w_encoding="true_form",
        col_accumulator_config=_accumulator_config(),
    )


def _ideal_macro_config(
    *,
    x_value_range: tuple[int, int] = (0, 1),
    w_value_range: tuple[int, int] = (-3, 3),
) -> IdealXbarMacroConfig:
    return IdealXbarMacroConfig(
        x_value_range=x_value_range,
        w_value_range=w_value_range,
    )


def _slice_config(
    *,
    w_slice_num: int,
    x_slice_num: int,
    x_range: tuple[int, int] = (0, 1),
    w_digit_count: int = 1,
) -> dict[str, Any]:
    return {
        "xbar_config": _ideal_xbar_config(x_range=x_range, w_digit_count=w_digit_count),
        "w_slice_num": w_slice_num,
        "x_slice_num": x_slice_num,
        "w_encoding": "true_form",
        "col_accumulator_config": _accumulator_config(),
        "sa_shift_adder_config": _shift_adder_config(),
        "sw_shift_adder_config": _shift_adder_config(),
    }


def _build_ideal(
    config: IdealXbarMacroConfig,
    *,
    name: str,
    w_logical_shape: tuple[int, ...],
) -> IdealXbarMacro:
    macro = IdealXbarMacro(
        config=config,
        policy=_IDEAL_MACRO_POLICY,
        name=name,
        w_logical_shape=w_logical_shape,
        dtype=torch.float32,
        T__K=300.0,
        ideal_xbar=False,
    )
    macro.eval()
    return macro


def _build_direct(
    config: DirectXbarMacroConfig,
    *,
    name: str,
    w_logical_shape: tuple[int, ...],
) -> DirectXbarMacro:
    macro = DirectXbarMacro(
        config=config,
        policy=_DIRECT_MACRO_POLICY,
        name=name,
        w_logical_shape=w_logical_shape,
        dtype=torch.float32,
        T__K=300.0,
        ideal_xbar=False,
    )
    macro.eval()
    return macro


def _build_inter(
    config: InterArraySliceXbarMacroConfig,
    *,
    name: str,
    w_logical_shape: tuple[int, ...],
) -> InterArraySliceXbarMacro:
    macro = InterArraySliceXbarMacro(
        config=config,
        policy=_INTER_MACRO_POLICY,
        name=name,
        w_logical_shape=w_logical_shape,
        dtype=torch.float32,
        T__K=300.0,
        ideal_xbar=False,
    )
    macro.eval()
    return macro


def _build_intra(
    config: IntraArraySliceXbarMacroConfig,
    *,
    name: str,
    w_logical_shape: tuple[int, ...],
) -> IntraArraySliceXbarMacro:
    macro = IntraArraySliceXbarMacro(
        config=config,
        policy=_INTRA_MACRO_POLICY,
        name=name,
        w_logical_shape=w_logical_shape,
        dtype=torch.float32,
        T__K=300.0,
        ideal_xbar=False,
    )
    macro.eval()
    return macro


def _assert_macro_matches_torch(macro: XbarMacro, weight: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
    macro.program(weight)
    actual = macro.matmul(activation, adc_operation_point=_TEST_ADC_OP)
    expected = torch.matmul(activation.to(torch.int64), weight.transpose(-1, -2).to(torch.int64))
    assert actual.shape == expected.shape
    assert torch.equal(actual.to(torch.int64), expected)
    return actual


def _build_macro_for_kind(
    macro_kind: str,
    config: IdealXbarMacroConfig
    | DirectXbarMacroConfig
    | InterArraySliceXbarMacroConfig
    | IntraArraySliceXbarMacroConfig,
    *,
    name: str,
    w_logical_shape: tuple[int, ...],
) -> XbarMacro:
    if macro_kind == "ideal":
        assert isinstance(config, IdealXbarMacroConfig)
        return _build_ideal(config, name=name, w_logical_shape=w_logical_shape)
    if macro_kind == "direct":
        assert isinstance(config, DirectXbarMacroConfig)
        return _build_direct(config, name=name, w_logical_shape=w_logical_shape)
    if macro_kind == "inter":
        assert isinstance(config, InterArraySliceXbarMacroConfig)
        return _build_inter(config, name=name, w_logical_shape=w_logical_shape)
    assert isinstance(config, IntraArraySliceXbarMacroConfig)
    return _build_intra(config, name=name, w_logical_shape=w_logical_shape)


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
    y = XbarMacro.chunk_pad_along(x, axis=0, chunk_size=4, pad_value=0)
    assert y.shape == (4, 4)
    assert torch.equal(y.flatten(), x)


def test_chunk_pad_along_with_padding() -> None:
    x = torch.arange(13)
    y = XbarMacro.chunk_pad_along(x, axis=0, chunk_size=16, pad_value=0)
    assert y.shape == (1, 16)
    assert torch.equal(y[0, :13], x)
    assert torch.equal(y[0, 13:], torch.zeros(3, dtype=x.dtype))


def test_chunk_pad_along_accepts_negative_axis() -> None:
    x = torch.arange(60).reshape(3, 4, 5)
    y = XbarMacro.chunk_pad_along(x, axis=-1, chunk_size=3, pad_value=0)
    assert y.shape == (3, 4, 2, 3)


@pytest.mark.parametrize(("n", "k", "m"), _SHAPE_CASES)
def test_ideal_xbar_macro_matches_torch_matmul_for_shape_cases(n: int, k: int, m: int) -> None:
    torch.manual_seed(900 + n * 13 + k * 7 + m)
    macro = _build_ideal(_ideal_macro_config(), name="ideal", w_logical_shape=(n, k))
    weight = _randint_in_range(macro.w_value_range, (n, k))
    activation = _randint_in_range(macro.x_value_range, (m, k))
    _assert_macro_matches_torch(macro, weight, activation)


def test_ideal_xbar_macro_uses_lossless_integer_matmul() -> None:
    macro = _build_ideal(
        _ideal_macro_config(x_value_range=(-4095, 4095), w_value_range=(-4095, 4095)),
        name="ideal_lossless",
        w_logical_shape=(3, 257),
    )
    weight = torch.full((3, 257), 4095, dtype=torch.int32)
    activation = torch.full((2, 257), 4095, dtype=torch.int32)
    _assert_macro_matches_torch(macro, weight, activation)


@pytest.mark.parametrize(("n", "k", "m"), _SHAPE_CASES)
def test_direct_xbar_macro_matches_torch_matmul_for_shape_cases(n: int, k: int, m: int) -> None:
    torch.manual_seed(1000 + n * 13 + k * 7 + m)
    macro = _build_direct(_direct_config(), name="direct", w_logical_shape=(n, k))
    weight = _randint_in_range(macro.w_value_range, (n, k))
    activation = _randint_in_range(macro.x_value_range, (m, k))
    _assert_macro_matches_torch(macro, weight, activation)


def test_direct_xbar_macro_handles_multi_digit_xbar_words() -> None:
    torch.manual_seed(1)
    n, k, m = 13, 20, 8
    macro = _build_direct(_direct_config(w_digit_count=2), name="direct_d2", w_logical_shape=(n, k))
    weight = torch.randint(-15, 16, (n, k), dtype=torch.int32)
    activation = torch.randint(0, 2, (m, k), dtype=torch.int32)
    _assert_macro_matches_torch(macro, weight, activation)


@pytest.mark.parametrize(("n", "k", "m"), _SHAPE_CASES)
def test_inter_array_slice_xbar_macro_matches_torch_matmul_for_shape_cases(n: int, k: int, m: int) -> None:
    torch.manual_seed(2000 + n * 13 + k * 7 + m)
    config = InterArraySliceXbarMacroConfig(**_slice_config(w_slice_num=3, x_slice_num=4))
    macro = _build_inter(config, name="inter", w_logical_shape=(n, k))
    weight = _randint_in_range(macro.w_value_range, (n, k))
    activation = _randint_in_range(macro.x_value_range, (m, k))
    _assert_macro_matches_torch(macro, weight, activation)


@pytest.mark.parametrize(("n", "k", "m"), _SHAPE_CASES)
def test_intra_array_slice_xbar_macro_matches_torch_matmul_for_shape_cases(n: int, k: int, m: int) -> None:
    torch.manual_seed(3000 + n * 13 + k * 7 + m)
    config = IntraArraySliceXbarMacroConfig(**_slice_config(w_slice_num=3, x_slice_num=4))
    macro = _build_intra(config, name="intra", w_logical_shape=(n, k))
    weight = _randint_in_range(macro.w_value_range, (n, k))
    activation = _randint_in_range(macro.x_value_range, (m, k))
    _assert_macro_matches_torch(macro, weight, activation)


@pytest.mark.parametrize(
    ("w_slice_num", "x_slice_num", "w_digit_count", "x_range"),
    [
        (1, 1, 2, (0, 1)),
        (2, 3, 2, (0, 1)),
        (3, 2, 2, (0, 3)),
    ],
)
def test_inter_array_slice_xbar_macro_matches_torch_for_slice_digit_cases(
    w_slice_num: int,
    x_slice_num: int,
    w_digit_count: int,
    x_range: tuple[int, int],
) -> None:
    torch.manual_seed(4000 + w_slice_num * 100 + x_slice_num * 10 + w_digit_count)
    n, k, m = 17, 19, 5
    config = InterArraySliceXbarMacroConfig(
        **_slice_config(
            w_slice_num=w_slice_num,
            x_slice_num=x_slice_num,
            w_digit_count=w_digit_count,
            x_range=x_range,
        )
    )
    macro = _build_inter(config, name="inter_digits", w_logical_shape=(n, k))
    weight = _randint_in_range(macro.w_value_range, (n, k))
    activation = _randint_in_range(macro.x_value_range, (m, k))
    _assert_macro_matches_torch(macro, weight, activation)


@pytest.mark.parametrize(
    ("w_slice_num", "x_slice_num", "w_digit_count", "x_range"),
    [
        (1, 1, 2, (0, 1)),
        (2, 3, 2, (0, 1)),
        (3, 2, 2, (0, 3)),
    ],
)
def test_intra_array_slice_xbar_macro_matches_torch_for_slice_digit_cases(
    w_slice_num: int,
    x_slice_num: int,
    w_digit_count: int,
    x_range: tuple[int, int],
) -> None:
    torch.manual_seed(5000 + w_slice_num * 100 + x_slice_num * 10 + w_digit_count)
    n, k, m = 17, 19, 5
    config = IntraArraySliceXbarMacroConfig(
        **_slice_config(
            w_slice_num=w_slice_num,
            x_slice_num=x_slice_num,
            w_digit_count=w_digit_count,
            x_range=x_range,
        )
    )
    macro = _build_intra(config, name="intra_digits", w_logical_shape=(n, k))
    weight = _randint_in_range(macro.w_value_range, (n, k))
    activation = _randint_in_range(macro.x_value_range, (m, k))
    _assert_macro_matches_torch(macro, weight, activation)


def test_direct_and_inter_slice_one_agree() -> None:
    torch.manual_seed(4)
    n, k, m = 13, 20, 8
    direct = _build_direct(_direct_config(), name="direct", w_logical_shape=(n, k))
    inter_config = InterArraySliceXbarMacroConfig(**_slice_config(w_slice_num=1, x_slice_num=1))
    inter = _build_inter(inter_config, name="inter11", w_logical_shape=(n, k))
    weight = torch.randint(-3, 4, (n, k), dtype=torch.int32)
    activation = torch.randint(0, 2, (m, k), dtype=torch.int32)

    direct.program(weight)
    inter.program(weight)
    assert torch.equal(
        direct.matmul(activation, adc_operation_point=_TEST_ADC_OP),
        inter.matmul(activation, adc_operation_point=_TEST_ADC_OP),
    )


def test_inter_and_intra_slice_macros_agree() -> None:
    torch.manual_seed(5)
    n, k, m = 13, 20, 8
    inter_config = InterArraySliceXbarMacroConfig(**_slice_config(w_slice_num=3, x_slice_num=4))
    intra_config = IntraArraySliceXbarMacroConfig(**_slice_config(w_slice_num=3, x_slice_num=4))
    inter = _build_inter(inter_config, name="inter", w_logical_shape=(n, k))
    intra = _build_intra(intra_config, name="intra", w_logical_shape=(n, k))
    weight = torch.randint(-63, 64, (n, k), dtype=torch.int32)
    activation = torch.randint(0, 16, (m, k), dtype=torch.int32)

    inter.program(weight)
    intra.program(weight)
    assert torch.equal(
        inter.matmul(activation, adc_operation_point=_TEST_ADC_OP),
        intra.matmul(activation, adc_operation_point=_TEST_ADC_OP),
    )


@pytest.mark.parametrize(
    ("macro_kind", "config"),
    [
        ("ideal", _ideal_macro_config()),
        ("direct", _direct_config()),
        ("inter", InterArraySliceXbarMacroConfig(**_slice_config(w_slice_num=3, x_slice_num=4))),
        ("intra", IntraArraySliceXbarMacroConfig(**_slice_config(w_slice_num=3, x_slice_num=4))),
    ],
)
def test_macro_program_replaces_owned_weight_state(
    macro_kind: str,
    config: IdealXbarMacroConfig
    | DirectXbarMacroConfig
    | InterArraySliceXbarMacroConfig
    | IntraArraySliceXbarMacroConfig,
) -> None:
    torch.manual_seed(6000)
    n, k, m = 13, 20, 8
    macro = _build_macro_for_kind(macro_kind, config, name=f"stateful_{macro_kind}", w_logical_shape=(n, k))
    activation = _randint_in_range(macro.x_value_range, (m, k))
    weight_a = _randint_in_range(macro.w_value_range, (n, k))
    weight_b = _randint_in_range(macro.w_value_range, (n, k))

    y_a = _assert_macro_matches_torch(macro, weight_a, activation)
    y_b = _assert_macro_matches_torch(macro, weight_b, activation)
    assert not torch.equal(y_a, y_b)


@pytest.mark.parametrize(
    ("macro_kind", "config", "weight_shape", "activation_shape"),
    [
        ("ideal", _ideal_macro_config(), (2, 13, 20), (8, 20)),
        ("ideal", _ideal_macro_config(), (2, 13, 20), (2, 8, 20)),
        ("direct", _direct_config(), (2, 13, 20), (8, 20)),
        ("direct", _direct_config(), (2, 13, 20), (2, 8, 20)),
        ("inter", InterArraySliceXbarMacroConfig(**_slice_config(w_slice_num=3, x_slice_num=4)), (2, 13, 20), (8, 20)),
        (
            "inter",
            InterArraySliceXbarMacroConfig(**_slice_config(w_slice_num=3, x_slice_num=4)),
            (2, 13, 20),
            (2, 8, 20),
        ),
        (
            "intra",
            IntraArraySliceXbarMacroConfig(**_slice_config(w_slice_num=3, x_slice_num=4)),
            (2, 13, 20),
            (8, 20),
        ),
        (
            "intra",
            IntraArraySliceXbarMacroConfig(**_slice_config(w_slice_num=3, x_slice_num=4)),
            (2, 13, 20),
            (2, 8, 20),
        ),
        (
            "inter",
            InterArraySliceXbarMacroConfig(**_slice_config(w_slice_num=2, x_slice_num=3)),
            (2, 3, 5, 7),
            (2, 3, 4, 7),
        ),
        (
            "intra",
            IntraArraySliceXbarMacroConfig(**_slice_config(w_slice_num=2, x_slice_num=3)),
            (2, 3, 5, 7),
            (2, 3, 4, 7),
        ),
    ],
)
def test_xbar_macro_supports_weight_and_activation_batch_prefixes(
    macro_kind: str,
    config: IdealXbarMacroConfig
    | DirectXbarMacroConfig
    | InterArraySliceXbarMacroConfig
    | IntraArraySliceXbarMacroConfig,
    weight_shape: tuple[int, ...],
    activation_shape: tuple[int, ...],
) -> None:
    torch.manual_seed(7000 + len(weight_shape) * 100 + len(activation_shape))
    macro = _build_macro_for_kind(macro_kind, config, name=f"batch_{macro_kind}", w_logical_shape=weight_shape)
    weight = _randint_in_range(macro.w_value_range, weight_shape)
    activation = _randint_in_range(macro.x_value_range, activation_shape)
    _assert_macro_matches_torch(macro, weight, activation)


def test_ideal_xbar_macro_public_properties() -> None:
    macro = _build_ideal(
        _ideal_macro_config(x_value_range=(-5, 7), w_value_range=(-11, 13)),
        name="ideal_props",
        w_logical_shape=(13, 20),
    )
    assert macro.w_value_range == (-11, 13)
    assert macro.x_value_range == (-5, 7)
    assert macro.adc_mode_num == 1
    assert macro.adc_max_bits == 0
    assert macro.adc_rescale_factor(AdcOperationPoint(adc_mode=0, adc_bits=0)) == 1.0


def test_direct_xbar_macro_public_properties() -> None:
    macro = _build_direct(
        _direct_config(x_range=(0, 3), w_digit_count=2),
        name="direct_props",
        w_logical_shape=(13, 20),
    )
    assert macro.w_value_range == (-15, 15)
    assert macro.x_value_range == (0, 3)
    assert macro.adc_mode_num == 1
    assert macro.adc_max_bits == _TEST_ADC_BITS
    # Ideal-backed macros derive rescale from bit width alone: bits == 0
    # (the full-precision sentinel) → identity rescale of 1.0.
    assert macro.adc_rescale_factor(_TEST_ADC_OP) == 1.0


def test_inter_array_slice_xbar_macro_public_properties() -> None:
    config = InterArraySliceXbarMacroConfig(
        **_slice_config(w_slice_num=3, x_slice_num=2, x_range=(0, 3), w_digit_count=2)
    )
    macro = _build_inter(config, name="inter_props", w_logical_shape=(13, 20))
    assert macro.w_value_range == (-4095, 4095)
    assert macro.x_value_range == (0, 15)
    # Ideal-backed → bits == 0 → identity rescale.
    assert macro.adc_rescale_factor(_TEST_ADC_OP) == 1.0


def test_intra_array_slice_xbar_macro_public_properties() -> None:
    config = IntraArraySliceXbarMacroConfig(
        **_slice_config(w_slice_num=3, x_slice_num=2, x_range=(0, 3), w_digit_count=2)
    )
    macro = _build_intra(config, name="intra_props", w_logical_shape=(13, 20))
    assert macro.w_value_range == (-4095, 4095)
    assert macro.x_value_range == (0, 15)
    # Ideal-backed → bits == 0 → identity rescale.
    assert macro.adc_rescale_factor(_TEST_ADC_OP) == 1.0


@pytest.mark.parametrize(
    ("config", "expected_type"),
    [
        (_ideal_macro_config(), IdealXbarMacro),
        (_direct_config(), DirectXbarMacro),
        (InterArraySliceXbarMacroConfig(**_slice_config(w_slice_num=2, x_slice_num=3)), InterArraySliceXbarMacro),
        (IntraArraySliceXbarMacroConfig(**_slice_config(w_slice_num=2, x_slice_num=3)), IntraArraySliceXbarMacro),
    ],
)
def test_xbar_macro_from_config_dispatches_to_registered_subclass(
    config: IdealXbarMacroConfig
    | DirectXbarMacroConfig
    | InterArraySliceXbarMacroConfig
    | IntraArraySliceXbarMacroConfig,
    expected_type: type[XbarMacro],
) -> None:
    policy_by_config = {
        IdealXbarMacroConfig: _IDEAL_MACRO_POLICY,
        DirectXbarMacroConfig: _DIRECT_MACRO_POLICY,
        InterArraySliceXbarMacroConfig: _INTER_MACRO_POLICY,
        IntraArraySliceXbarMacroConfig: _INTRA_MACRO_POLICY,
    }
    macro = XbarMacro.from_config(
        config=config,
        policy=policy_by_config[type(config)],
        name="from_config",
        w_logical_shape=(13, 20),
        dtype=torch.float32,
        T__K=300.0,
        ideal_xbar=False,
    )
    macro.eval()
    assert isinstance(macro, expected_type)
