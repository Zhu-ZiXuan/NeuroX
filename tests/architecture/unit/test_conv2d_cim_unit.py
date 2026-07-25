"""Exactness tests for the Toeplitz Conv2dCimUnit and the IdealConv2dUnit reference."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neurox.architecture.unit.cim import Conv2dCimUnit, Conv2dCimUnitConfig, Conv2dCimUnitPolicy
from neurox.architecture.unit.cim.engine import CimEngine, DirectCimEngineConfig, DirectCimEnginePolicy
from neurox.architecture.unit.ideal import IdealConv2dUnit, IdealConv2dUnitConfig, IdealConv2dUnitPolicy
from neurox.primitive.digital import AccumulatorConfig
from neurox.primitive.macro.cim import IdealCimMacroConfig, IdealCimMacroPolicy

_UNIT_POLICY = Conv2dCimUnitPolicy(
    engine=DirectCimEnginePolicy(cim_macro_policy=IdealCimMacroPolicy()),
)

# ``adc_bits == 0`` is the IdealCimMacro lossless sentinel: the unit pipeline
# must match the exact-integer references bit-exactly.
_ADC_MODE = 0
_ADC_BITS = 0


def _ideal_macro_config(
    *,
    max_active_num: int | None = None,
    x_value_range: tuple[int, int] = (0, 3),
) -> IdealCimMacroConfig:
    return IdealCimMacroConfig(
        max_active_num=16 if max_active_num is None else max_active_num,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
        x_value_range=x_value_range,
        w_value_range=(-3, 3),
        adc_mode_num=1,
        adc_max_bits=0,
    )


def _accumulator_config() -> AccumulatorConfig:
    return AccumulatorConfig(
        bit_width=32,
        energy_per_op__fJ=0.0,
        latency_per_op__ns=0.0,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )


def _unit_config(
    *,
    row_num: int = 16,
    col_num: int = 16,
    max_active_num: int | None = None,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    x_value_range: tuple[int, int] = (0, 3),
) -> Conv2dCimUnitConfig:
    return Conv2dCimUnitConfig(
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        engine=DirectCimEngineConfig(
            input_num=row_num,
            output_num=col_num,
            cim_macro_config=_ideal_macro_config(
                max_active_num=row_num if max_active_num is None else max_active_num,
                x_value_range=x_value_range,
            ),
            phase_accumulator_config=_accumulator_config(),
            col_accumulator_config=_accumulator_config(),
        ),
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


def _build_unit(
    config: Conv2dCimUnitConfig,
    *,
    w_logical_shape: tuple[int, ...],
) -> Conv2dCimUnit:
    unit = Conv2dCimUnit(
        config=config,
        policy=_UNIT_POLICY,
        w_logical_shape=w_logical_shape,
        dtype=torch.float32,
        T__K=300.0,
        ideal_macro=True,
    )
    unit.eval()
    return unit


def _build_ideal_unit(
    *,
    w_logical_shape: tuple[int, ...],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
) -> IdealConv2dUnit:
    unit = IdealConv2dUnit(
        config=IdealConv2dUnitConfig(
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
            x_value_range=(0, 3),
            w_value_range=(-3, 3),
            stride=stride,
            padding=padding,
            dilation=dilation,
        ),
        policy=IdealConv2dUnitPolicy(),
        w_logical_shape=w_logical_shape,
        dtype=torch.float32,
        T__K=300.0,
        ideal_macro=False,
    )
    unit.eval()
    return unit


def _random_weight(unit: Conv2dCimUnit | IdealConv2dUnit, shape: tuple[int, ...]) -> torch.Tensor:
    lo, hi = unit.w_value_range
    return torch.randint(lo, hi + 1, shape, dtype=torch.int32)


def _random_activation(unit: Conv2dCimUnit | IdealConv2dUnit, shape: tuple[int, ...]) -> torch.Tensor:
    lo, hi = unit.x_value_range
    return torch.randint(lo, hi + 1, shape, dtype=torch.int32)


def _conv2d_int64_oracle(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> torch.Tensor:
    """Explicit unfold-matmul reference in int64, independent of the unit."""
    s_h, s_w = stride
    p_h, p_w = padding
    d_h, d_w = dilation
    c_out, _, kh, kw = weight.shape
    x64 = F.pad(x.to(torch.int64), (p_w, p_w, p_h, p_h))
    w64 = weight.to(torch.int64)
    h_p, w_p = x64.shape[-2:]
    h_out = (h_p - d_h * (kh - 1) - 1) // s_h + 1
    w_out = (w_p - d_w * (kw - 1) - 1) // s_w + 1
    out = torch.zeros((*x.shape[:-3], c_out, h_out, w_out), dtype=torch.int64)
    for i in range(h_out):
        for j in range(w_out):
            patch = x64[
                ...,
                :,
                i * s_h : i * s_h + d_h * (kh - 1) + 1 : d_h,
                j * s_w : j * s_w + d_w * (kw - 1) + 1 : d_w,
            ]
            # Shape: [..., 1, C_in, kh, kw] * [C_out, C_in, kh, kw] -> [..., C_out]
            out[..., :, i, j] = (patch.unsqueeze(-4) * w64).sum(dim=(-3, -2, -1))
    return out


# --- 1. W_g geometry derivation ---


@pytest.mark.parametrize(
    ("c_in", "kh", "kw", "s_w", "d_w", "row_num", "col_num", "c_out", "expected"),
    [
        # g_k-limited: g_k = 1 + (4 // 1 - 2) // 1 = 3 < g_n = 16.
        (1, 1, 2, 1, 1, 4, 16, 1, (3, 4, 4, 3)),
        # g_n-limited: g_k = 1 + (16 - 2) // 1 = 15, g_n = 7 // 3 = 2.
        (1, 1, 2, 1, 1, 16, 7, 3, (2, 3, 3, 6)),
        # floor at 1: K'(1) = 2*3*3 = 18 > row_num = 4 (g_k <= 0).
        (2, 3, 3, 1, 1, 4, 16, 2, (1, 3, 18, 2)),
        # dilated, strided strip: kw_eff = 5, g_k = 1 + (12 - 5) // 2 = 4.
        (1, 1, 3, 2, 2, 12, 8, 1, (4, 11, 11, 4)),
    ],
)
def test_w_g_derivation(
    c_in: int,
    kh: int,
    kw: int,
    s_w: int,
    d_w: int,
    row_num: int,
    col_num: int,
    c_out: int,
    expected: tuple[int, int, int, int],
) -> None:
    unit = _build_unit(
        _unit_config(row_num=row_num, col_num=col_num, stride=(1, s_w), dilation=(1, d_w)),
        w_logical_shape=(c_out, c_in, kh, kw),
    )
    w_g, w_strip, k_prime, n_prime = expected
    assert unit._w_g == w_g
    assert unit._w_strip == w_strip
    assert unit._k_prime == k_prime
    assert unit._n_prime == n_prime
    assert unit.engine._w_logical_shape == (n_prime, k_prime)


# --- 2. IdealConv2dUnit exactness ---


@pytest.mark.parametrize(
    ("kernel", "stride", "padding", "dilation"),
    [
        ((3, 2), (1, 1), (0, 0), (1, 1)),  # kh != kw
        ((3, 3), (2, 1), (0, 0), (1, 1)),  # s_h != s_w
        ((3, 3), (1, 1), (1, 2), (1, 1)),  # p_h != p_w
        ((3, 3), (1, 1), (0, 0), (2, 1)),  # d_h != d_w
        ((2, 3), (2, 1), (1, 2), (1, 2)),  # combined asymmetry
    ],
)
def test_ideal_conv2d_exact(
    kernel: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> None:
    torch.manual_seed(400 + kernel[1] * 7 + stride[0] * 5 + padding[1] * 3 + dilation[0])
    c_out, c_in, h, w = 3, 2, 8, 9
    kh, kw = kernel
    unit = _build_ideal_unit(
        w_logical_shape=(c_out, c_in, kh, kw),
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    weight = _random_weight(unit, (c_out, c_in, kh, kw))
    x = _random_activation(unit, (2, c_in, h, w))  # leading batch dim
    unit.program(weight)
    actual = unit.conv2d(x, adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
    expected = _conv2d_int64_oracle(x, weight, stride=stride, padding=padding, dilation=dilation)
    assert actual.shape == expected.shape
    assert torch.equal(actual.to(torch.int64), expected)


def test_ideal_conv2d_bias_exact() -> None:
    torch.manual_seed(410)
    c_out, c_in, kh, kw, h, w = 3, 2, 3, 2, 8, 9
    stride, padding, dilation = (2, 1), (1, 2), (1, 2)
    unit = _build_ideal_unit(
        w_logical_shape=(c_out, c_in, kh, kw),
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    weight = _random_weight(unit, (c_out, c_in, kh, kw))
    bias = torch.randint(-7, 8, (c_out,), dtype=torch.int32)
    x = _random_activation(unit, (c_in, h, w))
    unit.program(weight, bias)
    actual = unit.conv2d(x, adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
    expected = _conv2d_int64_oracle(x, weight, stride=stride, padding=padding, dilation=dilation) + bias.to(
        torch.int64
    ).view(-1, 1, 1)
    assert torch.equal(actual.to(torch.int64), expected)


# --- 3. Toeplitz path vs IdealConv2dUnit ---


def _assert_matches_ideal(
    *,
    row_num: int,
    col_num: int,
    w_shape: tuple[int, int, int, int],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    x_shape: tuple[int, ...],
    bias: torch.Tensor | None = None,
    expected_w_g: int | None = None,
    seed: int,
) -> Conv2dCimUnit:
    torch.manual_seed(seed)
    unit = _build_unit(
        _unit_config(row_num=row_num, col_num=col_num, stride=stride, padding=padding, dilation=dilation),
        w_logical_shape=w_shape,
    )
    if expected_w_g is not None:
        assert unit._w_g == expected_w_g
    ideal = _build_ideal_unit(w_logical_shape=w_shape, stride=stride, padding=padding, dilation=dilation)
    weight = _random_weight(unit, w_shape)
    x = _random_activation(unit, x_shape)
    unit.program(weight, bias)
    ideal.program(weight, bias)
    actual = unit.conv2d(x, adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
    expected = ideal.conv2d(x, adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
    assert actual.shape == expected.shape
    assert torch.equal(actual.to(torch.int64), expected.to(torch.int64))
    return unit


def test_toeplitz_matches_ideal_multi_window() -> None:
    # (a) W_g > 1: g_k = 1 + (16//2 - 2) = 7, g_n = 8; W_out = 7 = W_g (no trim).
    _assert_matches_ideal(
        row_num=16,
        col_num=16,
        w_shape=(2, 1, 2, 2),
        x_shape=(1, 5, 8),
        expected_w_g=7,
        seed=500,
    )


def test_toeplitz_matches_ideal_single_window_degenerate() -> None:
    # (b) W_g = 1 (col_num < 2*C_out): g_n = 5 // 3 = 1 — im2col degenerate.
    _assert_matches_ideal(
        row_num=32,
        col_num=5,
        w_shape=(3, 2, 3, 3),
        x_shape=(2, 6, 7),
        expected_w_g=1,
        seed=510,
    )


def test_toeplitz_matches_ideal_last_segment_trim() -> None:
    # (c) W_out = 8, W_g = 7: T_seg = 2 with 6 surplus windows trimmed at fold.
    _assert_matches_ideal(
        row_num=16,
        col_num=16,
        w_shape=(2, 1, 2, 2),
        x_shape=(1, 5, 9),
        expected_w_g=7,
        seed=520,
    )


def test_toeplitz_matches_ideal_asymmetric_geometry() -> None:
    # (d) stride/padding/dilation asymmetry under W_g > 1:
    # kw_eff = 5, g_k = 1 + (16//2 - 5) = 4, g_n = 8; W_out = 9, T_seg = 3.
    _assert_matches_ideal(
        row_num=16,
        col_num=8,
        w_shape=(1, 1, 2, 3),
        stride=(2, 1),
        padding=(1, 2),
        dilation=(1, 2),
        x_shape=(1, 7, 9),
        expected_w_g=4,
        seed=530,
    )


def test_toeplitz_matches_ideal_with_bias() -> None:
    # (e) bias preload semantics survive the fold + trim exactly once per element.
    _assert_matches_ideal(
        row_num=16,
        col_num=16,
        w_shape=(2, 1, 2, 2),
        x_shape=(1, 5, 9),
        bias=torch.randint(-7, 8, (2,), dtype=torch.int32),
        expected_w_g=7,
        seed=540,
    )


def test_toeplitz_matches_ideal_batch_ride_through() -> None:
    # (f) caller-owned leading batch dims ride through untouched.
    _assert_matches_ideal(
        row_num=16,
        col_num=16,
        w_shape=(2, 1, 2, 2),
        x_shape=(2, 3, 1, 5, 9),
        expected_w_g=7,
        seed=550,
    )


# --- 4. Toeplitz matrix placement law ---


def test_toeplitz_matrix_placement() -> None:
    torch.manual_seed(600)
    c_out, c_in, kh, kw = 2, 2, 2, 3
    stride, dilation = (1, 2), (1, 2)
    # kw_eff = 5; g_k = 1 + (32//4 - 5) // 2 = 2; g_n = 16 // 2 = 8 -> W_g = 2;
    # W_strip = 5 + 2 = 7; K' = 4*7 = 28; N' = 4.
    unit = _build_unit(
        _unit_config(row_num=32, col_num=16, stride=stride, dilation=dilation),
        w_logical_shape=(c_out, c_in, kh, kw),
    )
    assert (unit._w_g, unit._w_strip, unit._k_prime, unit._n_prime) == (2, 7, 28, 4)
    # All-nonzero weight so the placement census is unambiguous.
    weight = torch.randint(1, 4, (c_out, c_in, kh, kw), dtype=torch.int32)
    matrix = unit._weight_to_matrix(weight)
    assert tuple(matrix.shape) == (unit._n_prime, unit._k_prime)
    s_w, d_w = stride[1], dilation[1]
    expected = torch.zeros(unit._n_prime, unit._k_prime, dtype=weight.dtype)
    for g in range(unit._w_g):
        for n in range(c_out):
            for ci in range(c_in):
                for i in range(kh):
                    for j in range(kw):
                        c = g * c_out + n
                        r = (ci * kh + i) * unit._w_strip + (g * s_w + j * d_w)
                        expected[c, r] = weight[n, ci, i, j]
    assert torch.equal(matrix, expected)
    # Collision-free placement: every mapped entry lands on its own cell.
    assert int((matrix != 0).sum()) == unit._w_g * c_out * c_in * kh * kw


# --- 5. Construction rejections ---


def test_rejects_non_4d_w_logical_shape() -> None:
    with pytest.raises(ValueError, match="C_out, C_in, kh, kw"):
        _build_unit(_unit_config(), w_logical_shape=(3, 18))


def test_conv2d_cim_rejects_float_weight() -> None:
    shape = (2, 1, 2, 2)
    unit = _build_unit(_unit_config(), w_logical_shape=shape)
    with pytest.raises(TypeError, match="integer weight tensor"):
        unit.program(torch.zeros(shape, dtype=torch.float32))


def test_padding_requires_x_value_range_covering_zero() -> None:
    with pytest.raises(ValueError, match="x_value_range"):
        _build_unit(
            _unit_config(padding=(1, 1), x_value_range=(1, 3)),
            w_logical_shape=(3, 2, 3, 3),
        )


def test_multi_window_requires_x_value_range_covering_zero() -> None:
    # W_g = 3 > 1 with zero padding: strip right-padding and surplus windows
    # still inject x = 0.
    with pytest.raises(ValueError, match="x_value_range"):
        _build_unit(
            _unit_config(row_num=4, col_num=16, x_value_range=(1, 3)),
            w_logical_shape=(1, 1, 1, 2),
        )


def test_toeplitz_requires_w_range_covering_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    # No shipped transcoder yields a zero-free weight range; force one through
    # the engine surface to pin the unconditional structural-zero check.
    monkeypatch.setattr(CimEngine, "w_value_range", property(lambda self: (1, 3)))
    with pytest.raises(ValueError, match="w_value_range"):
        _build_unit(_unit_config(), w_logical_shape=(3, 2, 3, 3))


# --- 6. Engine-tiling composition ---


def test_conv2d_accepts_non_divisor_input_blocking_and_stays_exact() -> None:
    # 16 % 6 != 0: the base macro accepts it and Conv2dCimUnitConfig adds no
    # divisor guard (the operator tolerates unused inputs). The engine must split
    # the 16 inputs into P = ceil(16/6) = 3 phases (short final block of 4),
    # so the Toeplitz path still matches the ideal conv2d oracle bit-exactly.
    # Under the old floor division (P = 2), inputs
    # 12..15 would be dropped and the result would be wrong.
    torch.manual_seed(700)
    c_out, c_in, kh, kw = 2, 1, 2, 2
    w_shape = (c_out, c_in, kh, kw)
    unit = _build_unit(
        _unit_config(row_num=16, col_num=16, max_active_num=6),
        w_logical_shape=w_shape,
    )
    assert unit.engine._input_phase_num == 3
    ideal = _build_ideal_unit(w_logical_shape=w_shape)
    weight = _random_weight(unit, w_shape)
    x = _random_activation(unit, (1, 5, 8))
    unit.program(weight)
    ideal.program(weight)
    actual = unit.conv2d(x, adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
    expected = ideal.conv2d(x, adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
    assert actual.shape == expected.shape
    assert torch.equal(actual.to(torch.int64), expected.to(torch.int64))


def test_engine_tiling_composition_matches_ideal() -> None:
    # K' = 18 > row_num = 8 (Tc = 3) and N' = 5 > col_num = 4 (Tr = 2): the
    # generic engine tiling splits the W_g = 1 Toeplitz matrix bit-exactly.
    unit = _assert_matches_ideal(
        row_num=8,
        col_num=4,
        w_shape=(5, 2, 3, 3),
        x_shape=(2, 6, 7),
        expected_w_g=1,
        seed=610,
    )
    assert unit.engine._w_logical_shape == (5, 18)
    assert unit.engine._row_tile_num == 2
