"""Exactness and mapping tests for the engine-backed Conv2dCimUnit."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neurox.architecture.unit.cim import Conv2dCimUnit, Conv2dCimUnitConfig, Conv2dCimUnitPolicy
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
from neurox.architecture.unit.conv2d import Conv2dUnit
from neurox.architecture.unit.ideal import IdealConv2dUnit, IdealConv2dUnitConfig, IdealConv2dUnitPolicy
from neurox.primitive.digital import AccumulatorConfig
from neurox.primitive.macro.cim import IdealCimMacroConfig, IdealCimMacroPolicy

_UNIT_POLICY = Conv2dCimUnitPolicy(
    engine=CimEnginePolicy(
        cim_macro_policy=IdealCimMacroPolicy(),
        placement=PlacementStagePolicy(),
        input_activation=InputActivationStagePolicy(),
        weight_slice=DirectWeightSliceStagePolicy(),
        x_slice=DirectXSliceStagePolicy(),
    ),
)
_QUANTIZATION_MODE = 0
_ADC_BITS: int | None = None


def _ideal_macro_config(
    *,
    max_active_num: int,
    x_value_range: tuple[int, int] = (0, 3),
) -> IdealCimMacroConfig:
    return IdealCimMacroConfig(
        max_active_num=max_active_num,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
        x_value_range=x_value_range,
        w_value_range=(-3, 3),
        # Only the lossless oracle is exercised here; the declared window and
        # width just have to be legal.
        quantization_input_ranges=((-256, 255),),
        adc_max_bits=8,
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
    input_num: int = 16,
    output_num: int = 16,
    max_active_num: int | None = None,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    x_value_range: tuple[int, int] = (0, 3),
) -> Conv2dCimUnitConfig:
    return Conv2dCimUnitConfig(
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        engine=CimEngineConfig(
            input_num=input_num,
            output_num=output_num,
            cim_macro_config=_ideal_macro_config(
                max_active_num=input_num if max_active_num is None else max_active_num,
                x_value_range=x_value_range,
            ),
            placement=PlacementStageConfig(
                contraction_accumulator_config=_accumulator_config(),
            ),
            input_activation=InputActivationStageConfig(
                phase_accumulator_config=_accumulator_config(),
            ),
            weight_slice=DirectWeightSliceStageConfig(),
            x_slice=DirectXSliceStageConfig(),
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
    """Explicit window-dot-product reference independent of the unit."""
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


def _assert_matches_ideal(
    *,
    input_num: int,
    output_num: int,
    w_shape: tuple[int, int, int, int],
    x_shape: tuple[int, ...],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    max_active_num: int | None = None,
    bias: torch.Tensor | None = None,
    seed: int,
) -> Conv2dCimUnit:
    torch.manual_seed(seed)
    unit = _build_unit(
        _unit_config(
            input_num=input_num,
            output_num=output_num,
            max_active_num=max_active_num,
            stride=stride,
            padding=padding,
            dilation=dilation,
        ),
        w_logical_shape=w_shape,
    )
    ideal = _build_ideal_unit(
        w_logical_shape=w_shape,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    weight = _random_weight(unit, w_shape)
    x = _random_activation(unit, x_shape)
    unit.program(weight, bias)
    ideal.program(weight, bias)
    actual = unit.conv2d(x, quantization_mode=_QUANTIZATION_MODE, adc_bits=_ADC_BITS)
    expected = ideal.conv2d(x, quantization_mode=_QUANTIZATION_MODE, adc_bits=_ADC_BITS)
    assert actual.shape == expected.shape
    assert torch.equal(actual.to(torch.int64), expected.to(torch.int64))
    return unit


# --- Ideal reference ---


@pytest.mark.parametrize(
    ("kernel", "stride", "padding", "dilation"),
    [
        ((3, 2), (1, 1), (0, 0), (1, 1)),
        ((3, 3), (2, 1), (0, 0), (1, 1)),
        ((3, 3), (1, 1), (1, 2), (1, 1)),
        ((3, 3), (1, 1), (0, 0), (2, 1)),
        ((2, 3), (2, 1), (1, 2), (1, 2)),
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
    x = _random_activation(unit, (2, c_in, h, w))
    unit.program(weight)
    actual = unit.conv2d(x, quantization_mode=_QUANTIZATION_MODE, adc_bits=_ADC_BITS)
    expected = _conv2d_int64_oracle(x, weight, stride=stride, padding=padding, dilation=dilation)
    assert actual.shape == expected.shape
    assert torch.equal(actual.to(torch.int64), expected)


# --- Kernel and window lowering ---


def test_kernel_matrix_stores_one_copy() -> None:
    shape = (5, 2, 2, 3)
    unit = _build_unit(_unit_config(input_num=32, output_num=4), w_logical_shape=shape)
    weight = torch.arange(torch.tensor(shape).prod(), dtype=torch.int32).reshape(shape)
    matrix = unit._weight_to_matrix(weight)
    assert matrix.shape == (5, 12)
    assert torch.equal(matrix, weight.flatten(start_dim=1))
    assert unit.engine._w_logical_shape == (5, 12)


def test_window_lowering_matches_explicit_patch_order() -> None:
    unit = _build_unit(
        _unit_config(input_num=16, output_num=8, stride=(2, 1), padding=(1, 2), dilation=(1, 2)),
        w_logical_shape=(3, 2, 2, 3),
    )
    x = torch.arange(2 * 5 * 7, dtype=torch.int32).reshape(1, 2, 5, 7)
    out_hw = unit._conv2d_out_hw(5, 7)
    planes = unit._conv2d_planes(x, out_hw=out_hw)
    assert planes.shape == (1, out_hw[0] * out_hw[1], 2 * 2 * 3)

    padded = F.pad(x, (2, 2, 1, 1))
    expected = [
        padded[0, :, i * 2 : i * 2 + 2, j : j + 5 : 2].flatten() for i in range(out_hw[0]) for j in range(out_hw[1])
    ]
    assert torch.equal(planes[0], torch.stack(expected))


# --- Engine-backed exactness ---


@pytest.mark.parametrize(
    ("input_num", "output_num", "w_shape", "x_shape", "stride", "padding", "dilation"),
    [
        (16, 16, (2, 1, 2, 2), (1, 5, 8), (1, 1), (0, 0), (1, 1)),
        (8, 4, (5, 2, 3, 3), (2, 6, 7), (1, 1), (0, 0), (1, 1)),
        (16, 8, (3, 1, 2, 3), (1, 7, 9), (2, 1), (1, 2), (1, 2)),
        (32, 5, (3, 2, 3, 3), (6, 2, 6, 7), (1, 1), (0, 0), (1, 1)),
        (16, 4, (20, 1, 2, 2), (2, 1, 5, 6), (1, 1), (0, 0), (1, 1)),
    ],
)
def test_conv2d_cim_matches_ideal(
    input_num: int,
    output_num: int,
    w_shape: tuple[int, int, int, int],
    x_shape: tuple[int, ...],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> None:
    _assert_matches_ideal(
        input_num=input_num,
        output_num=output_num,
        w_shape=w_shape,
        x_shape=x_shape,
        stride=stride,
        padding=padding,
        dilation=dilation,
        seed=500 + input_num + output_num,
    )


def test_conv2d_cim_bias_exact() -> None:
    bias = torch.randint(-7, 8, (3,), dtype=torch.int32)
    _assert_matches_ideal(
        input_num=16,
        output_num=8,
        w_shape=(3, 2, 2, 3),
        x_shape=(2, 2, 7, 9),
        stride=(2, 1),
        padding=(1, 2),
        dilation=(1, 2),
        bias=bias,
        seed=600,
    )


def test_conv2d_balances_input_axis_blocks() -> None:
    # K=4, input_num=16 gives block capacity C=4. Five output blocks require
    # two macros and are balanced as three plus two block slots.
    unit = _assert_matches_ideal(
        input_num=16,
        output_num=4,
        w_shape=(20, 1, 2, 2),
        x_shape=(2, 1, 5, 6),
        seed=610,
    )
    engine = unit.engine
    assert engine.placement.plan.output_block_num == 5
    assert engine.placement.plan.block_group_capacity == 4
    assert engine.placement.plan.block_group_num == 2
    assert engine.placement.plan.block_slot_num == 3
    assert engine.cim_macro.inst_shape == (1, 1, 1, 1, 2)


def test_conv2d_combines_block_steps_and_input_phases() -> None:
    # K=20 and input_num=64 pack three blocks per macro; max_active_num=6
    # splits each block step into four independently converted partial dots.
    unit = _assert_matches_ideal(
        input_num=64,
        output_num=2,
        max_active_num=6,
        w_shape=(12, 1, 4, 5),
        x_shape=(1, 7, 8),
        seed=620,
    )
    assert unit.engine.placement.plan.block_group_capacity == 3
    assert unit.engine.placement.plan.block_group_num == 2
    assert unit.engine.placement.plan.block_slot_num == 3
    assert unit.engine.input_activation._input_phase_num == 4


# --- Unbatched input ---


def _assert_unbatched_matches_batch_of_one(unit: Conv2dUnit, x: torch.Tensor) -> None:
    """A 3-D call returns 3-D and equals the `B = 1` call bit for bit."""
    unbatched = unit.conv2d(x, quantization_mode=_QUANTIZATION_MODE, adc_bits=_ADC_BITS)
    batched = unit.conv2d(x.unsqueeze(0), quantization_mode=_QUANTIZATION_MODE, adc_bits=_ADC_BITS)
    assert unbatched.ndim == 3
    assert batched.shape == (1, *unbatched.shape)
    assert torch.equal(unbatched.to(torch.int64), batched[0].to(torch.int64))


def test_unbatched_input_matches_batch_of_one() -> None:
    torch.manual_seed(630)
    w_shape = (3, 2, 2, 3)
    unit = _build_unit(
        _unit_config(input_num=16, output_num=8, stride=(2, 1), padding=(1, 2), dilation=(1, 2)),
        w_logical_shape=w_shape,
    )
    ideal = _build_ideal_unit(w_logical_shape=w_shape, stride=(2, 1), padding=(1, 2), dilation=(1, 2))
    weight = _random_weight(unit, w_shape)
    bias = torch.randint(-7, 8, (3,), dtype=torch.int32)
    x = _random_activation(unit, (2, 7, 9))
    units: tuple[Conv2dUnit, ...] = (unit, ideal)
    for candidate in units:
        candidate.program(weight, bias)
        _assert_unbatched_matches_batch_of_one(candidate, x)
    assert unit.initiation_interval__ns((2, 7, 9), adc_bits=_ADC_BITS) == unit.initiation_interval__ns(
        (1, 2, 7, 9), adc_bits=_ADC_BITS
    )


# --- Rejections ---


def test_rejects_non_4d_w_logical_shape() -> None:
    with pytest.raises(ValueError, match="C_out, C_in, kh, kw"):
        _build_unit(_unit_config(), w_logical_shape=(3, 18))


def test_rejects_input_outside_the_two_accepted_forms() -> None:
    unit = _build_unit(_unit_config(), w_logical_shape=(2, 1, 2, 2))
    unit.program(_random_weight(unit, (2, 1, 2, 2)))
    x = _random_activation(unit, (2, 3, 1, 5, 6))
    with pytest.raises(ValueError, match=r"\[C_in, H, W\] or \[B, C_in, H, W\]"):
        unit.conv2d(x, quantization_mode=_QUANTIZATION_MODE, adc_bits=_ADC_BITS)


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


def test_zero_free_input_range_is_valid_without_padding() -> None:
    unit = _build_unit(
        _unit_config(x_value_range=(1, 3)),
        w_logical_shape=(3, 1, 2, 2),
    )
    assert unit.x_value_range == (1, 3)
