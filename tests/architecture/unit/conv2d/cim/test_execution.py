"""Convolution lowering preserves window geometry, output layout and bias."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from neurox import Profiler
from neurox.architecture.unit.conv2d import (
    Conv2dCimUnit,
    Conv2dCimUnitConfig,
    Conv2dCimUnitPolicy,
)
from neurox.encoding import Encoding
from neurox.primitive.digital import AccumulatorConfig, AdderConfig, RadixAccumulatorConfig, RadixSummatorConfig
from neurox.primitive.macro.cim import (
    CimMacroQuantizationScheme,
    IdealCimMacroConfig,
    IdealCimMacroPolicy,
)

_UNIT_POLICY = Conv2dCimUnitPolicy(cim_macro_policy=IdealCimMacroPolicy())


def _ideal_macro_config(
    *,
    input_num: int,
    output_num: int,
    max_active_num: int,
    x_value_range: tuple[int, int] = (0, 3),
) -> IdealCimMacroConfig:
    return IdealCimMacroConfig(
        input_num=input_num,
        rescale_factors=(1.0,),
        max_active_num=max_active_num,
        lane_num=1,
        scan_num=output_num,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
        w_digit_num=2,
        w_digit_radix=2,
        w_encoding=Encoding.TRUE_FORM,
        w_signed=True,
        x_digit_num=2,
        x_digit_radix=2,
        x_encoding=Encoding.UNSIGNED,
        x_value_range=x_value_range,
        w_value_range=(-3, 3),
        adc_bits=8,
        quantization_scheme=CimMacroQuantizationScheme.ZERO_POINT,
    )


def _unit_config(
    *,
    input_num: int = 16,
    output_num: int = 16,
    max_active_num: int | None = None,
    x_value_range: tuple[int, int] = (0, 3),
) -> Conv2dCimUnitConfig:
    return Conv2dCimUnitConfig(
        clock_period__ns=1.0,
        cim_macro_config=_ideal_macro_config(
            input_num=input_num,
            output_num=output_num,
            max_active_num=input_num if max_active_num is None else max_active_num,
            x_value_range=x_value_range,
        ),
        merge=True,
        tile_accumulator_config=None,
        w_polarity_adder_config=None,
        local_accumulator_config=RadixAccumulatorConfig(
            bit_width=32, energy_per_op__fJ=0.0, leakage_per_inst__uW=0.0, area_per_inst__um2=0.0
        ),
        w_slice_num=1,
        w_slice_encoding=None,
        w_radix_summator_config=None,
        x_slice_num=1,
        x_slice_encoding=None,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
    )


def _build_unit(
    config: Conv2dCimUnitConfig,
    *,
    w_logical_shape: tuple[int, ...],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    groups: int = 1,
) -> Conv2dCimUnit:
    unit = Conv2dCimUnit(
        config=config,
        policy=_UNIT_POLICY,
        w_logical_shape=w_logical_shape,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dtype=torch.float32,
    )
    unit.eval()
    return unit


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
    x64 = F.pad(x.long(), (p_w, p_w, p_h, p_h))
    w64 = weight.long()
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
            # Shape: [..., C_out=1, C_in, kh, kw] * [C_out, C_in, kh, kw] -> [..., C_out]
            out[..., :, i, j] = (patch.unsqueeze(-4) * w64).sum(dim=(-3, -2, -1))
    return out


@pytest.mark.parametrize(
    ("input_num", "output_num", "w_shape", "x_shape", "stride", "padding", "dilation", "with_bias"),
    [
        (16, 16, (2, 1, 2, 2), (1, 5, 8), (1, 1), (0, 0), (1, 1), False),
        (16, 8, (3, 2, 2, 3), (2, 7, 9), (2, 1), (1, 2), (1, 2), True),
        (32, 5, (3, 2, 3, 3), (6, 2, 6, 7), (1, 1), (0, 0), (1, 1), True),
    ],
)
def test_conv2d_cim_matches_integer_oracle(
    input_num: int,
    output_num: int,
    w_shape: tuple[int, int, int, int],
    x_shape: tuple[int, ...],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    with_bias: bool,
) -> None:
    generator = torch.Generator().manual_seed(500 + input_num + output_num)
    unit = _build_unit(
        _unit_config(input_num=input_num, output_num=output_num),
        w_logical_shape=w_shape,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    w_lo, w_hi = unit.w_value_range
    x_lo, x_hi = unit.x_value_range
    weight = torch.randint(w_lo, w_hi + 1, w_shape, dtype=torch.int32, generator=generator)
    x = torch.randint(x_lo, x_hi + 1, x_shape, dtype=torch.int32, generator=generator)
    bias = torch.arange(w_shape[0], dtype=torch.int32) - 1 if with_bias else None
    unit.program(weight, bias=bias)
    actual = unit.conv2d(x, quantization_mode=0, adc_active_bits=None)
    expected = _conv2d_int64_oracle(x, weight, stride=stride, padding=padding, dilation=dilation)
    if bias is not None:
        expected = expected + bias.long().view(-1, 1, 1)
    assert actual.shape == expected.shape
    assert torch.equal(actual.long(), expected)


@pytest.mark.parametrize(
    ("per_scan__ns", "one_window__ns", "nine_windows__ns"),
    [(0.0, 1.0, 5.0), (0.5, 3.0, 21.0)],
)
def test_image_latency_preserves_window_pipeline_without_combining_samples(
    monkeypatch: pytest.MonkeyPatch,
    per_scan__ns: float,
    one_window__ns: float,
    nine_windows__ns: float,
) -> None:
    config = replace(_unit_config(input_num=8, output_num=4, max_active_num=8), clock_period__ns=0.25)
    unit = _build_unit(config, w_logical_shape=(8, 1, 3, 3))
    monkeypatch.setattr(unit.cim_macro, "_latency_per_scan__ns", lambda **kwargs: per_scan__ns)
    # Two input-tile rows transfer in 0.5 ns; local work takes 0.25 or 2.25 ns.
    # The image pipeline preserves either bottleneck and drains only once.
    assert unit.latency__ns((1, 3, 3), adc_active_bits=None) == one_window__ns
    assert unit.latency__ns((1, 5, 5), adc_active_bits=None) == nine_windows__ns
    assert unit.latency__ns((7, 1, 5, 5), adc_active_bits=None) == nine_windows__ns

    unit.program(torch.ones((8, 1, 3, 3), dtype=torch.int32))
    unit.set_profile_leading_rank(1)
    profiler = Profiler(concat_dim=0)
    profiler.collect_static_data(unit)
    for batch, side in ((2, 5), (1, 3)):
        with profiler:
            unit.conv2d(
                torch.ones((batch, 1, side, side), dtype=torch.int32), quantization_mode=0, adc_active_bits=None
            )
    durations = profiler.result[""].working_duration__ns
    assert durations is not None
    torch.testing.assert_close(
        durations, torch.tensor([nine_windows__ns, nine_windows__ns, one_window__ns], dtype=torch.float64)
    )


@pytest.mark.parametrize(
    ("groups", "w_shape", "batched", "merge", "differential", "slices", "input_num", "stride", "padding", "dilation"),
    [
        (2, (6, 2, 2, 3), True, False, False, 1, 8, (1, 1), (0, 1), (1, 1)),
        (3, (9, 2, 1, 1), False, True, True, 2, 8, (2, 1), (0, 0), (1, 1)),
        (4, (4, 1, 3, 3), True, False, True, 2, 8, (1, 2), (1, 1), (1, 1)),
        (4, (8, 1, 3, 3), False, True, True, 3, 18, (1, 1), (2, 2), (2, 2)),
        (1, (5, 3, 1, 1), True, True, False, 1, 8, (1, 1), (0, 0), (1, 1)),
    ],
)
def test_grouped_and_depthwise_convolution_match_torch_and_ideal(
    groups: int,
    w_shape: tuple[int, int, int, int],
    batched: bool,
    merge: bool,
    differential: bool,
    slices: int,
    input_num: int,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    device: torch.device,
) -> None:
    config = _unit_config(
        input_num=input_num,
        output_num=5,
        max_active_num=3,
    )
    config = replace(
        config,
        merge=merge,
        cim_macro_config=replace(
            config.cim_macro_config,
            w_signed=not differential,
            w_encoding=Encoding.UNSIGNED if differential else Encoding.TRUE_FORM,
            w_value_range=(0, 3) if differential else (-3, 3),
        ),
        w_slice_num=slices,
        w_slice_encoding=Encoding.TRUE_FORM,
        x_slice_num=2,
        x_slice_encoding=Encoding.UNSIGNED,
        w_polarity_adder_config=AdderConfig(
            bit_width=32, energy_per_op__fJ=1.0, area_per_inst__um2=1.0, leakage_per_inst__uW=1.0
        ),
        tile_accumulator_config=AccumulatorConfig(
            bit_width=32, energy_per_op__fJ=1.0, area_per_inst__um2=1.0, leakage_per_inst__uW=1.0
        ),
        w_radix_summator_config=RadixSummatorConfig(
            bit_width=32, energy_per_op__fJ=1.0, area_per_inst__um2=1.0, leakage_per_inst__uW=1.0
        ),
    )
    unit = _build_unit(
        config, w_logical_shape=w_shape, stride=stride, padding=padding, dilation=dilation, groups=groups
    ).to(device)
    generator = torch.Generator().manual_seed(831)
    weight = torch.randint(unit.w_value_range[0], unit.w_value_range[1] + 1, w_shape, generator=generator)
    x = torch.randint(0, unit.x_value_range[1] + 1, (2, w_shape[1] * groups, 6, 7), generator=generator)
    x = x if batched else x[0]
    bias = torch.arange(w_shape[0], dtype=torch.int64) - w_shape[0] // 2
    expected = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups).to(device)
    weight, x, bias = weight.to(device), x.to(device), bias.to(device)
    unit.program(weight, bias=bias)
    torch.testing.assert_close(unit.conv2d(x, quantization_mode=0, adc_active_bits=None), expected)

    ideal = unit.to_ideal()
    assert ideal.groups == groups
    ideal.program(weight, bias=bias)
    torch.testing.assert_close(ideal.conv2d(x, quantization_mode=0, adc_active_bits=None), expected)


def test_parallel_groups_replicate_circuit_costs_without_multiplying_latency(
    device: torch.device, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _unit_config(input_num=6, output_num=5, max_active_num=3)
    config = replace(
        config,
        area_per_inst__um2=3.0,
        leakage_per_inst__uW=2.0,
        cim_macro_config=replace(
            config.cim_macro_config,
            w_signed=False,
            w_encoding=Encoding.UNSIGNED,
            w_value_range=(0, 3),
            area_per_inst__um2=11.0,
            leakage_per_inst__uW=13.0,
        ),
        w_slice_num=2,
        w_slice_encoding=Encoding.TRUE_FORM,
        x_slice_num=2,
        x_slice_encoding=Encoding.UNSIGNED,
        local_accumulator_config=RadixAccumulatorConfig(
            bit_width=32, energy_per_op__fJ=3.0, area_per_inst__um2=7.0, leakage_per_inst__uW=2.0
        ),
        w_polarity_adder_config=AdderConfig(
            bit_width=32, energy_per_op__fJ=4.0, area_per_inst__um2=5.0, leakage_per_inst__uW=7.0
        ),
        tile_accumulator_config=AccumulatorConfig(
            bit_width=32, energy_per_op__fJ=2.0, area_per_inst__um2=17.0, leakage_per_inst__uW=3.0
        ),
        w_radix_summator_config=RadixSummatorConfig(
            bit_width=32, energy_per_op__fJ=5.0, area_per_inst__um2=19.0, leakage_per_inst__uW=11.0
        ),
    )
    weight = (torch.arange(24, device=device).reshape(3, 2, 2, 2) % 15 - 7).int()
    x = (torch.arange(80, device=device).reshape(2, 2, 4, 5) % 4).int()
    profiles = []
    units = []
    for groups in (1, 3):
        unit = _build_unit(config, w_logical_shape=(3 * groups, 2, 2, 2), groups=groups).to(device)
        monkeypatch.setattr(unit.cim_macro, "_latency_per_scan__ns", lambda **kwargs: 1.3)
        unit.program(weight.repeat(groups, 1, 1, 1))
        unit.set_profile_leading_rank(1)
        profiler = Profiler(concat_dim=0)
        profiler.collect_static_data(unit)
        with profiler:
            actual = unit.conv2d(x.repeat(1, groups, 1, 1), quantization_mode=0, adc_active_bits=None)
        expected = F.conv2d(x.cpu().long(), weight.cpu().long()).repeat(1, groups, 1, 1).to(device)
        torch.testing.assert_close(actual, expected)
        profiles.append(profiler.result)
        units.append(unit)
    for name, single in profiles[0].items():
        grouped = profiles[1][name]
        factor = 1 if name == "" else 3
        assert grouped.area__um2 == single.area__um2 * factor
        assert grouped.leakage__uW == single.leakage__uW * factor
        if single.dynamic_energy__fJ is not None:
            torch.testing.assert_close(
                grouped.dynamic_energy__fJ, single.dynamic_energy__fJ * factor, check_dtype=False
            )
    assert profiles[0][""].working_duration__ns is not None
    assert profiles[1][""].working_duration__ns is not None
    torch.testing.assert_close(profiles[1][""].working_duration__ns, profiles[0][""].working_duration__ns)
    assert units[1].cim_macro.inst_count == 3 * units[0].cim_macro.inst_count
    assert units[1].w_polarity_adder.inst_count == 3 * units[0].w_polarity_adder.inst_count


@pytest.mark.parametrize("weight_slices", [1, 4])
def test_depthwise_merge_reuses_only_tiles_within_each_group(weight_slices: int, device: torch.device) -> None:
    config = _unit_config(input_num=36, output_num=4, max_active_num=9)
    config = replace(
        config,
        cim_macro_config=replace(
            config.cim_macro_config, w_signed=False, w_encoding=Encoding.UNSIGNED, w_value_range=(0, 3)
        ),
        w_slice_num=weight_slices,
        w_slice_encoding=Encoding.TRUE_FORM,
    )
    units = [
        _build_unit(replace(config, merge=merge), w_logical_shape=(3, 1, 3, 3), groups=3).to(device)
        for merge in (False, True)
    ]
    weight = (torch.arange(27, device=device).reshape(3, 1, 3, 3) % 7 - 3).int()
    x = (torch.arange(3 * 5 * 6, device=device).reshape(1, 3, 5, 6) % 4).int()
    expected = F.conv2d(x.cpu().long(), weight.cpu().long(), groups=3).to(device)
    for unit in units:
        unit.program(weight)
        torch.testing.assert_close(unit.conv2d(x, quantization_mode=0, adc_active_bits=None), expected)
    unmerged, merged = units
    # One output tile per group cannot share across depth-wise groups.
    # Four weight slices require two tiles per group and reuse two input slots.
    assert merged.cim_macro.inst_count == 3
    assert unmerged.cim_macro.inst_count == (3 if weight_slices == 1 else 6)
    assert merged.merge.merge_step_num == (1 if weight_slices == 1 else 2)
    if weight_slices == 4:
        assert merged.latency__ns(x.shape, adc_active_bits=None) > unmerged.latency__ns(x.shape, adc_active_bits=None)
