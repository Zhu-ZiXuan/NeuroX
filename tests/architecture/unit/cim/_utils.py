"""Concrete host and explicit configurations for the shared CIM implementation."""

from __future__ import annotations

import torch

from neurox.architecture.unit.conv2d import Conv2dCimUnit, Conv2dCimUnitConfig, Conv2dCimUnitPolicy
from neurox.encoding import Encoding
from neurox.primitive.digital import AccumulatorConfig, RadixAccumulatorConfig, RadixSummatorConfig
from neurox.primitive.macro.cim import CimMacroQuantizationScheme, IdealCimMacroConfig, IdealCimMacroPolicy


def unit_config() -> Conv2dCimUnitConfig:
    return Conv2dCimUnitConfig(
        cim_macro_config=IdealCimMacroConfig(
            input_num=8,
            lane_num=1,
            scan_num=4,
            max_active_num=2,
            rescale_factors=(1.0,),
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
            w_digit_num=2,
            w_digit_radix=2,
            w_encoding=Encoding.TRUE_FORM,
            w_signed=True,
            x_digit_num=1,
            x_digit_radix=2,
            x_encoding=Encoding.UNSIGNED,
            x_value_range=(0, 1),
            w_value_range=(-3, 3),
            adc_bits=8,
            quantization_scheme=CimMacroQuantizationScheme.ZERO_POINT,
        ),
        local_accumulator_config=RadixAccumulatorConfig(
            bit_width=32, energy_per_op__fJ=0.0, area_per_inst__um2=0.0, leakage_per_inst__uW=0.0
        ),
        w_polarity_adder_config=None,
        tile_accumulator_config=AccumulatorConfig(
            bit_width=32, energy_per_op__fJ=0.0, area_per_inst__um2=0.0, leakage_per_inst__uW=0.0
        ),
        w_radix_summator_config=RadixSummatorConfig(
            bit_width=32, energy_per_op__fJ=0.0, area_per_inst__um2=0.0, leakage_per_inst__uW=0.0
        ),
        w_slice_num=2,
        w_slice_encoding=Encoding.TRUE_FORM,
        x_slice_num=3,
        x_slice_encoding=Encoding.UNSIGNED,
        clock_period__ns=1.0,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        merge=True,
    )


def build_unit(config: Conv2dCimUnitConfig, *, matrix_shape: tuple[int, int]) -> Conv2dCimUnit:
    unit = Conv2dCimUnit(
        config=config,
        policy=Conv2dCimUnitPolicy(cim_macro_policy=IdealCimMacroPolicy()),
        w_logical_shape=(*matrix_shape, 1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        dtype=torch.float32,
    )
    unit.eval()
    return unit
