"""Ideal conversion preserves operator semantics and leaves programmed hardware independent."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neurox.architecture.unit.conv2d import (
    Conv2dCimUnit,
    Conv2dCimUnitConfig,
    Conv2dCimUnitPolicy,
)
from neurox.architecture.unit.linear import (
    LinearCimUnit,
    LinearCimUnitConfig,
    LinearCimUnitPolicy,
)
from neurox.encoding import Encoding
from neurox.primitive.digital import RadixAccumulatorConfig
from neurox.primitive.macro.cim import CimMacroQuantizationScheme, IdealCimMacroConfig, IdealCimMacroPolicy


@pytest.mark.parametrize("role", ["linear", "conv2d"])
@pytest.mark.parametrize("differential", [False, True])
def test_ideal_conversion_keeps_geometry_cost_ownership_and_independent_state(
    role: str, differential: bool, device: torch.device
) -> None:
    common = {"area_per_inst__um2": 3.0, "leakage_per_inst__uW": 2.0}
    geometry = {"stride": (2, 1), "padding": (1, 0), "dilation": (1, 2), "groups": 1} if role == "conv2d" else {}
    config_type = LinearCimUnitConfig if role == "linear" else Conv2dCimUnitConfig
    policy_type = LinearCimUnitPolicy if role == "linear" else Conv2dCimUnitPolicy
    expected_type = LinearCimUnit if role == "linear" else Conv2dCimUnit
    config = config_type(
        clock_period__ns=1.0,
        **common,
        **({"merge": True} if role == "conv2d" else {}),
        cim_macro_config=IdealCimMacroConfig(
            input_num=8,
            lane_num=1,
            scan_num=8,
            max_active_num=4,
            area_per_inst__um2=100.0,
            leakage_per_inst__uW=50.0,
            w_digit_num=2,
            w_digit_radix=2,
            w_encoding=Encoding.UNSIGNED if differential else Encoding.TRUE_FORM,
            w_signed=not differential,
            x_digit_num=1,
            x_digit_radix=2,
            x_encoding=Encoding.UNSIGNED,
            x_value_range=(0, 1),
            w_value_range=(0, 3) if differential else (-3, 3),
            adc_bits=8,
            quantization_scheme=CimMacroQuantizationScheme.ZERO_POINT,
            rescale_factors=(1.0,),
        ),
        tile_accumulator_config=None,
        w_polarity_adder_config=None,
        local_accumulator_config=RadixAccumulatorConfig(
            bit_width=32,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
            energy_per_op__fJ=0.0,
        ),
        w_slice_num=2,
        w_slice_encoding=Encoding.TRUE_FORM,
        x_slice_num=2,
        x_slice_encoding=Encoding.UNSIGNED,
        w_radix_summator_config=None,
    )
    policy = policy_type(cim_macro_policy=IdealCimMacroPolicy())
    shape = (3, 2) if role == "linear" else (3, 2, 2, 2)
    w = torch.ones(shape, dtype=torch.int64, device=device)
    w[1] = -1
    unit = expected_type(config=config, policy=policy, w_logical_shape=w.shape, dtype=torch.float64, **geometry).to(
        device
    )
    unit.set_temperature(333.0)
    unit.program(w, bias=torch.full((3,), 7, dtype=torch.int64, device=device))

    ideal = unit.to_ideal()
    assert ideal.x_value_range == (0, 3)
    assert ideal.w_value_range == (-15, 15)
    assert ideal.config.area_per_inst__um2 == 3.0
    assert ideal.config.leakage_per_inst__uW == 2.0
    assert ideal.T__K == 333.0
    x_shape = (2, 2) if role == "linear" else (2, 2, 5, 6)
    x = torch.ones(x_shape, dtype=torch.int64, device=device)
    # Callers may compile a model around the unit's already-compiled entry.
    execute = torch.compile(
        lambda value: getattr(ideal, role)(value, quantization_mode=0, adc_active_bits=None),
        dynamic=False,
        fullgraph=True,
    )
    ideal.program(2 * w)
    expected = F.linear(x.cpu(), w.cpu()) if role == "linear" else F.conv2d(x.cpu(), w.cpu(), **geometry)
    expected = expected.to(device)
    torch.testing.assert_close(execute(x), 2 * expected)
    execute_original = torch.compile(
        lambda value: getattr(unit, role)(value, quantization_mode=0, adc_active_bits=None),
        dynamic=False,
        fullgraph=True,
    )
    original = execute_original(x)
    torch.testing.assert_close(original, expected + 7)
