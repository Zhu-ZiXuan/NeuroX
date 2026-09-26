"""Ideal integer reductions preserve cancellation beyond floating-point precision."""

from __future__ import annotations

import pytest
import torch

from neurox.architecture.unit.conv2d import IdealConv2dUnit, IdealConv2dUnitConfig, IdealConv2dUnitPolicy
from neurox.architecture.unit.linear import IdealLinearUnit, IdealLinearUnitConfig, IdealLinearUnitPolicy


@pytest.mark.parametrize("convolution", [False, True])
def test_integer_cancellation_preserves_low_bits_on_the_programmed_device(
    device: torch.device, convolution: bool
) -> None:
    large = 2**54
    config_type = IdealConv2dUnitConfig if convolution else IdealLinearUnitConfig
    unit_type = IdealConv2dUnit if convolution else IdealLinearUnit
    policy = IdealConv2dUnitPolicy() if convolution else IdealLinearUnitPolicy()
    config = config_type(
        x_value_range=(-large, large + 3),
        w_value_range=(-1, 1),
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
    )
    geometry = {"stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "groups": 1} if convolution else {}
    weight_shape = (1, 2, 1, 1) if convolution else (1, 2)
    unit = unit_type(config=config, policy=policy, w_logical_shape=weight_shape, dtype=torch.float64, **geometry)
    unit.program(
        torch.ones(weight_shape, dtype=torch.int64, device=device),
        bias=torch.tensor([5], dtype=torch.int64, device=device),
    )
    values = torch.tensor([[large + 1, -large], [large + 3, -large]], dtype=torch.int64, device=device)
    if convolution:
        values = values.T.reshape(1, 2, 1, 2)
        actual = unit.conv2d(values, quantization_mode=0, adc_active_bits=None)
        expected_shape = (1, 1, 1, 2)
    else:
        actual = unit.linear(values, quantization_mode=0, adc_active_bits=None)
        expected_shape = (2, 1)
    expected = torch.tensor([6, 8], dtype=torch.int64, device=device).reshape(expected_shape)
    torch.testing.assert_close(actual, expected)
