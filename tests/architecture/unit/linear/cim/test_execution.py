"""Linear operators preserve leading axes and replace their programmed bias."""

from __future__ import annotations

import math

import pytest
import torch

from neurox import Profiler
from neurox.architecture.unit.linear import LinearCimUnit, LinearCimUnitConfig, LinearCimUnitPolicy
from neurox.encoding import Encoding
from neurox.primitive.digital import RadixAccumulatorConfig
from neurox.primitive.macro.cim import CimMacroQuantizationScheme, IdealCimMacroConfig, IdealCimMacroPolicy


@pytest.mark.parametrize("batch", [(), (2, 3)])
@pytest.mark.parametrize("differential", [False, True])
def test_bias_is_per_output_and_reprogramming_can_clear_it(
    batch: tuple[int, ...], differential: bool, device: torch.device
) -> None:
    config = LinearCimUnitConfig(
        cim_macro_config=IdealCimMacroConfig(
            input_num=8,
            lane_num=1,
            scan_num=4,
            max_active_num=4,
            rescale_factors=(1.0,),
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
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
        ),
        local_accumulator_config=RadixAccumulatorConfig(
            bit_width=32, energy_per_op__fJ=0.0, area_per_inst__um2=0.0, leakage_per_inst__uW=0.0
        ),
        tile_accumulator_config=None,
        w_polarity_adder_config=None,
        w_radix_summator_config=None,
        w_slice_num=1,
        w_slice_encoding=Encoding.TRUE_FORM if differential else None,
        x_slice_num=1,
        x_slice_encoding=None,
        clock_period__ns=1.0,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
    )
    weight = torch.arange(35, dtype=torch.int32, device=device).reshape(5, 7) % 5 - 2
    bias = torch.arange(-2, 3, dtype=torch.int32, device=device)
    x = torch.arange(math.prod(batch) * 7, dtype=torch.int32, device=device).reshape(*batch, 7) % 2
    unit = LinearCimUnit(
        config=config,
        policy=LinearCimUnitPolicy(cim_macro_policy=IdealCimMacroPolicy()),
        w_logical_shape=weight.shape,
        dtype=torch.float32,
    ).to(device)
    expected = (x.cpu().long() @ weight.cpu().long().T).to(device)
    unit.program(weight, bias=bias)
    actual = unit.linear(x, quantization_mode=0, adc_active_bits=None)
    torch.testing.assert_close(actual, expected + bias)

    one_vector__ns = unit.latency__ns((7,), adc_active_bits=None)
    assert one_vector__ns > 0.0
    assert unit.latency__ns(x.shape, adc_active_bits=None) == one_vector__ns

    unit.set_profile_leading_rank(len(batch) + 1)
    rejected = Profiler(concat_dim=0)
    rejected.collect_static_data(unit)
    # Catch inside the context so any premature child submissions remain visible.
    # Eager execution exposes the ValueError without Dynamo's tracing wrapper.
    with (
        torch.compiler.set_stance("force_eager"),
        rejected,
        pytest.raises(ValueError, match="profile_leading_rank"),
    ):
        unit.linear(x, quantization_mode=0, adc_active_bits=None)
    assert all(
        item.dynamic_energy__fJ is None and item.working_duration__ns is None for item in rejected.result.values()
    )

    unit.program(weight)
    unit.set_profile_leading_rank(len(batch))
    profiler = Profiler(concat_dim=0)
    profiler.collect_static_data(unit)
    with profiler:
        actual = unit.linear(x, quantization_mode=0, adc_active_bits=None)
    torch.testing.assert_close(actual, expected)
    timing = profiler.result[""].working_duration__ns
    torch.testing.assert_close(timing, torch.full(batch or (1,), one_vector__ns, dtype=torch.float64, device="cpu"))
