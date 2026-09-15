"""Functional unit families load their implementations and construct fresh ideal twins."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from neurox import conv2d_unit_from_file, linear_unit_from_file
from neurox.architecture.mapping import TilingMode
from neurox.architecture.unit.conv2d import (
    Conv2dCimUnit,
    Conv2dCimUnitConfig,
    Conv2dCimUnitPolicy,
    IdealConv2dUnit,
    IdealConv2dUnitConfig,
    IdealConv2dUnitPolicy,
)
from neurox.architecture.unit.linear import (
    IdealLinearUnit,
    IdealLinearUnitConfig,
    IdealLinearUnitPolicy,
    LinearCimUnit,
    LinearCimUnitConfig,
    LinearCimUnitPolicy,
    LinearUnit,
    LinearUnitConfig,
    LinearUnitPolicy,
)
from neurox.encoding import Encoding
from neurox.primitive.digital import AccumulatorConfig
from neurox.primitive.macro.cim import CimMacroQuantizationScheme, IdealCimMacroConfig, IdealCimMacroPolicy


@pytest.mark.parametrize("role", ["linear", "conv2d"])
@pytest.mark.parametrize("implementation", ["cim", "ideal"])
def test_role_file_factory_and_fresh_ideal_conversion(role: str, implementation: str, tmp_path: Path) -> None:
    common = {"area_per_inst__um2": 3.0, "leakage_per_inst__uW": 2.0}
    geometry = {"stride": (2, 1), "padding": (1, 0), "dilation": (1, 2)} if role == "conv2d" else {}
    if implementation == "cim":
        config_type = LinearCimUnitConfig if role == "linear" else Conv2dCimUnitConfig
        policy_type = LinearCimUnitPolicy if role == "linear" else Conv2dCimUnitPolicy
        expected_type = LinearCimUnit if role == "linear" else Conv2dCimUnit
        config = config_type(
            **common,
            **geometry,
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
                w_encoding=Encoding.TRUE_FORM,
                x_digit_num=1,
                x_digit_radix=2,
                x_encoding=Encoding.UNSIGNED,
                x_value_range=(0, 1),
                w_value_range=(-3, 3),
                adc_bits=8,
                quantization_scheme=CimMacroQuantizationScheme.ZERO_POINT,
                rescale_factors=(1.0,),
            ),
            phase_accumulator_config=AccumulatorConfig(
                bit_width=32,
                area_per_inst__um2=0.0,
                leakage_per_inst__uW=0.0,
                energy_per_op__fJ=0.0,
                latency_per_op__ns=0.0,
            ),
            w_slice_num=2,
            w_slice_encoding=Encoding.TRUE_FORM,
            x_slice_num=2,
            x_slice_encoding=Encoding.UNSIGNED,
            tiling=TilingMode.SLICE_PLANES,
            w_shift_adder_config=None,
            x_shift_adder_config=None,
        )
        policy = policy_type(cim_macro_policy=IdealCimMacroPolicy())
    else:
        config_type = IdealLinearUnitConfig if role == "linear" else IdealConv2dUnitConfig
        policy_type = IdealLinearUnitPolicy if role == "linear" else IdealConv2dUnitPolicy
        expected_type = IdealLinearUnit if role == "linear" else IdealConv2dUnit
        config = config_type(**common, **geometry, x_value_range=(0, 3), w_value_range=(-15, 15))
        policy = policy_type()

    config_file, policy_file = tmp_path / "config.yaml", tmp_path / "policy.yaml"
    config.to_file(config_file)
    policy.to_file(policy_file)
    shape = (3, 2) if role == "linear" else (3, 2, 2, 2)
    load = linear_unit_from_file if role == "linear" else conv2d_unit_from_file
    unit = load(config_files=[config_file], policy_files=[policy_file], w_logical_shape=shape, dtype=torch.float64)
    assert type(unit) is expected_type
    unit.set_temperature(333.0)
    w = torch.ones(shape, dtype=torch.int64)
    unit.program(w, torch.full((3,), 7, dtype=torch.int64))

    ideal = unit.to_ideal()
    assert ideal is not unit
    assert type(ideal) is (IdealLinearUnit if role == "linear" else IdealConv2dUnit)
    assert ideal.x_value_range == (0, 3)
    assert ideal.w_value_range == (-15, 15)
    assert ideal.config.area_per_inst__um2 == 3.0
    assert ideal.config.leakage_per_inst__uW == 2.0
    assert ideal.T__K == 333.0
    assert ideal._dtype == torch.float64
    assert ideal._int_bias is None
    if role == "conv2d":
        assert ideal.config.stride == (2, 1)
        assert ideal.config.padding == (1, 0)
        assert ideal.config.dilation == (1, 2)
    x = torch.ones((2, 2), dtype=torch.int64) if role == "linear" else torch.ones((2, 2, 5, 6), dtype=torch.int64)
    execute = getattr(ideal, role)
    with pytest.raises(AttributeError):
        execute(x, quantization_mode=0, adc_active_bits=None)
    ideal.program(w)
    expected = F.linear(x, w) if role == "linear" else F.conv2d(x, w, **geometry)
    torch.testing.assert_close(execute(x, quantization_mode=0, adc_active_bits=None), expected)


class _FunctionalLinear(LinearUnit):
    is_profile_target = False

    @property
    def x_value_range(self) -> tuple[int, int]:
        return (0, 3)

    @property
    def w_value_range(self) -> tuple[int, int]:
        return (-3, 3)

    @property
    def adc_bits(self) -> None:
        return None

    def rescale_factor(self, *, quantization_mode: int, adc_active_bits: int | None) -> float:
        return 1.0

    def latency__ns(self, input_shape: tuple[int, ...], *, adc_active_bits: int | None) -> float:
        return 0.0

    def program(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        self._weight = weight.long()
        self._program_int_bias(bias, channels=weight.shape[0])

    def _linear_impl(self, input: torch.Tensor, *, quantization_mode: int, adc_active_bits: int | None) -> torch.Tensor:
        return F.linear(input.long(), self._weight, self._int_bias)


def test_functional_unit_needs_no_ppa_configuration() -> None:
    unit = _FunctionalLinear(
        config=LinearUnitConfig(),
        policy=LinearUnitPolicy(),
        w_logical_shape=(1, 2),
        dtype=torch.float32,
    )
    w = torch.tensor([[2, 3]])
    x = torch.tensor([[1, 2]])
    unit.program(w)
    torch.testing.assert_close(unit.linear(x, quantization_mode=0, adc_active_bits=None), torch.tensor([[8]]))
    ideal = unit.to_ideal()
    assert ideal.area__um2 == 0.0
    assert ideal.leakage__uW == 0.0
    ideal.program(w)
    torch.testing.assert_close(ideal.linear(x, quantization_mode=0, adc_active_bits=None), torch.tensor([[8]]))
