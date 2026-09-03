"""Input activation geometry on the CimMacro base.

- `CimMacroConfig` geometry guards for `max_active_num`.
- `max_active_num` exposes the per-conversion selection limit.
"""

from __future__ import annotations

import dataclasses
from typing import TypedDict

import pytest
import torch

from neurox.common.encoding import Encoding
from neurox.primitive.macro.cim import (
    CimMacroQuantizationScheme,
    IdealCimMacro,
    IdealCimMacroConfig,
    IdealCimMacroPolicy,
)


class _IdealCimMacroKwargs(TypedDict):
    input_num: int
    rescale_factors: tuple[float, ...]
    max_active_num: int
    lane_num: int
    scan_num: int
    area_per_inst__um2: float
    leakage_per_inst__uW: float
    w_digit_num: int
    w_digit_radix: int
    w_encoding: Encoding
    x_digit_num: int
    x_digit_radix: int
    x_encoding: Encoding
    x_value_range: tuple[int, int]
    w_value_range: tuple[int, int]
    adc_bits: int
    quantization_scheme: CimMacroQuantizationScheme


def _config_kwargs(
    *,
    max_active_num: int,
    input_num: int = 8,
    lane_num: int = 1,
    scan_num: int = 4,
) -> _IdealCimMacroKwargs:
    return {
        "input_num": input_num,
        "rescale_factors": (1.0,),
        "max_active_num": max_active_num,
        "lane_num": lane_num,
        "scan_num": scan_num,
        "area_per_inst__um2": 0.0,
        "leakage_per_inst__uW": 0.0,
        "w_digit_num": 1,
        "w_digit_radix": 2,
        "w_encoding": Encoding.UNSIGNED,
        "x_digit_num": 1,
        "x_digit_radix": 2,
        "x_encoding": Encoding.UNSIGNED,
        "x_value_range": (0, 1),
        "w_value_range": (0, 1),
        "adc_bits": 8,
        "quantization_scheme": CimMacroQuantizationScheme.ZERO_POINT,
    }


def _make_macro(
    *,
    input_num: int,
    max_active_num: int,
    lane_num: int = 1,
    scan_num: int = 4,
    inst_shape: tuple[int, ...] = (),
) -> IdealCimMacro:
    config = IdealCimMacroConfig(
        **_config_kwargs(
            max_active_num=max_active_num,
            input_num=input_num,
            lane_num=lane_num,
            scan_num=scan_num,
        )
    )
    macro = IdealCimMacro(
        config=config,
        policy=IdealCimMacroPolicy(),
        inst_shape=inst_shape,
        dtype=torch.float32,
        T__K=300.0,
    )
    macro.eval()
    return macro


# ---------------------------------------------------------------------------
# max_active_num geometry validation
# ---------------------------------------------------------------------------


class TestMaxActiveSizeValidation:
    def test_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"max_active_num"):
            IdealCimMacroConfig(**_config_kwargs(max_active_num=0))

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"max_active_num"):
            IdealCimMacroConfig(**_config_kwargs(max_active_num=-2))

    def test_above_input_num_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"max_active_num"):
            _make_macro(input_num=8, max_active_num=16)

    def test_non_divisor_accepted(self) -> None:
        cfg = IdealCimMacroConfig(**_config_kwargs(max_active_num=3))
        assert cfg.max_active_num == 3

    def test_full_activation_accepted(self) -> None:
        cfg = IdealCimMacroConfig(**_config_kwargs(max_active_num=8))
        assert cfg.max_active_num == 8

    def test_max_active_num_property(self) -> None:
        macro = _make_macro(input_num=8, max_active_num=2)
        assert macro.max_active_num == 2
        full = _make_macro(input_num=8, max_active_num=8)
        assert full.max_active_num == 8


class TestReadoutGeometry:
    def test_non_positive_lane_num_rejected(self) -> None:
        with pytest.raises(ValueError, match="lane_num"):
            IdealCimMacroConfig(**_config_kwargs(max_active_num=1, lane_num=0))

    def test_non_positive_scan_num_rejected(self) -> None:
        with pytest.raises(ValueError, match="scan_num"):
            IdealCimMacroConfig(**_config_kwargs(max_active_num=1, scan_num=0))

    def test_output_num_is_derived_from_readout_geometry(self) -> None:
        macro = _make_macro(input_num=4, max_active_num=4, lane_num=2, scan_num=4)
        assert macro.lane_num == 2
        assert macro.scan_num == 4
        assert macro.output_num == 8


class TestDigitGeometryValidation:
    @pytest.mark.parametrize(
        ("field_name", "property_name"),
        [("w_digit_num", "w_digit_n"), ("x_digit_num", "x_digit_n")],
    )
    def test_non_positive_digit_num_rejected(self, field_name: str, property_name: str) -> None:
        config = IdealCimMacroConfig(**_config_kwargs(max_active_num=1))
        with pytest.raises(ValueError, match=property_name):
            dataclasses.replace(config, **{field_name: 0})

    @pytest.mark.parametrize(
        ("field_name", "property_name"),
        [("w_digit_radix", "w_digit_r"), ("x_digit_radix", "x_digit_r")],
    )
    def test_sub_binary_digit_radix_rejected(self, field_name: str, property_name: str) -> None:
        config = IdealCimMacroConfig(**_config_kwargs(max_active_num=1))
        with pytest.raises(ValueError, match=property_name):
            dataclasses.replace(config, **{field_name: 1})
