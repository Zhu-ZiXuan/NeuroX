"""Ideal CIM-macro output quantization from calibrated mode scales."""

from __future__ import annotations

import pytest
import torch

from neurox.common.encoding import Encoding
from neurox.primitive.macro.cim import (
    CimMacroQuantizationScheme,
    IdealCimMacro,
    IdealCimMacroConfig,
    IdealCimMacroPolicy,
)


def _make_macro(
    *,
    factors: tuple[float, ...] = (1.0,),
    adc_bits: int = 4,
    x_value_range: tuple[int, int] = (-128, 127),
    w_value_range: tuple[int, int] = (-1, 1),
    scheme: CimMacroQuantizationScheme = CimMacroQuantizationScheme.ZERO_POINT,
) -> IdealCimMacro:
    macro = IdealCimMacro(
        config=IdealCimMacroConfig(
            input_num=1,
            rescale_factors=factors,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
            max_active_num=1,
            lane_num=1,
            scan_num=1,
            w_digit_num=1,
            w_digit_radix=2,
            w_encoding=Encoding.TRUE_FORM if w_value_range[0] < 0 else Encoding.UNSIGNED,
            x_digit_num=1,
            x_digit_radix=2,
            x_encoding=Encoding.COMPLEMENT if x_value_range[0] < 0 else Encoding.UNSIGNED,
            x_value_range=x_value_range,
            w_value_range=w_value_range,
            adc_bits=adc_bits,
            quantization_scheme=scheme,
        ),
        policy=IdealCimMacroPolicy(),
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    macro.program(torch.ones((1, 1), dtype=torch.int64))
    macro.eval()
    return macro


def _codes(
    macro: IdealCimMacro,
    dots: list[int],
    *,
    quantization_mode: int = 0,
    adc_active_bits: int | None,
) -> list[int]:
    x = torch.tensor(dots, dtype=torch.int64).unsqueeze(-1)
    return (
        macro.vec_mat_mul(
            x,
            quantization_mode=quantization_mode,
            adc_active_bits=adc_active_bits,
        )
        .squeeze(-1)
        .tolist()
    )


class TestHighestPrecision:
    def test_none_returns_exact_dots(self) -> None:
        macro = _make_macro(factors=(13.0,))
        assert _codes(macro, [-101, -1, 0, 7, 103], adc_active_bits=None) == [-101, -1, 0, 7, 103]

    def test_none_has_identity_factor(self) -> None:
        macro = _make_macro(factors=(13.0,))
        assert macro.rescale_factor(quantization_mode=0, adc_active_bits=None) == 1.0


class TestZeroPoint:
    def test_full_resolution_returns_centered_signed_codes(self) -> None:
        macro = _make_macro(factors=(2.0,), adc_bits=4)
        dots = [-100, -16, -15, -2, -1, 0, 1, 2, 14, 15, 16, 100]
        assert _codes(macro, dots, adc_active_bits=4) == [-8, -8, -8, -1, -1, 0, 0, 1, 7, 7, 7, 7]

    def test_lower_width_arithmetically_truncates_the_full_code(self) -> None:
        macro = _make_macro(factors=(2.0,), adc_bits=4)
        dots = [-16, -1, 0, 7, 8, 16]
        assert _codes(macro, dots, adc_active_bits=2) == [-2, -1, 0, 0, 1, 1]
        assert macro.rescale_factor(quantization_mode=0, adc_active_bits=2) == 8.0

    def test_logical_value_ranges_do_not_change_the_mapping(self) -> None:
        signed_domain = _make_macro(factors=(2.0,), adc_bits=4)
        nonnegative_domain = _make_macro(
            factors=(2.0,),
            adc_bits=4,
            x_value_range=(0, 255),
            w_value_range=(0, 1),
        )
        dots = [-16, -1, 0, 7, 8, 16]
        assert _codes(signed_domain, dots, adc_active_bits=4) == _codes(
            nonnegative_domain,
            dots,
            adc_active_bits=4,
        )


class TestSignMagnitude:
    def test_quantizes_magnitude_and_restores_sign(self) -> None:
        macro = _make_macro(
            factors=(10.0,),
            adc_bits=3,
            scheme=CimMacroQuantizationScheme.SIGN_MAGNITUDE,
        )
        dots = [-81, -80, -79, -70, -69, -10, -9, -1, 0, 1, 9, 10, 69, 70, 79, 80]
        assert _codes(macro, dots, adc_active_bits=3) == [
            -7,
            -7,
            -7,
            -7,
            -6,
            -1,
            0,
            0,
            0,
            0,
            0,
            1,
            6,
            7,
            7,
            7,
        ]

    def test_lower_width_truncates_the_full_magnitude(self) -> None:
        macro = _make_macro(
            factors=(10.0,),
            adc_bits=3,
            scheme=CimMacroQuantizationScheme.SIGN_MAGNITUDE,
        )
        assert _codes(macro, [-80, -50, -19, 0, 19, 50, 80], adc_active_bits=2) == [-3, -2, 0, 0, 0, 2, 3]
        assert macro.rescale_factor(quantization_mode=0, adc_active_bits=2) == 20.0


class TestStochasticRounding:
    def test_full_resolution_rounds_by_the_fractional_part(self) -> None:
        macro = _make_macro(factors=(3.0,), adc_bits=6)
        macro.train()
        torch.manual_seed(0)
        code = _codes(macro, [1] * 3_000, adc_active_bits=6)
        assert set(code) == {0, 1}
        assert sum(value == 1 for value in code) / len(code) == pytest.approx(1 / 3, abs=0.03)

    def test_rounding_precedes_active_width_truncation(self) -> None:
        macro = _make_macro(factors=(3.0,), adc_bits=6)
        macro.train()
        torch.manual_seed(0)
        code = _codes(macro, [4] * 3_000, adc_active_bits=5)
        assert set(code) == {0, 1}
        assert sum(value == 1 for value in code) / len(code) == pytest.approx(1 / 3, abs=0.03)

    def test_integral_full_resolution_codes_do_not_jitter(self) -> None:
        macro = _make_macro(factors=(2.0,), adc_bits=6)
        macro.train()
        for seed in range(4):
            torch.manual_seed(seed)
            assert _codes(macro, [-4, -2, 0, 2, 4], adc_active_bits=6) == [-2, -1, 0, 1, 2]


class TestValidation:
    @pytest.mark.parametrize("adc_active_bits", [-1, 5])
    def test_rejects_invalid_width(self, adc_active_bits: int) -> None:
        with pytest.raises(ValueError, match=r"adc_active_bits"):
            _codes(_make_macro(adc_bits=4), [0], adc_active_bits=adc_active_bits)

    def test_rejects_invalid_mode_even_for_exact_bypass(self) -> None:
        with pytest.raises(ValueError, match=r"quantization_mode"):
            _codes(_make_macro(), [0], quantization_mode=1, adc_active_bits=None)
