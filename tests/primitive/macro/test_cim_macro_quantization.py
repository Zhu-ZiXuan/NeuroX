"""Output-scale contracts of the CIM-macro base."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from neurox.primitive.macro.cim import (
    CimMacro,
    CimMacroConfig,
    CimMacroPolicy,
    CimMacroQuantizationScheme,
    IdealCimMacro,
)


class _StubMacroConfig(CimMacroConfig):
    adc_bits: int
    quantization_scheme: CimMacroQuantizationScheme


class _StubMacroPolicy(CimMacroPolicy):
    pass


class _StubMacro(CimMacro[_StubMacroConfig, _StubMacroPolicy]):
    @property
    def x_value_range(self) -> tuple[int, int]:
        return (0, 1)

    @property
    def w_value_range(self) -> tuple[int, int]:
        return (-1, 1)

    @property
    def adc_bits(self) -> int:
        return self.config.adc_bits

    @property
    def _quantization_scheme(self) -> CimMacroQuantizationScheme:
        return self.config.quantization_scheme

    def restore_adc_layout(self, value: Tensor) -> Tensor:
        return value

    def program(self, w: Tensor) -> None:
        raise NotImplementedError

    def vec_mat_mul(self, x: Tensor, *, quantization_mode: int, adc_active_bits: int) -> Tensor:
        raise NotImplementedError

    def initiation_interval__ns(self, *, adc_active_bits: int) -> float:
        del adc_active_bits
        return 0.0


def _stub_macro(
    *,
    factors: tuple[float, ...] = (1.0, 0.5),
    adc_bits: int = 4,
    input_num: int = 8,
    output_num: int = 4,
    max_active_num: int = 4,
    inst_shape: tuple[int, ...] = (),
    quantization_scheme: CimMacroQuantizationScheme = CimMacroQuantizationScheme.ZERO_POINT,
) -> _StubMacro:
    config = _StubMacroConfig(
        rescale_factors=factors,
        area_per_inst__um2=1.0,
        leakage_per_inst__uW=2.0,
        max_active_num=max_active_num,
        adc_bits=adc_bits,
        quantization_scheme=quantization_scheme,
    )
    return _StubMacro(
        config=config,
        policy=_StubMacroPolicy(),
        input_num=input_num,
        output_num=output_num,
        inst_shape=inst_shape,
        dtype=torch.float32,
        T__K=300.0,
    )


class TestCimMacroConfig:
    def test_rejects_empty_factors(self) -> None:
        with pytest.raises(ValueError, match=r"rescale_factors"):
            _stub_macro(factors=())

    @pytest.mark.parametrize("factor", [0.0, -1.0])
    def test_rejects_non_positive_factor(self, factor: float) -> None:
        with pytest.raises(ValueError, match=r"rescale_factors\[1\]"):
            _stub_macro(factors=(1.0, factor))


class TestRescaleFactor:
    def test_full_resolution_selects_the_mode_factor(self) -> None:
        macro = _stub_macro()
        assert macro.rescale_factor(quantization_mode=0, adc_active_bits=4) == 1.0
        assert macro.rescale_factor(quantization_mode=1, adc_active_bits=4) == 0.5

    def test_each_dropped_bit_doubles_the_factor(self) -> None:
        macro = _stub_macro()
        assert macro.rescale_factor(quantization_mode=1, adc_active_bits=3) == 1.0
        assert macro.rescale_factor(quantization_mode=1, adc_active_bits=2) == 2.0

    @pytest.mark.parametrize("adc_active_bits", [0, -1, 5])
    def test_rejects_invalid_width(self, adc_active_bits: int) -> None:
        with pytest.raises(ValueError, match=r"adc_active_bits"):
            _stub_macro().rescale_factor(quantization_mode=0, adc_active_bits=adc_active_bits)

    def test_rejects_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match=r"quantization_mode"):
            _stub_macro().rescale_factor(quantization_mode=2, adc_active_bits=4)


class TestToIdeal:
    def test_twin_preserves_scales_and_geometry(self) -> None:
        macro = _stub_macro(input_num=16, output_num=6, max_active_num=8, inst_shape=(2, 3))
        twin = macro.to_ideal()
        assert isinstance(twin, IdealCimMacro)
        assert twin.config.rescale_factors == macro.config.rescale_factors
        assert twin.adc_bits == macro.adc_bits
        assert twin.x_value_range == macro.x_value_range
        assert twin.w_value_range == macro.w_value_range
        assert twin.inst_shape == macro.inst_shape
        assert twin.max_active_num == macro.max_active_num

    def test_twin_uses_the_physical_mode_scale(self) -> None:
        macro = _stub_macro(
            factors=(10.0,),
            adc_bits=3,
            quantization_scheme=CimMacroQuantizationScheme.SIGN_MAGNITUDE,
            input_num=1,
            output_num=1,
            max_active_num=1,
        )
        twin = macro.to_ideal()
        twin.eval()
        twin.program(torch.ones((1, 1), dtype=torch.int32))
        dots = torch.tensor([[-80], [-69], [-10], [-9], [0], [9], [10], [69], [70], [80]])
        code = twin.vec_mat_mul(dots, quantization_mode=0, adc_active_bits=3)
        assert code.squeeze(-1).tolist() == [-7, -6, -1, 0, 0, 0, 1, 6, 7, 7]
        assert twin.rescale_factor(quantization_mode=0, adc_active_bits=3) == 10.0

    def test_ideal_macro_returns_itself(self) -> None:
        twin = _stub_macro().to_ideal()
        assert twin.to_ideal() is twin
