"""Output-scale contracts of the CIM-macro base."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from neurox.common.encoding import Encoding
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

    @property
    def w_digit_n(self) -> int:
        return 1

    @property
    def w_digit_r(self) -> int:
        return 2

    @property
    def w_enc(self) -> Encoding:
        return Encoding.TRUE_FORM

    @property
    def x_digit_n(self) -> int:
        return 1

    @property
    def x_digit_r(self) -> int:
        return 2

    @property
    def x_enc(self) -> Encoding:
        return Encoding.UNSIGNED

    @property
    def quant_scheme(self) -> CimMacroQuantizationScheme:
        return self.quantization_scheme


class _StubMacroPolicy(CimMacroPolicy):
    pass


class _StubMacro(CimMacro):
    @property
    def adc_bits(self) -> int:
        return self.config.adc_bits

    def program(self, w: Tensor) -> None:
        raise NotImplementedError

    def _vec_mat_mul_impl(
        self,
        x: Tensor,
        *,
        quantization_mode: int,
        adc_active_bits: int | None,
    ) -> Tensor:
        del quantization_mode, adc_active_bits
        return x[..., : self.output_num].unflatten(-1, (self.lane_num, self.scan_num))

    def latency__ns(self, *, adc_active_bits: int | None) -> float:
        del adc_active_bits
        return 0.0


class _BadOutputMacro(_StubMacro):
    def _vec_mat_mul_impl(
        self,
        x: Tensor,
        *,
        quantization_mode: int,
        adc_active_bits: int | None,
    ) -> Tensor:
        del quantization_mode, adc_active_bits
        return x[..., : self.output_num]


def _stub_macro(
    *,
    factors: tuple[float, ...] = (1.0, 0.5),
    adc_bits: int = 4,
    input_num: int = 8,
    lane_num: int = 1,
    scan_num: int = 4,
    max_active_num: int = 4,
    inst_shape: tuple[int, ...] = (),
    quantization_scheme: CimMacroQuantizationScheme = CimMacroQuantizationScheme.ZERO_POINT,
) -> _StubMacro:
    config = _StubMacroConfig(
        input_num=input_num,
        rescale_factors=factors,
        area_per_inst__um2=1.0,
        leakage_per_inst__uW=2.0,
        max_active_num=max_active_num,
        lane_num=lane_num,
        scan_num=scan_num,
        adc_bits=adc_bits,
        quantization_scheme=quantization_scheme,
    )
    return _StubMacro(
        config=config,
        policy=_StubMacroPolicy(),
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
        assert macro.rescale_factor(quantization_mode=1, adc_active_bits=None) == 0.5

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
        macro = _stub_macro(input_num=16, lane_num=2, scan_num=3, max_active_num=8, inst_shape=(2, 3))
        twin = macro.to_ideal()
        assert isinstance(twin, IdealCimMacro)
        assert twin.config.rescale_factors == macro.config.rescale_factors
        assert twin.adc_bits == macro.adc_bits
        assert twin.x_value_range == macro.x_value_range
        assert twin.w_value_range == macro.w_value_range
        assert twin.inst_shape == macro.inst_shape
        assert twin.max_active_num == macro.max_active_num
        assert twin.lane_num == macro.lane_num
        assert twin.scan_num == macro.scan_num

    def test_twin_uses_the_physical_mode_scale(self) -> None:
        macro = _stub_macro(
            factors=(10.0,),
            adc_bits=3,
            quantization_scheme=CimMacroQuantizationScheme.SIGN_MAGNITUDE,
            input_num=1,
            scan_num=1,
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


class TestVecMatMulTemplate:
    def test_flattens_the_canonical_readout_layout(self) -> None:
        macro = _stub_macro(input_num=8, lane_num=2, scan_num=3)
        x = torch.arange(16).reshape(2, 8)
        output = macro.vec_mat_mul(x, quantization_mode=0, adc_active_bits=None)
        assert tuple(output.shape) == (2, 6)
        assert torch.equal(output, x[..., :6])

    def test_rejects_wrong_input_width(self) -> None:
        with pytest.raises(ValueError, match=r"x\.shape\[-1\]"):
            _stub_macro(input_num=8).vec_mat_mul(torch.zeros(7), quantization_mode=0, adc_active_bits=None)

    def test_rejects_a_noncanonical_implementation_output(self) -> None:
        base = _stub_macro(input_num=8, lane_num=2, scan_num=2)
        macro = _BadOutputMacro(
            config=base.config,
            policy=base.policy,
            inst_shape=(),
            dtype=torch.float32,
            T__K=300.0,
        )
        with pytest.raises(ValueError, match=r"implementation output trailing shape"):
            macro.vec_mat_mul(torch.zeros(8), quantization_mode=0, adc_active_bits=None)
