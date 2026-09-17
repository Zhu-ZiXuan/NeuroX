"""Ideal-twin conversion and lane scheduling owned by the CIM-macro base."""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.encoding import Encoding
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

    def _phase_mask_from_effective_output_num(self, effective_output_num: Tensor) -> Tensor:
        indices = torch.arange(self.output_num, device=effective_output_num.device).view(self.lane_num, self.scan_num)
        return indices < effective_output_num[..., None, None]

    def program(self, w: Tensor) -> None:
        raise NotImplementedError

    def _vec_mat_mul_impl(
        self,
        x: Tensor,
        *,
        leading_shape: tuple[int, ...],
        quantization_mode: int,
        adc_active_bits: int | None,
        phase_mask: Tensor | None,
    ) -> Tensor:
        del quantization_mode, adc_active_bits
        return x[..., : self.output_num].expand(*leading_shape, self.output_num)

    def _latency_per_scan__ns(self, *, adc_active_bits: int | None) -> float:
        del adc_active_bits
        return 0.0


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
    )


class TestToIdeal:
    def test_twin_preserves_scales_and_geometry(self) -> None:
        macro = _stub_macro(input_num=16, lane_num=2, scan_num=3, max_active_num=8, inst_shape=(2, 3))
        macro.set_temperature(350.0)
        twin = macro.to_ideal()
        assert isinstance(twin, IdealCimMacro)
        assert twin.T__K == macro.T__K
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


def test_effective_outputs_broadcast_and_mask_in_logical_order(device: torch.device) -> None:
    macro = _stub_macro(input_num=4, lane_num=2, scan_num=2).to(device)
    x = torch.arange(1, 5, device=device).expand(3, 2, 4)
    counts = torch.tensor([[0, 1], [2, 3], [4, 2]], device=device)
    x = x.clone()
    x[0, 0] = 0
    actual = macro.vec_mat_mul(x, quantization_mode=0, adc_active_bits=None, effective_output_num=counts)
    assert torch.equal(actual, x.where(torch.arange(4, device=device) < counts.unsqueeze(-1), 0))


def test_latency_uses_longest_lane_not_total_output_count(device: torch.device, monkeypatch) -> None:
    macro = _stub_macro(input_num=4, lane_num=2, scan_num=2).to(device)
    monkeypatch.setattr(macro, "_latency_per_scan__ns", lambda **kwargs: 3.0)
    counts = torch.arange(5, device=device)
    actual = macro.latency__ns(adc_active_bits=None, effective_output_num=counts)
    assert torch.equal(actual, torch.tensor([0, 3, 6, 6, 6], device=device))
