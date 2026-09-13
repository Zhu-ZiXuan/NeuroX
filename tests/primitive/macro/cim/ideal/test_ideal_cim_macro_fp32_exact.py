"""Ideal macro dots and quantized codes agree with independent integer arithmetic."""

from __future__ import annotations

import pytest
import torch

from neurox.encoding import Encoding
from neurox.primitive.macro.cim import (
    CimMacroQuantizationScheme,
    IdealCimMacro,
    IdealCimMacroConfig,
    IdealCimMacroPolicy,
)


def _make_macro(
    *,
    input_num: int,
    max_active_num: int,
    output_num: int,
    x_value_range: tuple[int, int],
    w_value_range: tuple[int, int] = (-3, 3),
    rescale_factors: tuple[float, ...] = (1.0,),
    adc_bits: int = 6,
) -> IdealCimMacro:
    config = IdealCimMacroConfig(
        input_num=input_num,
        rescale_factors=rescale_factors,
        max_active_num=max_active_num,
        lane_num=1,
        scan_num=output_num,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        w_digit_num=1,
        w_digit_radix=2,
        w_encoding=Encoding.TRUE_FORM if w_value_range[0] < 0 else Encoding.UNSIGNED,
        x_digit_num=1,
        x_digit_radix=2,
        x_encoding=Encoding.COMPLEMENT if x_value_range[0] < 0 else Encoding.UNSIGNED,
        x_value_range=x_value_range,
        w_value_range=w_value_range,
        adc_bits=adc_bits,
        quantization_scheme=CimMacroQuantizationScheme.ZERO_POINT,
    )
    macro = IdealCimMacro(
        config=config,
        policy=IdealCimMacroPolicy(),
        inst_shape=(),
        dtype=torch.float32,
    )
    macro.eval()
    return macro


def _random_operands(
    macro: IdealCimMacro,
    *,
    batch: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate and program random in-range weights and inputs."""
    generator = torch.Generator().manual_seed(seed)
    w_lo, w_hi = macro.w_value_range
    w = torch.randint(
        w_lo,
        w_hi + 1,
        (macro.input_num, macro.output_num),
        dtype=torch.int32,
        generator=generator,
    )
    x_lo, x_hi = macro.x_value_range
    x = torch.randint(x_lo, x_hi + 1, (batch, macro.input_num), dtype=torch.int32, generator=generator)
    macro.program(w.to(device))
    return w, x


def _masked_planes(x: torch.Tensor, *, input_num: int, max_active_num: int) -> torch.Tensor:
    """Pre-masked WL planes via the engine mask formula."""
    p_num = input_num // max_active_num
    mask = torch.arange(input_num) // max_active_num == torch.arange(p_num).unsqueeze(-1)
    # Shape: [..., input] -> [..., P, input]
    return torch.where(mask, x.unsqueeze(-2), x.new_zeros(()))


def _plane_dot_oracle(weight: torch.Tensor, planes: torch.Tensor) -> torch.Tensor:
    """CPU int64 exact plane dots."""
    w = weight.cpu().long()
    return (planes.cpu().long().unsqueeze(-1) * w).sum(dim=-2)


_REPRESENTATIVE = [
    pytest.param((0, 1), (-3, 3), id="spike_x_signed_w"),
    pytest.param((-8, 7), (0, 15), id="signed_x_unsigned_w"),
    pytest.param((0, 255), (-255, 255), id="wide_x_wide_w"),
]


class TestExactDots:
    @pytest.mark.parametrize(
        ("x_value_range", "w_value_range"),
        _REPRESENTATIVE,
    )
    def test_device_matches_int64_oracle(
        self,
        device: torch.device,
        x_value_range: tuple[int, int],
        w_value_range: tuple[int, int],
    ) -> None:
        macro = _make_macro(
            input_num=64,
            max_active_num=16,
            output_num=8,
            x_value_range=x_value_range,
            w_value_range=w_value_range,
        )
        macro.to(device)
        weight, x = _random_operands(macro, batch=5, seed=202, device=device)
        planes = _masked_planes(x, input_num=64, max_active_num=16)
        oracle = _plane_dot_oracle(weight, planes)
        y = macro.vec_mat_mul(planes.to(device), quantization_mode=0, adc_active_bits=None)
        assert y.device.type == device.type
        assert y.dtype == torch.int64
        assert y.shape == (5, 4, 8)
        assert torch.equal(y.cpu(), oracle)


class TestQuantizedDots:
    def _quantized_macro(self) -> IdealCimMacro:
        return _make_macro(
            input_num=64,
            max_active_num=16,
            output_num=8,
            x_value_range=(-8, 7),
            w_value_range=(-15, 15),
            adc_bits=6,
        )

    def test_codes_match_integer_quantization_oracle(self, device: torch.device) -> None:
        macro = self._quantized_macro()
        macro.to(device)
        weight, x = _random_operands(macro, batch=5, seed=303, device=device)
        planes = _masked_planes(x, input_num=64, max_active_num=16)
        quantization_mode = 0
        adc_active_bits = 4
        actual = macro.vec_mat_mul(
            planes.to(device), quantization_mode=quantization_mode, adc_active_bits=adc_active_bits
        )
        dot = _plane_dot_oracle(weight, planes)
        rail = 1 << (macro.adc_bits - 1)
        drop_factor = 1 << (macro.adc_bits - adc_active_bits)
        expected = dot.clamp(-rail, rail - 1).div(drop_factor, rounding_mode="floor")
        assert actual.dtype == torch.int64
        assert actual.device.type == device.type
        assert torch.equal(actual.cpu(), expected)


class TestBeyondFp32Precision:
    """Integer outputs remain exact beyond fp32 precision."""

    def test_preserves_integers_beyond_fp32_precision(self) -> None:
        macro = _make_macro(
            input_num=3,
            max_active_num=3,
            output_num=2,
            x_value_range=(0, 1),
            w_value_range=(0, 2**23),
        )
        # The result lies between adjacent fp32-representable integers.
        w = torch.tensor([[2**23, 0], [2**23, 0], [1, 0]], dtype=torch.int32)
        macro.program(w)
        x = torch.ones(3, dtype=torch.int32)
        y = macro.vec_mat_mul(x, quantization_mode=0, adc_active_bits=None)
        assert y.shape == (2,)
        assert y[0].item() == 2**24 + 1
        assert y[1].item() == 0
