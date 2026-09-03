"""Ideal CIM macro with optional calibrated output quantization.

See Also:
    docs/reference/primitive/macro/cim/ideal.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.common import stochastic_round
from neurox.common.encoding import Encoding

from .base import (
    CimMacro,
    CimMacroConfig,
    CimMacroPolicy,
    CimMacroQuantizationScheme,
)


class IdealCimMacroConfig(CimMacroConfig):
    w_digit_num: int
    """Weight digits in the physical twin."""
    w_digit_radix: int
    """Weight-digit radix in the physical twin."""
    w_encoding: Encoding
    """Weight encoding in the physical twin."""
    x_digit_num: int
    """Input digits in the physical twin."""
    x_digit_radix: int
    """Input-digit radix in the physical twin."""
    x_encoding: Encoding
    """Input encoding in the physical twin."""
    x_value_range: tuple[int, int]
    """Inclusive single-cycle integer input range; `(0, 0)` is rejected."""
    w_value_range: tuple[int, int]
    """Inclusive integer weight range; `(0, 0)` is rejected."""
    adc_bits: int
    """Maximum selectable virtual ADC resolution."""
    quantization_scheme: CimMacroQuantizationScheme
    """Macro output quantization scheme."""

    @property
    def w_digit_n(self) -> int:
        return self.w_digit_num

    @property
    def w_digit_r(self) -> int:
        return self.w_digit_radix

    @property
    def w_enc(self) -> Encoding:
        return self.w_encoding

    @property
    def x_digit_n(self) -> int:
        return self.x_digit_num

    @property
    def x_digit_r(self) -> int:
        return self.x_digit_radix

    @property
    def x_enc(self) -> Encoding:
        return self.x_encoding

    @property
    def quant_scheme(self) -> CimMacroQuantizationScheme:
        return self.quantization_scheme

    def validate(self) -> None:
        super().validate()

        # --- Value ranges ---

        if self.x_value_range == (0, 0):
            raise ValueError("require: x_value_range cannot be (0, 0) — a zero-only macro carries no signal")
        if self.w_value_range == (0, 0):
            raise ValueError("require: w_value_range cannot be (0, 0) — a zero-only macro carries no signal")

        # --- Quantization ---

        self._require_ge(self.adc_bits, "adc_bits", 1)


class IdealCimMacroPolicy(CimMacroPolicy):
    pass


@CimMacro.register_neurox_module(config_type=IdealCimMacroConfig, policy_type=IdealCimMacroPolicy)
class IdealCimMacro(CimMacro[IdealCimMacroConfig, IdealCimMacroPolicy]):
    """Ideal macro VMM whose highest precision is the exact integer result."""

    _w: Tensor  # Shape: [*inst_shape, input, output]

    def __init__(
        self,
        *,
        config: IdealCimMacroConfig,
        policy: IdealCimMacroPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )
        w_lo, w_hi = self.w_value_range
        max_w_abs = max(abs(w_lo), abs(w_hi))
        x_lo, x_hi = self.x_value_range
        max_x_abs = max(abs(x_lo), abs(x_hi))
        self._max_plane_dot_abs = config.max_active_num * max_w_abs * max_x_abs
        # Integers below 2^24 are exactly representable by IEEE fp32.
        self._fp32_exact = self._max_plane_dot_abs < 2**24

    def latency__ns(self, *, adc_active_bits: int | None) -> float:
        """Zero — an arithmetic oracle has no circuit latency."""
        self._check_adc_active_bits(adc_active_bits)
        return 0.0

    @property
    def x_value_range(self) -> tuple[int, int]:
        return self.config.x_value_range

    @property
    def w_value_range(self) -> tuple[int, int]:
        return self.config.w_value_range

    @property
    def adc_bits(self) -> int:
        return self.config.adc_bits

    def to_ideal(self) -> IdealCimMacro:
        """Return this already ideal macro."""
        return self

    def rescale_factor(
        self,
        *,
        quantization_mode: int,
        adc_active_bits: int | None,
    ) -> float:
        if adc_active_bits is None:
            self._check_quantization_mode(quantization_mode)
            return 1.0
        return super().rescale_factor(
            quantization_mode=quantization_mode,
            adc_active_bits=adc_active_bits,
        )

    def program(self, w: Tensor) -> None:
        expected_shape = (*self.inst_shape, self.input_num, self.output_num)
        if tuple(w.shape) != expected_shape:
            raise ValueError(f"program() expects w.shape {expected_shape}; got {tuple(w.shape)}")
        self._w = w.detach().clone()

    def _vec_mat_mul_impl(
        self,
        x: Tensor,
        *,
        quantization_mode: int,
        adc_active_bits: int | None,
    ) -> Tensor:
        factor = self.config.rescale_factors[quantization_mode]
        w = self._w.to(torch.int64)

        if self._fp32_exact:
            # Shape: [..., input] -> [..., output]
            plane_dot = torch.einsum("...io,...i->...o", w.to(torch.float32), x.to(torch.float32)).to(torch.int64)
        else:
            # Shape: [..., input] -> [..., input, output=1]
            x = x.to(torch.int64).unsqueeze(-1)
            # Shape: [..., input, output] -> [..., output]
            plane_dot = (w * x).sum(dim=-2)

        if adc_active_bits is None:
            code = plane_dot
        elif self.config.quant_scheme is CimMacroQuantizationScheme.SIGN_MAGNITUDE:
            code = self._convert_sign_magnitude(plane_dot, factor=factor, adc_active_bits=adc_active_bits)
        else:
            code = self._convert_zero_point(plane_dot, factor=factor, adc_active_bits=adc_active_bits)
        return code.reshape(*code.shape[:-1], self.lane_num, self.scan_num)

    def _quantize(self, value: Tensor, factor: float, min_code: int, max_code: int, drop_bits: int) -> Tensor:
        code = stochastic_round(value.to(torch.float32) / factor, enabled=self.training)
        return code.to(torch.int32).clamp(min_code, max_code) >> drop_bits

    def _convert_zero_point(
        self,
        plane_dot: Tensor,
        *,
        factor: float,
        adc_active_bits: int,
    ) -> Tensor:
        """Quantize to a centered two's-complement code and truncate low bits."""
        zero_point = 1 << (self.adc_bits - 1)
        min_code = -zero_point
        max_code = zero_point - 1
        drop_bits = self.adc_bits - adc_active_bits
        return self._quantize(plane_dot, factor, min_code, max_code, drop_bits)

    def _convert_sign_magnitude(
        self,
        plane_dot: Tensor,
        *,
        factor: float,
        adc_active_bits: int,
    ) -> Tensor:
        """Quantize the magnitude, truncate it, then restore the sign."""
        min_code = 0
        max_code = (1 << self.adc_bits) - 1
        drop_bits = self.adc_bits - adc_active_bits
        return plane_dot.sign() * self._quantize(plane_dot.abs(), factor, min_code, max_code, drop_bits)
