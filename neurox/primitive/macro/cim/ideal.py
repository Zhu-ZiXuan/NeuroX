"""Ideal CIM macro with calibrated output quantization.

See Also:
    docs/reference/primitive/macro/cim/ideal.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.common import stochastic_round

from .base import (
    CimMacro,
    CimMacroConfig,
    CimMacroPolicy,
    CimMacroQuantizationScheme,
)


class IdealCimMacroConfig(CimMacroConfig):
    x_value_range: tuple[int, int]
    """Inclusive single-cycle integer input range; `(0, 0)` is rejected."""
    w_value_range: tuple[int, int]
    """Inclusive integer weight range; `(0, 0)` is rejected."""
    adc_bits: int
    """Maximum selectable virtual ADC resolution."""
    quantization_scheme: CimMacroQuantizationScheme
    """Macro output quantization scheme."""

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
    """Ideal macro VMM with per-mode output quantization."""

    _w: Tensor  # Shape: [*inst_shape, input_num, output_num]

    def __init__(
        self,
        *,
        config: IdealCimMacroConfig,
        policy: IdealCimMacroPolicy,
        input_num: int,
        output_num: int,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            input_num=input_num,
            output_num=output_num,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )
        if config.max_active_num > input_num:
            raise ValueError(f"require: max_active_num ({config.max_active_num}) <= input_num ({input_num})")
        self.input_num = input_num
        self.output_num = output_num

        w_lo, w_hi = config.w_value_range
        max_w_abs = max(abs(w_lo), abs(w_hi))
        x_lo, x_hi = config.x_value_range
        max_x_abs = max(abs(x_lo), abs(x_hi))
        self._max_plane_dot_abs = config.max_active_num * max_w_abs * max_x_abs
        # Integers below 2^24 are exactly representable by IEEE fp32.
        self._fp32_exact = self._max_plane_dot_abs < 2**24

    def initiation_interval__ns(self, *, adc_active_bits: int) -> float:
        """Zero — an arithmetic oracle occupies no execution interval."""
        if adc_active_bits != 0:
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

    @property
    def _quantization_scheme(self) -> CimMacroQuantizationScheme:
        return self.config.quantization_scheme

    def to_ideal(self) -> IdealCimMacro:
        """Return this already ideal macro."""
        return self

    def rescale_factor(self, *, quantization_mode: int, adc_active_bits: int) -> float:
        if adc_active_bits == 0:
            self._check_quantization_mode(quantization_mode)
            return 1.0
        return super().rescale_factor(
            quantization_mode=quantization_mode,
            adc_active_bits=adc_active_bits,
        )

    def restore_adc_layout(self, value: Tensor) -> Tensor:
        """Return the already-logical ideal output layout."""
        return value

    def program(self, w: Tensor) -> None:
        expected_shape = (*self.inst_shape, self.input_num, self.output_num)
        if tuple(w.shape) != expected_shape:
            raise ValueError(f"program() expects w.shape {expected_shape}; got {tuple(w.shape)}")
        self._w = w.detach().clone()

    def vec_mat_mul(self, x: Tensor, *, quantization_mode: int, adc_active_bits: int) -> Tensor:
        """Ideal per-plane VMM followed by one calibrated conversion.

        Args:
            x: Logical input tensor; at most `max_active_num` positions may
                be selected.
                Shape: `[..., input_num]`.
            quantization_mode: Index selecting the full-resolution rescale factor.
            adc_active_bits: Active ADC resolution in `[1, adc_bits]`, or `0`
                to bypass the virtual ADC and return the exact integer result.

        Returns:
            Integer code tensor whose leading axes broadcast the input's against
            `inst_shape`. At `adc_active_bits = 0`, returns the exact `int64`
            plane dots. Otherwise, zero-point mapping returns a centered signed
            code in `[-2^(b - 1), 2^(b - 1) - 1]`, while sign-magnitude mapping
            returns a signed magnitude in `[-(2^b - 1), 2^b - 1]` for `b = adc_active_bits`.
            Shape: `[..., output_num]`.

        Raises:
            ValueError: The mode index is outside the declared modes, or the
                resolution is outside `[0, adc_bits]`.
        """
        self._check_quantization_mode(quantization_mode)
        if adc_active_bits != 0:
            self._check_adc_active_bits(adc_active_bits)
        factor = self.config.rescale_factors[quantization_mode]
        w = self._w.to(torch.int64)

        if self._fp32_exact:
            # Shape: [..., input_num] -> [..., output_num]
            plane_dot = torch.einsum("...io,...i->...o", w.to(torch.float32), x.to(torch.float32)).to(torch.int64)
        else:
            # Shape: [..., input_num] -> [..., input_num, 1]
            x = x.to(torch.int64).unsqueeze(-1)
            # Shape: [..., input_num, output_num] -> [..., output_num]
            plane_dot = (w * x).sum(dim=-2)

        if adc_active_bits == 0:
            return plane_dot

        if self._quantization_scheme is CimMacroQuantizationScheme.SIGN_MAGNITUDE:
            return self._convert_sign_magnitude(plane_dot, factor=factor, adc_active_bits=adc_active_bits)
        return self._convert_zero_point(plane_dot, factor=factor, adc_active_bits=adc_active_bits)

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
