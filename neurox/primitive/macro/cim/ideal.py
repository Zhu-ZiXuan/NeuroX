"""Tile-level ideal crossbar with adc_bits-driven output quantization.

See also:
    docs/internals/primitive/macro/cim/ideal.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.common.quant import stochastic_floor_to_int

from .base import CimMacro, CimMacroConfig, CimMacroPolicy


class IdealCimMacroConfig(CimMacroConfig):
    """Configuration for :class:`IdealCimMacro`.

    Attributes:
        x_range: Inclusive single-cycle integer input range.
        w_digit_count: Digits per ``w``.
        w_digit_radix: In-tile positional radix.
        w_digit_range: Inclusive integer range a single digit can carry.
        adc_mode_num: Number of supported ADC operating points.
        adc_max_bits: Maximum supported ``adc_bits`` value; ``0`` is the
            lossless-sentinel bit width.
    """

    x_range: tuple[int, int]
    w_digit_count: int
    w_digit_radix: int
    w_digit_range: tuple[int, int]
    adc_mode_num: int
    adc_max_bits: int

    def validate(self) -> None:
        super().validate()
        self.validate_value_grid()
        if not (self.adc_mode_num >= 1):
            raise ValueError(f"require: adc_mode_num ({self.adc_mode_num}) >= 1")
        if not (self.adc_max_bits >= 0):
            raise ValueError(f"require: adc_max_bits ({self.adc_max_bits}) >= 0")


class IdealCimMacroPolicy(CimMacroPolicy):
    """Empty nonideality policy — ideal xbar has no nonidealities to toggle."""


@CimMacro.register_key(IdealCimMacroConfig)
class IdealCimMacro(CimMacro[IdealCimMacroConfig, IdealCimMacroPolicy]):
    """Ideal tile VMM with per-plane output quantization.

    Args:
        config: Concrete configuration dataclass.
        policy: Empty :class:`IdealCimMacroPolicy` marker.
        inst_shape: Per-instance multiplicity prefix; trailing
            ``(col_num, w_digit_count, row_num)`` is derived from config.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    # --- Immutable model buffers ---

    digit_weights: Tensor

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
        self._area_per_inst__um2 = 0.0
        self._leakage_per_inst__uW = 0.0
        if config.w_digit_count <= 0:
            raise ValueError(f"require: w_digit_count ({config.w_digit_count}) > 0")
        if config.w_digit_radix <= 1:
            raise ValueError(f"require: w_digit_radix ({config.w_digit_radix}) > 1")

        digit_weights = torch.tensor(
            [config.w_digit_radix**k for k in range(config.w_digit_count)],
            dtype=torch.int32,
        )
        self.register_buffer("digit_weights", digit_weights, persistent=False)

        d_lo, d_hi = config.w_digit_range
        max_digit_abs = max(abs(d_lo), abs(d_hi))
        max_w_logical_abs = max_digit_abs * int(digit_weights.sum().item())
        x_lo, x_hi = config.x_range
        max_x_abs = max(abs(x_lo), abs(x_hi))
        # One conversion covers at most ``max_active_rows`` nonzero rows.
        self._max_plane_dot_abs = config.active_row_num * max_w_logical_abs * max_x_abs
        # Integers below 2^24 are exactly representable by IEEE fp32.
        self._fp32_exact = self._max_plane_dot_abs < 2**24

        # A zero bit width selects lossless output; one bit has no signed range.
        self._rescale_by_bits = {0: 1.0}
        self._scale_by_bits = {}
        for bits in range(2, config.adc_max_bits + 1):
            half_range = (1 << (bits - 1)) - 1
            rescale = self._max_plane_dot_abs / half_range
            self._rescale_by_bits[bits] = rescale
            self._scale_by_bits[bits] = 1.0 / rescale

    @property
    def x_range(self) -> tuple[int, int]:
        return self.config.x_range

    @property
    def w_digit_count(self) -> int:
        return self.config.w_digit_count

    @property
    def w_digit_radix(self) -> int:
        return self.config.w_digit_radix

    @property
    def w_digit_range(self) -> tuple[int, int]:
        return self.config.w_digit_range

    @property
    def adc_mode_num(self) -> int:
        return self.config.adc_mode_num

    @property
    def adc_max_bits(self) -> int:
        return self.config.adc_max_bits

    def to_ideal(self) -> IdealCimMacro:
        """An ideal xbar is its own ideal counterpart."""
        return self

    def adc_rescale_factor(self, *, adc_mode: int, adc_bits: int) -> float:
        del adc_mode
        return self._rescale_by_bits[adc_bits]

    def program(self, w: Tensor) -> None:
        if tuple(w.shape) != self._w_layout_shape:
            raise ValueError(f"program() expects w.shape {self._w_layout_shape}; got {tuple(w.shape)}")
        if w.is_floating_point() or w.is_complex():
            raise TypeError(f"program() expects an integer digit tensor; got dtype {w.dtype}")
        self.digits = w.detach().clone()

    def vec_mat_mul(self, x: Tensor, *, adc_mode: int, adc_bits: int) -> Tensor:
        """Ideal per-plane VMM with adc_bits-driven output quantization.

        Args:
            x: WL plane tensor with primitive trailing ``[row_num]``;
                at most :attr:`max_active_rows` rows may be nonzero.
            adc_mode: Accepted and ignored.
            adc_bits: ADC resolution [bits]. Zero returns lossless dots.

        Returns:
            Signed ADC-code tensor with the leading order preserved and
            primitive trailing ``[col_num]``. When ``adc_bits > 0`` each
            code is clamped to
            ``[-2^(adc_bits-1), 2^(adc_bits-1) - 1]``; when
            ``adc_bits == 0`` the lossless integer plane dots are
            returned unmodified.
        """
        del adc_mode

        digits = self.digits.to(torch.int64)

        # Shape: [w_digit_count] -> [1, ..., w_digit_count, 1]
        digit_weights = self.digit_weights.to(torch.int64).view(*([1] * (digits.ndim - 2)), -1, 1)

        # Shape: [..., col_num, w_digit_count, row_num] -> [..., col_num, row_num]
        w = (digits * digit_weights).sum(dim=-2)

        if self._fp32_exact:
            # Shape: [..., col_num, row_num] x [..., row_num] -> [..., col_num]
            plane_dot = torch.einsum("...cr,...r->...c", w.to(torch.float32), x.to(torch.float32)).to(torch.int64)
        else:
            # Shape: [..., row_num] -> [..., 1, row_num]
            x = x.to(torch.int64).unsqueeze(-2)
            full_shape = torch.broadcast_shapes(w.shape, x.shape)
            # Shape: [..., col_num, row_num] -> [..., col_num]
            plane_dot = (w.expand(full_shape) * x.expand(full_shape)).sum(dim=-1)

        if adc_bits == 0:
            return plane_dot

        scale = self._scale_by_bits[adc_bits]
        code = stochastic_floor_to_int(
            plane_dot.to(torch.float32),
            scale,
            out_dtype=torch.int16,
            training=self.training,
        )
        bound = 1 << (adc_bits - 1)
        code = code.clamp(-bound, bound - 1)
        return code
