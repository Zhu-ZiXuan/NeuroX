"""Tile-level ideal crossbar with adc_operation_point-driven output quantization.

See also:
    docs/dev/modules/xbar/ideal.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.analog.adc import AdcOperationPoint
from neurox.common.quant import stochastic_floor_to_int

from .base import Xbar, XbarConfig, XbarPolicy


@dataclass(frozen=True, kw_only=True)
class IdealXbarConfig(XbarConfig):
    """Configuration for :class:`IdealXbar`.

    Attributes:
        x_range: Inclusive single-cycle integer input range.
        w_digit_count: Digits per ``w``.
        w_digit_radix: In-tile positional radix.
        w_digit_range: Inclusive integer range a single digit can carry.
        adc_mode_num: Number of supported ADC operating points.
        adc_max_bits: Maximum supported ``adc_bits`` value.
    """

    x_range: tuple[int, int]
    w_digit_count: int
    w_digit_radix: int
    w_digit_range: tuple[int, int]
    adc_mode_num: int
    adc_max_bits: int


@dataclass(frozen=True)
class IdealXbarPolicy(XbarPolicy):
    """Empty nonideality policy — ideal xbar has no nonidealities to toggle."""


@Xbar.register_key(IdealXbarConfig)
class IdealXbar(Xbar):
    """Tile-level ideal VMM with adc_operation_point-driven output quantization.

    Args:
        config: Concrete configuration dataclass.
        policy: Empty :class:`IdealXbarPolicy` marker.
        name: Hierarchical instance name used by the profiler.
        inst_shape: Per-instance multiplicity prefix; trailing
            ``(col_num, w_digit_count, row_num)`` is derived from config.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature [K].
    """

    config: IdealXbarConfig
    nominal_digits: Tensor
    digits: Tensor
    digit_weights: Tensor

    def __init__(
        self,
        *,
        config: IdealXbarConfig,
        policy: IdealXbarPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            name=name,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )
        if config.w_digit_count <= 0:
            raise ValueError(f"require: w_digit_count ({config.w_digit_count}) > 0")
        if config.w_digit_radix <= 1:
            raise ValueError(f"require: w_digit_radix ({config.w_digit_radix}) > 1")

        # 0-d nominal digit template (zero = unprogrammed weight).
        self.register_buffer(
            "nominal_digits",
            torch.zeros((), dtype=torch.int32),
            persistent=False,
        )
        # Actual digits: starts at nominal; rewritten by ``program(w)``.
        self.register_buffer(
            "digits",
            self.nominal_digits.clone(),
            persistent=False,
        )
        # LSB-first positional weights ``(1, r, r², ..., r^(D-1))``.
        digit_weights = torch.tensor(
            [config.w_digit_radix**k for k in range(config.w_digit_count)],
            dtype=torch.int32,
        )
        self.register_buffer("digit_weights", digit_weights, persistent=False)

        # Quantize-side multiplier — ideal xbar derives this **purely from
        # bit width**, not from the chip's calibrated ``adc_calibration``
        # table. The recovery rescale is the inverse of:
        #
        #   max_dot = row_num · max|w_logical| · max|x|
        #   scale   = (2 ** (bits - 1) - 1) / max_dot
        #
        # so the lossless integer dot product clips at exactly the signed
        # N-bit endpoints. This keeps :class:`IdealXbar` purely a function
        # of (config geometry, requested bits) — chip-agnostic by design.
        max_w_digit_abs = config.w_digit_radix ** config.w_digit_count - 1
        x_lo, x_hi = config.x_range
        max_x_abs = max(abs(x_lo), abs(x_hi))
        self._max_dot_abs: int = config.row_num * max_w_digit_abs * max_x_abs

        self._scale_lut: dict[AdcOperationPoint, float] = {}
        self._rescale_lut = {}  # override base's chip-calibrated table
        for op in (AdcOperationPoint(adc_mode=e.adc_mode, adc_bits=e.adc_bits) for e in config.adc_calibration):
            if op.adc_bits == 0:
                # Full-precision sentinel: lossless int dot = ideal VMM
                # exactly, so the recovery rescale is identity. The
                # ``vec_mat_mul`` path also skips quantization (and
                # therefore the scale lookup) when ``adc_bits == 0``.
                self._rescale_lut[op] = 1.0
                continue
            half_range = (1 << (op.adc_bits - 1)) - 1
            rescale = self._max_dot_abs / half_range  # ideal_dot per ADC code
            self._rescale_lut[op] = rescale
            self._scale_lut[op] = 1.0 / rescale

        self._log_static()

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

    def to_ideal(self) -> IdealXbar:
        """An ideal xbar is its own ideal counterpart."""
        return self

    def program(self, w: Tensor) -> None:
        """Store the xbar-native digit tensor as the tile weight.

        Args:
            w: Integer digit tensor whose shape matches
                :attr:`_w_layout_shape` —
                ``(*inst_shape, col_num, w_digit_count, row_num)``.
        """
        if tuple(w.shape) != self._w_layout_shape:
            raise ValueError(f"program() expects w.shape {self._w_layout_shape}; got {tuple(w.shape)}")
        self.digits = w

    def vec_mat_mul(self, x: Tensor, *, adc_operation_point: AdcOperationPoint) -> Tensor:
        """Ideal VMM with operation-point-driven output quantization.

        ``adc_operation_point.adc_bits == 0`` is a sentinel: skip ADC quantization and
        the signed clamp, returning the lossless integer dot product so
        results match :class:`IdealXbarMacro` exactly.

        Args:
            x: Activation tensor with primitive trailing
                ``[row_num]``.
            adc_operation_point: Runtime ADC operating point.

        Returns:
            Signed ADC-code tensor with primitive trailing ``[col_num]``.
            When ``adc_bits > 0`` the output is clamped to
            ``[-2^(adc_bits-1), 2^(adc_bits-1) - 1]``; when ``adc_bits == 0``
            the lossless integer dot product is returned unmodified.
        """
        # Widen to int64 before any integer arithmetic so per-cell products
        # and the row-num / digit-num reductions cannot overflow.
        digits = self.digits.to(torch.int64)
        # Shape: [..., col_num, w_digit_count, row_num] -> [..., col_num, row_num].
        digit_weights = self.digit_weights.to(torch.int64).view(*([1] * (digits.ndim - 2)), -1, 1)
        w = (digits * digit_weights).sum(dim=-2)

        # Shape: [..., row_num] -> [..., 1, row_num].
        x = x.to(torch.int64).unsqueeze(-2)

        full_shape = torch.broadcast_shapes(w.shape, x.shape)
        w = w.expand(full_shape)
        x = x.expand(full_shape)
        # Shape: [..., col_num]
        dot = (x * w).sum(dim=-1)

        if adc_operation_point.adc_bits == 0:
            return dot

        scale = self._scale_lut[adc_operation_point]
        code = stochastic_floor_to_int(
            dot.to(torch.float32),
            scale,
            out_dtype=torch.int16,
            training=self.training,
        )
        bound = 1 << (adc_operation_point.adc_bits - 1)
        return code.clamp(-bound, bound - 1)
