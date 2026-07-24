"""Tile-level ideal crossbar with adc_bits-driven output quantization.

See also:
    docs/reference/primitive/macro/cim/README.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common.quant import stochastic_floor_to_int

from .base import CimMacro, CimMacroConfig, CimMacroPolicy


@dataclass(frozen=True, kw_only=True)
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


@dataclass(frozen=True)
class IdealCimMacroPolicy(CimMacroPolicy):
    """Empty nonideality policy — ideal xbar has no nonidealities to toggle."""


@CimMacro.register_key(IdealCimMacroConfig)
class IdealCimMacro(CimMacro[IdealCimMacroConfig, IdealCimMacroPolicy]):
    """Tile-level ideal VMM with adc_bits-driven per-plane quantization.

    One call quantizes each WL plane's dot product independently into
    one code per column; any accumulation of plane codes happens outside
    the macro, on the caller's leading axes. The rescale at a given
    ``adc_bits`` is derived in :meth:`__init__` from integer geometry
    alone (per conversion, i.e. per plane); no chip calibration enters
    the computation. ``adc_mode`` is opaque and not read at runtime.

    Args:
        config: Concrete configuration dataclass.
        policy: Empty :class:`IdealCimMacroPolicy` marker.
        inst_shape: Per-instance multiplicity prefix; trailing
            ``(col_num, w_digit_count, row_num)`` is derived from config.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    nominal_digits: Tensor
    digits: Tensor
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

        # ``max|w_logical|`` uses the actual per-digit range so signed /
        # offset digit encodings get the right bound.
        d_lo, d_hi = config.w_digit_range
        max_digit_abs = max(abs(d_lo), abs(d_hi))
        max_w_logical_abs = max_digit_abs * int(digit_weights.sum().item())
        x_lo, x_hi = config.x_range
        max_x_abs = max(abs(x_lo), abs(x_hi))
        # One ADC conversion digitizes one WL plane; with at most
        # ``max_active_rows`` live rows per conformant plane, the
        # per-conversion range is active_row_num * max|w_logical| * max|x|.
        self._max_plane_dot_abs: int = config.active_row_num * max_w_logical_abs * max_x_abs
        # fp32-exact fast-path eligibility: CUDA has no integer-matmul
        # kernel. Every per-cell product and every partial sum of the
        # plane dot is bounded by ``_max_plane_dot_abs`` (a conformant
        # plane has at most ``active_row_num`` non-zero rows), so with
        # ``_max_plane_dot_abs < 2^24`` IEEE fp32 einsum accumulation
        # (the framework default; TF32 disabled) reproduces the int64
        # plane dots bit-exactly for range-conformant digits and inputs.
        self._fp32_exact: bool = self._max_plane_dot_abs < 2**24

        # Bit-width-keyed rescale / scale tables. ``adc_bits == 0`` is the
        # lossless sentinel — ``vec_mat_mul`` returns the integer plane
        # dots unmodified and the rescale is identity. Skip
        # ``bits == 1``: the signed 1-bit endpoint ``2^0 - 1 == 0`` makes
        # the rescale formula degenerate; callers asking for it hit a
        # natural ``KeyError`` at lookup time.
        self._rescale_by_bits: dict[int, float] = {0: 1.0}
        self._scale_by_bits: dict[int, float] = {}
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
        del adc_mode  # opaque to the ideal tile; only adc_bits is consumed
        return self._rescale_by_bits[adc_bits]

    def program(self, w: Tensor) -> None:
        if tuple(w.shape) != self._w_layout_shape:
            raise ValueError(f"program() expects w.shape {self._w_layout_shape}; got {tuple(w.shape)}")
        if w.is_floating_point() or w.is_complex():
            raise TypeError(f"program() expects an integer digit tensor; got dtype {w.dtype}")
        self.digits = w.detach().clone().to(self.digit_weights.device)

    def vec_mat_mul(self, x: Tensor, *, adc_mode: int, adc_bits: int) -> Tensor:
        """Ideal per-plane VMM with adc_bits-driven output quantization.

        One call performs one independent ADC conversion per column per
        WL plane. Quantize-then-accumulate is the modeled physical
        semantics (``Q(sum) != sum(Q)`` in general); it is preserved
        because the caller presents each sub-phase as its own plane.
        Only ``adc_bits`` enters the computation; ``adc_mode`` is opaque
        to the ideal tile and not read.
        ``adc_bits == 0`` is a sentinel: skip ADC quantization and the
        signed clamp, returning the lossless integer plane dots.

        Args:
            x: WL plane tensor with primitive trailing ``[row_num]``;
                rows outside the caller's active window (at most
                :attr:`max_active_rows` live rows per plane) must arrive
                zeroed (WL off). Leading axes are anonymous broadcast
                batch.
            adc_mode: Opaque to the ideal tile; not read.
            adc_bits: ADC resolution [bits]; the sole operating-point
                input consumed.

        Returns:
            Signed ADC-code tensor with the leading order preserved and
            primitive trailing ``[col_num]``. When ``adc_bits > 0`` each
            code is clamped to
            ``[-2^(adc_bits-1), 2^(adc_bits-1) - 1]``; when
            ``adc_bits == 0`` the lossless integer plane dots are
            returned unmodified.
        """
        del adc_mode  # opaque to the ideal tile; only adc_bits is consumed
        # Widen to int64 before any integer arithmetic so per-cell products
        # and the row-num / digit-num reductions cannot overflow. The digit
        # reduction is elementwise (CUDA-safe at int64); only the row dot
        # below needs the fp32-exact fast path.
        digits = self.digits.to(torch.int64)
        # Shape: [..., col_num, w_digit_count, row_num] -> [..., col_num, row_num]
        digit_weights = self.digit_weights.to(torch.int64).view(*([1] * (digits.ndim - 2)), -1, 1)
        w = (digits * digit_weights).sum(dim=-2)

        if self._fp32_exact:
            # fp32-exact fast path (bound checked in ``__init__``):
            # full-row contraction; zeroed rows contribute nothing, so
            # the dot equals the per-conversion partial dot. The cast
            # back to int64 is lossless.
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
