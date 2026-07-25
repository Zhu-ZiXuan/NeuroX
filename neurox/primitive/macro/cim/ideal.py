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
        x_value_range: Inclusive single-cycle integer input range.
        w_value_range: Inclusive integer weight range.
        adc_mode_num: Number of supported ADC operating points.
        adc_max_bits: Maximum supported ``adc_bits`` value; ``0`` is the
            lossless-sentinel bit width.
    """

    x_value_range: tuple[int, int]
    w_value_range: tuple[int, int]
    adc_mode_num: int
    adc_max_bits: int

    def validate(self) -> None:
        super().validate()

        # --- Value ranges ---

        if self.x_value_range == (0, 0):
            raise ValueError("require: x_value_range cannot be (0, 0) — collapses rescale math")
        if self.w_value_range == (0, 0):
            raise ValueError("require: w_value_range cannot be (0, 0) — collapses rescale math")
        # --- ADC ---

        if not (self.adc_mode_num >= 1):
            raise ValueError(f"require: adc_mode_num ({self.adc_mode_num}) >= 1")
        if not (self.adc_max_bits >= 0):
            raise ValueError(f"require: adc_max_bits ({self.adc_max_bits}) >= 0")


class IdealCimMacroPolicy(CimMacroPolicy):
    """Empty nonideality policy for the ideal CIM macro."""


@CimMacro.register_neurox_module(config_type=IdealCimMacroConfig, policy_type=IdealCimMacroPolicy)
class IdealCimMacro(CimMacro[IdealCimMacroConfig, IdealCimMacroPolicy]):
    """Ideal tile VMM with per-plane output quantization.

    Args:
        config: Concrete configuration dataclass.
        policy: Empty :class:`IdealCimMacroPolicy` marker.
        input_num: Logical input-vector length.
        output_num: Logical output-vector length.
        inst_shape: Per-instance multiplicity prefix.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    # --- Programmed state ---

    _w: Tensor

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
            raise ValueError(
                f"require: max_active_num ({config.max_active_num}) <= input_num ({input_num})"
            )
        self.input_num = input_num
        self.output_num = output_num
        self._area_per_inst__um2 = 0.0
        self._leakage_per_inst__uW = 0.0

        w_lo, w_hi = config.w_value_range
        max_w_abs = max(abs(w_lo), abs(w_hi))
        x_lo, x_hi = config.x_value_range
        max_x_abs = max(abs(x_lo), abs(x_hi))
        self._max_plane_dot_abs = config.max_active_num * max_w_abs * max_x_abs
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
    def x_value_range(self) -> tuple[int, int]:
        return self.config.x_value_range

    @property
    def w_value_range(self) -> tuple[int, int]:
        return self.config.w_value_range

    @property
    def adc_mode_num(self) -> int:
        return self.config.adc_mode_num

    @property
    def adc_max_bits(self) -> int:
        return self.config.adc_max_bits

    def to_ideal(self) -> IdealCimMacro:
        """Return this already ideal macro."""
        return self

    def adc_rescale_factor(self, *, adc_mode: int, adc_bits: int) -> float:
        del adc_mode
        return self._rescale_by_bits[adc_bits]

    def program(self, w: Tensor) -> None:
        expected_shape = (*self.inst_shape, self.input_num, self.output_num)
        if tuple(w.shape) != expected_shape:
            raise ValueError(f"program() expects w.shape {expected_shape}; got {tuple(w.shape)}")
        self._w = w.detach().clone()

    def vec_mat_mul(self, x: Tensor, *, adc_mode: int, adc_bits: int) -> Tensor:
        """Ideal per-plane VMM with adc_bits-driven output quantization.

        Args:
            x: Logical input tensor with primitive trailing ``[input_num]``;
                at most :attr:`max_active_num` positions may be selected.
            adc_mode: Accepted and ignored.
            adc_bits: ADC resolution [bits]. Zero returns lossless dots.

        Returns:
            Signed ADC-code tensor with the leading order preserved and
            primitive trailing ``[output_num]``. When ``adc_bits > 0`` each
            code is clamped to
            ``[-2^(adc_bits-1), 2^(adc_bits-1) - 1]``; when
            ``adc_bits == 0`` the lossless integer plane dots are
            returned unmodified.
        """
        del adc_mode

        w = self._w.to(torch.int64)

        if self._fp32_exact:
            # Shape: [..., input_num, output_num] x [..., input_num] -> [..., output_num]
            plane_dot = torch.einsum("...io,...i->...o", w.to(torch.float32), x.to(torch.float32)).to(torch.int64)
        else:
            # Shape: [..., input_num] -> [..., input_num, 1]
            x = x.to(torch.int64).unsqueeze(-1)
            full_shape = torch.broadcast_shapes(w.shape, x.shape)
            # Shape: [..., input_num, output_num] -> [..., output_num]
            plane_dot = (w.expand(full_shape) * x.expand(full_shape)).sum(dim=-2)

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
