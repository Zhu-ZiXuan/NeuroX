"""Tile-level ideal crossbar with window-driven output quantization.

See Also:
    docs/reference/primitive/macro/cim/ideal.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import (
    CimMacro,
    CimMacroConfig,
    CimMacroPolicy,
    map_zero_point_input_code,
    validate_quantization_input_range,
)


class IdealCimMacroConfig(CimMacroConfig):
    """Configuration for `IdealCimMacro`."""

    x_value_range: tuple[int, int]
    """Inclusive single-cycle integer input range; `(0, 0)` is rejected."""
    w_value_range: tuple[int, int]
    """Inclusive integer weight range; `(0, 0)` is rejected."""
    quantization_input_ranges: tuple[tuple[int, int], ...]
    """One canonical inclusive conversion window `(lower, upper)` per
    quantization mode, in MAC units. The tuple position is the
    `quantization_mode` index and the tuple length is the mode count."""
    adc_max_bits: int
    """Largest supported `adc_bits` value. The lossless oracle `adc_bits is
    None` lies outside this bound and is always accepted at runtime."""

    def validate(self) -> None:
        super().validate()

        # --- Value ranges ---

        if self.x_value_range == (0, 0):
            raise ValueError("require: x_value_range cannot be (0, 0) — a zero-only tile carries no signal")
        if self.w_value_range == (0, 0):
            raise ValueError("require: w_value_range cannot be (0, 0) — a zero-only tile carries no signal")

        # --- Quantization ---

        if not (self.adc_max_bits >= 1):
            raise ValueError(f"require: adc_max_bits ({self.adc_max_bits}) >= 1")
        if not self.quantization_input_ranges:
            raise ValueError("require: quantization_input_ranges must declare at least one window")
        for window in self.quantization_input_ranges:
            validate_quantization_input_range(window)


class IdealCimMacroPolicy(CimMacroPolicy):
    """Empty nonideality policy for the ideal CIM macro."""


@CimMacro.register_neurox_module(config_type=IdealCimMacroConfig, policy_type=IdealCimMacroPolicy)
class IdealCimMacro(CimMacro[IdealCimMacroConfig, IdealCimMacroPolicy]):
    """Ideal tile VMM with per-plane, window-driven output quantization.

    One conversion reads the exact integer plane dot through the canonical
    window `[lower, upper]` that `quantization_mode` selects, as the unsigned
    reading `code_u = clamp(floor((dot - lower) · 2^b / W), 0, 2^b - 1)` with
    `W = upper - lower + 1` and `b = adc_bits`, less the window zero code —
    `0` unsigned, `2^(b-1)` mid-zero, always computed, never configured. A
    window with `W > 2^b` is a legal lossy operating point, and the whole
    conversion evaluates in integers without ever forming the fractional step.
    """

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

    @property
    def _area_per_inst__um2(self) -> float:
        return 0.0

    @property
    def _leakage_per_inst__uW(self) -> float:
        return 0.0

    def latency__ns(self, *, adc_bits: int | None) -> float:
        """Zero — an arithmetic oracle has no circuit to take time."""
        del adc_bits
        return 0.0

    @property
    def x_value_range(self) -> tuple[int, int]:
        return self.config.x_value_range

    @property
    def w_value_range(self) -> tuple[int, int]:
        return self.config.w_value_range

    @property
    def quantization_input_ranges(self) -> tuple[tuple[int, int], ...]:
        return self.config.quantization_input_ranges

    @property
    def adc_max_bits(self) -> int:
        return self.config.adc_max_bits

    def to_ideal(self) -> IdealCimMacro:
        """Return this already ideal macro."""
        return self

    def _max_bits_rescale_factor(self, quantization_mode: int) -> float:
        """Return `1.0`: the ideal macro's codes are the rescale reference.

        Args:
            quantization_mode: Window index in
                `[0, len(quantization_input_ranges))`.

        Raises:
            ValueError: The mode index is outside the declared modes.
        """
        self._window(quantization_mode)
        return 1.0

    def map_quantization_input_code(self, code: Tensor, *, quantization_mode: int) -> tuple[Tensor, tuple[int, int]]:
        """Map exact plane dots onto the window's zero-point input grid.

        Args:
            code: Exact integer plane dots.
            quantization_mode: Window index in
                `[0, len(quantization_input_ranges))`.

        Returns:
            The offset codes and their inclusive range `(0, W - 1)`.
        """
        return map_zero_point_input_code(code, code_range=self._window(quantization_mode))

    def program(self, w: Tensor) -> None:
        expected_shape = (*self.inst_shape, self.input_num, self.output_num)
        if tuple(w.shape) != expected_shape:
            raise ValueError(f"program() expects w.shape {expected_shape}; got {tuple(w.shape)}")
        self._w = w.detach().clone()

    def vec_mat_mul(self, x: Tensor, *, quantization_mode: int, adc_bits: int | None) -> Tensor:
        """Ideal per-plane VMM followed by one windowed conversion.

        Args:
            x: Logical input tensor; at most `max_active_num` positions may
                be selected.
                Shape: `[..., input_num]`.
            quantization_mode: Window index in
                `[0, len(quantization_input_ranges))`.
            adc_bits: Conversion resolution [bits] in `[1, adc_max_bits]`,
                or `None` for the lossless oracle.

        Returns:
            Signed `int64` code tensor whose leading axes broadcast the
            input's against `inst_shape`. The lossless oracle returns the
            exact integer plane dots unmodified. Otherwise a mid-zero window
            yields codes in `[-2^(b-1), 2^(b-1) - 1]` and an unsigned window
            codes in `[0, 2^b - 1]`: dots below the window clip to the bottom
            code, dots above it to the top one.
            Shape: `[..., output_num]`.

        Raises:
            ValueError: The mode index is outside the declared modes, or the
                resolution is neither `None` nor in `[1, adc_max_bits]`.
        """
        lower, upper = self._window(quantization_mode)

        w = self._w.to(torch.int64)

        if self._fp32_exact:
            # Shape: [..., input_num] @ [..., input_num, output_num] -> [..., output_num]
            plane_dot = torch.einsum("...io,...i->...o", w.to(torch.float32), x.to(torch.float32)).to(torch.int64)
        else:
            # Shape: [..., input_num] -> [..., input_num, 1]
            x = x.to(torch.int64).unsqueeze(-1)
            full_shape = torch.broadcast_shapes(w.shape, x.shape)
            # Shape: [..., input_num, output_num] -> [..., output_num]
            plane_dot = (w.expand(full_shape) * x.expand(full_shape)).sum(dim=-2)

        if adc_bits is None:
            return plane_dot

        self._check_bits(adc_bits)
        return self._convert(plane_dot, lower=lower, upper=upper, adc_bits=adc_bits)

    def _window(self, quantization_mode: int) -> tuple[int, int]:
        """Return the inclusive conversion window of one quantization mode."""
        ranges = self.config.quantization_input_ranges
        if not (0 <= quantization_mode < len(ranges)):
            raise ValueError(f"require: quantization_mode ({quantization_mode}) in [0, {len(ranges)})")
        return ranges[quantization_mode]

    def _check_bits(self, adc_bits: int) -> None:
        """Require a converting bit width; `None` is the lossless oracle."""
        if not (1 <= adc_bits <= self.config.adc_max_bits):
            raise ValueError(
                f"require: adc_bits ({adc_bits}) in [1, adc_max_bits ({self.config.adc_max_bits})] "
                f"or None for the lossless oracle"
            )

    def _convert(self, plane_dot: Tensor, *, lower: int, upper: int, adc_bits: int) -> Tensor:
        """Convert exact plane dots into signed codes of one window.

        In training mode a uniform integer jitter in `[0, W - 1]` is added
        before the floor, firing its remainder as a Bernoulli trial: the
        expected code equals the unrounded ratio wherever the window does not
        clip, while on-grid dots keep their deterministic code. Evaluation
        mode takes the bare floor, which rounds toward negative infinity.

        Args:
            plane_dot: Exact integer plane dots.
                Shape: `[..., output_num]`.
            lower: Window minimum, inclusive, in MAC units.
            upper: Window maximum, inclusive, in MAC units.
            adc_bits: ADC resolution [bits], at least `1`.

        Returns:
            Signed `int64` code tensor, one code per plane dot.
            Shape: `[..., output_num]`.
        """
        level_num = 1 << adc_bits
        width = upper - lower + 1
        numerator = (plane_dot - lower) * level_num
        if self.training:
            numerator = numerator + torch.randint(
                low=0,
                high=width,
                size=numerator.shape,
                dtype=numerator.dtype,
                device=numerator.device,
            )
        code_u = torch.div(numerator, width, rounding_mode="floor").clamp_(0, level_num - 1)
        zero_code = 0 if lower == 0 else level_num >> 1
        return code_u - zero_code
