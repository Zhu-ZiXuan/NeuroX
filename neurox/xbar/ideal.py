"""Tile-level lossless ideal crossbar.

See also:
    docs/dev/modules/xbar/ideal.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common.quant import stochastic_floor_to_int

from .base import Xbar, XbarConfig


@dataclass(frozen=True, kw_only=True)
class IdealXbarConfig(XbarConfig):
    """Configuration for :class:`IdealXbar`.

    Carries the structural fields that ``XbarConfig`` does not — the
    same surface every physical xbar publishes through its abstract
    properties. ``IdealXbar`` is the only registered ``Xbar`` subclass
    that consumes these fields directly from a config.

    Attributes:
        x_range: Inclusive single-cycle integer input range.
        w_digit_count: Digits per ``w``.
        w_digit_radix: In-tile positional radix.
        w_digit_range: Inclusive integer range a single digit can carry.
    """

    x_range: tuple[int, int]
    w_digit_count: int
    w_digit_radix: int
    w_digit_range: tuple[int, int]


@Xbar.register_key(IdealXbarConfig)
class IdealXbar(Xbar):
    """Tile-level ideal VMM with output quantization.

    Args:
        cfg: Ideal-xbar configuration carrying both the base
            :class:`XbarConfig` fields and the four structural fields.
        name: Hierarchical profiler name.
        T__K: Operating temperature [K].
        dtype: Tensor dtype for internal buffers.
    """

    cfg: IdealXbarConfig
    digits: Tensor
    digit_weights: Tensor

    def __init__(
        self,
        *,
        cfg: IdealXbarConfig,
        name: str,
        T__K: float,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(cfg=cfg, name=name, T__K=T__K, dtype=dtype)
        if cfg.w_digit_count <= 0:
            raise ValueError(f"require: w_digit_count ({cfg.w_digit_count}) > 0")
        if cfg.w_digit_radix <= 1:
            raise ValueError(f"require: w_digit_radix ({cfg.w_digit_radix}) > 1")

        # ``digits`` placeholder; ``fabricate`` overwrites.
        self.register_buffer("digits", torch.empty(0, dtype=torch.int32), persistent=False)
        # LSB-first digit weights ``(1, r, r², ..., r^(D-1))``.
        digit_weights = torch.tensor(
            [cfg.w_digit_radix**k for k in range(cfg.w_digit_count)],
            dtype=torch.int32,
        )
        self.register_buffer("digit_weights", digit_weights, persistent=False)

    @property
    def x_range(self) -> tuple[int, int]:
        return self.cfg.x_range

    @property
    def w_digit_count(self) -> int:
        return self.cfg.w_digit_count

    @property
    def w_digit_radix(self) -> int:
        return self.cfg.w_digit_radix

    @property
    def w_digit_range(self) -> tuple[int, int]:
        return self.cfg.w_digit_range

    def to_ideal(self) -> IdealXbar:
        """An ideal xbar is its own ideal counterpart."""
        return self

    def fabricate(self, w: Tensor) -> None:
        """Register the xbar-native digit tensor as the tile weight.

        Args:
            w: Integer digit tensor with primitive trailing
                ``[data_num, digit_num, row_num]``.
        """
        self._record_xbar_inst_count(w)
        self.register_buffer("digits", w, persistent=False)

    def vec_mat_mul(self, x: Tensor) -> Tensor:
        """Ideal VMM with output quantisation.

        Args:
            x: Activation tensor with primitive trailing
                ``[row_num]``.

        Returns:
            ADC-code tensor with primitive trailing ``[data_num]``.
        """
        digits = self.digits
        # Shape: [..., data_num, digit_num, row_num] -> [..., data_num, row_num].
        digit_weights = self.digit_weights.to(digits.dtype).view(*([1] * (digits.ndim - 2)), -1, 1)
        w_logic = (digits * digit_weights).sum(dim=-2)

        # Shape: [..., row_num] -> [..., 1, row_num].
        x = x.unsqueeze(-2)

        full_shape = torch.broadcast_shapes(w_logic.shape, x.shape)
        w_logic = w_logic.expand(full_shape)
        x = x.expand(full_shape)
        # Shape: [..., data_num]
        dot = (x * w_logic).sum(dim=-1)

        rf = self.output_rescale_factor
        code = stochastic_floor_to_int(
            dot.to(torch.float32),
            rf,
            out_dtype=torch.int16,
            training=self.training,
        )
        return code
