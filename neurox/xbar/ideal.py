"""Tile-level lossless ideal crossbar.

See also:
    docs/dev/modules/xbar/ideal.md
"""

import torch
from torch import Tensor

from neurox.common.quant import stochastic_floor_to_int

from .base import Xbar, XbarConfig


class IdealXbar(Xbar):
    """Tile-level ideal VMM with output quantization.

    Args:
        cfg: Base :class:`XbarConfig`.
        name: Hierarchical profiler name.
        x_range: Inclusive single-cycle integer input range.
        w_digit_count: Number of digits per ``w``.
        w_digit_radix: Positional base ``r`` of the in-tile digit
            combination.
        w_digit_range: Inclusive integer range a single digit can
            carry.
        stochastic: Optional override for stochastic-floor rounding;
            ``None`` defers to ``self.training``.
    """

    weight: Tensor
    digit_weights: Tensor

    def __init__(
        self,
        cfg: XbarConfig,
        *,
        name: str = "",
        x_range: tuple[int, int],
        w_digit_count: int,
        w_digit_radix: int,
        w_digit_range: tuple[int, int],
        stochastic: bool | None = None,
    ) -> None:
        super().__init__(cfg, name=name)
        if w_digit_count <= 0:
            raise ValueError(f"require: w_digit_count ({w_digit_count}) > 0")
        if w_digit_radix <= 1:
            raise ValueError(f"require: w_digit_radix ({w_digit_radix}) > 1")
        self._x_range = x_range
        self._w_digit_count = w_digit_count
        self._w_digit_radix = w_digit_radix
        self._w_digit_range = w_digit_range
        self._stochastic_override = stochastic

        # ``weight`` placeholder; ``fabricate`` overwrites.
        self.register_buffer("weight", torch.empty(0, dtype=torch.int32), persistent=False)
        # LSB-first digit weights ``(1, r, r², ..., r^(D-1))``.
        digit_weights = torch.tensor(
            [w_digit_radix**k for k in range(w_digit_count)],
            dtype=torch.int32,
        )
        self.register_buffer("digit_weights", digit_weights, persistent=False)

    @property
    def x_range(self) -> tuple[int, int]:
        return self._x_range

    @property
    def w_digit_count(self) -> int:
        return self._w_digit_count

    @property
    def w_digit_radix(self) -> int:
        return self._w_digit_radix

    @property
    def w_digit_range(self) -> tuple[int, int]:
        return self._w_digit_range

    def fabricate(self, w: Tensor) -> None:
        """Register the xbar-native digit tensor as the tile weight.

        Args:
            w: Integer digit tensor with primitive trailing
                ``[data_num, digit_num, row_num]``.
        """
        self._record_xbar_inst_count(w)
        self.register_buffer("weight", w, persistent=False)

    def vec_mat_mul(self, x: Tensor) -> Tensor:
        """Ideal VMM with output quantisation.

        Args:
            x: Activation tensor with primitive trailing
                ``[row_num]``.

        Returns:
            ADC-code tensor with primitive trailing ``[data_num]``.
        """
        digits = self.weight
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
            override=self._stochastic_override,
        )
        return code
