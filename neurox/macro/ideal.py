"""Ideal macro baseline."""

import torch
import torch.nn as nn
from torch import Tensor


class IdealMacro(nn.Module):
    """Exact integer matmul + requantization baseline."""

    def __init__(
        self,
        *,
        x_value_range: tuple[int, int],
        w_value_range: tuple[int, int],
    ) -> None:
        """Construct one ideal macro.

        Args:
            x_value_range: Inclusive integer activation range.
            w_value_range: Inclusive integer weight range.
        """
        super().__init__()
        self._x_value_range = x_value_range
        self._w_value_range = w_value_range

    @property
    def w_value_range(self) -> tuple[int, int]:
        return self._w_value_range

    @property
    def x_value_range(self) -> tuple[int, int]:
        return self._x_value_range

    @property
    def output_rescale_factor(self) -> float:
        return 1.0

    def fabricate(self, weight: Tensor) -> None:
        """Prepare the macro for one weight tensor.

        Args:
            weight: Integer weight tensor. Shape: [..., N, K]. Ignored by the
                ideal macro.
        """

    @torch.no_grad()
    @torch.compile(dynamic=True)
    def matmul(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        rescale_multiplier: Tensor,
        rescale_rshift: Tensor,
        output_zero_point: Tensor | None,
    ) -> Tensor:
        """Run one integer matmul with fixed-point requantization.

        Args:
            input: Integer activation tensor. Shape: [..., M, K].
            weight: Integer weight tensor. Shape: [..., N, K].
            bias: Optional integer bias tensor. Shape: [..., N].
            rescale_multiplier: Per-output fixed-point multiplier.
            rescale_rshift: Per-output right-shift amount.
            output_zero_point: Optional output zero point.

        Returns:
            Integer output tensor with shape [..., M, N].
        """
        x_dtype = input.dtype

        y = torch.matmul(input.to(torch.float32), weight.to(torch.float32).transpose(-2, -1))
        y = torch.round(y).to(torch.int32)

        if bias is not None:
            y = y + bias.to(torch.int32).unsqueeze(-2)

        m = rescale_multiplier.to(torch.int32).unsqueeze(-2)
        s = rescale_rshift.to(torch.int32).unsqueeze(-2)
        y = (y.to(torch.int32) * m) >> s

        if output_zero_point is not None:
            y = y + output_zero_point.to(torch.int32)

        return y.to(x_dtype)
