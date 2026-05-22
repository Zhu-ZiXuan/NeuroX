"""Ideal macro baseline."""

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.mixin import FabricateMixin


class IdealMacro(FabricateMixin, nn.Module):
    """Exact integer matmul + requantization baseline."""

    nominal_weight: Tensor
    weight: Tensor

    def __init__(
        self,
        *,
        x_value_range: tuple[int, int],
        w_value_range: tuple[int, int],
        w_logical_shape: tuple[int, ...],
    ) -> None:
        """Construct one ideal macro.

        Args:
            x_value_range: Inclusive integer activation range.
            w_value_range: Inclusive integer weight range.
            w_logical_shape: Shape of the logical weight tensor the macro
                expects in ``program(...)``, typically ``(*prefix, N, K)``.
        """
        super().__init__()
        self._x_value_range = x_value_range
        self._w_value_range = w_value_range
        self._w_logical_shape = tuple(w_logical_shape)
        self._inst_shape = self._w_logical_shape[:-2]

        # 0-d nominal weight: broadcasts to a zero-weight matmul before any
        # ``program(...)`` call.
        self.register_buffer("nominal_weight", torch.zeros((), dtype=torch.int32), persistent=False)
        self.register_buffer("weight", self.nominal_weight.clone(), persistent=False)

    @property
    def w_value_range(self) -> tuple[int, int]:
        return self._w_value_range

    @property
    def x_value_range(self) -> tuple[int, int]:
        return self._x_value_range

    @property
    def output_rescale_factor(self) -> float:
        return 1.0

    def program(self, weight: Tensor) -> None:
        """Store the integer weight tensor verbatim.

        Args:
            weight: Integer weight tensor. Shape: must match ``w_logical_shape``.
        """
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        self.weight = weight

    @torch.no_grad()
    @torch.compile(dynamic=True)
    def matmul(
        self,
        input: Tensor,
        bias: Tensor | None,
        rescale_multiplier: Tensor,
        rescale_rshift: Tensor,
        output_zero_point: Tensor | None,
    ) -> Tensor:
        """Run one integer matmul with fixed-point requantization.

        Args:
            input: Integer activation tensor. Shape: [..., M, K].
            bias: Optional integer bias tensor. Shape: [..., N].
            rescale_multiplier: Per-output fixed-point multiplier.
            rescale_rshift: Per-output right-shift amount.
            output_zero_point: Optional output zero point.

        Returns:
            Integer output tensor with shape [..., M, N].
        """
        x_dtype = input.dtype
        weight = self.weight

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
