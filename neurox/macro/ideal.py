"""Ideal macro baseline."""

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.mixin import FabricateMixin


class IdealMacro(FabricateMixin, nn.Module):
    """Exact integer matmul baseline; returns the pre-requantize int output."""

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
            w_logical_shape: Logical weight shape ``(*prefix, N, K)`` bound to ``program(...)``.
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
        """Inclusive integer weight range accepted by the macro."""
        return self._w_value_range

    @property
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive integer activation range accepted by the macro."""
        return self._x_value_range

    @property
    def output_rescale_factor(self) -> float:
        """Ratio of the ideal partial-product max to the actual tile output max."""
        return 1.0

    def program(self, weight: Tensor) -> None:
        """Write the macro's static weight state from one logical weight tensor.

        Args:
            weight: Integer weight tensor whose shape matches
                ``self._w_logical_shape``.
        """
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        self.weight = weight

    @torch.no_grad()
    @torch.compile(dynamic=True)
    def matmul(self, input: Tensor) -> Tensor:
        """Execute one integer matrix multiply against the programmed weight state.

        Matches ``torch.matmul`` semantics (pure matmul, no bias). Bias add
        and requantize live in the operator layer.

        Args:
            input: Integer activation tensor. Shape: ``[..., M, K]``.

        Returns:
            Integer pre-requantize output tensor. Shape: ``[..., M, N]``.
        """
        weight = self.weight
        y = torch.matmul(input.to(torch.float32), weight.to(torch.float32).transpose(-2, -1))
        return torch.round(y).to(torch.int32)
