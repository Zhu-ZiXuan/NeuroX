"""LinearUnit operator interface.

See Also:
    docs/reference/architecture/unit/family.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .base import UnitBase


class LinearUnit(UnitBase, ABC):
    """Interface for an integer `torch.nn.functional.linear` replacement."""

    @abstractmethod
    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        """Write the unit's static weight state and optional integer bias.

        Args:
            weight: Integer weight values.
                Shape: `[N, K]`.
            bias: Per-channel integer bias added in the int64 accumulation
                domain; `None` clears any programmed bias.
                Shape: `[N]`.
        """
        raise NotImplementedError

    def _activation_to_planes(self, input: Tensor) -> Tensor:
        # Shape: [..., K] -> [..., 1, K]
        return input.unsqueeze(-2)

    def _undo_aggregation(self, output: Tensor) -> Tensor:
        # Shape: [..., 1, N] -> [..., N]
        return output.squeeze(-2)

    @torch.no_grad()
    def linear(self, input: Tensor, *, quantization_mode: int, adc_bits: int | None) -> Tensor:
        """Execute one integer linear operator against the programmed state.

        Args:
            input: Integer activation values.
                Shape: `[..., K]`.
            quantization_mode: Index selecting the runtime quantization window.
            adc_bits: Runtime ADC resolution, or `None` for the lossless
                oracle.

        Returns:
            Integer pre-requantize output tensor. Leading dimensions are
            preserved, exactly as `torch.nn.functional.linear`.
            Shape: `[..., N]`.
        """
        y = self._lower_matmul(input, quantization_mode=quantization_mode, adc_bits=adc_bits)
        int_bias = self._int_bias
        if int_bias is not None:
            # Shape: [..., N] + [N] -> [..., N]
            y = y + int_bias
        return y
