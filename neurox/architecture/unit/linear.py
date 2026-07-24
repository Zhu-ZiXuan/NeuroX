"""LinearUnit operator interface.

See also:
    docs/internals/architecture/unit/linear.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .base import UnitBase


class LinearUnit(UnitBase, ABC):
    """Interface for an integer ``torch.nn.functional.linear`` replacement."""

    @abstractmethod
    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        """Write the unit's static weight state and optional integer bias.

        Args:
            weight: Integer weight tensor of shape ``(*prefix, N, K)``.
            bias: Optional integer bias tensor of shape ``(N,)``, added in
                the int64 accumulation domain by :meth:`linear`; ``None``
                clears any programmed bias.
        """
        raise NotImplementedError

    def _activation_to_planes(self, input: Tensor) -> Tensor:
        # Shape: [..., K] -> [..., 1, K]
        return input.unsqueeze(-2)

    def _undo_aggregation(self, output: Tensor) -> Tensor:
        # Shape: [..., 1, N] -> [..., N]
        return output.squeeze(-2)

    @torch.no_grad()
    def linear(self, input: Tensor, *, adc_mode: int, adc_bits: int) -> Tensor:
        """Execute one integer linear operator against the programmed state.

        Args:
            input: Integer activation tensor with trailing ``[K]``.
            adc_mode: Runtime ADC operating-point index.
            adc_bits: Runtime ADC resolution.

        Returns:
            Integer pre-requantize output tensor with trailing ``[N]``;
            leading dims mirror ``input``.
        """
        y = self._lower_matmul(input, adc_mode=adc_mode, adc_bits=adc_bits)
        int_bias = self._int_bias
        if int_bias is not None:
            # Shape: [..., N] + [N] -> [..., N]
            y = y + int_bias
        return y
