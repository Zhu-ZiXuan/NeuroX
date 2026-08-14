"""Common interface for integer compute units.

See Also:
    docs/internals/architecture/unit/base.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


def _validate_int_bias(bias: Tensor, *, channels: int) -> Tensor:
    """Validate an integer per-channel bias vector and return it as int64.

    Args:
        bias: Bias values in any integer dtype.
            Shape: `[channels]`.
        channels: Output channels the vector must cover.

    Returns:
        `bias` cast to `torch.int64`, the accumulation domain.

    Raises:
        TypeError: `bias` has a non-integer dtype.
        ValueError: `bias` has a shape other than the per-channel vector stated
            above.
    """
    if bias.dtype.is_floating_point or bias.dtype.is_complex or bias.dtype == torch.bool:
        raise TypeError(f"require: integer bias dtype; got {bias.dtype}")
    if tuple(bias.shape) != (channels,):
        raise ValueError(f"require: bias.shape ({tuple(bias.shape)}) == ({channels},)")
    return bias.to(torch.int64)


class UnitBase(ABC):
    """Base interface for programmed integer operators.

    Leading input dimensions pass through unchanged. Optional bias is
    accumulated in the `torch.int64` output domain.
    """

    # === Programmed state ===

    _int_bias: Tensor | None = None  # Shape: [channels]

    @property
    @abstractmethod
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive integer weight range accepted by the unit."""
        raise NotImplementedError

    @property
    @abstractmethod
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive integer activation range accepted by the unit."""
        raise NotImplementedError

    @property
    @abstractmethod
    def adc_max_bits(self) -> int | None:
        """Maximum supported `adc_bits` value; `None` when the unit never quantizes its output."""
        raise NotImplementedError

    @abstractmethod
    def rescale_factor(self, *, quantization_mode: int, adc_bits: int | None) -> float:
        """Return the unit's output code expressed in ideal-macro codes.

        Args:
            quantization_mode: Index selecting the runtime quantization window.
            adc_bits: Runtime ADC resolution, or `None` for the lossless
                oracle.
        """
        raise NotImplementedError

    @abstractmethod
    def fabricate(self) -> None:
        """Re-sample static manufacturing variation across self and descendants."""
        raise NotImplementedError

    @abstractmethod
    def _matmul(self, input: Tensor, *, quantization_mode: int, adc_bits: int | None) -> Tensor:
        """Multiply integer input planes by the programmed weight.

        Args:
            input: Integer activation planes.
                Shape: `[..., M, K]`.
            quantization_mode: Index selecting the runtime quantization window.
            adc_bits: Runtime ADC resolution, or `None` for the lossless
                oracle.

        Returns:
            Integer pre-requantize output tensor; leading order preserved.
            Shape: `[..., M, N]`.
        """
        raise NotImplementedError

    def _weight_to_matrix(self, weight: Tensor) -> Tensor:
        """Convert an operator weight to a matmul weight matrix.

        Returns:
            Weight matrix in the matmul contraction layout.
            Shape: `[N, K]`.
        """
        return weight

    def _activation_to_planes(self, input: Tensor) -> Tensor:
        """Convert operator input to matmul-shaped planes."""
        return input

    def _undo_aggregation(self, output: Tensor) -> Tensor:
        """Convert matmul output back to the operator output layout."""
        return output

    def _lower_matmul(self, input: Tensor, *, quantization_mode: int, adc_bits: int | None) -> Tensor:
        """Run one operator call through the unit's matmul contract."""
        planes = self._activation_to_planes(input)
        y = self._matmul(planes, quantization_mode=quantization_mode, adc_bits=adc_bits)
        return self._undo_aggregation(y)

    def _program_int_bias(self, bias: Tensor | None, *, channels: int) -> None:
        """Store the validated int64 bias; `None` clears it."""
        self._int_bias = None if bias is None else _validate_int_bias(bias, channels=channels)
