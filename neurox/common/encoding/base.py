"""Transcoder ABC and encoding identifiers.

See also:
    docs/internals/common/encoding/encodings.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

from torch import Tensor


class Encoding(StrEnum):
    """Supported signed-digit encoding algorithms."""

    TRUE_FORM = "true_form"
    COMPLEMENT = "complement"
    CANONICAL = "canonical"


class Transcoder(ABC):
    """Fixed-length positional signed-digit transcoder.

    Args:
        radix: Positional base ``r`` of the digit representation.
        digit_count: Number of digits produced by ``encode``.
    """

    def __init__(self, *, radix: int, digit_count: int) -> None:
        if radix < 2:
            raise ValueError(f"require: radix ({radix}) >= 2")
        if digit_count < 1:
            raise ValueError(f"require: digit_count ({digit_count}) >= 1")
        self._radix = radix
        self._digit_count = digit_count

    @property
    def radix(self) -> int:
        """Positional radix ``r`` of the digit representation."""
        return self._radix

    @property
    def digit_count(self) -> int:
        """Number of digits produced by ``encode``."""
        return self._digit_count

    @abstractmethod
    def encode(self, x: Tensor, *, dim: int = -1) -> Tensor:
        """Encode ``x`` into the target representation.

        Args:
            x: Integer tensor to encode.
            dim: Axis at which the digit dimension is inserted.

        Returns:
            Encoded tensor carrying a new digit axis at ``dim``.
            Shape: ``[..., digit_count, ...]``.
        """
        raise NotImplementedError

    def decode(self, digits: Tensor, *, dim: int = -1) -> Tensor:
        """Reduce a digit tensor back to integers via positional weights.

        Args:
            digits: Digit tensor produced by ``encode``.
                Shape: ``[..., digit_count, ...]``.
            dim: Axis of the digit dimension to reduce.

        Returns:
            Integer tensor with ``dim`` removed.
        """
        parts = digits.unbind(dim=dim)
        decoded = parts[-1]
        for part in reversed(parts[:-1]):
            decoded = decoded * self._radix + part
        return decoded

    @property
    @abstractmethod
    def value_range(self) -> tuple[int, int]:
        """Inclusive integer range one digit string can losslessly represent."""
        raise NotImplementedError
