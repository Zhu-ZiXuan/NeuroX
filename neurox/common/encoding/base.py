"""Transcoder ABC and encoding identifiers."""

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

    `encode` and `decode` are mutual inverses inside `value_range`:
    `decode(encode(x))` equals `x` exactly for every `x` in that band. Outside it
    the encoded value wraps silently and no error is raised. `decode` is the
    shared positional reduction and serves every encoding unchanged.

    Args:
        radix: Positional base `r` of the digit representation, `r >= 2`.
        digit_count: Number of digits `D` produced by `encode`, `D >= 1`.

    Raises:
        ValueError: `radix < 2` or `digit_count < 1`.
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
        return self._radix

    @property
    def digit_count(self) -> int:
        return self._digit_count

    @abstractmethod
    def encode(self, x: Tensor, *, dim: int = -1) -> Tensor:
        """Encode integers into the target digit representation.

        Args:
            x: Integer tensor to encode.
            dim: Axis at which the digit dimension is inserted.

        Returns:
            Encoded tensor carrying a new digit axis at `dim`.
            Shape: `[..., digit_count, ...]`.
        """
        raise NotImplementedError

    def decode(self, digits: Tensor, *, dim: int = -1) -> Tensor:
        """Reduce a digit tensor back to integers via positional weights.

        Every encoding shares the reduction `M = Σ_i d_i·r^i` for
        `i in {0, ..., D - 1}`, with `d_0` the least-significant digit. Horner
        evaluation keeps the arithmetic exact integer, free of floating-point
        error.

        Args:
            digits: Digit tensor produced by `encode`.
                Shape: `[..., digit_count, ...]`.
            dim: Axis of the digit dimension to reduce.

        Returns:
            Integer tensor with `dim` removed.
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
