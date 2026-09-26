"""Transcoder ABC and encoding identifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import assert_never

from torch import Tensor


class Encoding(StrEnum):
    """Supported positional digit encodings."""

    UNSIGNED = "unsigned"
    TRUE_FORM = "true_form"
    COMPLEMENT = "complement"
    CANONICAL = "canonical"


class Transcoder(ABC):
    """Fixed-length positional signed-digit transcoder.

    `encode` and `decode` are mutual inverses throughout the continuous
    `value_range`: `decode(encode(x))` equals `x` exactly for every integer in
    that range. Encodings may be redundant, but may not leave holes. Outside the
    range the encoded value may wrap. `decode` is the shared positional
    reduction and serves every encoding unchanged.

    Subclass authors implement `encode`, `value_range`, and `has_signed_digits`.
    Emit exactly `digit_count` digits in least-significant-first order so the
    shared `decode` can recover legal inputs. Keep transforms device-preserving
    and choose an integer dtype wide enough for all intermediate positional
    products. These functions do not validate every input value or saturate it.

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
        self.__radix = radix
        self.__digit_count = digit_count

    # === Public API ===

    @staticmethod
    def from_encoding(encoding: Encoding, *, radix: int, digit_count: int) -> Transcoder:
        """Construct the implementation selected by `encoding`."""
        from .canonical import CanonicalTranscoder
        from .complement import ComplementTranscoder
        from .true_form import TrueFormTranscoder
        from .unsigned import UnsignedTranscoder

        match encoding:
            case Encoding.UNSIGNED:
                return UnsignedTranscoder(radix=radix, digit_count=digit_count)
            case Encoding.TRUE_FORM:
                return TrueFormTranscoder(radix=radix, digit_count=digit_count)
            case Encoding.COMPLEMENT:
                return ComplementTranscoder(radix=radix, digit_count=digit_count)
            case Encoding.CANONICAL:
                return CanonicalTranscoder(radix=radix, digit_count=digit_count)
        assert_never(encoding)

    @property
    def radix(self) -> int:
        return self.__radix

    @property
    def digit_count(self) -> int:
        return self.__digit_count

    @property
    def place_values(self) -> tuple[int, ...]:
        """LSB-first positional weights."""
        return tuple(self.radix**digit for digit in range(self.digit_count))

    def decode(self, digits: Tensor, *, dim: int = -1) -> Tensor:
        """Reduce a digit tensor back to integers via positional weights.

        Integer Horner evaluation preserves the input dtype.

        Args:
            digits: Digit tensor produced by `encode`.
                Shape: `[..., digit, ...]`.
            dim: Axis of the digit dimension to reduce.

        Returns:
            Integer tensor with `dim` removed.
        """
        parts = digits.unbind(dim=dim)
        decoded = parts[-1]
        for part in reversed(parts[:-1]):
            decoded = decoded * self.radix + part
        return decoded

    # === For subclass to implement or override ===

    @property
    @abstractmethod
    def has_signed_digits(self) -> bool:
        """Whether this encoding can emit negative digit values."""
        raise NotImplementedError

    @property
    @abstractmethod
    def value_range(self) -> tuple[int, int]:
        """Inclusive continuous integer range the encoding represents."""
        raise NotImplementedError

    @abstractmethod
    def encode(self, x: Tensor, *, dim: int = -1) -> Tensor:
        """Encode integers into the target digit representation.

        Args:
            x: Integer tensor to encode.
            dim: Axis at which the digit dimension is inserted.

        Returns:
            Encoded tensor carrying a new digit axis at `dim`.
            Shape: `[..., digit, ...]`.
        """
        raise NotImplementedError
