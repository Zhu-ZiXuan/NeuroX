"""Transcoder ABC and ``Encoding`` discriminator.

See also:
    docs/dev/modules/mapper/transcoder/README.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, TypeAlias

import torch
from torch import Tensor

from neurox.common.mixin import RegistryMixin

Encoding: TypeAlias = Literal["true_form", "complement", "canonical"]


class Transcoder(RegistryMixin[Encoding, "Transcoder"], ABC):
    """Fixed-length positional signed-digit transcoder.

    Holds the shared positional-radix state ``(radix, digit_num)`` and
    the encoding-agnostic ``decode`` reduction. Subclasses supply the
    encoding-specific ``encode`` and ``value_range``, and self-register
    against their :data:`Encoding` discriminator.

    Args:
        radix: Positional base ``r`` of the digit representation.
        digit_num: Number of digits produced by ``encode``.
    """

    def __init__(self, *, radix: int, digit_num: int) -> None:
        if radix < 2:
            raise ValueError(f"require: radix ({radix}) >= 2")
        if digit_num < 1:
            raise ValueError(f"require: digit_num ({digit_num}) >= 1")
        self._radix = radix
        self._digit_num = digit_num

    @property
    def radix(self) -> int:
        """Positional radix ``r`` of the digit representation."""
        return self._radix

    @property
    def digit_num(self) -> int:
        """Number of digits produced by ``encode``."""
        return self._digit_num

    @abstractmethod
    def encode(self, x: Tensor, *, dim: int = -1) -> Tensor:
        """Encode ``x`` into the target representation.

        Args:
            x: Integer tensor to encode.
            dim: Axis at which the digit dimension is inserted.

        Returns:
            Encoded tensor with a new size-``digit_num`` axis at ``dim``.
        """
        raise NotImplementedError

    def decode(self, digits: Tensor, *, dim: int = -1) -> Tensor:
        """Reduce a digit tensor back to integers via positional weights.

        Args:
            digits: Digit tensor produced by ``encode``.
            dim: Axis of the digit dimension to reduce.

        Returns:
            Integer tensor with ``dim`` removed.
        """
        scales = torch.tensor(
            [self._radix**i for i in range(digits.size(dim))],
            device=digits.device,
            dtype=digits.dtype,
        )
        shape = [1] * digits.ndim
        shape[dim] = digits.size(dim)
        return (digits * scales.view(*shape)).sum(dim=dim)

    @property
    @abstractmethod
    def value_range(self) -> tuple[int, int]:
        """Inclusive integer range one digit string can losslessly represent."""
        raise NotImplementedError

    @classmethod
    def create(cls, encoding: Encoding, *, radix: int, digit_num: int) -> Transcoder:
        """Build the concrete subclass registered for ``encoding``.

        Args:
            encoding: One of ``"true_form"``, ``"complement"``, ``"canonical"``.
            radix: Positional base ``r``.
            digit_num: Number of digits.

        Returns:
            Concrete transcoder instance bound to ``(radix, digit_num)``.
        """
        impl = cls._lookup_impl(encoding)
        return impl(radix=radix, digit_num=digit_num)
