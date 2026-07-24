"""Slicer ABC for value-domain decomposition.

See also:
    docs/reference/architecture/unit/family.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class Slicer(ABC):
    """Decompose integers into trailing ``[slice_num, digit_count]`` axes."""

    @property
    @abstractmethod
    def value_range(self) -> tuple[int, int]:
        """Inclusive algorithm-side integer range this slicer can encode."""
        raise NotImplementedError

    @property
    @abstractmethod
    def slice_radix(self) -> int:
        """Return the positional radix between adjacent slices."""
        raise NotImplementedError

    @property
    @abstractmethod
    def slice_weights(self) -> tuple[int, ...]:
        """Return the LSB-first positional weight of each slice."""
        raise NotImplementedError

    @abstractmethod
    def slice(self, x: Tensor) -> Tensor:
        """Decompose ``x`` into trailing-2 ``[slice_num, digit_count]`` digit slots."""
        raise NotImplementedError
