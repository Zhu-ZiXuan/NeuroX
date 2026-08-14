"""Slicer ABC for value-domain decomposition.

See Also:
    docs/internals/architecture/unit/cim/slicer/base.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class Slicer(ABC):
    """Decompose integers into a trailing slice axis."""

    @property
    @abstractmethod
    def value_range(self) -> tuple[int, int]:
        """Inclusive algorithm-side integer range this slicer can encode."""
        raise NotImplementedError

    @property
    @abstractmethod
    def slice_radix(self) -> int:
        """Positional radix between adjacent slices."""
        raise NotImplementedError

    @property
    @abstractmethod
    def slice_weights(self) -> tuple[int, ...]:
        """LSB-first positional weight of each slice."""
        raise NotImplementedError

    @abstractmethod
    def slice(self, x: Tensor) -> Tensor:
        """Decompose `x` into trailing positional slice values.

        Returns:
            Positional slice values.
            Shape: `[..., slice_num]`.
        """
        raise NotImplementedError
