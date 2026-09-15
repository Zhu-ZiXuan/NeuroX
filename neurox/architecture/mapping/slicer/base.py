"""Slicer ABC for value-domain decomposition."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Slicer(ABC):
    """Decompose integers into a trailing slice axis."""

    # === For subclass to implement or override ===

    @property
    @abstractmethod
    def value_range(self) -> tuple[int, int]:
        """Inclusive algorithm-side integer range this slicer can encode.

        A caller contract: `slice` neither clamps nor rejects a value outside
        it.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def slice_num(self) -> int:
        """Number of positional slices."""
        raise NotImplementedError

    @property
    @abstractmethod
    def slice_radix(self) -> int:
        """Positional radix between adjacent slices."""
        raise NotImplementedError

    @property
    @abstractmethod
    def slice_weights(self) -> tuple[int, ...]:
        """LSB-first positional weight of each slice.

        Plain Python integers: a slicer owns no device and no dtype, so a
        consumer materializes the tensor at its own boundary.
        """
        raise NotImplementedError

    @abstractmethod
    def slice(self, x: Tensor) -> Tensor:
        """Decompose `x` into trailing positional slice values.

        Returns:
            Positional slice values.
            Shape: `[..., slice]`.
        """
        raise NotImplementedError

    def recover(self, values: Tensor, *, dim: int) -> Tensor:
        """Sum partial results using the positional weights of this decomposition.

        Args:
            values: Slice values or linear-operation results for each slice.
                Shape: `[..., slice, ...]`.
            dim: Axis indexing slices in least-significant-first order.

        Returns:
            Weighted sum with the slice axis removed, preserving the dtype.
            No hardware register-width wrap is applied.
        """
        weights = torch.tensor(self.slice_weights, dtype=values.dtype, device=values.device)
        # Shape: [..., slice, ...] -> [..., slice]
        aligned = values.movedim(dim, -1)
        # Shape: [..., slice] -> [...]
        return (aligned * weights).sum(dim=-1, dtype=values.dtype)
