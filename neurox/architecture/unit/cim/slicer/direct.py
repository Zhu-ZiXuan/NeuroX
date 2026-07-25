"""DirectSlicer — identity value decomposition.

See also:
    docs/reference/architecture/unit/family.md
"""

from __future__ import annotations

from torch import Tensor

from .base import Slicer


class DirectSlicer(Slicer):
    """Append one size-one slice axis.

    Args:
        value_range: Inclusive integer range represented by the direct path.
    """

    def __init__(self, *, value_range: tuple[int, int]) -> None:
        lo, hi = value_range
        if lo >= hi:
            raise ValueError(f"require: value_range lo ({lo}) < hi ({hi})")
        self._value_range = (int(lo), int(hi))

    @property
    def value_range(self) -> tuple[int, int]:
        return self._value_range

    @property
    def slice_radix(self) -> int:
        lo, hi = self._value_range
        return hi - lo + 1

    @property
    def slice_weights(self) -> tuple[int, ...]:
        return (1,)

    def slice(self, x: Tensor) -> Tensor:
        """Append the structural slice axis.

        Args:
            x: Integer tensor.

        Returns:
            ``x`` with shape ``[..., slice_num=1]``.
        """
        # Shape: [...] -> [..., slice_num=1]
        return x.unsqueeze(-1)
