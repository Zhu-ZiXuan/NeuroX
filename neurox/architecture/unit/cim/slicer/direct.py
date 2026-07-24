"""DirectSlicer — identity value decomposition.

See also:
    docs/reference/architecture/unit/family.md
"""

from __future__ import annotations

from torch import Tensor

from .base import Slicer


class DirectSlicer(Slicer):
    """Validate values and append one slice and one digit axis.

    Args:
        value_range: Inclusive integer ``(lo, hi)`` accepted by ``slice``;
            requires ``lo < hi``.
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
        """Validate the value range and append the two structural trailing axes.

        Args:
            x: Integer tensor with every value inside ``value_range``.

        Returns:
            ``x`` with two size-1 trailing axes: shape
            ``[..., slice_num=1, digit_count=1]``, same dtype and values.
        """
        lo, hi = self._value_range
        if bool((x < lo).any()) or bool((x > hi).any()):
            raise ValueError(f"require: x values within value_range [{lo}, {hi}]")
        # Shape: [...] -> [..., slice_num=1, digit_count=1]
        return x.unsqueeze(-1).unsqueeze(-1)
