"""DirectSlicer — identity value decomposition."""

from __future__ import annotations

from torch import Tensor

from .base import Slicer


class DirectSlicer(Slicer):
    """Insert one size-one slice axis.

    Recovery removes that axis without arithmetic or circuit cost.
    """

    def __init__(self, *, value_range: tuple[int, int]) -> None:
        super().__init__(recovery_circuit=None)
        lo, hi = value_range
        if lo >= hi:
            raise ValueError(f"require: value_range lo ({lo}) < hi ({hi})")
        self._value_range = (int(lo), int(hi))

    @property
    def place_values(self) -> tuple[int, ...]:
        return (1,)

    @property
    def has_signed_slices(self) -> bool:
        return self.value_range[0] < 0

    @property
    def value_range(self) -> tuple[int, int]:
        return self._value_range

    @property
    def slice_num(self) -> int:
        return 1

    @property
    def slice_radix(self) -> int:
        lo, hi = self._value_range
        return hi - lo + 1

    def slice(self, x: Tensor, *, dim: int = -1) -> Tensor:
        # Shape: [...] -> [..., slice=1, ...]
        return x.unsqueeze(dim)
