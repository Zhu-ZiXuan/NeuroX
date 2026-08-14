"""SerialSlicer — true-form radix-`r` positional decomposition.

See Also:
    docs/internals/architecture/unit/cim/slicer/serial.md
"""

from __future__ import annotations

from torch import Tensor

from neurox.common.encoding import TrueFormTranscoder

from .base import Slicer


class SerialSlicer(Slicer):
    """Radix-`r` serial decomposition.

    Args:
        slice_num: Number of per-cycle digits — the slice-count axis in shape notation.
        digit_radix: Activation-cell positional radix `r`. The implied
            unsigned digit range is `[0, r - 1]`.
    """

    def __init__(self, *, slice_num: int, digit_radix: int) -> None:
        if slice_num < 1:
            raise ValueError(f"require: slice_num ({slice_num}) >= 1")
        if digit_radix < 2:
            raise ValueError(f"require: digit_radix ({digit_radix}) >= 2")
        self._slice_num = slice_num
        self._digit_radix = digit_radix
        self._transcoder = TrueFormTranscoder(radix=digit_radix, digit_count=slice_num)

    @property
    def value_range(self) -> tuple[int, int]:
        return 0, self._digit_radix**self._slice_num - 1

    @property
    def slice_radix(self) -> int:
        return self._digit_radix

    @property
    def slice_weights(self) -> tuple[int, ...]:
        r = self._digit_radix
        return tuple(r**i for i in range(self._slice_num))

    def slice(self, x: Tensor) -> Tensor:
        # Shape: [...] -> [..., slice_num]
        return self._transcoder.encode(x, dim=-1)
