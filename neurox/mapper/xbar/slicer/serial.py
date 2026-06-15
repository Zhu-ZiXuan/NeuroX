"""SerialSlicer — true-form radix-``r`` positional decomposition.

See also:
    docs/reference/mapper/xbar/slicer/README.md
"""

from __future__ import annotations

from torch import Tensor

from neurox.mapper.transcoder import TrueFormTranscoder

from .base import Slicer


class SerialSlicer(Slicer):
    """Radix-``r`` serial decomposition with structural ``digit_num = 1``.

    The activation grid is unsigned by construction, so the sign-magnitude
    (``true_form``) encoding is the only compatible policy — non-negative
    inputs decompose into non-negative digits that fit directly into
    the xbar's unsigned primitive cell.

    Args:
        slice_num: Number of per-cycle digits (shape shorthand ``Sa``).
        digit_radix: Activation-cell positional radix ``r``. The implied
            unsigned digit range is ``[0, r - 1]``.
    """

    def __init__(self, *, slice_num: int, digit_radix: int) -> None:
        if slice_num < 1:
            raise ValueError(f"require: slice_num ({slice_num}) >= 1")
        if digit_radix < 2:
            raise ValueError(f"require: digit_radix ({digit_radix}) >= 2")
        self._slice_num = slice_num
        self._digit_radix = digit_radix
        self._transcoder = TrueFormTranscoder(radix=digit_radix, digit_num=slice_num)

    @property
    def value_range(self) -> tuple[int, int]:
        # Unsigned positional decomposition: [0, r^Sa - 1].
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
        encoded = self._transcoder.encode(x, dim=-1)
        # Shape: [..., slice_num] -> [..., slice_num, digit_num=1]
        return encoded.unsqueeze(-1)
