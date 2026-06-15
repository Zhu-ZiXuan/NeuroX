"""SimpleSlicer — direct digitise-then-group decomposer.

See also:
    docs/reference/mapper/xbar/slicer/README.md
"""

from __future__ import annotations

from torch import Tensor

from neurox.mapper.transcoder import Encoding, Transcoder

from .base import Slicer


class SimpleSlicer(Slicer):
    """Direct digitise-then-group signed-digit decomposer.

    Strategy: encode the input into a single radix-``digit_radix`` digit
    string of length ``slice_num * digit_count``, then unflatten the
    trailing axis into ``[slice_num, digit_count]``. Digit-grid
    compatibility with the xbar's primitive cell is a caller-side
    invariant.

    Args:
        slice_num: Number of macro-external xbar-word slices (shape
            shorthand ``Sw``). Any ``slice_num >= 1`` is supported.
        digit_count: Xbar-internal digit count per xbar-word.
        digit_radix: Xbar-internal per-digit positional radix.
        encoding: Signed-digit encoding policy for the unified digit string.
    """

    def __init__(
        self,
        *,
        slice_num: int,
        digit_count: int,
        digit_radix: int,
        encoding: Encoding,
    ) -> None:
        if slice_num < 1:
            raise ValueError(f"require: slice_num ({slice_num}) >= 1")
        if digit_count < 1:
            raise ValueError(f"require: digit_count ({digit_count}) >= 1")
        if digit_radix < 2:
            raise ValueError(f"require: digit_radix ({digit_radix}) >= 2")
        self._slice_num = slice_num
        self._digit_count = digit_count
        # One transcoder covers the entire ``slice_num * digit_count``
        # digit string; the slicer just regroups its output.
        self._transcoder = Transcoder.create(
            encoding,
            radix=digit_radix,
            digit_num=slice_num * digit_count,
        )
        self._slice_radix = int(digit_radix**digit_count)

    @property
    def value_range(self) -> tuple[int, int]:
        return self._transcoder.value_range

    @property
    def slice_radix(self) -> int:
        return self._slice_radix

    @property
    def slice_weights(self) -> tuple[int, ...]:
        r = self._slice_radix
        return tuple(r**i for i in range(self._slice_num))

    def slice(self, x: Tensor) -> Tensor:
        # LSB-first digit ordering.
        # Shape: [...] -> [..., slice_num * digit_count]
        flat_digits = self._transcoder.encode(x, dim=-1)
        # Shape: [..., slice_num * digit_count] -> [..., slice_num, digit_count]
        regrouped: Tensor = flat_digits.unflatten(-1, (self._slice_num, self._digit_count))
        return regrouped
