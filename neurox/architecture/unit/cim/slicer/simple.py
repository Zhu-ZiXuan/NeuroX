"""SimpleSlicer — direct digitise-then-group decomposer.

See also:
    docs/reference/architecture/unit/family.md
"""

from __future__ import annotations

from torch import Tensor

from neurox.common.encoding import Encoding, Transcoder

from .base import Slicer


class SimpleSlicer(Slicer):
    """Direct digitise-then-group signed-digit decomposer.

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
        self._transcoder = Transcoder.create(
            encoding,
            radix=digit_radix,
            digit_count=slice_num * digit_count,
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
        # Shape: [...] -> [..., slice_num * digit_count]
        flat_digits = self._transcoder.encode(x, dim=-1)
        # Shape: [..., slice_num * digit_count] -> [..., slice_num, digit_count]
        regrouped: Tensor = flat_digits.unflatten(-1, (self._slice_num, self._digit_count))
        return regrouped
