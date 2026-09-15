"""Positional decomposition into bounded integer slices."""

from __future__ import annotations

from typing import assert_never

from torch import Tensor

from neurox.encoding import Encoding, Transcoder

from .base import Slicer


class SimpleSlicer(Slicer):
    """Decompose integer values into slices bounded by a carrier's range.

    Select the largest radix whose digit intervals fit `slice_value_range`.
    Every emitted digit fits that carrier range, including for integer inputs
    outside `value_range`. Such inputs may not recover their original value.
    True-form, canonical, and complement encoding require a carrier that admits
    negative digits.

    Args:
        slice_num: Number of positional slices.
        slice_value_range: Inclusive integer interval carried by every slice.
            It must contain the digit ranges required by the selected encoding.
        encoding: Positional encoding used across slices.
    """

    def __init__(
        self,
        *,
        slice_num: int,
        slice_value_range: tuple[int, int],
        encoding: Encoding,
    ) -> None:
        if slice_num < 1:
            raise ValueError(f"require: slice_num ({slice_num}) >= 1")

        lo, hi = slice_value_range

        match encoding:
            case Encoding.UNSIGNED:
                if not (lo <= 0 and hi >= 1):
                    raise ValueError(f"unsigned slices require a range containing [0, 1]; got {slice_value_range}")
                radix = hi + 1
            case Encoding.TRUE_FORM:
                if not (lo <= -1 and hi >= 1):
                    raise ValueError(f"true-form slices require a range containing [-1, 1]; got {slice_value_range}")
                radix = min(-lo, hi) + 1
            case Encoding.CANONICAL:
                if not (lo <= -1 and hi >= 1):
                    raise ValueError(f"canonical slices require a range containing [-1, 1]; got {slice_value_range}")
                radix = min(-lo, hi) + 1
            case Encoding.COMPLEMENT:
                min_hi = 0 if slice_num == 1 else 1
                if not (lo <= -1 and hi >= min_hi):
                    raise ValueError(
                        f"{slice_num} complement slices require a range containing [-1, {min_hi}]; "
                        f"got {slice_value_range}"
                    )
                # Only the highest digit is signed; lower digits must fit [0, r-1].
                radix = min(-2 * lo + 1, 2 * hi + 2 if slice_num == 1 else hi + 1)
            case _:
                assert_never(encoding)

        self._slice_num = slice_num
        self._transcoder = Transcoder.from_encoding(
            encoding=encoding,
            radix=radix,
            digit_count=slice_num,
        )
        self._slice_radix = radix

    @property
    def value_range(self) -> tuple[int, int]:
        return self._transcoder.value_range

    @property
    def slice_num(self) -> int:
        return self._slice_num

    @property
    def slice_radix(self) -> int:
        return self._slice_radix

    @property
    def slice_weights(self) -> tuple[int, ...]:
        r = self._slice_radix
        return tuple(r**i for i in range(self._slice_num))

    def slice(self, x: Tensor) -> Tensor:
        # Shape: [...] -> [..., slice]
        return self._transcoder.encode(x, dim=-1)
