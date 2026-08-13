"""SimpleSlicer — positional decomposition into macro-level slices.

See also:
    docs/internals/architecture/unit/cim/slicer/simple.md
"""

from __future__ import annotations

from torch import Tensor

from neurox.common.encoding import Encoding, create_transcoder

from .base import Slicer


class SimpleSlicer(Slicer):
    """Decompose values into macro-carriable positional slices.

    Args:
        slice_num: Number of macro-level weight slices.
        slice_value_range: Inclusive value range accepted by one macro slice.
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
        if lo == 0:
            if encoding is not Encoding.TRUE_FORM:
                raise ValueError("an unsigned slice_value_range requires true-form encoding")
            radix = hi + 1
            if radix < 2:
                raise ValueError(f"require: slice_value_range upper bound ({hi}) >= 1")
            value_range = (0, radix**slice_num - 1)
        elif lo == -hi and hi >= 1:
            radix = hi + 1
            value_range = create_transcoder(
                encoding=encoding,
                radix=radix,
                digit_count=slice_num,
            ).value_range
        else:
            radix = hi - lo + 1
            one_slice_range = create_transcoder(
                encoding=encoding,
                radix=radix,
                digit_count=1,
            ).value_range
            if one_slice_range != slice_value_range:
                raise ValueError(
                    f"{encoding.value} cannot represent slice_value_range {slice_value_range}; "
                    f"expected {one_slice_range}"
                )
            value_range = create_transcoder(
                encoding=encoding,
                radix=radix,
                digit_count=slice_num,
            ).value_range

        self._slice_num = slice_num
        self._transcoder = create_transcoder(
            encoding=encoding,
            radix=radix,
            digit_count=slice_num,
        )
        self._slice_radix = radix
        self._value_range = value_range

    @property
    def value_range(self) -> tuple[int, int]:
        return self._value_range

    @property
    def slice_radix(self) -> int:
        return self._slice_radix

    @property
    def slice_weights(self) -> tuple[int, ...]:
        r = self._slice_radix
        return tuple(r**i for i in range(self._slice_num))

    def slice(self, x: Tensor) -> Tensor:
        # Shape: [...] -> [..., slice_num]
        return self._transcoder.encode(x, dim=-1)
