"""Positional decomposition into bounded integer slices."""

from __future__ import annotations

from typing import assert_never

from torch import Tensor

from neurox.encoding import Encoding, Transcoder
from neurox.primitive.digital import RadixAccumulator, RadixSummator

from .base import Slicer


class SimpleSlicer(Slicer):
    """Decompose integers into a fixed number of slices bounded by a carrier.

    Compute the largest radix, at most the carrier maximum plus one, from
    the encoding's digit bounds and `slice_value_range`.
    Every emitted digit fits that carrier range, including for integer inputs
    outside `value_range`. Such inputs may not recover their original value.

    Args:
        slice_num: Exact number of emitted slices; greater than one.
        slice_value_range: Inclusive integer interval carried by every slice.
            It must contain the digit ranges required by the selected encoding.
        encoding: `UNSIGNED`, `TRUE_FORM`, or `CANONICAL`.

    Raises:
        ValueError: `COMPLEMENT` is not supported for slicing.
    """

    def __init__(
        self,
        *,
        slice_num: int,
        slice_value_range: tuple[int, int],
        encoding: Encoding,
        recovery_circuit: RadixSummator | RadixAccumulator | None = None,
    ) -> None:
        if slice_num <= 1:
            raise ValueError(f"require: slice_num ({slice_num}) > 1")
        super().__init__(recovery_circuit=recovery_circuit)
        lo, hi = slice_value_range
        match encoding:
            case Encoding.UNSIGNED:
                if not (lo <= 0 and hi >= 1):
                    raise ValueError(f"unsigned slices require a range containing [0, 1]; got {slice_value_range}")
                radix = hi + 1
            case Encoding.TRUE_FORM | Encoding.CANONICAL:
                if not (lo <= -1 and hi >= 1):
                    raise ValueError(f"{encoding} slices require a range containing [-1, 1]; got {slice_value_range}")
                radix = min(-lo, hi) + 1
            case Encoding.COMPLEMENT:
                raise ValueError("complement encoding is not supported for slicing")
            case _:
                assert_never(encoding)

        self._slice_num = slice_num
        self._transcoder = Transcoder.from_encoding(encoding, radix=radix, digit_count=slice_num)
        self._slice_radix = radix

    @property
    def place_values(self) -> tuple[int, ...]:
        return self._transcoder.place_values

    @property
    def has_signed_slices(self) -> bool:
        return self._transcoder.has_signed_digits

    @property
    def value_range(self) -> tuple[int, int]:
        return self._transcoder.value_range

    @property
    def slice_num(self) -> int:
        return self._slice_num

    @property
    def slice_radix(self) -> int:
        return self._slice_radix

    def slice(self, x: Tensor, *, dim: int = -1) -> Tensor:
        # Shape: [...] -> [..., slice, ...]
        return self._transcoder.encode(x, dim=dim)
