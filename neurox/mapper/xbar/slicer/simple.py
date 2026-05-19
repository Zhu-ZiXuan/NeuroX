"""SimpleSlicer — direct digitise-then-group decomposer.

See also:
    docs/dev/modules/mapper/xbar/slicer/README.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.mapper.transcoder import Encoding, SignedDigitTranscoder

from .base import Slicer, SlicingPlan


class SimpleSlicer(Slicer):
    """Direct digitise-then-group signed-digit decomposer.

    Strategy: encode the input into a single radix-``digit_radix``
    digit string of length ``slice_num * digit_count``, then
    unflatten the trailing axis into ``[slice_num, digit_count]``.

    Args:
        slice_num: Number of macro-external xbar-word slices
            (shape shorthand ``Sw``).  Any ``slice_num >= 1`` is
            supported.
        encoding: Signed-digit encoding policy for the unified
            digit string.
    """

    def __init__(
        self,
        *,
        slice_num: int,
        encoding: Encoding,
    ) -> None:
        if slice_num < 1:
            raise ValueError(f"require: slice_num ({slice_num}) >= 1")
        self._slice_num = slice_num
        self._encoding: Encoding = encoding

    @property
    def slice_num(self) -> int:
        return self._slice_num

    @property
    def encoding(self) -> Encoding:
        return self._encoding

    def value_range(
        self,
        *,
        digit_count: int,
        digit_radix: int,
        digit_range: tuple[int, int],
    ) -> tuple[int, int]:
        del digit_range
        return self._build_transcoder(
            digit_count=digit_count,
            digit_radix=digit_radix,
        ).value_range()

    def slice_radix(
        self,
        *,
        digit_count: int,
        digit_radix: int,
        digit_range: tuple[int, int],
    ) -> int:
        del digit_range
        return self._slice_radix(digit_count=digit_count, digit_radix=digit_radix)

    def slice(
        self,
        x: Tensor,
        *,
        digit_count: int,
        digit_radix: int,
        digit_range: tuple[int, int],
    ) -> SlicingPlan:
        """Slice ``x`` into ``[..., slice_num, digit_count]`` digit slots.

        Args:
            x: Integer weight tensor of arbitrary shape.
            digit_count: Xbar-internal digit count per xbar-word.
            digit_radix: Xbar-internal per-digit radix.
            digit_range: Unused; kept for the unified 3-kwarg contract.
        """
        del digit_range
        transcoder = self._build_transcoder(
            digit_count=digit_count,
            digit_radix=digit_radix,
        )

        # LSB-first digit ordering.
        # Shape: [...] -> [..., slice_num * digit_count]
        flat_digits = transcoder.encode(x, dim=-1)
        # Shape: [..., slice_num * digit_count] -> [..., slice_num, digit_count]
        values = flat_digits.unflatten(-1, (self._slice_num, digit_count))

        slice_radix = self._slice_radix(digit_count=digit_count, digit_radix=digit_radix)
        slice_weights = torch.tensor(
            [slice_radix**i for i in range(self._slice_num)],
            dtype=values.dtype,
            device=values.device,
        )
        digit_weights = torch.tensor(
            [digit_radix**i for i in range(digit_count)],
            dtype=values.dtype,
            device=values.device,
        )
        return SlicingPlan(
            values=values,
            slice_weights=slice_weights,
            digit_weights=digit_weights,
            value_range=transcoder.value_range(),
        )

    def _build_transcoder(
        self,
        *,
        digit_count: int,
        digit_radix: int,
    ) -> SignedDigitTranscoder:
        # One transcoder covers the entire ``slice_num * digit_count``
        # digit string; the slicer just regroups its output.
        if digit_count < 1:
            raise ValueError(f"require: digit_count ({digit_count}) >= 1")
        if digit_radix < 2:
            raise ValueError(f"require: digit_radix ({digit_radix}) >= 2")
        return SignedDigitTranscoder(
            self._encoding,
            radix=digit_radix,
            digit_num=self._slice_num * digit_count,
        )

    @staticmethod
    def _slice_radix(*, digit_count: int, digit_radix: int) -> int:
        if digit_count < 1:
            raise ValueError(f"require: digit_count ({digit_count}) >= 1")
        if digit_radix < 2:
            raise ValueError(f"require: digit_radix ({digit_radix}) >= 2")
        return digit_radix**digit_count
