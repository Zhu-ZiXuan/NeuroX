"""SerialSlicer — radix-``r`` positional decomposition with ``digit_num=1``.

See also:
    docs/dev/modules/mapper/xbar/slicer/README.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.mapper.transcoder import Encoding, SignedDigitTranscoder

from .base import Slicer, SlicingPlan


class SerialSlicer(Slicer):
    """Radix-``r`` serial decomposition with structural ``digit_num=1``.

    Args:
        slice_num: Number of per-cycle digits (shape shorthand ``Sa``).
        encoding: Signed-digit encoding policy.
    """

    def __init__(self, *, slice_num: int, encoding: Encoding) -> None:
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
        radix = self._validate(digit_count, digit_radix, digit_range)
        # SignedDigitTranscoder.value_range gives the symmetric
        # envelope of one radix-r digit string; the activation grid
        # is unsigned, so this slicer publishes only the non-negative
        # half.
        envelope = SignedDigitTranscoder(
            self._encoding,
            radix=radix,
            digit_num=self._slice_num,
        ).value_range()
        return 0, envelope[1]

    def slice_radix(
        self,
        *,
        digit_count: int,
        digit_radix: int,
        digit_range: tuple[int, int],
    ) -> int:
        return self._validate(digit_count, digit_radix, digit_range)

    def slice(
        self,
        x: Tensor,
        *,
        digit_count: int,
        digit_radix: int,
        digit_range: tuple[int, int],
    ) -> SlicingPlan:
        """Decompose ``x`` into ``slice_num`` radix-``r`` digits.

        Args:
            x: Integer activation tensor of arbitrary shape.
            digit_count: Inner digit slots per slice; must be ``1``.
            digit_radix: Per-digit radix; must equal
                ``len(digit_range)``.
            digit_range: Xbar's primitive input grid ``(0, r - 1)``.
        """
        radix = self._validate(digit_count, digit_radix, digit_range)
        transcoder = SignedDigitTranscoder(
            self._encoding,
            radix=radix,
            digit_num=self._slice_num,
        )
        # Shape: [...] -> [..., slice_num].
        encoded = transcoder.encode(x, dim=-1)
        # Shape: [..., slice_num] -> [..., slice_num, digit_num=1].
        values = encoded.unsqueeze(-1)
        slice_weights = torch.tensor(
            [radix**i for i in range(self._slice_num)],
            dtype=values.dtype,
            device=values.device,
        )
        digit_weights = torch.ones(1, dtype=values.dtype, device=values.device)
        envelope = transcoder.value_range()
        return SlicingPlan(
            values=values,
            slice_weights=slice_weights,
            digit_weights=digit_weights,
            value_range=(0, envelope[1]),
        )

    @staticmethod
    def _validate(
        digit_count: int,
        digit_radix: int,
        digit_range: tuple[int, int],
    ) -> int:
        # SerialSlicer emits one positional digit per slice — the
        # inner digit axis is always a structural singleton.
        if digit_count != 1:
            raise ValueError(
                f"require: digit_count ({digit_count}) == 1 for SerialSlicer",
            )
        # The activation grid is unsigned by construction; signed
        # grids would silently produce wrong digit weights downstream.
        lo, hi = digit_range
        if lo != 0:
            raise ValueError(f"require: digit_range lo ({lo}) == 0 (unsigned grid)")
        if not hi > lo:
            raise ValueError(f"require: digit_range hi ({hi}) > lo ({lo})")
        radix = hi - lo + 1
        # SerialSlicer's radix is determined by the grid cardinality;
        # any mismatch in the caller's view points at a real bug.
        if digit_radix != radix:
            raise ValueError(
                f"require: digit_radix ({digit_radix}) == len(digit_range) ({radix})",
            )
        return radix
