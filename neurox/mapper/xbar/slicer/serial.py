"""SerialSlicer — true-form radix-``r`` positional decomposition.

See also:
    docs/dev/modules/mapper/xbar/slicer/README.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.mapper.transcoder import TrueFormTranscoder

from .base import Slicer, SlicingPlan


class SerialSlicer(Slicer):
    """Radix-``r`` serial decomposition with structural ``digit_num = 1``.

    The activation grid is unsigned by construction, so only the
    sign-magnitude (``true_form``) encoding is compatible — non-negative
    inputs decompose into non-negative digits that fit directly into
    the xbar's unsigned primitive cell. Other encodings would produce
    mixed-sign digits and silently corrupt half the value range; they
    are therefore not exposed.

    Args:
        slice_num: Number of per-cycle digits (shape shorthand ``Sa``).
    """

    def __init__(self, *, slice_num: int) -> None:
        if slice_num < 1:
            raise ValueError(f"require: slice_num ({slice_num}) >= 1")
        self._slice_num = slice_num

    @property
    def slice_num(self) -> int:
        return self._slice_num

    def value_range(
        self,
        *,
        digit_count: int,
        digit_radix: int,
        digit_range: tuple[int, int],
    ) -> tuple[int, int]:
        radix = self._validate(digit_count, digit_radix, digit_range)
        # Unsigned positional decomposition: [0, r^Sa - 1].
        return 0, radix**self._slice_num - 1

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
            digit_radix: Per-digit radix; must equal ``len(digit_range)``.
            digit_range: Xbar's primitive input grid ``(0, r - 1)``.
        """
        radix = self._validate(digit_count, digit_radix, digit_range)
        transcoder = TrueFormTranscoder(radix=radix, digit_num=self._slice_num)
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
        return SlicingPlan(
            values=values,
            slice_weights=slice_weights,
            digit_weights=digit_weights,
            value_range=(0, radix**self._slice_num - 1),
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
