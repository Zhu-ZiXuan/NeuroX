"""Radix-complement signed-digit transcoder.

See also:
    docs/reference/mapper/transcoder/README.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import Transcoder


@Transcoder.register_key("complement")
class ComplementTranscoder(Transcoder):
    """Radix-complement encoding (two's-complement when ``r = 2``).

    Low ``D - 1`` digits are in ``{0, ..., r - 1}``; the MSB is folded
    into ``{-⌊r/2⌋, ..., ⌈r/2⌉ - 1}``. Representable range is the
    asymmetric envelope ``[-⌊r/2⌋·r^(D-1), ⌈r/2⌉·r^(D-1) - 1]`` — values
    outside that band silently wrap.
    """

    def encode(self, x: Tensor, *, dim: int = -1) -> Tensor:
        """Encode integer tensor into radix-complement digit form.

        Args:
            x: Integer tensor to encode.
            dim: Axis at which the digit dimension is inserted.

        Returns:
            Digit tensor with a new size-``digit_num`` axis at ``dim``.
        """
        radix = self._radix
        all_digits: list[Tensor] = []
        for _ in range(self._digit_num):
            rem = x % radix
            x = x // radix
            all_digits.append(rem)
        msb = all_digits[-1]
        all_digits[-1] = torch.where(msb >= (radix + 1) // 2, msb - radix, msb)
        return torch.stack(all_digits, dim=dim)

    @property
    def value_range(self) -> tuple[int, int]:
        """Asymmetric envelope ``[-⌊r/2⌋·r^(D-1), ⌈r/2⌉·r^(D-1) - 1]``."""
        r = self._radix
        top = r ** (self._digit_num - 1)
        lo = -(r // 2) * top
        hi = ((r + 1) // 2) * top - 1
        return lo, hi
