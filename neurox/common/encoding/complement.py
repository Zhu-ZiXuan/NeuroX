"""Radix-complement signed-digit transcoder."""

from __future__ import annotations

import torch
from torch import Tensor

from .base import Transcoder


class ComplementTranscoder(Transcoder):
    """Radix-complement encoding (two's-complement when `r = 2`).

    Low `D - 1` digits are in `{0, ..., r - 1}`; the MSB is folded into
    `{-floor(r/2), ..., ceil(r/2) - 1}`. The representable range is an
    asymmetric envelope — values outside that band silently wrap.
    """

    def encode(self, x: Tensor, *, dim: int = -1) -> Tensor:
        radix = self.radix
        all_digits: list[Tensor] = []
        for _ in range(self.digit_count):
            rem = x % radix
            x = x // radix
            all_digits.append(rem)
        msb = all_digits[-1]
        all_digits[-1] = torch.where(msb >= (radix + 1) // 2, msb - radix, msb)
        # Shape: [...] -> [..., digit_count, ...]
        return torch.stack(all_digits, dim=dim)

    @property
    def value_range(self) -> tuple[int, int]:
        """Asymmetric envelope `[-floor(r/2)·r^(D - 1), ceil(r/2)·r^(D - 1) - 1]`."""
        r = self.radix
        top = r ** (self.digit_count - 1)
        lo = -(r // 2) * top
        hi = ((r + 1) // 2) * top - 1
        return lo, hi
