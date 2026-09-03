"""Canonical (non-adjacent-form) signed-digit transcoder."""

from __future__ import annotations

import torch
from torch import Tensor

from .base import Transcoder


class CanonicalTranscoder(Transcoder):
    """Non-adjacent-form-style canonical signed-digit encoding.

    Each digit lies in `{-(r-1), ..., r-1}`. At `r = 2` this is the non-adjacent
    form, where no two consecutive positions are non-zero.
    """

    def encode(self, x: Tensor, *, dim: int = -1) -> Tensor:
        radix = self.radix
        all_digits: list[Tensor] = []
        for _ in range(self.digit_count):
            rem = x % radix
            x = x // radix
            mod = x % radix
            carry = rem > (radix // 2)
            carry = carry & (mod != 0)
            carry = carry | (mod == (radix - 1))
            carry = carry & (rem != 0)
            x = x + carry
            all_digits.append(torch.where(carry, rem - radix, rem))
        # Shape: [...] -> [..., digit, ...]
        return torch.stack(all_digits, dim=dim)

    @property
    def value_range(self) -> tuple[int, int]:
        """Symmetric envelope `[-M, M]`, `M = Σ_j (r - 1)·r^(D - 1 - 2j)` over `j in {0, ..., ceil(D/2) - 1}`."""
        max_abs = sum((self.radix - 1) * (self.radix**power) for power in range(self.digit_count - 1, -1, -2))
        return -max_abs, max_abs
