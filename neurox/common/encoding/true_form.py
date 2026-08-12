"""True-form signed-digit transcoder."""

from __future__ import annotations

import torch
from torch import Tensor

from .base import Transcoder


class TrueFormTranscoder(Transcoder):
    """Sign-magnitude signed-digit encoding.

    All non-zero digits share the input sign; magnitude is an ordinary
    base-``r`` decomposition, ``d_i = sign(x)·(floor(|x| / r^i) mod r)``.
    Representable range is the symmetric envelope ``[-(r^D - 1), r^D - 1]``.
    """

    def encode(self, x: Tensor, *, dim: int = -1) -> Tensor:
        sign = x.sign()
        x = x.abs()
        all_digits: list[Tensor] = []
        for _ in range(self._digit_count):
            rem = x % self._radix
            x = x // self._radix
            all_digits.append(rem * sign)
        # Shape: [...] -> [..., digit_count, ...]
        return torch.stack(all_digits, dim=dim)

    @property
    def value_range(self) -> tuple[int, int]:
        """Symmetric envelope ``[-(r^D - 1), r^D - 1]``."""
        n_max = self._radix**self._digit_count - 1
        return -n_max, n_max
