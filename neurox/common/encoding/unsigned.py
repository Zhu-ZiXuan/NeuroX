"""Unsigned positional transcoders."""

from __future__ import annotations

import torch
from torch import Tensor

from .base import Transcoder


class UnsignedTranscoder(Transcoder):
    """Ordinary unsigned base-`r` encoding."""

    def encode(self, x: Tensor, *, dim: int = -1) -> Tensor:
        all_digits: list[Tensor] = []
        for _ in range(self.digit_count):
            all_digits.append(x % self.radix)
            x = x // self.radix
        return torch.stack(all_digits, dim=dim)

    @property
    def value_range(self) -> tuple[int, int]:
        return 0, self.radix**self.digit_count - 1
