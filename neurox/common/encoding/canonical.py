"""Canonical (non-adjacent-form) signed-digit transcoder.

See also:
    docs/internals/common/encoding/README.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import Transcoder


@Transcoder.register_key("canonical")
class CanonicalTranscoder(Transcoder):
    """Non-adjacent-form-style canonical signed-digit encoding.

    Each digit lies in ``{-(r-1), ..., r-1}`` but at most every other
    position is non-zero, so the representable envelope is strictly
    tighter than the true-form sign-magnitude bound.
    """

    def encode(self, x: Tensor, *, dim: int = -1) -> Tensor:
        """Encode integer tensor into canonical signed-digit form.

        Args:
            x: Integer tensor to encode.
            dim: Axis at which the digit dimension is inserted.

        Returns:
            Digit tensor with a new size-``digit_count`` axis at ``dim``.
        """
        radix = self._radix
        all_digits: list[Tensor] = []
        for _ in range(self._digit_count):
            rem = x % radix
            x = x // radix
            mod = x % radix
            carry = rem > (radix // 2)
            carry = carry & (mod != 0)
            carry = carry | (mod == (radix - 1))
            carry = carry & (rem != 0)
            x = x + carry
            all_digits.append(torch.where(carry, rem - radix, rem))
        return torch.stack(all_digits, dim=dim)

    @property
    def value_range(self) -> tuple[int, int]:
        """Symmetric envelope ``[-M, M]``."""
        max_abs = sum((self._radix - 1) * (self._radix**power) for power in range(self._digit_count - 1, -1, -2))
        return -max_abs, max_abs
