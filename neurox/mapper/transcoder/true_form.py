"""True-form signed-digit transcoder.

See also:
    docs/dev/modules/mapper/transcoder/README.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import Transcoder


@Transcoder.register_key("true_form")
class TrueFormTranscoder(Transcoder):
    """Sign-magnitude signed-digit encoding.

    All non-zero digits share the input sign; magnitude is an ordinary
    base-``r`` decomposition. Representable range is the symmetric
    envelope ``[-(r^D - 1), r^D - 1]``.
    """

    def encode(self, x: Tensor, *, dim: int = -1) -> Tensor:
        """Encode integer tensor into sign-magnitude digit form.

        Args:
            x: Integer tensor to encode.
            dim: Axis at which the digit dimension is inserted.

        Returns:
            Digit tensor with a new size-``digit_num`` axis at ``dim``.
        """
        sign = x.sign()
        x = x.abs()
        all_digits: list[Tensor] = []
        for _ in range(self._digit_num):
            rem = x % self._radix
            x = x // self._radix
            all_digits.append(rem * sign)
        return torch.stack(all_digits, dim=dim)

    def value_range(self) -> tuple[int, int]:
        """Symmetric envelope ``[-(r^D - 1), r^D - 1]``."""
        n_max = self._radix**self._digit_num - 1
        return -n_max, n_max
