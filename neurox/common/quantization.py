"""Quantization helpers."""

import torch
from torch import Tensor


def stochastic_round(value: Tensor, *, enabled: bool) -> Tensor:
    """Round down, or stochastically according to the fractional part.

    For `value = q + r`, with integer `q` and fractional `r` in `[0, 1)`,
    stochastic rounding returns `q + 1` with probability `r` and `q`
    otherwise. Integer values remain deterministic.

    Args:
        value: Floating-point values to round.
        enabled: Enable stochastic rather than floor rounding.

    Returns:
        Integral values represented in the input dtype.
    """
    lower = torch.floor(value)
    if not enabled:
        return lower
    return lower + (torch.rand_like(value) < (value - lower))
