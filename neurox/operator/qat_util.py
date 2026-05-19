"""Fixed-point scale derivation for hardware-compatible integer rescaling.

See also:
    docs/dev/modules/operator/README.md
"""

import math

import torch
from torch import Tensor

# Default multiplier precision.  8 bits keeps x * mult in int32 for
# accumulators up to 24 bits (2^24 * 2^8 = 2^32).
DEFAULT_MULT_BITS: int = 8


def derive_multiplier_and_shift(
    scale: float,
    mult_bits: int = DEFAULT_MULT_BITS,
) -> tuple[int, int]:
    """Convert a floating-point scale into a fixed-point multiplier and right shift.

    The result satisfies ``scale ≈ multiplier / 2^shift``, so that
    ``(x * multiplier) >> shift ≈ x * scale`` for integer ``x``.

    Args:
        scale: The floating-point scale to encode.
        mult_bits: Number of bits for the multiplier (default 8).
            The multiplier is clamped to ``[0, 2^mult_bits - 1]``.
    """
    if scale == 0:
        return 0, 0
    mult_max = (1 << mult_bits) - 1
    significand, exponent = math.frexp(scale)
    multiplier = min(round(significand * (1 << mult_bits)), mult_max)
    shift = mult_bits - exponent
    return multiplier, shift


def derive_multiplier_and_shift_tensor(
    scale_tensor: Tensor,
    mult_bits: int = DEFAULT_MULT_BITS,
) -> tuple[Tensor, Tensor]:
    """Batched fixed-point decomposition for per-channel scale tensors.

    Args:
        scale_tensor: 1-D float tensor of per-channel scale factors.
        mult_bits: Number of bits for the multiplier (default 8).

    Returns:
        Tuple ``(multiplier, rshift)`` of int32 tensors, each with the
        same shape as ``scale_tensor``.
    """
    mult_max = (1 << mult_bits) - 1
    significand, exponent = torch.frexp(scale_tensor)
    multiplier = torch.round(significand.to(torch.float64) * (1 << mult_bits))
    multiplier = torch.clamp(multiplier, max=mult_max).to(torch.int32)
    shift = (mult_bits - exponent).to(torch.int32)
    return multiplier, shift
