"""Stochastic-rounding helpers shared by every NeuroX quantizer.

See also:
    docs/dev/modules/common/quant.md
"""

from __future__ import annotations

import torch
from torch import Tensor


def stochastic_floor_div(
    numerator: Tensor,
    rshift: Tensor | int,
    *,
    training: bool,
) -> Tensor:
    """Compute ``numerator >> rshift`` with optional unbiased jitter.

    Stochastic rounding is applied when ``training`` is ``True`` and
    skipped otherwise.

    Args:
        numerator: Integer tensor to be shifted.
        rshift: Right-shift amount; scalar or broadcastable tensor.
        training: ``module.training`` flag from the caller.

    Returns:
        Quotient tensor (same dtype as ``numerator``).
    """
    if not training:
        return numerator >> rshift

    if isinstance(rshift, int):
        denom = 1 << rshift
        jitter = torch.randint(
            low=0,
            high=denom,
            size=numerator.shape,
            dtype=numerator.dtype,
            device=numerator.device,
        )
        return (numerator + jitter) >> rshift

    # Tensor-valued rshift: per-element denom; sample from uniform[0,1)
    # in float and multiply through, then cast back to integer to keep
    # the result exact-modulo-denom.
    rshift_t = rshift.to(numerator.dtype)
    denom = (torch.ones_like(rshift_t) << rshift_t).to(torch.float64)
    jitter_float = torch.rand(numerator.shape, device=numerator.device, dtype=torch.float64)
    jitter = (jitter_float * denom).to(numerator.dtype)
    return (numerator + jitter) >> rshift


def stochastic_floor_to_int(
    signal: Tensor,
    lsb: Tensor | float,
    *,
    out_dtype: torch.dtype,
    training: bool,
) -> Tensor:
    """Float→int floor quantizer with optional unbiased jitter.

    Stochastic rounding is applied when ``training`` is ``True``.

    Args:
        signal: Float input.
        lsb: Per-bin step size in ``signal``'s units.
        out_dtype: Target integer dtype.
        training: ``module.training`` flag.

    Returns:
        Integer code tensor with dtype ``out_dtype``.
    """
    if training:
        jitter = torch.rand(signal.shape, device=signal.device, dtype=signal.dtype) * lsb
        signal = signal + jitter
    return torch.floor(signal / lsb).to(out_dtype)


def floor_bucketize(
    signal: Tensor,
    boundaries: Tensor,
    *,
    out_dtype: torch.dtype,
    training: bool,
    lsb: Tensor | float,
) -> Tensor:
    """Bucketize against ``boundaries`` with floor semantics + optional jitter.

    Stochastic rounding is applied when ``training`` is ``True``.

    Args:
        signal: Float input.
        boundaries: Sorted ascending threshold tensor of shape
            ``[n_codes - 1]``. Code edges: ``B_c = c · LSB``.
        out_dtype: Target integer dtype.
        training: ``module.training`` flag.
        lsb: Bin width used to size the stochastic jitter.

    Returns:
        Code tensor in ``[0, n_codes - 1]``.
    """
    if training:
        jitter = torch.rand(signal.shape, device=signal.device, dtype=signal.dtype) * lsb
        signal = signal + jitter
    # ``right=True`` gives floor semantics: signal at an exact
    # boundary lands in the upper bin (code = C when signal == C·LSB).
    # The default ``right=False`` would round-to-nearest at boundaries.
    return torch.bucketize(signal, boundaries, right=True, out_int32=True).to(out_dtype)
