"""Stochastic-rounding helpers shared by every NeuroX quantizer.

See also:
    docs/dev/modules/common/quant.md
"""

from __future__ import annotations

import torch
from torch import Tensor


def use_stochastic(*, training: bool, override: bool | None) -> bool:
    """Resolve the active stochastic-rounding flag.

    Args:
        training: ``module.training`` from the calling ``nn.Module``.
        override: Per-config force flag; ``None`` follows ``training``.

    Returns:
        ``True`` when stochastic rounding should be applied.
    """
    return training if override is None else override


def stochastic_floor_div(
    numerator: Tensor,
    rshift: Tensor | int,
    *,
    training: bool,
    override: bool | None,
) -> Tensor:
    """Compute ``numerator >> rshift`` with optional unbiased jitter.

    Args:
        numerator: Integer tensor to be shifted.
        rshift: Right-shift amount; scalar or broadcastable tensor.
        training: ``module.training`` flag from the caller.
        override: Per-config force flag; ``None`` follows ``training``.

    Returns:
        Quotient tensor (same dtype as ``numerator``).
    """
    if not use_stochastic(training=training, override=override):
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
    override: bool | None,
) -> Tensor:
    """Float→int floor quantizer with optional unbiased jitter.

    Args:
        signal: Float input.
        lsb: Per-bin step size in ``signal``'s units.
        out_dtype: Target integer dtype.
        training: ``module.training`` flag.
        override: Per-config force flag.

    Returns:
        Integer code tensor with dtype ``out_dtype``.
    """
    if use_stochastic(training=training, override=override):
        jitter = torch.rand(signal.shape, device=signal.device, dtype=signal.dtype) * lsb
        signal = signal + jitter
    return torch.floor(signal / lsb).to(out_dtype)


def floor_bucketize(
    signal: Tensor,
    boundaries: Tensor,
    *,
    out_dtype: torch.dtype,
    training: bool,
    override: bool | None,
    lsb: Tensor | float | None = None,
) -> Tensor:
    """Bucketize against ``boundaries`` with floor semantics + optional jitter.

    Args:
        signal: Float input.
        boundaries: Sorted ascending threshold tensor of shape
            ``[n_codes - 1]``. Code edges: ``B_c = c · LSB``.
        out_dtype: Target integer dtype.
        training: ``module.training`` flag.
        override: Per-config force flag.
        lsb: Bin width used to size the stochastic jitter; required
            iff stochastic rounding is active.

    Returns:
        Code tensor in ``[0, n_codes - 1]``.
    """
    if use_stochastic(training=training, override=override):
        if lsb is None:
            raise ValueError("floor_bucketize: lsb is required when stochastic rounding is active")
        jitter = torch.rand(signal.shape, device=signal.device, dtype=signal.dtype) * lsb
        signal = signal + jitter
    # ``right=True`` gives floor semantics: signal at an exact
    # boundary lands in the upper bin (code = C when signal == C·LSB).
    # The default ``right=False`` would round-to-nearest at boundaries.
    return torch.bucketize(signal, boundaries, right=True, out_int32=True).to(out_dtype)
