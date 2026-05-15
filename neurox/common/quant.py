"""Stochastic-rounding helpers shared by every NeuroX quantizer.

In low-precision pipelines (HAT, ADC bucketize, fixed-point requantize),
nearest-bin / floor rounding introduces a deterministic bias that can
stall training under coarse grids.  The helpers below implement
*unbiased* stochastic rounding: draw a uniform jitter inside one LSB
and floor the perturbed signal — the expected output equals the
ideal-real value, so gradients are unbiased.

Two flavours are exposed because the consuming primitives speak
different units:

* :func:`stochastic_floor_div` — integer arithmetic right-shift with
  uniform jitter in ``[0, 1 << rshift)``.  Used by
  :class:`neurox.digital.Requantizer` and :class:`neurox.xbar.IdealXbar`
  where the upstream tensor is already integer multiplied by an
  integer scale.

* :func:`stochastic_floor_to_int` — float-domain quantizer that adds
  ``uniform(0, lsb)`` then floors and casts.  Used by ADC subclasses
  whose comparator stage receives a float voltage / current.

Both helpers fall back to the deterministic floor when the call site
opts out (``override=False`` or training/eval flag says no).  Branch-
free implementation (no Python ``if`` on tensor data) lets every site
trace cleanly through ``torch.compile``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def use_stochastic(*, training: bool, override: bool | None) -> bool:
    """Resolve the active stochastic-rounding flag.

    ``override`` wins when set; otherwise stochastic rounding follows
    ``module.training``.  Centralising the rule keeps every quantizer
    site consistent.

    Args:
        training: ``module.training`` from the calling ``nn.Module``.
        override: Per-config force flag (``None`` means auto).

    Returns:
        True if the call site should apply stochastic rounding.
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

    When ``use_stochastic(training, override)`` is True, draws a
    uniform integer in ``[0, 1 << rshift)`` and adds it to
    ``numerator`` before the right shift.  The expected value of
    ``(numerator + jitter) >> rshift`` equals the ideal real
    ``numerator / (1 << rshift)``, so the rounding is unbiased.

    Args:
        numerator: Integer tensor to be shifted (e.g. ``x * mult``).
        rshift: Right-shift amount.  Scalar (Python ``int``) or
            broadcastable integer tensor.  When tensor-valued the
            jitter draws a per-element uniform mask.
        training: ``module.training`` flag from the caller.
        override: Per-config force flag (``None`` = follow ``training``).

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

    Equivalent to ``floor(signal / lsb)`` cast to ``out_dtype``.  When
    stochastic, adds ``uniform(0, lsb)`` to ``signal`` before the
    floor.  The expected output equals ``signal / lsb`` so the
    rounding is unbiased.

    Args:
        signal: Float input (e.g. comparator voltage, post-noise BL
            current).  Any shape.
        lsb: Per-bin step size in ``signal``'s units.  Scalar float
            or broadcastable float tensor.
        out_dtype: Target integer dtype (``torch.int16`` /
            ``torch.int32``).
        training: ``module.training`` flag.
        override: Per-config force flag.

    Returns:
        Integer code tensor of shape ``signal.shape`` and dtype
        ``out_dtype``.
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

    Floor semantics requires boundaries placed at code edges
    (``B_C = C · LSB`` for ``C ∈ {1, …, n_codes - 1}``) — *not* at
    midpoints (``(C - 0.5) · LSB``) which would round-to-nearest.

    When stochastic, the caller supplies the bin width via ``lsb`` so
    we can add ``uniform(0, lsb)`` jitter before the threshold
    comparison.  ``lsb`` may be a scalar (uniform-bin ADCs) or a
    per-element tensor (non-uniform calibrated ADCs); see
    :func:`stochastic_floor_to_int` for the uniform-bin shortcut.

    Args:
        signal: Float input. Any shape.
        boundaries: Sorted ascending threshold tensor of shape
            ``[n_codes - 1]``.
        out_dtype: Target integer dtype.
        training: ``module.training`` flag.
        override: Per-config force flag.
        lsb: Bin width used to size the stochastic jitter.  Required
            iff stochastic rounding is active; ignored otherwise.

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
