"""Quantization primitives shared by every NeuroX user pipeline.

The module groups three categories of helpers, all stateless except the
two observer classes (which carry frozen-EMA buffers):

- **Stochastic rounding**: ``stochastic_floor_div`` / ``stochastic_floor_to_int``
  / ``floor_bucketize`` — used by macro internals and any QAT path that
  wants dithering during training.
- **Fixed-point scale conversion**: ``derive_multiplier_and_shift`` /
  ``derive_multiplier_and_shift_tensor`` — convert a float scale factor
  into ``(multiplier, rshift)`` for hardware-compatible integer rescaling.
- **QAT-side training primitives**: ``PerTensorObserver`` /
  ``PerChannelSymmObserver`` (EMA min/max trackers with a freeze flag);
  ``fake_quant_ste`` / ``fake_quant_symm_per_channel_ste`` (STE
  fake-quantize for forward, identity backward).

The contract is intentionally narrow: **none of these helpers know about
the macro or the ADC**. They are pure quantization math + observer
state. Each user example (e.g. ``example/lenet``, ``example/bert``)
composes them into its own training / inference pipeline.

See also:
    docs/dev/modules/common/quant.md
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

# Default multiplier precision. 8 bits keeps x * mult in int32 for
# accumulators up to 24 bits (2^24 * 2^8 = 2^32).
DEFAULT_MULT_BITS: int = 8


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
    scale: Tensor | float,
    *,
    out_dtype: torch.dtype,
    training: bool,
) -> Tensor:
    """Float→int floor quantizer ``code = floor(signal · scale)`` with optional jitter.

    Stochastic rounding is applied when ``training`` is ``True``: a uniform
    ``U(0, 1)`` jitter is added in code space before the floor.

    Args:
        signal: Float input.
        scale: Codes per ``signal`` unit; i.e. the reciprocal of one
            LSB step in ``signal``'s units.
        out_dtype: Target integer dtype.
        training: ``module.training`` flag.

    Returns:
        Integer code tensor with dtype ``out_dtype``.
    """
    scaled = signal * scale
    if training:
        scaled = scaled + torch.rand(scaled.shape, device=scaled.device, dtype=scaled.dtype)
    return torch.floor(scaled).to(out_dtype)


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


# ---------------------------------------------------------------------------
# Fixed-point scale → (multiplier, rshift)
# ---------------------------------------------------------------------------


def derive_multiplier_and_shift(
    scale: float,
    mult_bits: int = DEFAULT_MULT_BITS,
) -> tuple[int, int]:
    """Convert one float scale into a fixed-point ``(multiplier, rshift)``.

    The result satisfies ``scale ≈ multiplier / 2^shift`` so that
    ``(x * multiplier) >> shift ≈ x * scale`` for integer ``x``.

    Args:
        scale: Float scale to encode.
        mult_bits: Multiplier precision (default 8). The multiplier is
            clamped to ``[0, 2^mult_bits - 1]``.
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
        mult_bits: Multiplier precision (default 8).

    Returns:
        ``(multiplier, rshift)`` int32 tensors, both shape-matching ``scale_tensor``.
    """
    mult_max = (1 << mult_bits) - 1
    significand, exponent = torch.frexp(scale_tensor)
    multiplier = torch.round(significand.to(torch.float64) * (1 << mult_bits))
    multiplier = torch.clamp(multiplier, max=mult_max).to(torch.int32)
    shift = (mult_bits - exponent).to(torch.int32)
    return multiplier, shift


# ---------------------------------------------------------------------------
# Min/max observers (per-tensor asymmetric, per-channel symmetric)
# ---------------------------------------------------------------------------


class PerTensorObserver(nn.Module):
    """Per-tensor asymmetric affine min/max observer with EMA tracking.

    The ``frozen`` 0-d bool buffer pins ``(min, max)`` after calibration
    so the stats survive subsequent ``model.train()`` calls.

    Args:
        qmin: Integer min of the target grid.
        qmax: Integer max of the target grid.
        momentum: EMA weight on the newest batch.
    """

    min_val: Tensor
    max_val: Tensor
    frozen: Tensor

    def __init__(self, qmin: int, qmax: int, momentum: float = 0.1) -> None:
        super().__init__()
        self.qmin = qmin
        self.qmax = qmax
        self.momentum = momentum
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("frozen", torch.tensor(False))

    def freeze(self) -> None:
        """Pin current stats so further forwards skip EMA updates."""
        self.frozen.fill_(True)

    def unfreeze(self) -> None:
        """Resume EMA updates (e.g. for a second calibration pass)."""
        self.frozen.fill_(False)

    @torch.no_grad()
    def forward(self, x: Tensor) -> None:
        """EMA update of ``(min_val, max_val)`` from ``x``.

        No-op when ``self.training`` is ``False`` or ``self.frozen`` is set.
        """
        if not self.training or bool(self.frozen):
            return
        new_min = x.detach().amin()
        new_max = x.detach().amax()
        if torch.isinf(self.min_val):
            self.min_val.copy_(new_min)
            self.max_val.copy_(new_max)
        else:
            self.min_val.lerp_(new_min, self.momentum)
            self.max_val.lerp_(new_max, self.momentum)

    def qparams(self) -> tuple[Tensor, Tensor]:
        """Return ``(scale, zero_point)`` as ``(float32, int32)`` tensors."""
        min_val = torch.minimum(self.min_val, torch.zeros_like(self.min_val))
        max_val = torch.maximum(self.max_val, torch.zeros_like(self.max_val))
        span = (max_val - min_val).clamp(min=1e-8)
        scale = span / (self.qmax - self.qmin)
        zp = torch.round(self.qmin - min_val / scale).clamp(self.qmin, self.qmax).to(torch.int32)
        return scale.detach().to(torch.float32), zp.detach()


class PerChannelSymmObserver(nn.Module):
    """Per-channel symmetric min/max observer with EMA tracking.

    Symmetric grid ``[-qmax, +qmax]`` with ``zero_point = 0``.

    Args:
        num_channels: Output-channel count.
        qmax: Symmetric grid half-width.
        momentum: EMA weight on the newest batch.
    """

    abs_max: Tensor
    frozen: Tensor

    def __init__(self, num_channels: int, qmax: int, momentum: float = 0.1) -> None:
        super().__init__()
        self.qmax = qmax
        self.momentum = momentum
        self.register_buffer("abs_max", torch.full((num_channels,), float("-inf")))
        self.register_buffer("frozen", torch.tensor(False))

    def freeze(self) -> None:
        """Pin current ``abs_max`` so further forwards skip EMA updates."""
        self.frozen.fill_(True)

    def unfreeze(self) -> None:
        """Resume EMA updates."""
        self.frozen.fill_(False)

    @torch.no_grad()
    def forward(self, weight: Tensor) -> None:
        """EMA update of the per-channel absolute max from ``weight``."""
        if not self.training or bool(self.frozen):
            return
        dims = tuple(range(1, weight.ndim))
        batch_abs_max = weight.detach().abs().amax(dim=dims) if dims else weight.detach().abs()
        if torch.isinf(self.abs_max).any():
            self.abs_max.copy_(batch_abs_max)
        else:
            self.abs_max.lerp_(batch_abs_max, self.momentum)

    def qparams(self) -> tuple[Tensor, Tensor]:
        """Return ``(per_channel_scale, per_channel_zero_point=0)``."""
        scale = (self.abs_max / self.qmax).clamp(min=1e-8)
        zp = torch.zeros_like(scale, dtype=torch.int32)
        return scale.detach().to(torch.float32), zp


# ---------------------------------------------------------------------------
# STE fake-quantize
# ---------------------------------------------------------------------------


def fake_quant_ste(x: Tensor, scale: Tensor, zero_point: Tensor, qmin: int, qmax: int) -> Tensor:
    """Differentiable asymmetric fake-quantize: forward quantizes, backward is identity.

    Args:
        x: Float input tensor.
        scale: Per-tensor scale (float32 scalar).
        zero_point: Per-tensor zero-point (int32 scalar).
        qmin: Integer grid minimum (inclusive).
        qmax: Integer grid maximum (inclusive).
    """
    x_int = torch.clamp(torch.round(x / scale + zero_point.float()), qmin, qmax)
    x_fq = (x_int - zero_point.float()) * scale
    return x + (x_fq - x).detach()


def fake_quant_symm_per_channel_ste(weight: Tensor, scale: Tensor, qmax: int) -> Tensor:
    """Differentiable symmetric per-output-channel fake-quantize.

    Args:
        weight: Float weight tensor, output-channel axis is ``0``.
        scale: Per-channel scale, shape ``[C_out]``.
        qmax: Symmetric grid half-width — values clamp into ``[-qmax, +qmax]``.
    """
    shape = [scale.shape[0]] + [1] * (weight.ndim - 1)
    sw = scale.view(shape)
    w_int = torch.round(weight / sw).clamp(-qmax, qmax)
    w_fq = w_int * sw
    return weight + (w_fq - weight).detach()
