"""Differentiable fake-quantize helpers (STE) for hardware-aware training.

The straight-through estimator (STE) lets HAT operators see the
quantization grid in the forward pass while keeping the backward
identity, so gradients still match the float reference.  The forward
quantizes to integer codes and dequantizes back; the backward swap
``(x_fq - x).detach()`` flips the gradient path through ``x``
unchanged.

Two flavors mirror the observer side:

- :func:`fake_quant_ste` — per-tensor asymmetric (activations).
- :func:`fake_quant_symm_per_channel_ste` — per-output-channel
  symmetric (weights).
"""

import torch
from torch import Tensor


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
