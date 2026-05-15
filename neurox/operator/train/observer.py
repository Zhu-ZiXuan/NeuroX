"""Tiny min/max observer for hardware-aware training.

The observer tracks a running ``(min, max)`` of the values it sees via
exponential moving average and derives ``(scale, zero_point)`` pairs
on demand.  It is deliberately minimal — no ties to ``torch.ao``
(deprecated) and no histogram / percentile complexity.  Two observer
flavors are exposed:

- ``PerTensorObserver``: one ``(scale, zp)`` pair across the whole
  tensor, asymmetric affine.  Used for activations.
- ``PerChannelSymmObserver``: per-output-channel ``(scale, zp=0)`` pair,
  symmetric range ``[-qmax, qmax]``.  Used for weights.

Both implement ``forward(x)`` as an observer hook (no gradient; called
before the fake-quant step) and ``qparams()`` to fetch the current
stats.  When ``training=False`` the ``forward`` call is a no-op, so an
eval pass uses the last-learned ``(scale, zp)`` and doesn't drift.
"""

import torch
import torch.nn as nn
from torch import Tensor


class PerTensorObserver(nn.Module):
    """Per-tensor asymmetric affine min/max observer with EMA tracking.

    Args:
        qmin: Integer min of the target grid (e.g. ``0``).
        qmax: Integer max of the target grid (e.g. ``15``).
        momentum: EMA weight on the newest batch.  ``0.1`` tracks slowly
            and tolerates outliers; ``1.0`` replaces the running stats
            every call.

    The observer has an explicit ``frozen`` flag (persisted as a 0-d
    bool buffer) so an external caller can pin ``(min, max)`` after a
    calibration phase and have the stats survive later ``model.train()``
    calls.  Pure ``self.training``-gating would be reset by the first
    ``model.train()`` that follows calibration, which destabilises
    integer-pipeline HAT where observer drift amplifies STE gradient
    noise batch-over-batch.
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

        No-op when ``self.training`` is ``False`` OR ``self.frozen`` is
        set, so the observer is frozen at inference time and stays
        frozen across ``model.train()`` calls after explicit freezing.
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
        # Pull zero into the observed range so zero can be represented exactly.
        min_val = torch.minimum(self.min_val, torch.zeros_like(self.min_val))
        max_val = torch.maximum(self.max_val, torch.zeros_like(self.max_val))
        span = (max_val - min_val).clamp(min=1e-8)
        scale = span / (self.qmax - self.qmin)
        zp = torch.round(self.qmin - min_val / scale).clamp(self.qmin, self.qmax).to(torch.int32)
        return scale.detach().to(torch.float32), zp.detach()


class PerChannelSymmObserver(nn.Module):
    """Per-channel symmetric min/max observer with EMA tracking.

    Used for weights under the symmetric per-output-channel grid
    ``[-qmax, +qmax]``; the zero-point is always ``0``.  Carries the
    same ``frozen`` flag as :class:`PerTensorObserver` so the two
    observer flavors share a uniform freeze contract.
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
