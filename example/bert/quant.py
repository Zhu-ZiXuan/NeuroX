"""BERT quantization layers — self-contained, Linear only.

Constants match the chip preset's exposed value ranges (4-bit unsigned x,
ternary w, signed 4-bit y).
"""

from __future__ import annotations

from typing import Any, Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neurox.architecture.unit import LinearUnit
from neurox.common import TensorDataClassBase

X_QMIN = 0
X_QMAX = 15
W_QMAX = 1
Y_QMIN = -15
Y_QMAX = 15


# ---------------------------------------------------------------------------
# Quantization primitives
# ---------------------------------------------------------------------------


def stochastic_floor_div(
    numerator: Tensor,
    rshift: Tensor | int,
    *,
    training: bool,
) -> Tensor:
    """Compute `numerator >> rshift`, optionally with unbiased jitter.

    Args:
        numerator: Integer tensor to be shifted.
        rshift: Right-shift amount; scalar or broadcastable tensor.
        training: Adds uniform jitter below the shifted LSB, turning the
            truncation into stochastic rounding.

    Returns:
        Quotient tensor, dtype of `numerator`.
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
    denom_t = (torch.ones_like(rshift_t) << rshift_t).to(torch.float64)
    jitter_float = torch.rand(numerator.shape, device=numerator.device, dtype=torch.float64)
    jitter = (jitter_float * denom_t).to(numerator.dtype)
    return (numerator + jitter) >> rshift


def derive_multiplier_and_shift_tensor(
    scale_tensor: Tensor,
    mult_bits: int,
) -> tuple[Tensor, Tensor]:
    """Batched fixed-point decomposition for per-channel scale tensors.

    Args:
        scale_tensor: Per-channel scale factors.
            Shape: `[channel]`.
        mult_bits: Multiplier precision. 8 bits keeps `x × mult` in int32 for
            accumulators up to 24 bits (2²⁴ × 2⁸ = 2³²).

    Returns:
        `(multiplier, rshift)` int32 tensors reproducing the scale as
        `x × multiplier >> rshift`.
        Shape: `[channel]`.
    """
    mult_max = (1 << mult_bits) - 1
    significand, exponent = torch.frexp(scale_tensor)
    multiplier = torch.round(significand.to(torch.float64) * (1 << mult_bits))
    multiplier = torch.clamp(multiplier, max=mult_max).to(torch.int32)
    shift = (mult_bits - exponent).to(torch.int32)
    return multiplier, shift


class PerTensorObserver(nn.Module):
    """Per-tensor asymmetric affine min/max observer with EMA tracking.

    The `frozen` buffer pins `(min_val, max_val)` after calibration so the
    stats survive subsequent `model.train()` calls.

    Args:
        qmin: Integer min of the target grid.
        qmax: Integer max of the target grid.
        momentum: EMA weight on the newest batch.
    """

    # === Runtime buffers ===

    min_val: Tensor  # Shape: []
    max_val: Tensor  # Shape: []
    frozen: Tensor  # Shape: []

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
        """EMA update of `(min_val, max_val)` from `x`.

        No-op outside training mode and once frozen.
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
        """Return `(scale, zero_point)` as `(float32, int32)` tensors."""
        min_val = torch.minimum(self.min_val, torch.zeros_like(self.min_val))
        max_val = torch.maximum(self.max_val, torch.zeros_like(self.max_val))
        span = (max_val - min_val).clamp(min=1e-8)
        scale = span / (self.qmax - self.qmin)
        zp = torch.round(self.qmin - min_val / scale).clamp(self.qmin, self.qmax).to(torch.int32)
        return scale.detach().to(torch.float32), zp.detach()


class PerChannelSymmObserver(nn.Module):
    """Per-channel symmetric min/max observer with EMA tracking.

    Symmetric grid `[-qmax, +qmax]` with `zero_point = 0`.

    Args:
        num_channels: Output-channel count.
        qmax: Symmetric grid half-width.
        momentum: EMA weight on the newest batch.
    """

    # === Runtime buffers ===

    abs_max: Tensor  # Shape: [channel]
    frozen: Tensor  # Shape: []

    def __init__(self, num_channels: int, qmax: int, momentum: float = 0.1) -> None:
        super().__init__()
        self.qmax = qmax
        self.momentum = momentum
        self.register_buffer("abs_max", torch.full((num_channels,), float("-inf")))
        self.register_buffer("frozen", torch.tensor(False))

    def freeze(self) -> None:
        """Pin current `abs_max` so further forwards skip EMA updates."""
        self.frozen.fill_(True)

    def unfreeze(self) -> None:
        """Resume EMA updates."""
        self.frozen.fill_(False)

    @torch.no_grad()
    def forward(self, weight: Tensor) -> None:
        """EMA update of the per-channel absolute max from `weight`.

        No-op outside training mode and once frozen.
        """
        if not self.training or bool(self.frozen):
            return
        dims = tuple(range(1, weight.ndim))
        batch_abs_max = weight.detach().abs().amax(dim=dims) if dims else weight.detach().abs()
        if torch.isinf(self.abs_max).any():
            self.abs_max.copy_(batch_abs_max)
        else:
            self.abs_max.lerp_(batch_abs_max, self.momentum)

    def qparams(self) -> tuple[Tensor, Tensor]:
        """Return `(per_channel_scale, per_channel_zero_point=0)`."""
        scale = (self.abs_max / self.qmax).clamp(min=1e-8)
        zp = torch.zeros_like(scale, dtype=torch.int32)
        return scale.detach().to(torch.float32), zp


def fake_quant_ste(x: Tensor, scale: Tensor, zero_point: Tensor, qmin: int, qmax: int) -> Tensor:
    """Differentiable asymmetric fake-quantize: forward quantizes, backward is identity.

    Args:
        x: Float input tensor.
        scale: Per-tensor float32 scale.
            Shape: `[]`.
        zero_point: Per-tensor int32 zero-point.
            Shape: `[]`.
        qmin: Integer grid minimum, inclusive.
        qmax: Integer grid maximum, inclusive.
    """
    x_int = torch.clamp(torch.round(x / scale + zero_point.float()), qmin, qmax)
    x_fq = (x_int - zero_point.float()) * scale
    return x + (x_fq - x).detach()


def fake_quant_symm_per_channel_ste(weight: Tensor, scale: Tensor, qmax: int) -> Tensor:
    """Differentiable symmetric per-output-channel fake-quantize.

    Args:
        weight: Float weight tensor; axis 0 is the output channel.
            Shape: `[channel, ...]`.
        scale: Per-channel scale.
            Shape: `[channel]`.
        qmax: Symmetric grid half-width — values clamp into `[-qmax, +qmax]`.
    """
    # Shape: [channel] -> [channel, ...]
    scale_shape = (scale.shape[0], *(1,) * (weight.ndim - 1))
    sw = scale.view(scale_shape)
    w_int = torch.round(weight / sw).clamp(-qmax, qmax)
    w_fq = w_int * sw
    return weight + (w_fq - weight).detach()


# ---------------------------------------------------------------------------
# Training-time layer
# ---------------------------------------------------------------------------


class QATLinear(nn.Linear):
    """Training-time fake-quantized linear with input / weight / output observers."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.act_observer = PerTensorObserver(X_QMIN, X_QMAX)
        self.weight_observer = PerChannelSymmObserver(out_features, W_QMAX)
        self.out_observer = PerTensorObserver(Y_QMIN, Y_QMAX)

    def forward(self, x: Tensor) -> Tensor:
        self.act_observer(x)
        s_x, zp_x = self.act_observer.qparams()
        self.weight_observer(self.weight)
        s_w, _ = self.weight_observer.qparams()
        x_fq = fake_quant_ste(x, s_x, zp_x, X_QMIN, X_QMAX)
        w_fq = fake_quant_symm_per_channel_ste(self.weight, s_w, W_QMAX)
        y = F.linear(x_fq, w_fq, self.bias)
        self.out_observer(y)
        return y

    @torch.no_grad()
    def export_state(self) -> dict[str, Any]:
        s_x, zp_x = self.act_observer.qparams()
        s_w, _ = self.weight_observer.qparams()
        s_y, zp_y = self.out_observer.qparams()
        weight_int = torch.round(self.weight / s_w.unsqueeze(-1)).clamp(-W_QMAX, W_QMAX).to(torch.int8)
        return {
            "kind": "linear",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "weight_int": weight_int.cpu(),
            "bias_float": self.bias.detach().cpu() if self.bias is not None else None,
            "s_x": s_x.cpu(),
            "zp_x": zp_x.cpu(),
            "s_w": s_w.cpu(),
            "s_y": s_y.cpu(),
            "zp_y": zp_y.cpu(),
        }


# ---------------------------------------------------------------------------
# Inference-time layer
# ---------------------------------------------------------------------------


def _default_op(macro: LinearUnit, quantization_mode: int | None) -> tuple[int, int | None]:
    """Resolve mode 0 and select the unit's highest available precision."""
    return (0 if quantization_mode is None else quantization_mode, macro.adc_bits)


def _mac_per_code(
    macro: LinearUnit,
    *,
    quantization_mode: int,
    adc_active_bits: int | None,
) -> float:
    """Return the MAC units one final output code carries."""
    return macro.rescale_factor(quantization_mode=quantization_mode, adc_active_bits=adc_active_bits)


class _FoldedScales(TensorDataClassBase):
    """Integer rescale terms the runtime forward applies to macro codes."""

    mult: Tensor
    """Int32 multiplier of the folded scale.
    Shape: `[channel]`."""
    rshift: Tensor
    """Right-shift paired with `mult`.
    Shape: `[channel]`."""
    bias_int: Tensor
    """Bias plus input zero-point correction, in macro codes.
    Shape: `[channel]`."""
    mac_per_code: float


def _fold_for_macro(
    *,
    weight_int: Tensor,
    bias_float: Tensor | None,
    s_x: Tensor,
    zp_x: Tensor,
    s_w: Tensor,
    s_y: Tensor,
    mac_per_code: float,
) -> _FoldedScales:
    """Fold the macro's MAC-units-per-code into `(mult, rshift, bias_int)`."""
    w_sum = weight_int.to(torch.int64).sum(dim=tuple(range(1, weight_int.ndim)))
    sx = s_x.to(torch.float64)
    zp = zp_x.to(torch.float64)
    sw = s_w.to(torch.float64)
    sy = s_y.to(torch.float64).clamp(min=1e-30)
    bias_ideal = (
        bias_float.to(torch.float64) / (sx * sw).clamp(min=1e-30)
        if bias_float is not None
        else torch.zeros(weight_int.shape[0], dtype=torch.float64)
    )
    folded = torch.round((bias_ideal - zp * w_sum.to(torch.float64)) / mac_per_code)
    int32 = torch.iinfo(torch.int32)
    bias_int = folded.clamp(min=int32.min, max=int32.max).to(torch.int32)
    combined = (sx * sw * mac_per_code / sy).to(torch.float32)
    mult, rshift = derive_multiplier_and_shift_tensor(combined, mult_bits=8)
    return _FoldedScales(
        mult=mult.to(torch.int32),
        rshift=rshift.to(torch.int32),
        bias_int=bias_int,
        mac_per_code=mac_per_code,
    )


def _quantize_input(x: Tensor, s_x: Tensor, zp_x: Tensor) -> Tensor:
    q = torch.round(x / s_x + zp_x.to(x.dtype))
    return q.clamp(X_QMIN, X_QMAX).to(torch.int32)


def _dequantize_output(y_int: Tensor, s_y: Tensor, zp_y: Tensor) -> Tensor:
    return (y_int.to(torch.float32) - zp_y.to(torch.float32)) * s_y


class QuantLinear(nn.Module):
    """Inference: macro-backed integer linear with folded rescale."""

    weight_int: Tensor
    bias_int: Tensor
    mult: Tensor
    rshift: Tensor
    s_x: Tensor
    zp_x: Tensor
    s_y: Tensor
    zp_y: Tensor

    def __init__(
        self,
        *,
        macro: LinearUnit,
        weight_int: Tensor,
        bias_float: Tensor | None,
        s_x: Tensor,
        zp_x: Tensor,
        s_w: Tensor,
        s_y: Tensor,
        zp_y: Tensor,
        in_features: int,
        out_features: int,
        quantization_mode: int | None = None,
    ) -> None:
        super().__init__()
        self.macro = macro
        self.in_features = in_features
        self.out_features = out_features
        self.quantization_mode, self.adc_active_bits = _default_op(macro, quantization_mode)
        folded = _fold_for_macro(
            weight_int=weight_int,
            bias_float=bias_float,
            s_x=s_x,
            zp_x=zp_x,
            s_w=s_w,
            s_y=s_y,
            mac_per_code=_mac_per_code(
                macro,
                quantization_mode=self.quantization_mode,
                adc_active_bits=self.adc_active_bits,
            ),
        )
        self.register_buffer("weight_int", weight_int.to(torch.int8))
        self.register_buffer("bias_int", folded.bias_int)
        self.register_buffer("mult", folded.mult)
        self.register_buffer("rshift", folded.rshift)
        self.register_buffer("s_x", s_x.to(torch.float32).reshape(()))
        self.register_buffer("zp_x", zp_x.to(torch.int32).reshape(()))
        self.register_buffer("s_y", s_y.to(torch.float32).reshape(()))
        self.register_buffer("zp_y", zp_y.to(torch.int32).reshape(()))

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        # Linear passes the leading dims through.
        # Shape: [..., K] -> [..., M=1, K]
        x_int = _quantize_input(x, self.s_x, self.zp_x).unsqueeze(-2)
        code = (
            self.macro.linear(
                x_int,
                quantization_mode=self.quantization_mode,
                adc_active_bits=self.adc_active_bits,
            )
            .to(torch.int32)
            .squeeze(-2)
        )
        y = (code + self.bias_int) * self.mult
        y = stochastic_floor_div(y, self.rshift, training=False)
        y = (y + self.zp_y.to(torch.int32)).clamp(Y_QMIN, Y_QMAX)
        return _dequantize_output(y, self.s_y, self.zp_y)

    @classmethod
    def from_state(
        cls,
        *,
        macro: LinearUnit,
        state: dict[str, Any],
        quantization_mode: int | None = None,
    ) -> Self:
        return cls(
            macro=macro,
            weight_int=state["weight_int"],
            bias_float=state["bias_float"],
            s_x=state["s_x"],
            zp_x=state["zp_x"],
            s_w=state["s_w"],
            s_y=state["s_y"],
            zp_y=state["zp_y"],
            in_features=int(state["in_features"]),
            out_features=int(state["out_features"]),
            quantization_mode=quantization_mode,
        )


# ---------------------------------------------------------------------------
# Observer freeze + state export
# ---------------------------------------------------------------------------


def freeze_observers(model: nn.Module) -> int:
    """Pin every QAT observer in `model` and return how many were frozen."""
    n = 0
    for m in model.modules():
        if isinstance(m, PerTensorObserver | PerChannelSymmObserver):
            m.freeze()
            n += 1
    return n


def export_qat_state(model: nn.Module) -> dict[str, dict]:
    """Walk `model`, collect `{qualified_layer_name: layer_state}` from QAT layers."""
    out: dict[str, dict] = {}
    for name, module in model.named_modules():
        if isinstance(module, QATLinear):
            out[name] = module.export_state()
    return out
