"""LeNet quantization layers — self-contained.

Two layer families live here:

- `QATConv2d` / `QATLinear` — training-time. `nn.Conv2d` / `nn.Linear`
  subclasses with input / weight / output observers attached. `forward`
  fake-quantizes input and weight, then runs the float `F.conv2d` / `F.linear`
  rather than the macro, while the output observer tracks the per-tensor y
  range. Backward is STE.
- `QuantConv2d` / `QuantLinear` — inference-time. Macro-backed integer matmul;
  the MAC units one output code carries are folded into
  `(mult, rshift, bias_int)` at construction, so the runtime forward is
  `quantize_input → macro.linear → (code + bias_int) · mult >> rshift + zp_y →
  dequantize`.

The bridge between training and inference is a flat per-layer state dict
produced by `export_state` and consumed by `from_state`.

The quant-grid constants below match the `x_value_range` / `w_value_range` the
shipped configs expose.
"""

from __future__ import annotations

from typing import Any, Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neurox.architecture.unit import LinearUnit
from neurox.common import TensorDataClassBase

# --- LeNet quantization grid ---
X_QMIN = 0
X_QMAX = 15  # 4-bit unsigned activation, x ∈ [0, 15]
W_QMAX = 1  # symmetric ternary weight, w ∈ {-1, 0, 1}
Y_QMIN = -15  # signed 4-bit pre-ReLU output grid
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
            Shape: `[num_channels]`.
        mult_bits: Multiplier precision.

    Returns:
        `(multiplier, rshift)` int32 tensors reproducing the scale as
        `x × multiplier >> rshift`.
        Shape: `[num_channels]`.
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

    abs_max: Tensor  # Shape: [num_channels]
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
            Shape: `[num_channels, ...]`.
        scale: Per-channel scale.
            Shape: `[num_channels]`.
        qmax: Symmetric grid half-width — values clamp into `[-qmax, +qmax]`.
    """
    shape = [scale.shape[0]] + [1] * (weight.ndim - 1)
    sw = scale.view(shape)
    w_int = torch.round(weight / sw).clamp(-qmax, qmax)
    w_fq = w_int * sw
    return weight + (w_fq - weight).detach()


# ---------------------------------------------------------------------------
# Training-time layers
# ---------------------------------------------------------------------------


class QATConv2d(nn.Conv2d):
    """Training-time fake-quantized conv2d with input / weight / output observers.

    The observers are module children: `model.train()` enables their EMA
    update, `freeze_observers(model)` pins them after calibration.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.act_observer = PerTensorObserver(X_QMIN, X_QMAX)
        self.weight_observer = PerChannelSymmObserver(out_channels, W_QMAX)
        self.out_observer = PerTensorObserver(Y_QMIN, Y_QMAX)

    def forward(self, x: Tensor) -> Tensor:
        self.act_observer(x)
        s_x, zp_x = self.act_observer.qparams()
        self.weight_observer(self.weight)
        s_w, _ = self.weight_observer.qparams()
        x_fq = fake_quant_ste(x, s_x, zp_x, X_QMIN, X_QMAX)
        w_fq = fake_quant_symm_per_channel_ste(self.weight, s_w, W_QMAX)
        y = F.conv2d(x_fq, w_fq, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self.out_observer(y)
        return y

    @torch.no_grad()
    def export_state(self) -> dict[str, Any]:
        """Per-layer state dict consumed by `QuantConv2d.from_state`."""
        s_x, zp_x = self.act_observer.qparams()
        s_w, _ = self.weight_observer.qparams()
        s_y, zp_y = self.out_observer.qparams()
        sw_expanded = s_w.view([-1] + [1] * (self.weight.ndim - 1))
        weight_int = torch.round(self.weight / sw_expanded).clamp(-W_QMAX, W_QMAX).to(torch.int8)
        return {
            "kind": "conv2d",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": tuple(self.kernel_size),
            "stride": tuple(self.stride),
            "padding": tuple(self.padding) if isinstance(self.padding, tuple) else self.padding,
            "weight_int": weight_int.cpu(),
            "bias_float": self.bias.detach().cpu() if self.bias is not None else None,
            "s_x": s_x.cpu(),
            "zp_x": zp_x.cpu(),
            "s_w": s_w.cpu(),
            "s_y": s_y.cpu(),
            "zp_y": zp_y.cpu(),
        }


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
        weight_int = torch.round(self.weight / s_w.view(-1, 1)).clamp(-W_QMAX, W_QMAX).to(torch.int8)
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
# Inference-time layers
# ---------------------------------------------------------------------------


def _default_op(macro: LinearUnit, quantization_mode: int | None) -> tuple[int, int]:
    """Resolve mode 0 by default and select the full ADC resolution."""
    return (0 if quantization_mode is None else quantization_mode, macro.adc_bits or 0)


def _mac_per_code(macro: LinearUnit, *, quantization_mode: int, adc_active_bits: int) -> float:
    """Return the MAC units one final output code carries."""
    return macro.rescale_factor(quantization_mode=quantization_mode, adc_active_bits=adc_active_bits)


class _FoldedScales(TensorDataClassBase):
    """Integer rescale terms the runtime forward applies to macro codes."""

    mult: Tensor
    """Int32 multiplier of the folded scale. Shape: `[num_channels]`."""
    rshift: Tensor
    """Right-shift paired with `mult`. Shape: `[num_channels]`."""
    bias_int: Tensor
    """Bias plus input zero-point correction, in macro codes. Shape: `[num_channels]`."""
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
    """Fold the macro's MAC-units-per-code into `(mult, rshift, bias_int)`.

    A code carries `ideal_dot ≈ code · mac_per_code`, zero through the origin
    by architectural invariant. With `mult >> rshift` standing for
    `(s_x · s_w / s_y) · mac_per_code` and
    `bias_int = round((bias_fp / (s_x · s_w) - zp_x · Σ_k w_int) / mac_per_code)`,
    the runtime `((code + bias_int) · mult) >> rshift + zp_y` reproduces the
    float `round((s_x · s_w · ideal_dot + bias) / s_y) + zp_y`.
    """
    k_axes = tuple(range(1, weight_int.ndim))
    w_sum = weight_int.to(torch.int64).sum(dim=k_axes) if k_axes else weight_int.to(torch.int64)
    sx = s_x.to(torch.float64)
    zp = zp_x.to(torch.float64)
    sw = s_w.to(torch.float64)
    sy = s_y.to(torch.float64).clamp(min=1e-30)
    if bias_float is not None:
        bias_ideal = bias_float.to(torch.float64) / (sx * sw).clamp(min=1e-30)
    else:
        bias_ideal = torch.zeros(weight_int.shape[0], dtype=torch.float64)
    folded = torch.round((bias_ideal - zp * w_sum.to(torch.float64)) / mac_per_code)
    int32 = torch.iinfo(torch.int32)
    bias_int = folded.clamp(min=int32.min, max=int32.max).to(torch.int32)

    combined = (sx * sw * mac_per_code / sy).to(torch.float32)
    # 8 multiplier bits keep `code × mult` in int32 for accumulators
    # up to 24 bits (2²⁴ × 2⁸ = 2³²).
    mult, rshift = derive_multiplier_and_shift_tensor(combined, mult_bits=8)
    return _FoldedScales(
        mult=mult.to(torch.int32),
        rshift=rshift.to(torch.int32),
        bias_int=bias_int,
        mac_per_code=mac_per_code,
    )


def _quantize_input(x: Tensor, s_x: Tensor, zp_x: Tensor) -> Tensor:
    """Asymmetric float → int32 quantizer; clamps to `[X_QMIN, X_QMAX]`."""
    q = torch.round(x / s_x + zp_x.to(x.dtype))
    return q.clamp(X_QMIN, X_QMAX).to(torch.int32)


def _dequantize_output(y_int: Tensor, s_y: Tensor, zp_y: Tensor) -> Tensor:
    return (y_int.to(torch.float32) - zp_y.to(torch.float32)) * s_y


# Conv2d-as-matmul shape glue ------------------------------------------------


def _unfold_conv_input(
    x: Tensor,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> tuple[Tensor, tuple[int, ...], int, int]:
    """Unfold the conv input into matmul-style row blocks.

    Returns:
        The block tensor and the `(batch_shape, out_h, out_w)` that inverting
        the unfold needs.
        Shape: `[N, OH*OW, C*kH*kW]`.
    """
    batch_shape = x.shape[:-3]
    if x.ndim > 4:
        x = x.flatten(0, x.ndim - 4)
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding
    _n, _c, h, w = x.shape
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    # Shape: [N, C, H, W] -> [N, C*kH*kW, OH*OW]
    cols = F.unfold(x, kernel_size=(kh, kw), padding=(ph, pw), stride=(sh, sw))
    # Shape: [N, C*kH*kW, OH*OW] -> [N, OH*OW, C*kH*kW]
    cols = cols.transpose(1, 2)
    return cols, batch_shape, out_h, out_w


def _fold_conv_output(y: Tensor, out_channels: int, batch_shape: tuple[int, ...], out_h: int, out_w: int) -> Tensor:
    """Reverse of `_unfold_conv_input` for the per-row matmul output."""
    n = y.shape[0]
    # Shape: [N, OH*OW, out_channels] -> [N, out_channels, OH, OW]
    y = y.transpose(1, 2).reshape(n, out_channels, out_h, out_w)
    if batch_shape:
        y = y.reshape(*batch_shape, out_channels, out_h, out_w)
    return y


# Quant* layers --------------------------------------------------------------


class QuantConv2d(nn.Module):
    """Inference: macro-backed integer conv2d with folded rescale.

    `quantization_mode` picks the conversion window; `None` resolves to mode 0.
    """

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
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        quantization_mode: int | None = None,
    ) -> None:
        super().__init__()
        self.macro = macro
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
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
        cols, batch_shape, out_h, out_w = _unfold_conv_input(x, self.kernel_size, self.stride, self.padding)
        # Per-row linear through the macro.
        # Shape: [N, OH*OW, K]
        x_int = _quantize_input(cols, self.s_x, self.zp_x)
        code = self.macro.linear(
            x_int,
            quantization_mode=self.quantization_mode,
            adc_active_bits=self.adc_active_bits,
        ).to(torch.int32)
        # Apply the per-out-channel mult / rshift.
        # Shape: [N, OH*OW, out_channels]
        y = (code + self.bias_int.view(1, 1, -1)) * self.mult.view(1, 1, -1)
        y = stochastic_floor_div(y, self.rshift.view(1, 1, -1), training=False)
        y = (y + self.zp_y.to(torch.int32)).clamp(Y_QMIN, Y_QMAX)
        y_float = _dequantize_output(y, self.s_y, self.zp_y)
        return _fold_conv_output(y_float, self.out_channels, batch_shape, out_h, out_w)

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
            in_channels=int(state["in_channels"]),
            out_channels=int(state["out_channels"]),
            kernel_size=tuple(state["kernel_size"]),
            stride=tuple(state["stride"]),
            padding=tuple(state["padding"]) if isinstance(state["padding"], tuple) else (int(state["padding"]),) * 2,
            quantization_mode=quantization_mode,
        )


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
        # Shape: [..., K] -> [..., 1, K]
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
# Calibration / freeze helpers
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
    """Walk `model`, collect `{layer_name: layer_state}` from QAT layers."""
    out: dict[str, dict] = {}
    for name, module in model.named_modules():
        if isinstance(module, QATConv2d | QATLinear):
            out[name] = module.export_state()
    return out
