"""LeNet quantization layers — self-contained.

Two layer families live here:

- ``QATConv2d`` / ``QATLinear`` — training-time. Plain ``nn.Conv2d`` /
  ``nn.Linear`` subclasses with input / weight / output observers
  attached. ``forward`` runs ``fake_quant`` on input + weight, then
  the standard float ``F.conv2d`` / ``F.linear`` (NOT the macro);
  output observer tracks the per-tensor y range. Backward is STE.
- ``QuantConv2d`` / ``QuantLinear`` — inference-time. Macro-backed
  integer matmul; macro's ``adc_rescale_factor`` is folded into
  ``(mult, rshift, bias_int)`` at construction time so the runtime
  forward is just ``quantize_input → macro.matmul → (code + bias_int)
  · mult >> rshift + zp_y → dequantize``.

The bridge between training and inference is a flat per-layer state
dict produced by ``QATLayer.export_state`` and consumed by
``QuantLayer.__init__``.

NOTE on the quant grid: the constants below match what
``macro_with_ideal_xbar.toml`` / ``macro_with_physical_xbar.toml``
expose as ``x_value_range`` / ``w_value_range``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neurox.architecture.unit.matmul import QuantMatMul
from neurox.common.quant import (
    PerChannelSymmObserver,
    PerTensorObserver,
    derive_multiplier_and_shift_tensor,
    fake_quant_ste,
    fake_quant_symm_per_channel_ste,
    stochastic_floor_div,
)
from neurox.primitive.analog.adc import AdcOperationPoint

# --- LeNet quantization grid ---
X_QMIN = 0
X_QMAX = 15  # 4-bit unsigned activation, x ∈ [0, 15]
W_QMAX = 1  # symmetric ternary weight, w ∈ {-1, 0, 1}
Y_QMIN = -15  # signed 4-bit pre-ReLU output grid
Y_QMAX = 15


# ---------------------------------------------------------------------------
# Training-time layers
# ---------------------------------------------------------------------------


class QATConv2d(nn.Conv2d):
    """Training-time fake-quantized conv2d.

    Observers are nn.Module children: ``model.train()`` enables their EMA
    update; ``freeze_observers(model)`` pins them after calibration.
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
        """Per-layer dict consumed by :meth:`QuantConv2d.from_state`."""
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
    """Training-time fake-quantized linear. Same pattern as :class:`QATConv2d`."""

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


def _default_op_point(macro: QuantMatMul, adc_mode: int | None) -> AdcOperationPoint:
    mode = 0 if adc_mode is None else adc_mode
    return AdcOperationPoint(adc_mode=mode, adc_bits=macro.adc_max_bits)


@dataclass
class _FoldedScales:
    """Per-channel ``(mult, rshift, bias_int)`` after folding macro's r_ADC."""

    mult: Tensor
    rshift: Tensor
    bias_int: Tensor
    r_adc: float


def _fold_for_macro(
    *,
    weight_int: Tensor,
    bias_float: Tensor | None,
    s_x: Tensor,
    zp_x: Tensor,
    s_w: Tensor,
    s_y: Tensor,
    r_adc: float,
) -> _FoldedScales:
    """Fold macro's r_ADC into ``(mult, rshift, bias_int)``.

    Math: with ``ideal_dot ≈ code · r_ADC`` (zero-through-origin by
    architectural invariant), ``combined' = (s_x · s_w / s_y) · r_ADC``,
    and ``bias_int_folded = round((bias_fp/(s_x·s_w) - zp_x · Σ_k w_int) / r_ADC)``,
    the runtime expression ``((code + bias_int_folded) · mult) >> rshift + zp_y``
    reproduces the float math ``round((s_x·s_w·ideal_dot + bias) / s_y) + zp_y``.
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
    folded = torch.round((bias_ideal - zp * w_sum.to(torch.float64)) / r_adc)
    int32 = torch.iinfo(torch.int32)
    bias_int = folded.clamp(min=int32.min, max=int32.max).to(torch.int32)

    combined = (sx * sw * r_adc / sy).to(torch.float32)
    mult, rshift = derive_multiplier_and_shift_tensor(combined)
    return _FoldedScales(mult=mult.to(torch.int32), rshift=rshift.to(torch.int32), bias_int=bias_int, r_adc=r_adc)


def _quantize_input(x: Tensor, s_x: Tensor, zp_x: Tensor) -> Tensor:
    """Asymmetric float → int32 quantizer; clamps to ``[X_QMIN, X_QMAX]``."""
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
    """Unfold (N, C, H, W) → (N, OH*OW, C*kH*kW) for matmul-style conv.

    Returns the reshaped block tensor plus ``(batch_shape, out_h, out_w)``
    needed by :func:`_fold_conv_output`.
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
    cols = F.unfold(x, kernel_size=(kh, kw), padding=(ph, pw), stride=(sh, sw))  # (N, C*kH*kW, OH*OW)
    cols = cols.transpose(1, 2)  # (N, OH*OW, C*kH*kW)
    return cols, batch_shape, out_h, out_w


def _fold_conv_output(y: Tensor, out_channels: int, batch_shape: tuple[int, ...], out_h: int, out_w: int) -> Tensor:
    """Reverse of :func:`_unfold_conv_input` for the per-row matmul output."""
    # y: (N, OH*OW, out_channels) → (N, out_channels, OH, OW)
    n = y.shape[0]
    y = y.transpose(1, 2).reshape(n, out_channels, out_h, out_w)
    if batch_shape:
        y = y.reshape(*batch_shape, out_channels, out_h, out_w)
    return y


# Quant* layers --------------------------------------------------------------


class QuantConv2d(nn.Module):
    """Inference: macro-backed integer conv2d with folded rescale.

    ``adc_mode`` (default 0) is the per-layer hardware mode pick.
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
        macro: QuantMatMul,
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
        adc_mode: int | None = None,
    ) -> None:
        super().__init__()
        self.macro = macro
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.adc_operation_point = _default_op_point(macro, adc_mode)
        r_adc = macro.adc_rescale_factor(self.adc_operation_point)
        folded = _fold_for_macro(
            weight_int=weight_int,
            bias_float=bias_float,
            s_x=s_x,
            zp_x=zp_x,
            s_w=s_w,
            s_y=s_y,
            r_adc=r_adc,
        )
        self.register_buffer("weight_int", weight_int.to(torch.int8))
        self.register_buffer("bias_int", folded.bias_int)
        self.register_buffer("mult", folded.mult)
        self.register_buffer("rshift", folded.rshift)
        self.register_buffer("s_x", s_x.to(torch.float32).reshape(()))
        self.register_buffer("zp_x", zp_x.to(torch.int32).reshape(()))
        self.register_buffer("s_y", s_y.to(torch.float32).reshape(()))
        self.register_buffer("zp_y", zp_y.to(torch.int32).reshape(()))
        # Reshape to the macro's logical weight shape (out_channels, C*kH*kW).
        w_for_macro = weight_int.reshape(out_channels, in_channels * kernel_size[0] * kernel_size[1])
        macro.fabricate()
        macro.program(w_for_macro.to(torch.int32))

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        cols, batch_shape, out_h, out_w = _unfold_conv_input(x, self.kernel_size, self.stride, self.padding)
        # cols: (N, OH*OW, K). Per-row matmul through macro.
        x_int = _quantize_input(cols, self.s_x, self.zp_x)
        code = self.macro.matmul(x_int, adc_operation_point=self.adc_operation_point).to(torch.int32)
        # code: (N, OH*OW, out_channels). Apply per-out-channel mult/rshift.
        y = (code + self.bias_int.view(1, 1, -1)) * self.mult.view(1, 1, -1)
        y = stochastic_floor_div(y, self.rshift.view(1, 1, -1), training=False)
        y = (y + self.zp_y.to(torch.int32)).clamp(Y_QMIN, Y_QMAX)
        y_float = _dequantize_output(y, self.s_y, self.zp_y)
        return _fold_conv_output(y_float, self.out_channels, batch_shape, out_h, out_w)

    @classmethod
    def from_state(
        cls,
        *,
        macro: QuantMatMul,
        state: dict[str, Any],
        adc_mode: int | None = None,
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
            adc_mode=adc_mode,
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
        macro: QuantMatMul,
        weight_int: Tensor,
        bias_float: Tensor | None,
        s_x: Tensor,
        zp_x: Tensor,
        s_w: Tensor,
        s_y: Tensor,
        zp_y: Tensor,
        in_features: int,
        out_features: int,
        adc_mode: int | None = None,
    ) -> None:
        super().__init__()
        self.macro = macro
        self.in_features = in_features
        self.out_features = out_features
        self.adc_operation_point = _default_op_point(macro, adc_mode)
        r_adc = macro.adc_rescale_factor(self.adc_operation_point)
        folded = _fold_for_macro(
            weight_int=weight_int,
            bias_float=bias_float,
            s_x=s_x,
            zp_x=zp_x,
            s_w=s_w,
            s_y=s_y,
            r_adc=r_adc,
        )
        self.register_buffer("weight_int", weight_int.to(torch.int8))
        self.register_buffer("bias_int", folded.bias_int)
        self.register_buffer("mult", folded.mult)
        self.register_buffer("rshift", folded.rshift)
        self.register_buffer("s_x", s_x.to(torch.float32).reshape(()))
        self.register_buffer("zp_x", zp_x.to(torch.int32).reshape(()))
        self.register_buffer("s_y", s_y.to(torch.float32).reshape(()))
        self.register_buffer("zp_y", zp_y.to(torch.int32).reshape(()))
        macro.fabricate()
        macro.program(weight_int.to(torch.int32))

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        # x: (..., in_features). Macro expects ((..., M, K)); treat trailing as M=1.
        x_int = _quantize_input(x, self.s_x, self.zp_x).unsqueeze(-2)  # (..., 1, K)
        code = self.macro.matmul(x_int, adc_operation_point=self.adc_operation_point).to(torch.int32).squeeze(-2)
        y = (code + self.bias_int) * self.mult
        y = stochastic_floor_div(y, self.rshift, training=False)
        y = (y + self.zp_y.to(torch.int32)).clamp(Y_QMIN, Y_QMAX)
        return _dequantize_output(y, self.s_y, self.zp_y)

    @classmethod
    def from_state(
        cls,
        *,
        macro: QuantMatMul,
        state: dict[str, Any],
        adc_mode: int | None = None,
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
            adc_mode=adc_mode,
        )


# ---------------------------------------------------------------------------
# Calibration / freeze helpers
# ---------------------------------------------------------------------------


def freeze_observers(model: nn.Module) -> int:
    """Pin every QAT observer in ``model``. Returns the count frozen."""
    n = 0
    for m in model.modules():
        if isinstance(m, PerTensorObserver | PerChannelSymmObserver):
            m.freeze()
            n += 1
    return n


def export_qat_state(model: nn.Module) -> dict[str, dict]:
    """Walk ``model``, collect ``{layer_name: layer_state}`` from QAT layers."""
    out: dict[str, dict] = {}
    for name, module in model.named_modules():
        if isinstance(module, QATConv2d | QATLinear):
            out[name] = module.export_state()
    return out
