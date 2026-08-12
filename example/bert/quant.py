"""BERT quantization layers — self-contained, Linear only.

Constants match the chip preset's exposed value ranges (4-bit unsigned x,
ternary w, signed 4-bit y).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neurox.architecture.unit import LinearUnit
from neurox.architecture.unit.cim import EngineBackedCimUnit
from neurox.common import stochastic_floor_div
from neurox.common.quant import (
    PerChannelSymmObserver,
    PerTensorObserver,
    derive_multiplier_and_shift_tensor,
    fake_quant_ste,
    fake_quant_symm_per_channel_ste,
)

X_QMIN = 0
X_QMAX = 15
W_QMAX = 1
Y_QMIN = -15
Y_QMAX = 15


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
# Inference-time layer
# ---------------------------------------------------------------------------


def _default_op(macro: LinearUnit, quantization_mode: int | None) -> tuple[int, int | None]:
    """Resolve the operating point: mode 0 by default, macro's max bits.

    A unit without output quantization publishes ``adc_max_bits is None``;
    the resolved ``adc_bits`` is then ``None``, the lossless oracle, whose
    rescale factor is ``1.0``.
    """
    return (0 if quantization_mode is None else quantization_mode, macro.adc_max_bits)


def _mac_per_code(macro: LinearUnit, *, quantization_mode: int, adc_bits: int | None) -> float:
    """Return the MAC units one output code carries at this operating point.

    The simulator exports codes plus ``rescale_factor``, which states a code
    in ideal-macro codes; turning those into MAC units is the algorithm's own
    job and takes the ideal twin's window step ``W / 2**B``. A unit that never
    quantizes its output returns exact dots, so one code is one MAC unit.
    """
    factor = macro.rescale_factor(quantization_mode=quantization_mode, adc_bits=adc_bits)
    if adc_bits is None or not isinstance(macro, EngineBackedCimUnit):
        return factor
    twin = macro.engine.cim_macro.to_ideal()
    lower, upper = twin.quantization_input_ranges[quantization_mode]
    return factor * (upper - lower + 1) / float(1 << twin.adc_max_bits)


@dataclass
class _FoldedScales:
    mult: Tensor
    rshift: Tensor
    bias_int: Tensor
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
    """Fold the macro's MAC-units-per-code into ``(mult, rshift, bias_int)``."""
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
    mult, rshift = derive_multiplier_and_shift_tensor(combined)
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
        self.quantization_mode, self.adc_bits = _default_op(macro, quantization_mode)
        folded = _fold_for_macro(
            weight_int=weight_int,
            bias_float=bias_float,
            s_x=s_x,
            zp_x=zp_x,
            s_w=s_w,
            s_y=s_y,
            mac_per_code=_mac_per_code(macro, quantization_mode=self.quantization_mode, adc_bits=self.adc_bits),
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
        # Linear passes the leading dims through.
        # Shape: [..., K] -> [..., 1, K]
        x_int = _quantize_input(x, self.s_x, self.zp_x).unsqueeze(-2)
        code = (
            self.macro.linear(x_int, quantization_mode=self.quantization_mode, adc_bits=self.adc_bits)
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
    """Pin every QAT observer in ``model``."""
    n = 0
    for m in model.modules():
        if isinstance(m, PerTensorObserver | PerChannelSymmObserver):
            m.freeze()
            n += 1
    return n


def export_qat_state(model: nn.Module) -> dict[str, dict]:
    """Walk ``model``, collect ``{qualified_layer_name: layer_state}`` from QAT layers."""
    out: dict[str, dict] = {}
    for name, module in model.named_modules():
        if isinstance(module, QATLinear):
            out[name] = module.export_state()
    return out
