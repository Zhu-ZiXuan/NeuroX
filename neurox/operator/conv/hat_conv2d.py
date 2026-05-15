"""HAT (hardware-aware training) replacement for ``nn.Conv2d``.

Structured the same way as :class:`neurox.operator.linear.HATLinear`:
observers learn activation ranges, fake-quant via STE drives the
gradient path, and the macro runs the same im2col + grouped matmul
the inference operator does — with its hardware effects included —
to drive the forward value.

Reuses the inference-side im2col plumbing from :mod:`._shared` so the
two operators share the exact same unfold / reshape / fold geometry,
and the int-domain pipeline + int-params derivation from
:mod:`neurox.operator.linear` (``run_matmul_pipeline``,
``derive_layer_int_params``).
"""

from typing import Literal, Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neurox.macro.base import NeuroxMacroQuantMatMul

from ..base import NeuroxOperator
from ..linear import derive_layer_int_params, run_matmul_pipeline
from ..spec import QuantSpec
from ..train.fake_quant import fake_quant_ste, fake_quant_symm_per_channel_ste
from ..train.observer import PerChannelSymmObserver, PerTensorObserver
from ._shared import (
    _build_reversed_padding,
    _conv_padding_args,
    _fold_output,
    _reshape_weight,
    _unfold_input,
)


class HATConv2d(nn.Conv2d):
    """Hardware-aware QAT replacement for ``nn.Conv2d``.

    Observers learn activation ranges; fake-quant via STE drives the
    gradient path; the macro runs the same im2col + grouped matmul the
    inference operator does — with its hardware effects included — to
    drive the forward value.  See :class:`HATLinear` for the
    step-by-step rationale of each stage.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: str | int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        *,
        macro: NeuroxMacroQuantMatMul,
        spec: QuantSpec,
        name: str = "",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device=device,
            dtype=dtype,
        )
        NeuroxOperator.validate_spec_against_macro(macro=macro, spec=spec)
        self.name = name
        self.spec = spec
        self.macro = macro
        self.act_observer = PerTensorObserver(spec.x_qmin, spec.x_qmax)
        self.out_observer = PerTensorObserver(spec.y_qmin, spec.y_qmax)
        self.weight_observer = PerChannelSymmObserver(out_channels, spec.w_qmax)
        self._reversed_padding_repeated_twice = _build_reversed_padding(
            self.padding if isinstance(self.padding, str) else (self.padding[0], self.padding[1]),
            (self.kernel_size[0], self.kernel_size[1]),
            (self.dilation[0], self.dilation[1]),
        )

    @classmethod
    def from_torch(
        cls,
        module: nn.Conv2d,
        macro: NeuroxMacroQuantMatMul,
        spec: QuantSpec,
        name: str = "",
    ) -> Self:
        """Build a HATConv2d from an ``nn.Conv2d``, carrying over float params."""
        ks, st, pad, dl = _conv_padding_args(module)
        hat = cls(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=ks,
            stride=st,
            padding=pad,
            dilation=dl,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            macro=macro,
            spec=spec,
            name=name,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        hat.weight = module.weight
        if module.bias is not None:
            hat.bias = module.bias
        return hat

    def train(self, mode: bool = True) -> Self:
        """Toggle observer training while pinning the macro to train mode.

        See ``HATLinear.train`` for the rationale.
        """
        super().train(mode)
        self.macro.train(True)
        return self

    def forward(self, input: Tensor) -> Tensor:
        # --- 1. Observe + derive activation qparams --- #
        self.act_observer(input)
        s_x, zp_x = self.act_observer.qparams()

        # --- 2. Observe + derive per-channel weight qparams --- #
        self.weight_observer(self.weight)
        s_w, _ = self.weight_observer.qparams()

        # --- 3. Float reference via F.conv2d on fake-quant inputs + weight --- #
        x_fq = fake_quant_ste(input, s_x, zp_x, self.spec.x_qmin, self.spec.x_qmax)
        w_fq = fake_quant_symm_per_channel_ste(self.weight, s_w, self.spec.w_qmax)
        y_float = F.conv2d(
            x_fq,
            w_fq,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        # --- 4. Observe + derive output qparams --- #
        self.out_observer(y_float)
        s_y, zp_y = self.out_observer.qparams()

        # --- 5. Hardware forward via unfold/fold + macro --- #
        with torch.no_grad():
            unfolded, batch_shape, out_h, out_w = _unfold_input(
                input,
                (self.kernel_size[0], self.kernel_size[1]),
                (self.stride[0], self.stride[1]),
                self.padding if isinstance(self.padding, str) else (self.padding[0], self.padding[1]),
                (self.dilation[0], self.dilation[1]),
                self.groups,
                self.padding_mode,
                self._reversed_padding_repeated_twice,
            )
            unfolded_int = torch.clamp(
                torch.round(unfolded / s_x + zp_x.float()), self.spec.x_qmin, self.spec.x_qmax
            ).to(torch.int32)

            weight_dtype = NeuroxOperator.weight_dtype_for_range(*self.macro.w_value_range)
            w_int, b_int, mult, rsh = derive_layer_int_params(
                float_weight=self.weight.detach(),
                float_bias=self.bias.detach() if self.bias is not None else None,
                input_scale=s_x,
                input_zero_point=zp_x,
                weight_scale=s_w,
                output_scale=s_y,
                w_qmax=self.spec.w_qmax,
                weight_dtype=weight_dtype,
                rescale_factor=self.macro.output_rescale_factor,
            )

            out_per_group = self.out_channels // self.groups
            w_grouped = _reshape_weight(w_int, self.groups)
            b_grouped = b_int.view(self.groups, out_per_group)
            mult_grouped = mult.view(self.groups, out_per_group)
            rsh_grouped = rsh.view(self.groups, out_per_group)

            output_deq = run_matmul_pipeline(
                unfolded_int,
                w_grouped,
                b_grouped,
                mult_grouped,
                rsh_grouped,
                zp_y.reshape(1),
                s_y,
                self.spec.y_qmin,
                self.spec.y_qmax,
                self.macro,
            )
            y_hw = _fold_output(output_deq, self.out_channels, batch_shape, out_h, out_w)

        return y_float + (y_hw - y_float).detach()

    @torch.no_grad()
    def extract_neurox_state(self, prefix: str) -> dict[str, Tensor]:
        """Emit the 12-buffer NeuroX-flat entries for this layer.

        Stored with ``rescale_factor=1.0`` — macro-agnostic ideal
        integer scale.  ``build_evaluator``'s ``bind_output_calibration``
        folds the target macro's ``output_rescale_factor`` in at load
        time.
        """
        s_x, zp_x = self.act_observer.qparams()
        s_w, _ = self.weight_observer.qparams()
        s_y, zp_y = self.out_observer.qparams()
        weight_dtype = NeuroxOperator.weight_dtype_for_range(*self.macro.w_value_range)
        w_int, b_int, mult, rsh = derive_layer_int_params(
            float_weight=self.weight.detach(),
            float_bias=self.bias.detach() if self.bias is not None else None,
            input_scale=s_x,
            input_zero_point=zp_x,
            weight_scale=s_w,
            output_scale=s_y,
            w_qmax=self.spec.w_qmax,
            weight_dtype=weight_dtype,
            rescale_factor=1.0,
        )
        return {
            f"{prefix}.weight_int": w_int.cpu(),
            f"{prefix}.bias_int": b_int.cpu(),
            f"{prefix}.rescale_multiplier": mult.cpu(),
            f"{prefix}.rescale_rshift": rsh.cpu(),
            f"{prefix}.output_zero_point": zp_y.reshape(1).cpu(),
            f"{prefix}.output_qmin": torch.tensor(self.spec.y_qmin, dtype=torch.int32),
            f"{prefix}.output_qmax": torch.tensor(self.spec.y_qmax, dtype=torch.int32),
            f"{prefix}.input_scale": s_x.reshape(1).cpu(),
            f"{prefix}.input_zero_point": zp_x.reshape(1).cpu(),
            f"{prefix}.input_qmin": torch.tensor(self.spec.x_qmin, dtype=torch.int32),
            f"{prefix}.input_qmax": torch.tensor(self.spec.x_qmax, dtype=torch.int32),
            f"{prefix}.output_scale": s_y.reshape(1).cpu(),
        }
