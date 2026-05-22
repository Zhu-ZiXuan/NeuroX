"""HAT (hardware-aware training) replacement for ``nn.Linear``.

See also:
    docs/dev/modules/operator/train/README.md
"""

from typing import Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neurox.macro.base import NeuroxMacroQuantMatMul
from neurox.operator.base import NeuroxOperator
from neurox.operator.spec import QuantSpec
from neurox.operator.train.fake_quant import fake_quant_ste, fake_quant_symm_per_channel_ste
from neurox.operator.train.observer import PerChannelSymmObserver, PerTensorObserver

from ._shared import derive_layer_int_params, run_matmul_pipeline


class HATLinear(nn.Linear):
    """Hardware-aware QAT replacement for ``nn.Linear``.

    Float ``weight`` / ``bias`` stay trainable; each forward observes
    input / weight / output ranges, fake-quantizes input + weight,
    runs both a float reference (gradient path) and the
    macro-backed hardware path, and STE-combines them so the
    backward uses the smooth float gradient.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        macro: NeuroxMacroQuantMatMul,
        spec: QuantSpec,
        name: str = "",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
        NeuroxOperator.validate_spec_against_macro(macro=macro, spec=spec)
        self.name = name
        self.spec = spec
        self.macro = macro
        self.act_observer = PerTensorObserver(spec.x_qmin, spec.x_qmax)
        self.out_observer = PerTensorObserver(spec.y_qmin, spec.y_qmax)
        self.weight_observer = PerChannelSymmObserver(out_features, spec.w_qmax)

    @classmethod
    def from_torch(
        cls,
        module: nn.Linear,
        macro: NeuroxMacroQuantMatMul,
        spec: QuantSpec,
        name: str = "",
    ) -> Self:
        """Build a HATLinear from an ``nn.Linear``, carrying over float params."""
        hat = cls(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
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

        The HAT operator's ``training`` flag drives observer EMA updates
        and STE behavior.  The macro, however, must always re-fabricate
        from the *current* weight (which is changing every optimizer
        step), so we keep it in training mode regardless — that branch
        of the macro's ``matmul`` is what re-derives the physical state
        from the freshly-updated weight tensor.
        """
        super().train(mode)
        self.macro.train(True)
        return self

    def forward(self, input: Tensor) -> Tensor:
        # --- 1. Observe + derive activation qparams ---
        self.act_observer(input)
        s_x, zp_x = self.act_observer.qparams()

        # --- 2. Observe + derive per-channel weight qparams ---
        self.weight_observer(self.weight)
        s_w, _ = self.weight_observer.qparams()

        # --- 3. Float reference forward with fake-quant (STE) ---
        x_fq = fake_quant_ste(input, s_x, zp_x, self.spec.x_qmin, self.spec.x_qmax)
        w_fq = fake_quant_symm_per_channel_ste(self.weight, s_w, self.spec.w_qmax)
        y_float = F.linear(x_fq, w_fq, self.bias)

        # --- 4. Observe + derive output qparams ---
        self.out_observer(y_float)
        s_y, zp_y = self.out_observer.qparams()

        # --- 5. Hardware forward through the macro ---
        with torch.no_grad():
            x_int = torch.clamp(torch.round(input / s_x + zp_x.float()), self.spec.x_qmin, self.spec.x_qmax).to(
                torch.int32
            )
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
            self.macro.program(w_int)
            self.macro.fabricate()
            y_hw = run_matmul_pipeline(
                x_int,
                b_int,
                mult,
                rsh,
                zp_y.reshape(1),
                s_y,
                self.spec.y_qmin,
                self.spec.y_qmax,
                self.macro,
            )

        # --- 6. STE: forward is hardware, backward is float ---
        return y_float + (y_hw - y_float).detach()

    @torch.no_grad()
    def extract_neurox_state(self, prefix: str) -> dict[str, Tensor]:
        """Emit the 12-buffer NeuroX-flat entries for this layer.

        Called after training to produce the state_dict that
        ``neurox.build_evaluator`` consumes.  The stored
        ``(rescale_multiplier, rescale_rshift, bias_int)`` use
        ``rescale_factor=1.0`` — i.e. the ideal-integer scale, with no
        macro-specific ADC rescale baked in.  ``build_evaluator``'s
        ``bind_output_calibration`` folds the *target* macro's
        ``output_rescale_factor`` into the buffers at load time, so one
        checkpoint serves any macro backend.
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
