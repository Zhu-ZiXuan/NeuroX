"""Inference-only crossbar-backed replacement for ``nn.Linear``."""

from typing import Self

import torch
import torch.nn as nn
from torch import Tensor

from neurox.macro.base import NeuroxMacroQuantMatMul
from neurox.operator.base import NeuroxOperator


class QuantLinear(NeuroxOperator):
    """Inference-only crossbar-backed replacement for ``nn.Linear``.

    Each forward quantizes the float input, runs ``macro.matmul``,
    adds bias, rescales, and dequantizes back to float.

    Attributes:
        weight_int: Quantized integer weight ``[out, in]``.
        bias_int: Folded int32 bias.
        rescale_multiplier: Per-output-channel int32 multiplier.
        rescale_rshift: Per-output-channel int32 right-shift.
        output_zero_point: Output zero-point.
        output_qmin: Post-requantize clamp lower bound.
        output_qmax: Post-requantize clamp upper bound.
        input_scale: Float32 input scale.
        input_zero_point: Int32 input zero-point.
        input_qmin: Int32 input quantization minimum.
        input_qmax: Int32 input quantization maximum.
        output_scale: Float32 output scale for dequantization.
    """

    weight_int: Tensor
    bias_int: Tensor
    rescale_multiplier: Tensor
    rescale_rshift: Tensor
    output_zero_point: Tensor
    output_qmin: Tensor
    output_qmax: Tensor
    input_scale: Tensor
    input_zero_point: Tensor
    input_qmin: Tensor
    input_qmax: Tensor
    output_scale: Tensor

    def __init__(
        self,
        macro: NeuroxMacroQuantMatMul,
        name: str,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.macro = macro
        self.adc_operation_point = self.default_adc_operation_point(macro)
        self.name = name
        self.in_features = in_features
        self.out_features = out_features

        w_dtype = self.weight_dtype_for_range(*macro.w_value_range)
        self.register_buffer("weight_int", torch.zeros((out_features, in_features), dtype=w_dtype, device=device))
        self.register_buffer("bias_int", torch.zeros(out_features, dtype=torch.int32, device=device))
        self.register_buffer("rescale_multiplier", torch.zeros(out_features, dtype=torch.int32, device=device))
        self.register_buffer("rescale_rshift", torch.zeros(out_features, dtype=torch.int32, device=device))
        self.register_buffer("output_zero_point", torch.zeros(1, dtype=torch.int32, device=device))
        self.register_buffer("output_qmin", torch.tensor(-(1 << 31), dtype=torch.int32, device=device))
        self.register_buffer("output_qmax", torch.tensor((1 << 31) - 1, dtype=torch.int32, device=device))

        self.register_buffer("input_scale", torch.ones(1, dtype=torch.float32, device=device))
        self.register_buffer("input_zero_point", torch.zeros(1, dtype=torch.int32, device=device))
        self.register_buffer("input_qmin", torch.tensor(-127, dtype=torch.int32, device=device))
        self.register_buffer("input_qmax", torch.tensor(127, dtype=torch.int32, device=device))
        self.register_buffer("output_scale", torch.ones(1, dtype=torch.float32, device=device))

    @classmethod
    def from_torch(cls, module: nn.Linear, macro: NeuroxMacroQuantMatMul, name: str) -> Self:
        """Create a QuantLinear with matching shapes from an ``nn.Linear``."""
        return cls(
            macro=macro,
            name=name,
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
        )

    def fabricate(self) -> None:
        """Resample the macro's static manufacturing variation."""
        self.macro.fabricate()

    def program(self) -> None:
        """Program the macro's static weight state from the loaded int weights."""
        self.assert_integer_tensor(self.weight_int, "weight")
        self._validate_weight_range(self.weight_int)
        self.macro.program(self.weight_int)

    @torch.no_grad()
    def forward(self, input: Tensor) -> Tensor:
        input_int = self.quantize_input(
            input,
            self.input_scale,
            self.input_zero_point,
            self.input_qmin,
            self.input_qmax,
        )
        return self.run_matmul_pipeline(
            input_int,
            self.bias_int,
            self.rescale_multiplier,
            self.rescale_rshift,
            self.output_zero_point,
            self.output_scale,
            self.output_qmin,
            self.output_qmax,
        )
