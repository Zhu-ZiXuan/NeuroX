"""Inference-only crossbar-backed replacement for ``nn.Conv2d``.

Mirror of :class:`neurox.operator.linear.QuantLinear` for 2-D
convolutions.  Forward: ``im2col → asymmetric int32 quantize →
grouped macro.matmul → fold → dequantize``.  State is loaded from a
NeuroX-flat checkpoint.

Reuses ``run_matmul_pipeline`` from :mod:`neurox.operator.linear` —
conv is just a regrouping of the same int matmul kernel.
"""

from typing import Literal, Self

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.utils import _pair

from neurox.macro import NeuroxMacroQuantMatMul

from ..base import NeuroxOperator
from ..linear import run_matmul_pipeline
from ._shared import (
    _build_reversed_padding,
    _conv_padding_args,
    _fold_output,
    _reshape_weight,
    _unfold_input,
)


class QuantConv2d(NeuroxOperator):
    """Inference-only crossbar-backed replacement for ``nn.Conv2d``.

    Forward: unfold float input → asymmetric int32 quantize → grouped
    ``macro.matmul`` → fold back → dequantize.  State is loaded from a
    NeuroX-flat checkpoint via ``load_state_dict``.
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
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if groups <= 0:
            raise ValueError("groups must be a positive integer.")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups.")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups.")

        self.macro = macro
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding if isinstance(padding, str) else _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self._reversed_padding_repeated_twice = _build_reversed_padding(self.padding, self.kernel_size, self.dilation)

        weight_shape = (out_channels, in_channels // groups, *self.kernel_size)
        w_dtype = self.weight_dtype_for_range(*macro.w_value_range)
        self.register_buffer("weight_int", torch.zeros(weight_shape, dtype=w_dtype, device=device))
        self.register_buffer("bias_int", torch.zeros(out_channels, dtype=torch.int32, device=device))
        self.register_buffer("rescale_multiplier", torch.zeros(out_channels, dtype=torch.int32, device=device))
        self.register_buffer("rescale_rshift", torch.zeros(out_channels, dtype=torch.int32, device=device))
        self.register_buffer("output_zero_point", torch.zeros(1, dtype=torch.int32, device=device))
        self.register_buffer("output_qmin", torch.tensor(-(1 << 31), dtype=torch.int32, device=device))
        self.register_buffer("output_qmax", torch.tensor((1 << 31) - 1, dtype=torch.int32, device=device))

        self.register_buffer("input_scale", torch.ones(1, dtype=torch.float32, device=device))
        self.register_buffer("input_zero_point", torch.zeros(1, dtype=torch.int32, device=device))
        self.register_buffer("input_qmin", torch.tensor(-127, dtype=torch.int32, device=device))
        self.register_buffer("input_qmax", torch.tensor(127, dtype=torch.int32, device=device))
        self.register_buffer("output_scale", torch.ones(1, dtype=torch.float32, device=device))

    @classmethod
    def from_torch(cls, module: nn.Conv2d, macro: NeuroxMacroQuantMatMul, name: str) -> Self:
        """Create a QuantConv2d with correct shapes from an ``nn.Conv2d``."""
        ks, st, pad, dl = _conv_padding_args(module)
        return cls(
            macro=macro,
            name=name,
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=ks,
            stride=st,
            padding=pad,
            dilation=dl,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            device=module.weight.device,
        )

    def extra_repr(self) -> str:
        parts = [
            f"{self.in_channels}",
            f"{self.out_channels}",
            f"kernel_size={self.kernel_size}",
            f"stride={self.stride}",
        ]
        if self.padding != (0, 0):
            parts.append(f"padding={self.padding}")
        if self.dilation != (1, 1):
            parts.append(f"dilation={self.dilation}")
        if self.groups != 1:
            parts.append(f"groups={self.groups}")
        if self.padding_mode != "zeros":
            parts.append(f"padding_mode={self.padding_mode}")
        return ", ".join(parts)

    def fabricate(self) -> None:
        """Program the macro's physical state from the loaded int weights.

        Profiler static aggregation is driven by per-physical-module
        ``_record_inst_count`` calls inside the macro/xbar cascade —
        no operator-side ``log_*`` call.
        """
        self.assert_integer_tensor(self.weight_int, "weight")
        self._validate_weight_range(self.weight_int)
        self.macro.fabricate(_reshape_weight(self.weight_int, self.groups))

    @torch.no_grad()
    def forward(self, input: Tensor) -> Tensor:
        unfolded, batch_shape, out_h, out_w = _unfold_input(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.padding_mode,
            self._reversed_padding_repeated_twice,
        )
        unfolded_int = self.quantize_input(
            unfolded,
            self.input_scale,
            self.input_zero_point,
            self.input_qmin,
            self.input_qmax,
        )
        out_per_group = self.out_channels // self.groups
        w_grouped = _reshape_weight(self.weight_int, self.groups)
        bias_grouped = self.bias_int.view(self.groups, out_per_group)
        rescale_m = self.rescale_multiplier.view(self.groups, out_per_group)
        rescale_s = self.rescale_rshift.view(self.groups, out_per_group)

        # Dynamic energy flows through the ProfiledModule side channel
        # from every physical leaf; the operator does not log here.
        output_int_float = run_matmul_pipeline(
            unfolded_int,
            w_grouped,
            bias_grouped,
            rescale_m,
            rescale_s,
            self.output_zero_point,
            self.output_scale,
            self.output_qmin,
            self.output_qmax,
            self.macro,
        )
        # ``output_int_float`` is already dequantized floats; fold back to conv layout.
        output = _fold_output(output_int_float, self.out_channels, batch_shape, out_h, out_w)
        return output
