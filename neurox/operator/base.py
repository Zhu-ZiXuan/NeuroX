"""Base class for quantized NeuroX operators."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.quant import stochastic_floor_div
from neurox.macro import NeuroxMacroQuantMatMul

from .spec import QuantSpec


def weight_dtype_for_range(w_min: int, w_max: int) -> torch.dtype:
    """Pick the smallest signed integer dtype that covers ``[w_min, w_max]``."""
    if w_min >= -128 and w_max <= 127:
        return torch.int8
    if w_min >= -32768 and w_max <= 32767:
        return torch.int16
    return torch.int32


class NeuroxOperator(nn.Module, ABC):
    """Base class for crossbar-backed inference-only operators.

    Subclasses hold integer weight buffers and quantization parameters
    (scales, zero-points, pre-derived fixed-point rescale coefficients)
    populated by ``load_state_dict`` from a NeuroX-flat state_dict.

    Attributes:
        name: Qualified module name used for profiler event labelling.
        macro: Crossbar macro that owns physical state and executes matmul.
    """

    name: str
    macro: NeuroxMacroQuantMatMul

    @staticmethod
    def assert_integer_tensor(tensor: Tensor, tensor_name: str) -> None:
        """Validate that one tensor uses an integer dtype."""
        if tensor.dtype not in {
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.uint16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError(f"Expected integer {tensor_name}, got {tensor.dtype}.")

    @staticmethod
    def weight_dtype_for_range(w_min: int, w_max: int) -> torch.dtype:
        """Pick the smallest signed integer dtype that covers ``[w_min, w_max]``."""
        return weight_dtype_for_range(w_min, w_max)

    @staticmethod
    def validate_spec_against_macro(
        *,
        macro: NeuroxMacroQuantMatMul,
        spec: QuantSpec,
    ) -> None:
        """Fail fast if ``spec`` cannot be mapped onto ``macro``'s grids.

        Checks that ``[-w_qmax, +w_qmax]`` fits inside ``macro.w_value_range``
        and ``[x_qmin, x_qmax]`` fits inside ``macro.x_value_range``. Output
        grid is not checked.

        Raises:
            ValueError: If either weight or activation range exceeds
                what ``macro`` accepts.
        """
        w_lo, w_hi = macro.w_value_range
        if -spec.w_qmax < w_lo or spec.w_qmax > w_hi:
            raise ValueError(
                f"QuantSpec.w_qmax ({spec.w_qmax}) yields weight range "
                f"[-{spec.w_qmax}, +{spec.w_qmax}] which exceeds "
                f"macro.w_value_range {macro.w_value_range}"
            )
        x_lo, x_hi = macro.x_value_range
        if spec.x_qmin < x_lo or spec.x_qmax > x_hi:
            raise ValueError(
                f"QuantSpec x-range [{spec.x_qmin}, {spec.x_qmax}] exceeds macro.x_value_range {macro.x_value_range}"
            )

    @abstractmethod
    def fabricate(self) -> None:
        """Re-sample the macro's static manufacturing variation.

        Drives ``self.macro.fabricate()`` (FabricateMixin auto-cascade). Called
        once at inference setup and once before each forward in noise-aware
        training.
        """
        raise NotImplementedError

    @abstractmethod
    def program(self) -> None:
        """Write the macro's static weight state from the loaded int weights.

        Drives ``self.macro.program(weight_int)``. Called once at inference
        setup and once after every weight update in QAT.
        """
        raise NotImplementedError

    def _validate_weight_range(self, weight_int: Tensor) -> None:
        """Validate that weight values are within the macro's supported range."""
        w_min, w_max = self.macro.w_value_range
        actual_min = weight_int.min().item()
        actual_max = weight_int.max().item()
        if actual_min < w_min or actual_max > w_max:
            raise ValueError(f"weight_int values [{actual_min}, {actual_max}] exceed macro range [{w_min}, {w_max}].")

    def quantize_input(
        self,
        input: Tensor,
        scale: Tensor,
        zero_point: Tensor,
        qmin: int | Tensor,
        qmax: int | Tensor,
    ) -> Tensor:
        """Quantize float input to int32 using the loaded activation scale/zp.

        Pass tensor buffers (not ``.item()``) when inside a compiled region.
        """
        return torch.clamp(torch.round(input / scale + zero_point.float()), qmin, qmax).to(torch.int32)

    @staticmethod
    def dequantize_output(output_int: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        """Dequantize int output to float using the loaded output scale/zp."""
        return (output_int.float() - zero_point.float()) * scale

    def run_matmul_pipeline(
        self,
        input_int: Tensor,
        bias_int: Tensor,
        rescale_multiplier: Tensor,
        rescale_rshift: Tensor,
        output_zero_point: Tensor,
        output_scale: Tensor,
        output_qmin: int | Tensor,
        output_qmax: int | Tensor,
    ) -> Tensor:
        """Run macro matmul → bias add → fixed-point requantize → clamp → float dequant.

        The macro returns the pre-requantize integer output. The bias is added
        on the pre-rescale grid; the multiply-shift then applies the per-channel
        rescale, adds the output zero-point, and clamps to the activation grid
        before dequantization. Stochastic rounding tracks ``self.training``.

        Args:
            input_int: Integer activation tensor already quantized to the grid.
            bias_int: Folded int32 bias (``round(b / (sx·sw)) - zp_x · sum(w_int)``).
            rescale_multiplier: Per-channel int32 fixed-point multiplier.
            rescale_rshift: Per-channel int32 right-shift.
            output_zero_point: Scalar int32 output zero-point.
            output_scale: Scalar float32 output scale.
            output_qmin: Post-rescale clamp lower bound. Pass a tensor inside a
                ``@torch.compile`` region to avoid CPU-sync.
            output_qmax: Post-rescale clamp upper bound.

        Returns:
            Dequantized float output.
        """
        y_int = self.macro.matmul(input_int)
        y_int = y_int + bias_int.to(torch.int32).unsqueeze(-2)
        y_int = stochastic_floor_div(
            y_int.to(torch.int32) * rescale_multiplier.to(torch.int32).unsqueeze(-2),
            rescale_rshift.to(torch.int32).unsqueeze(-2),
            training=self.training,
        )
        y_int = y_int + output_zero_point.to(torch.int32)
        # Match pt2e Q/DQ saturation at the layer boundary.
        y_int = torch.clamp(y_int, output_qmin, output_qmax)
        return (y_int.float() - output_zero_point.float()) * output_scale
