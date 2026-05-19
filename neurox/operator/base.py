"""Base class for quantized NeuroX operators."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

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
        """Fabricate the macro's physical state from the loaded int weights."""
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
