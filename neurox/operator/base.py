"""Base class for quantized NeuroX operators.

``NeuroxOperator`` is the eval-only base for the crossbar-backed
replacements of ``nn.Linear`` / ``nn.Conv2d``.  Concrete subclasses
hold integer weight buffers and quantization parameters populated by
``load_state_dict`` from a pre-computed NeuroX-flat state_dict (see
``neurox.loader.pt2e.pt2e_to_neurox_state``).  NeuroX does not run QAT
itself — the quantization-aware training step is handled by the user's
own pipeline (e.g. ``torchao.quantization.pt2e``).

``forward`` quantizes float input to int32, delegates to
``macro.matmul`` for crossbar simulation, and dequantizes the int32
output back to float.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from neurox.macro import NeuroxMacroQuantMatMul

from .spec import QuantSpec


def weight_dtype_for_range(w_min: int, w_max: int) -> torch.dtype:
    """Pick the smallest signed integer dtype that covers ``[w_min, w_max]``.

    The integer dtype is the encoding carrier for the algorithm-side
    weight range published by the macro (``w_value_range``).  Range is
    the source of truth; the dtype is purely derived from it.
    """
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

        Called from each HAT operator's ``__init__`` so range mismatches
        surface at model-build time, never deep in a training forward.
        Checks two contracts:

        1. Symmetric weight bound ``[-w_qmax, +w_qmax]`` must fit inside
           ``macro.w_value_range``.
        2. Activation grid ``[x_qmin, x_qmax]`` must fit inside
           ``macro.x_value_range``.

        Output grid ``[y_qmin, y_qmax]`` is intentionally not checked —
        it is the operator-side boundary between layers and need not
        equal the macro's input grid.

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
                f"QuantSpec x-range [{spec.x_qmin}, {spec.x_qmax}] "
                f"exceeds macro.x_value_range {macro.x_value_range}"
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

        Accepts ``qmin`` / ``qmax`` as either Python ints or 0-d int tensors.
        Callers in compiled regions should pass the tensor buffers directly
        (via ``self.input_qmin``, not ``self.input_qmin.item()``) to avoid
        an implicit CPU sync inside the compiled graph.
        """
        return torch.clamp(torch.round(input / scale + zero_point.float()), qmin, qmax).to(torch.int32)

    @staticmethod
    def dequantize_output(output_int: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        """Dequantize int output to float using the loaded output scale/zp."""
        return (output_int.float() - zero_point.float()) * scale
