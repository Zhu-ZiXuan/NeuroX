"""Protocol definitions for NeuroX macro hardware models.

This module defines the structural protocol any macro implementation
must satisfy.  Using ``Protocol`` rather than an abstract base lets
``XbarMacro`` (full hardware simulation) and ``IdealMacro`` (functional
reference) satisfy the interface without sharing inheritance.

Static PPA — area and leakage power — flows through the side-channel
profiler (``NeuroxProfiler.analyze_static``) and is owned by every
``ProfiledModule`` inside the macro.  The macro itself does **not**
expose aggregated area / leakage / latency properties; the profiler
walks the model tree to aggregate.

NeuroxMacroQuantMatMul
    Full interface required by ``neurox.operator`` layer wrappers:
    weight / activation level ranges, ``output_rescale_factor`` for
    the operator's integer-quantization fold-in, one-time
    ``fabricate``, and a per-inference ``matmul`` call.
"""

from typing import Protocol

from torch import Tensor


class NeuroxMacroQuantMatMul(Protocol):
    """Protocol for macros that perform quantized-integer matrix multiply.

    Properties:
        w_value_range: ``(min_val, max_val)`` inclusive integer range for weights.
        x_value_range: ``(min_val, max_val)`` inclusive integer range for activations.
        output_rescale_factor: ADC-code-to-ideal-integer scale folded into
            the operator's ``(rescale_multiplier, rescale_rshift, bias_int)``
            buffers by :func:`neurox.replace.replace.bind_output_calibration`.

    Methods:
        fabricate: Program physical state from an integer weight tensor.
            Called once before eval-mode inference; ignored or re-called
            every step in training mode.
        matmul: Compute an integer matmul with fixed-point requantization.
            Returns the integer output tensor.  Dynamic energy and
            latency emit through the profiler side channel.
    """

    @property
    def w_value_range(self) -> tuple[int, int]: ...

    @property
    def x_value_range(self) -> tuple[int, int]: ...

    @property
    def output_rescale_factor(self) -> float: ...

    def fabricate(self, weight: Tensor) -> None: ...

    def matmul(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        rescale_multiplier: Tensor,
        rescale_rshift: Tensor,
        output_zero_point: Tensor | None,
    ) -> Tensor: ...
