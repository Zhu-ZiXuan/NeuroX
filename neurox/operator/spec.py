"""Quantization-grid specification shared by the HAT operators."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantSpec:
    """Uniform quantization grid for one HAT layer.

    Attributes:
        x_qmin: Minimum integer code for input activations.
        x_qmax: Maximum integer code for input activations.
        w_qmax: Symmetric positive integer bound for weights. The weight range
            is `[-w_qmax, +w_qmax]`.
        y_qmin: Minimum integer code for outputs.
        y_qmax: Maximum integer code for outputs.
    """

    x_qmin: int = 0
    x_qmax: int = 15
    w_qmax: int = 63
    y_qmin: int = 0
    y_qmax: int = 15
