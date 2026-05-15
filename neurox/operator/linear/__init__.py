"""Linear operators: inference ``QuantLinear``, training ``HATLinear``,
and their shared int-domain math.
"""

from ._shared import derive_layer_int_params, run_matmul_pipeline
from .hat_linear import HATLinear
from .quant_linear import QuantLinear

__all__ = [
    "HATLinear",
    "QuantLinear",
    "derive_layer_int_params",
    "run_matmul_pipeline",
]
