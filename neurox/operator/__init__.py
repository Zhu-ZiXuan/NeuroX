"""Crossbar-backed PyTorch operators.

Organised by operator type:

- :mod:`neurox.operator.linear` — :class:`QuantLinear` (inference) and
  :class:`HATLinear` (training), plus their shared int-domain math
  (``run_matmul_pipeline``, ``derive_layer_int_params``).
- :mod:`neurox.operator.conv` — :class:`QuantConv2d` (inference) and
  :class:`HATConv2d` (training), sharing one set of im2col helpers.
- :mod:`neurox.operator.train` — training-only utilities: observers,
  fake-quant STE helpers, BN folding, and the HAT pipeline functions
  (``replace_for_hat`` / ``freeze_hat_observers`` /
  ``extract_neurox_state``).

The common base / spec / fixed-point helpers stay at this package's
top level (``base.py``, ``spec.py``, ``qat_util.py``).
"""

from .base import NeuroxOperator
from .conv import HATConv2d, QuantConv2d
from .linear import HATLinear, QuantLinear
from .spec import QuantSpec

__all__ = [
    "HATConv2d",
    "HATLinear",
    "NeuroxOperator",
    "QuantConv2d",
    "QuantLinear",
    "QuantSpec",
]
