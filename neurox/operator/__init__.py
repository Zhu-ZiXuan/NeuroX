"""Crossbar-backed PyTorch operators."""

from ._shape import w_logical_shape_for
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
    "w_logical_shape_for",
]
