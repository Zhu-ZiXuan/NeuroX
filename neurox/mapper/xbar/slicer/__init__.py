"""Value-domain slicers for the xbar mapper."""

from .base import Slicer
from .serial import SerialSlicer
from .simple import SimpleSlicer

__all__ = [
    "SerialSlicer",
    "SimpleSlicer",
    "Slicer",
]
