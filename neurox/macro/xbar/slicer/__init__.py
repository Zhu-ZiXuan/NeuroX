"""Value-domain slicers for the xbar macro.

See also:
    docs/internals/macro/xbar/slicer/README.md
"""

from .base import Slicer
from .serial import SerialSlicer
from .simple import SimpleSlicer

__all__ = [
    "SerialSlicer",
    "SimpleSlicer",
    "Slicer",
]
