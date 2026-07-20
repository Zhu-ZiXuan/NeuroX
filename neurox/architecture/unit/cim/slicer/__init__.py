"""Value-domain slicers for the xbar macro.

See also:
    docs/internals/architecture/unit/cim/slicer/README.md
"""

from .base import Slicer
from .direct import DirectSlicer
from .serial import SerialSlicer
from .simple import SimpleSlicer

__all__ = [
    "DirectSlicer",
    "SerialSlicer",
    "SimpleSlicer",
    "Slicer",
]
