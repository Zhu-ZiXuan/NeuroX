"""Value-domain slicers for the xbar mapper.

A :class:`Slicer` decomposes each scalar of an integer tensor into
``[slice_num, digit_num]`` signed digits.  Matrix tiling is
handled separately by :mod:`neurox.mapper.xbar.tiler`.

Concrete impls:

* :class:`SerialSlicer` — radix-``r`` decomposition with
  structural ``digit_num = 1`` (activation path).
* :class:`SimpleSlicer` — slice-first-then-digitize (weight path).
"""

from .base import Slicer, SlicingResult
from .serial import SerialSlicer
from .simple import SimpleSlicer

__all__ = [
    "SerialSlicer",
    "SimpleSlicer",
    "Slicer",
    "SlicingResult",
]
