"""Matrix tilers for the xbar mapper.

A :class:`Tiler` chops a logical ``[N, K]`` weight matrix or
``[M, K]`` activation matrix into xbar tiles of size
``(data_num, row_num)``.  Value-domain decomposition is the
slicer's job; the tiler ignores the slicer's trailing
``[slice_num, digit_num]`` dims.

Concrete impl:

* :class:`SimpleTiler` — right-pad-and-unflatten.
"""

from .base import TilePlan, Tiler
from .simple import SimpleTiler

__all__ = [
    "SimpleTiler",
    "TilePlan",
    "Tiler",
]
