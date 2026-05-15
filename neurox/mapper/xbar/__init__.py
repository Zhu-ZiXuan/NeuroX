"""Xbar-macro mapping (tiler + slicer + mapper).

Three sub-layers:

* :mod:`neurox.mapper.xbar.slicer` — value-domain decomposers
  (:class:`SerialSlicer` for activation, :class:`SimpleSlicer`
  for weight).  Output: ``[..., slice_num, digit_num]``.
* :mod:`neurox.mapper.xbar.tiler` — matrix tiling primitives
  (:class:`SimpleTiler`).  Operates on slicer outputs.
* :class:`XbarMapper` abstract base + :class:`SimpleMapper`
  concrete impl — combine a tiler + two slicers into the
  macro-facing mapping surface (``map_x`` / ``map_w`` /
  ``x_value_range`` / ``w_value_range``).

Result dataclasses :class:`XMappingResult` / :class:`WMappingResult`
carry the mapped tensor plus the positional weights and tile
geometry the macro reads.

The mapper sub-tree is pure-tool: no buffers, no fabricated
state, no profiler participation, no :class:`torch.nn.Module`
inheritance.  Static strategy parameters live on the relevant
sub-component; runtime xbar capabilities flow in as explicit
keyword arguments at every call.

User-facing imports flow through this subpackage:
``from neurox.mapper.xbar import <name>``.
"""

from .base import WMappingResult, XbarMapper, XMappingResult
from .simple_mapper import SimpleMapper
from .slicer import SerialSlicer, SimpleSlicer, Slicer, SlicingResult
from .tiler import SimpleTiler, TilePlan, Tiler

__all__ = [
    "SerialSlicer",
    "SimpleMapper",
    "SimpleSlicer",
    "SimpleTiler",
    "Slicer",
    "SlicingResult",
    "TilePlan",
    "Tiler",
    "WMappingResult",
    "XMappingResult",
    "XbarMapper",
]
