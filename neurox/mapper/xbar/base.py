"""Abstract base for the unified xbar mapper.

The mapper is the **owner of complete mapping semantics**: it
internally composes a shared :class:`~neurox.mapper.xbar.tiler.Tiler`
plus an activation-side :class:`~neurox.mapper.xbar.slicer.Slicer`
and a weight-side :class:`~neurox.mapper.xbar.slicer.Slicer`, and
turns algorithm-side tensors into the xbar primitive's
broadcast-ready layouts.

Decoupling contract
-------------------
* **mapper** owns its tiler + slicer instances and any static
  strategy parameters those carry.  It does *not* hold a
  reference to the xbar and does *not* snapshot any xbar
  capability at construction.
* **macro** owns the xbar and the mapper.  Every mapper call
  receives the **full** xbar capability set as explicit keyword
  arguments — never a subset.  A concrete mapper is free to
  ignore parameters it does not consume, but the macro must
  not pre-select which parameters are passed.  This keeps the
  macro completely uncoupled from any specific mapper's
  internal needs: changing the mapper does not change the macro.

The full capability set is split into two groups by semantic
purpose (value-domain vs. geometry):

* ``x_range: tuple[int, int]`` — xbar input grid (value-domain).
* ``w_digit_count: int`` — digit slots per xbar-word
  (value-domain).
* ``w_digit_radix: int`` — radix per xbar-internal digit
  (value-domain).
* ``w_digit_range: tuple[int, int]`` — one digit's physical range
  (value-domain).
* ``col_num: int`` — xbar tile output-side geometry; equals the
  primitive weight tail's ``data_num`` slot count (geometry).
* ``row_num: int`` — xbar tile input-side geometry; size of the
  primitive's ``[..., row_num]`` activation tail (geometry).

``col_num`` / ``row_num`` size the xbar tile, not its value grid;
they are *not* value-domain capabilities.

Shape contract
--------------
Full layouts (macro-side, used for broadcast):

* weight ``w_xbar``:
  ``[Bw, M=1, Tc, Tr, Sa=1, Sw, data_num, digit_num, row_num]``
* activation ``x_xbar``:
  ``[Bx, M, Tc, Tr=1, Sa, Sw=1, row_num]``

Structural singleton broadcast axes — kept so the weight and
activation paths share one broadcast contract:

* weight: ``M = 1`` (no per-sample weight state), ``Sa = 1``
  (weights don't depend on activation digit index).
* activation: ``Tr = 1`` (activation doesn't depend on output
  tile), ``Sw = 1`` (no macro-external activation slicing).

The shorthand axis labels ``Bw / Bx / M / Tc / Tr / Sa / Sw`` are
shape-annotation only.  Variable-name counterparts are
``x_slice_num``, ``w_slice_num``, ``col_tile_num``,
``row_tile_num``.

The xbar primitive contracts (see :class:`neurox.xbar.Xbar`) ride
on the trailing dims of these full layouts:

* weight primitive trailing: ``[data_num, digit_num, row_num]`` —
  consumed by :meth:`~neurox.xbar.Xbar.fabricate`.
* activation primitive trailing: ``[row_num]`` — consumed by
  :meth:`~neurox.xbar.Xbar.vec_mat_mul`.

Range terminology
-----------------
The *xbar* layer uses ``x_range`` / ``w_digit_range`` for its
primitive grids.  Everything above the xbar (mapper / slicer /
macro) uses ``value_range`` to talk about the algorithm-side
representable range; e.g. :meth:`XbarMapper.x_value_range` and
:meth:`XbarMapper.w_value_range`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class XMappingResult:
    """Result of :meth:`XbarMapper.map_x`.

    Attributes:
        x_xbar: Tile-mapped activation digit tensor with full
            macro-side shape ``[Bx, M, Tc, Tr=1, Sa, Sw=1, row_num]``.
            ``Tr = 1`` and ``Sw = 1`` are structural singletons.
            The xbar primitive consumes only the trailing
            ``[..., row_num]``; macro-side axes are broadcast batch
            from the xbar's point of view.  Each entry already lies
            in the xbar's input grid.
        slice_weights: Length-``Sa`` int tensor of per-digit
            positional weights (LSB first) ``[1, r, r^2, ...]``.
            Same device / dtype as ``x_xbar``.  Drives the macro's
            ``x_shift_adder`` reduction.
        logical_batch_shape: Leading batch shape ``Bx`` (everything
            before ``M``) of the original activation tensor.
    """

    x_xbar: Tensor
    slice_weights: Tensor
    logical_batch_shape: tuple[int, ...]


@dataclass(frozen=True)
class WMappingResult:
    """Result of :meth:`XbarMapper.map_w`.

    Attributes:
        w_xbar: Xbar-native integer digit tensor consumed by
            :meth:`neurox.xbar.Xbar.fabricate`.  Full macro-side
            shape ``[Bw, M=1, Tc, Tr, Sa=1, Sw, data_num, digit_num,
            row_num]``; the xbar primitive reads only the trailing
            ``[..., data_num, digit_num, row_num]`` and treats every
            preceding axis as a broadcast batch dim.  ``M = 1`` and
            ``Sa = 1`` are structural singletons.
        slice_weights: Per-slice combination weights for the macro-
            external ``Sw`` shift-add reduction.  Length-``Sw`` int
            tensor of positional weights
            ``[1, w_slice_radix, w_slice_radix^2, ...]``.
        logical_out_dim: Original ``N`` (output channel count)
            before tile padding.
        row_tile_num: Number of output tiles along ``N``
            (``Tr`` in shape annotations).
        col_tile_num: Number of input tiles along ``K``
            (``Tc`` in shape annotations).
    """

    w_xbar: Tensor
    slice_weights: Tensor
    logical_out_dim: int
    row_tile_num: int
    col_tile_num: int


class XbarMapper(ABC):
    """Abstract mapper — owner of complete mapping semantics.

    Every runtime method takes the same full xbar capability set
    via keyword arguments (``x_range``, ``col_num``, ``row_num``,
    ``w_digit_count``, ``w_digit_radix``, ``w_digit_range``).  A
    concrete mapper ignores the parameters it does not consume; the
    macro never pre-selects a subset.  This contract decouples the
    macro from any specific mapper's needs and lets a different
    mapper drop in without macro-side changes.
    """

    @abstractmethod
    def x_value_range(
        self,
        *,
        x_range: tuple[int, int],
        col_num: int,
        row_num: int,
        w_digit_count: int,
        w_digit_radix: int,
        w_digit_range: tuple[int, int],
    ) -> tuple[int, int]:
        """Inclusive algorithm-side integer activation range."""
        raise NotImplementedError

    @abstractmethod
    def w_value_range(
        self,
        *,
        x_range: tuple[int, int],
        col_num: int,
        row_num: int,
        w_digit_count: int,
        w_digit_radix: int,
        w_digit_range: tuple[int, int],
    ) -> tuple[int, int]:
        """Inclusive algorithm-side integer weight range."""
        raise NotImplementedError

    @abstractmethod
    def x_slice_radix(
        self,
        *,
        x_range: tuple[int, int],
        col_num: int,
        row_num: int,
        w_digit_count: int,
        w_digit_radix: int,
        w_digit_range: tuple[int, int],
    ) -> int:
        """Per-digit positional radix; drives the ``x_shift_adder`` reduction."""
        raise NotImplementedError

    @abstractmethod
    def w_slice_radix(
        self,
        *,
        x_range: tuple[int, int],
        col_num: int,
        row_num: int,
        w_digit_count: int,
        w_digit_radix: int,
        w_digit_range: tuple[int, int],
    ) -> int:
        """Per-slice positional radix; drives the ``w_shift_adder`` reduction."""
        raise NotImplementedError

    @abstractmethod
    def map_x(
        self,
        x: Tensor,
        *,
        x_range: tuple[int, int],
        col_num: int,
        row_num: int,
        w_digit_count: int,
        w_digit_radix: int,
        w_digit_range: tuple[int, int],
    ) -> XMappingResult:
        """Map an algorithm-side activation into the xbar input layout."""
        raise NotImplementedError

    @abstractmethod
    def map_w(
        self,
        w: Tensor,
        *,
        x_range: tuple[int, int],
        col_num: int,
        row_num: int,
        w_digit_count: int,
        w_digit_radix: int,
        w_digit_range: tuple[int, int],
    ) -> WMappingResult:
        """Map an algorithm-side weight into xbar-native tile + digit form."""
        raise NotImplementedError
