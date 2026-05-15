"""Tiler ABC + :class:`TilePlan`.

A :class:`Tiler` is the *matrix-tiling* primitive in the mapping
pipeline.  Its job is to chop an algorithm-side weight matrix
``[N, K]`` into xbar tiles of shape ``[data_num, row_num]`` and
chop an algorithm-side activation matrix ``[M, K]`` along the
shared ``K`` axis.  It does **not** do any value-domain
decomposition — that is the slicer's job (see
:mod:`neurox.mapper.xbar.slicer`).

The tiler is stateless and topology-agnostic.  Weight and
activation tile geometries are computed by two distinct factory
methods (:meth:`make_w_plan`, :meth:`make_x_plan`) so neither path
needs to pass placeholder values such as ``n = 0`` for the other.

Slicer / tiler composition
--------------------------
The tiler operates on tensors whose trailing-2 dims are
``[slice_num, digit_num]`` (the slicer's uniform output shape).
The tile-axes (``N`` / ``K``) sit at axis -4 / -3 for weight or
axis -3 for activation.  The tiler ignores the trailing slice /
digit dims; the mapper handles their final placement in the macro
canonical layout afterwards.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class TilePlan:
    """Geometry of how a logical matrix splits into xbar tiles.

    Weight plans (from :meth:`Tiler.make_w_plan`) carry the full
    ``N / K`` geometry.  Activation plans (from
    :meth:`Tiler.make_x_plan`) carry only the ``K`` geometry and
    expose ``row_tile_num = 0``, ``data_num = 0``, ``n_pad = 0``,
    ``logical_out_dim = 0``; consumers MUST NOT read those fields
    on activation plans.

    Attributes:
        logical_out_dim: Original ``N`` (output channels) before
            tile padding (weight plans only; ``0`` for activation
            plans).
        logical_in_dim: Original ``K`` (input channels) before
            tile padding.
        row_tile_num: Number of output tiles along ``N``
            (``Tr`` in shape annotations).  ``0`` for activation
            plans.
        col_tile_num: Number of input tiles along ``K``
            (``Tc`` in shape annotations).
        data_num: Output-direction tile size (= xbar's
            ``col_num``).  ``0`` for activation plans.
        row_num: Input-direction tile size (= xbar's ``row_num``).
        n_pad: Right pad on ``N`` to reach ``row_tile_num * data_num``.
            ``0`` for activation plans.
        k_pad: Right pad on ``K`` to reach ``col_tile_num * row_num``.
    """

    logical_out_dim: int
    logical_in_dim: int
    row_tile_num: int
    col_tile_num: int
    data_num: int
    row_num: int
    n_pad: int
    k_pad: int


class Tiler(ABC):
    """Abstract matrix tiling primitive.

    Stateless tool.  Concrete subclasses define how a logical
    ``[N, K]`` / ``[M, K]`` matrix lines up with the xbar tile
    geometry.  The two plan factories
    (:meth:`make_w_plan` / :meth:`make_x_plan`) are kept separate
    so neither path leaks placeholder geometry into the other.
    """

    @abstractmethod
    def make_w_plan(
        self,
        *,
        n: int,
        k: int,
        col_num: int,
        row_num: int,
    ) -> TilePlan:
        """Compute weight-side tile geometry for an ``[N, K]`` matrix."""
        raise NotImplementedError

    @abstractmethod
    def make_x_plan(
        self,
        *,
        k: int,
        row_num: int,
    ) -> TilePlan:
        """Compute activation-side tile geometry for the shared ``K``.

        Activation plans only describe how ``K`` splits across
        xbar input tiles; the output direction (``N``) is not part
        of the activation geometry.  The returned ``TilePlan``
        sets every ``N``-side field to ``0``.
        """
        raise NotImplementedError

    @abstractmethod
    def tile_w(self, w: Tensor, *, plan: TilePlan) -> Tensor:
        """Split ``w`` along ``N`` and ``K`` using ``plan``.

        Args:
            w: Tensor of shape ``[..., N, K, slice_num, digit_num]``.
            plan: Weight plan from :meth:`make_w_plan`.

        Returns:
            Tensor of shape
            ``[..., row_tile_num, data_num, col_tile_num, row_num,
            slice_num, digit_num]``.
        """
        raise NotImplementedError

    @abstractmethod
    def tile_x(self, x: Tensor, *, plan: TilePlan) -> Tensor:
        """Split ``x`` along ``K`` using ``plan``.

        Args:
            x: Tensor of shape ``[..., M, K, slice_num, digit_num]``.
            plan: Activation plan from :meth:`make_x_plan`.

        Returns:
            Tensor of shape ``[..., M, col_tile_num, row_num,
            slice_num, digit_num]``.
        """
        raise NotImplementedError
