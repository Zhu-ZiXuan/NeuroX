"""Tiler ABC + :class:`TilePlan`.

See also:
    docs/dev/modules/mapper/xbar/tiler/README.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class TilePlan:
    """Geometry of how a logical matrix splits into xbar tiles.

    Activation plans set every ``N``-side field to ``0``.

    Attributes:
        logical_out_dim: Original ``N`` before tile padding.
        logical_in_dim: Original ``K`` before tile padding.
        row_tile_num: Number of output tiles along ``N``.
        col_tile_num: Number of input tiles along ``K``.
        data_num: Output-direction tile size (= xbar's ``col_num``).
        row_num: Input-direction tile size (= xbar's ``row_num``).
        n_pad: Right pad on ``N``.
        k_pad: Right pad on ``K``.
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
    """Abstract matrix tiling primitive."""

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

        ``N``-side fields are ``0`` on activation plans.
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
