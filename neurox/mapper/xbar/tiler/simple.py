"""SimpleTiler — plain right-pad-and-unflatten N/K tiling.

Operates on tensors with the slicer's trailing-2
``[slice_num, digit_num]`` dims already appended.  Tiles ``N`` at
axis -4 and ``K`` at axis -3 (for weight), or ``K`` at axis -3
(for activation).  Padding is added on the high side only.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from .base import TilePlan, Tiler


class SimpleTiler(Tiler):
    """Right-pad-and-unflatten tiler."""

    def make_w_plan(
        self,
        *,
        n: int,
        k: int,
        col_num: int,
        row_num: int,
    ) -> TilePlan:
        if col_num < 1:
            raise ValueError(f"require: col_num ({col_num}) >= 1")
        if row_num < 1:
            raise ValueError(f"require: row_num ({row_num}) >= 1")
        if n < 1:
            raise ValueError(f"require: n ({n}) >= 1")
        if k < 1:
            raise ValueError(f"require: k ({k}) >= 1")
        row_tile_num = (n + col_num - 1) // col_num
        col_tile_num = (k + row_num - 1) // row_num
        return TilePlan(
            logical_out_dim=n,
            logical_in_dim=k,
            row_tile_num=row_tile_num,
            col_tile_num=col_tile_num,
            data_num=col_num,
            row_num=row_num,
            n_pad=row_tile_num * col_num - n,
            k_pad=col_tile_num * row_num - k,
        )

    def make_x_plan(
        self,
        *,
        k: int,
        row_num: int,
    ) -> TilePlan:
        if row_num < 1:
            raise ValueError(f"require: row_num ({row_num}) >= 1")
        if k < 1:
            raise ValueError(f"require: k ({k}) >= 1")
        col_tile_num = (k + row_num - 1) // row_num
        return TilePlan(
            logical_out_dim=0,
            logical_in_dim=k,
            row_tile_num=0,
            col_tile_num=col_tile_num,
            data_num=0,
            row_num=row_num,
            n_pad=0,
            k_pad=col_tile_num * row_num - k,
        )

    def tile_w(self, w: Tensor, *, plan: TilePlan) -> Tensor:
        """Tile ``w`` along ``N`` (axis -4) and ``K`` (axis -3).

        Args:
            w: Shape ``[..., N, K, slice_num, digit_num]``.
            plan: Weight plan from :meth:`make_w_plan`.

        Returns:
            Shape ``[..., row_tile_num, data_num, col_tile_num,
            row_num, slice_num, digit_num]``.
        """
        # ``F.pad`` indexes dims from the last; for last-4 dims we
        # supply 8 numbers in (digit_lo, digit_hi, slice_lo,
        # slice_hi, K_lo, K_hi, N_lo, N_hi) order.  We right-pad
        # ``N`` and ``K`` only.
        # Shape: [..., N, K, slice_num, digit_num] ->
        #        [..., row_tile_num*data_num, col_tile_num*row_num, slice_num, digit_num].
        padded = F.pad(w, (0, 0, 0, 0, 0, plan.k_pad, 0, plan.n_pad))
        # Shape: [..., row_tile_num*data_num, ...] ->
        #        [..., row_tile_num, data_num, ...].
        tiled = padded.unflatten(-4, (plan.row_tile_num, plan.data_num))
        # Shape: [..., row_tile_num, data_num, col_tile_num*row_num, slice_num, digit_num] ->
        #        [..., row_tile_num, data_num, col_tile_num, row_num, slice_num, digit_num].
        tiled = tiled.unflatten(-3, (plan.col_tile_num, plan.row_num))
        return tiled  # type: ignore[no-any-return]

    def tile_x(self, x: Tensor, *, plan: TilePlan) -> Tensor:
        """Tile ``x`` along ``K`` (axis -3).

        Args:
            x: Shape ``[..., M, K, slice_num, digit_num]``.
            plan: Activation plan from :meth:`make_x_plan`.

        Returns:
            Shape ``[..., M, col_tile_num, row_num, slice_num,
            digit_num]``.
        """
        # ``F.pad`` last-3 dims: (digit_lo, digit_hi, slice_lo,
        # slice_hi, K_lo, K_hi).  Pad ``K`` only.
        # Shape: [..., M, K, slice_num, digit_num] ->
        #        [..., M, col_tile_num*row_num, slice_num, digit_num].
        padded = F.pad(x, (0, 0, 0, 0, 0, plan.k_pad))
        # Shape: [..., M, col_tile_num*row_num, slice_num, digit_num] ->
        #        [..., M, col_tile_num, row_num, slice_num, digit_num].
        tiled = padded.unflatten(-3, (plan.col_tile_num, plan.row_num))
        return tiled  # type: ignore[no-any-return]
