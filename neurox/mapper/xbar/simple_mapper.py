"""Concrete :class:`XbarMapper` built from a tiler and two slicers."""

from __future__ import annotations

from torch import Tensor

from .base import WMappingResult, XbarMapper, XMappingResult
from .slicer.base import Slicer
from .tiler.base import Tiler


class SimpleMapper(XbarMapper):
    """Concrete mapper composing a shared tiler + activation / weight slicers.

    Args:
        tiler: Matrix-tiling primitive shared by the x and w paths.
        x_slicer: Activation-path value decomposer.
        w_slicer: Weight-path value decomposer.
    """

    def __init__(
        self,
        *,
        tiler: Tiler,
        x_slicer: Slicer,
        w_slicer: Slicer,
    ) -> None:
        self._tiler = tiler
        self._x_slicer = x_slicer
        self._w_slicer = w_slicer

    @property
    def tiler(self) -> Tiler:
        return self._tiler

    @property
    def x_slicer(self) -> Slicer:
        return self._x_slicer

    @property
    def w_slicer(self) -> Slicer:
        return self._w_slicer

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
        del col_num, row_num, w_digit_count, w_digit_radix, w_digit_range
        x_lo, x_hi = x_range
        return self._x_slicer.value_range(
            digit_count=1,
            digit_radix=x_hi - x_lo + 1,
            digit_range=(x_lo, x_hi),
        )

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
        del x_range, col_num, row_num
        return self._w_slicer.value_range(
            digit_count=w_digit_count,
            digit_radix=w_digit_radix,
            digit_range=w_digit_range,
        )

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
        del col_num, row_num, w_digit_count, w_digit_radix, w_digit_range
        x_lo, x_hi = x_range
        return self._x_slicer.slice_radix(
            digit_count=1,
            digit_radix=x_hi - x_lo + 1,
            digit_range=(x_lo, x_hi),
        )

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
        del x_range, col_num, row_num
        return self._w_slicer.slice_radix(
            digit_count=w_digit_count,
            digit_radix=w_digit_radix,
            digit_range=w_digit_range,
        )

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
        del col_num, w_digit_count, w_digit_radix, w_digit_range
        leading_batch = tuple(x.shape[:-2])
        K = x.shape[-1]
        plan = self._tiler.make_x_plan(k=K, row_num=row_num)

        x_lo, x_hi = x_range
        # Shape: [Bx, M, K] -> [Bx, M, K, Sa, digit_num=1]
        sliced = self._x_slicer.slice(
            x,
            digit_count=1,
            digit_radix=x_hi - x_lo + 1,
            digit_range=(x_lo, x_hi),
        )
        # Shape: [Bx, M, K, Sa, 1] -> [Bx, M, Tc, row_num, Sa, 1]
        tiled = self._tiler.tile_x(sliced.values, plan=plan)

        # Shape: [Bx, M, Tc, row_num, Sa, 1] -> [Bx, M, Tc, row_num, Sa]
        squeezed = tiled.squeeze(-1)
        # Shape: [Bx, M, Tc, row_num, Sa] -> [Bx, M, Tc, Sa, row_num]
        transposed = squeezed.transpose(-2, -1)
        # Shape: [Bx, M, Tc, Sa, row_num] -> [Bx, M, Tc, Sa, Sw=1, row_num]
        with_sw = transposed.unsqueeze(-2)
        # Shape: [Bx, M, Tc, Sa, Sw=1, row_num] -> [Bx, M, Tc, Tr=1, Sa, Sw=1, row_num]
        x_xbar = with_sw.unsqueeze(-4)
        return XMappingResult(
            x_xbar=x_xbar,
            slice_weights=sliced.slice_weights,
            logical_batch_shape=leading_batch,
        )

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
        del x_range
        N = w.shape[-2]
        K = w.shape[-1]
        plan = self._tiler.make_w_plan(
            n=N,
            k=K,
            col_num=col_num,
            row_num=row_num,
        )

        # Shape: [Bw, N, K] -> [Bw, N, K, Sw, D]
        sliced = self._w_slicer.slice(
            w,
            digit_count=w_digit_count,
            digit_radix=w_digit_radix,
            digit_range=w_digit_range,
        )
        # Shape: [Bw, N, K, Sw, D] -> [Bw, Tr, data_num, Tc, row_num, Sw, D]
        tiled = self._tiler.tile_w(sliced.values, plan=plan)

        # Permute trailing-6 (Tr, data_num, Tc, row_num, Sw, D) to
        # (Tc, Tr, Sw, data_num, D, row_num).  Leading batch dims
        # stay in place.
        B = tiled.ndim - 6
        perm = list(range(B)) + [B + 2, B + 0, B + 4, B + 1, B + 5, B + 3]
        # Shape: [Bw, Tr, data_num, Tc, row_num, Sw, D] -> [Bw, Tc, Tr, Sw, data_num, D, row_num]
        arranged = tiled.permute(perm)
        # Shape: [Bw, Tc, Tr, Sw, data_num, D, row_num] -> [Bw, Tc, Tr, Sa=1, Sw, data_num, D, row_num]
        arranged = arranged.unsqueeze(B + 2)
        # Shape: [Bw, Tc, Tr, Sa=1, Sw, data_num, D, row_num] -> [Bw, M=1, Tc, Tr, Sa=1, Sw, data_num, D, row_num]
        w_xbar = arranged.unsqueeze(B)

        return WMappingResult(
            w_xbar=w_xbar,
            slice_weights=sliced.slice_weights,
            logical_out_dim=N,
            row_tile_num=plan.row_tile_num,
            col_tile_num=plan.col_tile_num,
        )
