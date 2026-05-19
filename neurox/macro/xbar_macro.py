"""Logical crossbar macro bridging a physical xbar tile and matrix ops."""

import math

import torch
import torch.nn as nn
from torch import Tensor

from neurox.digital import Accumulator, Requantizer, ShiftAdder
from neurox.mapper.xbar import XbarMapper
from neurox.xbar import Xbar


class XbarMacro(nn.Module):
    """Logical macro built from an xbar, a mapper, and digital reducers."""

    def __init__(
        self,
        *,
        xbar: Xbar,
        mapper: XbarMapper,
        col_accumulator: Accumulator,
        w_shift_adder: ShiftAdder,
        x_shift_adder: ShiftAdder,
        requantizer: Requantizer,
    ) -> None:
        """Construct one xbar-backed macro.

        Args:
            xbar: Physical xbar tile model.
            mapper: Logical-to-physical mapper.
            col_accumulator: Tile-column reducer.
            w_shift_adder: Weight-slice reducer.
            x_shift_adder: Activation-slice reducer.
            requantizer: Output requantizer.
        """
        super().__init__()

        self.xbar = xbar
        self.mapper = mapper

        self.col_accumulator = col_accumulator
        self.w_shift_adder = w_shift_adder
        self.x_shift_adder = x_shift_adder
        self.requantizer = requantizer

        self._serial_op_num: int = 0
        self._w_parallel_size: int = 0
        self._x_shape_cached: tuple[int, ...] = ()

    def extra_repr(self) -> str:
        """One-line summary of this macro shown by ``print(model)``."""
        return (
            f"xbar={type(self.xbar).__name__}, "
            f"row_num={self.xbar.row_num}, col_num={self.xbar.col_num}, "
            f"w_value_range={self.w_value_range}, x_value_range={self.x_value_range}"
        )

    def __repr__(self) -> str:
        """Compact repr that hides macro internals from ``print(model)``."""
        return f"{type(self).__name__}({self.extra_repr()})"

    @property
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive algorithm-side integer weight range — forwards to the mapper."""
        return self.mapper.w_value_range(
            x_range=self.xbar.x_range,
            col_num=self.xbar.col_num,
            row_num=self.xbar.row_num,
            w_digit_count=self.xbar.w_digit_count,
            w_digit_radix=self.xbar.w_digit_radix,
            w_digit_range=self.xbar.w_digit_range,
        )

    @property
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive algorithm-side integer activation range — forwards to the mapper."""
        return self.mapper.x_value_range(
            x_range=self.xbar.x_range,
            col_num=self.xbar.col_num,
            row_num=self.xbar.row_num,
            w_digit_count=self.xbar.w_digit_count,
            w_digit_radix=self.xbar.w_digit_radix,
            w_digit_range=self.xbar.w_digit_range,
        )

    @property
    def output_rescale_factor(self) -> float:
        """Ratio of ideal integer partial-product max to actual tile output max."""
        return self.xbar.output_rescale_factor

    def fabricate(self, weight: Tensor) -> None:
        """Map one weight tensor and program the xbar.

        Args:
            weight: Integer weight tensor. Shape: [..., N, K].
        """
        mapped = self.mapper.map_w(
            weight,
            x_range=self.xbar.x_range,
            col_num=self.xbar.col_num,
            row_num=self.xbar.row_num,
            w_digit_count=self.xbar.w_digit_count,
            w_digit_radix=self.xbar.w_digit_radix,
            w_digit_range=self.xbar.w_digit_range,
        )
        self.xbar.fabricate(mapped.w_xbar)

        # Strip the trailing ``data_num, digit_num, row_num`` axes.
        *w_batch, _M, _col_tile_num, row_tile_num, _Sa, w_slice_num = mapped.w_xbar.shape[:-3]

        self._w_parallel_size = math.prod(w_batch)
        N = mapped.logical_out_dim
        self.col_accumulator.fabricate((self._w_parallel_size, w_slice_num, row_tile_num))
        self.w_shift_adder.fabricate((self._w_parallel_size, row_tile_num))
        self.x_shift_adder.fabricate((self._w_parallel_size, row_tile_num))
        self.requantizer.fabricate((self._w_parallel_size, N))

    def _eval_tiles(
        self,
        x: Tensor,
        N: int,
        bias: Tensor | None,
        x_slice_radix: int,
        w_slice_radix: int,
    ) -> Tensor:
        """Run the xbar and digital reduction pipeline.

        Args:
            x: Mapped activation tensor in xbar-native layout.
            N: Logical output dimension.
            bias: Optional integer bias tensor. Shape: [..., N].
            x_slice_radix: Activation slice radix.
            w_slice_radix: Weight slice radix.

        Returns:
            Integer partial result before requantization.
        """
        y = self.xbar.vec_mat_mul(x)
        y = self._digital_aggregate(y, N, bias, x_slice_radix, w_slice_radix)
        return y

    def _digital_aggregate(
        self,
        y: Tensor,
        N: int,
        bias: Tensor | None,
        x_slice_radix: int,
        w_slice_radix: int,
    ) -> Tensor:
        """Reduce xbar outputs across slice and tile dimensions.

        Args:
            y: Raw xbar output tensor.
            N: Logical output dimension.
            bias: Optional integer bias tensor. Shape: [..., N].
            x_slice_radix: Activation slice radix.
            w_slice_radix: Weight slice radix.

        Returns:
            Integer accumulated output tensor with shape [..., M, N].
        """
        # Shape: [..., M, Tc, Tr, Sa, Sw, data_num] -> [..., M, Tc, Tr, Sw, data_num]
        y = self.x_shift_adder.operate(y, x_slice_radix, dim=-3)
        # Shape: [..., M, Tc, Tr, Sw, data_num] -> [..., M, Tc, Tr, data_num]
        y = self.w_shift_adder.operate(y, w_slice_radix, dim=-2)
        # Shape: [..., M, Tc, Tr, data_num] -> [..., M, Tr, data_num]
        y = self.col_accumulator.operate(y, dim=-3)
        # Shape: [..., M, Tr, data_num] -> [..., M, Tr * data_num] -> [..., M, N]
        y = y.flatten(start_dim=-2)[..., :N]
        if bias is not None:
            y = y + bias
        return y

    @torch.no_grad()
    @torch.compile(dynamic=True)
    def matmul(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        rescale_multiplier: Tensor,
        rescale_rshift: Tensor,
        output_zero_point: Tensor | None,
    ) -> Tensor:
        """Run one integer matmul through the xbar macro.

        Args:
            input: Integer activation tensor. Shape: [..., M, K].
            weight: Integer weight tensor. Shape: [..., N, K].
            bias: Optional integer bias tensor. Shape: [..., N].
            rescale_multiplier: Per-output fixed-point multiplier.
            rescale_rshift: Per-output right-shift amount.
            output_zero_point: Optional output zero point.

        Returns:
            Integer output tensor with shape [..., M, N].
        """
        N = weight.shape[-2]
        x_dtype = input.dtype

        # Snapshot primitive capabilities once per call.
        x_range = self.xbar.x_range
        col_num = self.xbar.col_num
        row_num = self.xbar.row_num
        w_digit_count = self.xbar.w_digit_count
        w_digit_radix = self.xbar.w_digit_radix
        w_digit_range = self.xbar.w_digit_range

        if self.training:
            self.xbar.fabricate(
                self.mapper.map_w(
                    weight,
                    x_range=x_range,
                    col_num=col_num,
                    row_num=row_num,
                    w_digit_count=w_digit_count,
                    w_digit_radix=w_digit_radix,
                    w_digit_range=w_digit_range,
                ).w_xbar,
            )

        x_mapped = self.mapper.map_x(
            input,
            x_range=x_range,
            col_num=col_num,
            row_num=row_num,
            w_digit_count=w_digit_count,
            w_digit_radix=w_digit_radix,
            w_digit_range=w_digit_range,
        )
        x = x_mapped.x_xbar
        if not self.training and self._x_shape_cached != x.shape:
            self._x_shape_cached = x.shape
            x_slice_num = self._x_shape_cached[-4]
            batch_M_prod = math.prod(self._x_shape_cached[:-6])
            self._serial_op_num = batch_M_prod * x_slice_num // self._w_parallel_size

        # Query slice radixes once per call.
        x_slice_radix = self.mapper.x_slice_radix(
            x_range=x_range,
            col_num=col_num,
            row_num=row_num,
            w_digit_count=w_digit_count,
            w_digit_radix=w_digit_radix,
            w_digit_range=w_digit_range,
        )
        w_slice_radix = self.mapper.w_slice_radix(
            x_range=x_range,
            col_num=col_num,
            row_num=row_num,
            w_digit_count=w_digit_count,
            w_digit_radix=w_digit_radix,
            w_digit_range=w_digit_range,
        )

        y = self._eval_tiles(x, N, bias, x_slice_radix, w_slice_radix)
        y = self.requantizer.operate(y, rescale_multiplier, rescale_rshift, output_zero_point)
        return y.to(x_dtype)
