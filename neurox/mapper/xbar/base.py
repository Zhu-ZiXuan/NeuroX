"""Abstract base for xbar mappers.

See also:
    docs/dev/modules/mapper/xbar/base.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class XMappingResult:
    """Result of :meth:`XbarMapper.map_x`.

    Attributes:
        x_xbar: Activation tensor in xbar-native layout.
        slice_weights: Positional weights for the activation slice axis.
        logical_batch_shape: Logical batch dimensions before the mapper-specific
            trailing layout.
    """

    x_xbar: Tensor
    slice_weights: Tensor
    logical_batch_shape: tuple[int, ...]


@dataclass(frozen=True)
class WMappingResult:
    """Result of :meth:`XbarMapper.map_w`.

    Attributes:
        w_xbar: Weight tensor in xbar-native layout.
        slice_weights: Positional weights for the weight slice axis.
        logical_out_dim: Logical output dimension `N` before tiling/padding.
        row_tile_num: Number of row/output tiles.
        col_tile_num: Number of column/input tiles.
    """

    w_xbar: Tensor
    slice_weights: Tensor
    logical_out_dim: int
    row_tile_num: int
    col_tile_num: int


class XbarMapper(ABC):
    """Abstract mapper interface."""

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
        """Return the inclusive algorithm-side activation range.

        Args:
            x_range: Primitive xbar input range.
            col_num: Primitive xbar column capacity.
            row_num: Primitive xbar row capacity.
            w_digit_count: Digits per primitive weight cell.
            w_digit_radix: Primitive weight-digit radix.
            w_digit_range: Primitive weight-digit range.

        Returns:
            Inclusive integer range that the mapper accepts for one logical
            activation value.
        """
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
        """Return the inclusive algorithm-side weight range.

        Args:
            x_range: Primitive xbar input range.
            col_num: Primitive xbar column capacity.
            row_num: Primitive xbar row capacity.
            w_digit_count: Digits per primitive weight cell.
            w_digit_radix: Primitive weight-digit radix.
            w_digit_range: Primitive weight-digit range.

        Returns:
            Inclusive integer range that the mapper accepts for one logical
            weight value.
        """
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
        """Return the activation slice radix.

        Args:
            x_range: Primitive xbar input range.
            col_num: Primitive xbar column capacity.
            row_num: Primitive xbar row capacity.
            w_digit_count: Digits per primitive weight cell.
            w_digit_radix: Primitive weight-digit radix.
            w_digit_range: Primitive weight-digit range.

        Returns:
            Positional radix used by the activation-side digital reduction.
        """
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
        """Return the weight slice radix.

        Args:
            x_range: Primitive xbar input range.
            col_num: Primitive xbar column capacity.
            row_num: Primitive xbar row capacity.
            w_digit_count: Digits per primitive weight cell.
            w_digit_radix: Primitive weight-digit radix.
            w_digit_range: Primitive weight-digit range.

        Returns:
            Positional radix used by the weight-side digital reduction.
        """
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
        """Map an activation tensor into xbar-native layout.

        Args:
            x: Algorithm-side activation tensor. Shape: [..., M, K].
            x_range: Primitive xbar input range.
            col_num: Primitive xbar column capacity.
            row_num: Primitive xbar row capacity.
            w_digit_count: Digits per primitive weight cell.
            w_digit_radix: Primitive weight-digit radix.
            w_digit_range: Primitive weight-digit range.

        Returns:
            Activation tensor plus slice metadata in xbar-native layout.
        """
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
        """Map a weight tensor into xbar-native layout.

        Args:
            w: Algorithm-side weight tensor. Shape: [..., N, K].
            x_range: Primitive xbar input range.
            col_num: Primitive xbar column capacity.
            row_num: Primitive xbar row capacity.
            w_digit_count: Digits per primitive weight cell.
            w_digit_radix: Primitive weight-digit radix.
            w_digit_range: Primitive weight-digit range.

        Returns:
            Weight tensor plus tiling metadata in xbar-native layout.
        """
        raise NotImplementedError
