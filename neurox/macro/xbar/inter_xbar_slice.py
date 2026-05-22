"""Inter-xbar slice macro: ``Sw`` distributed across xbar planes (Strategy 1).

See also:
    docs/dev/modules/macro/xbar/inter_xbar_slice.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.digital import (
    Accumulator,
    AccumulatorConfig,
    Requantizer,
    RequantizerConfig,
    ShiftAdder,
    ShiftAdderConfig,
)
from neurox.mapper.transcoder import Encoding
from neurox.mapper.xbar.slicer import SerialSlicer, SimpleSlicer

from .base import XbarMacro, XbarMacroConfig


@dataclass(frozen=True)
class InterXbarSliceMacroConfig(XbarMacroConfig):
    """Configuration for :class:`InterXbarSliceMacro`.

    Attributes:
        w_slice_num: Per-weight Sw slice count.
        x_slice_num: Per-activation Sa slice count.
        w_encoding: Signed-digit encoding for the weight slicer.
        col_accumulator_cfg: Tc-axis cross-tile accumulator config.
        sa_shift_adder_cfg: Sa-axis intra-xbar shift-adder config.
        sw_shift_adder_cfg: Sw-axis cross-xbar shift-adder config.
        requantizer_cfg: Output requantizer config.
    """

    w_slice_num: int
    x_slice_num: int
    w_encoding: Encoding

    col_accumulator_cfg: AccumulatorConfig
    sa_shift_adder_cfg: ShiftAdderConfig
    sw_shift_adder_cfg: ShiftAdderConfig
    requantizer_cfg: RequantizerConfig

    def validate(self) -> None:
        super().validate()
        self._require_pos(self.w_slice_num, "w_slice_num")
        self._require_pos(self.x_slice_num, "x_slice_num")


@XbarMacro.register_key(InterXbarSliceMacroConfig)
class InterXbarSliceMacro(XbarMacro):
    """Xbar macro that distributes weight slices across separate xbar planes.

    One xbar plane holds one ``Sw`` slice index across every logical weight.
    """

    cfg: InterXbarSliceMacroConfig

    def __init__(
        self,
        *,
        cfg: InterXbarSliceMacroConfig,
        name: str,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_xbar: bool,
    ) -> None:
        super().__init__(
            cfg=cfg,
            name=name,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_xbar=ideal_xbar,
        )
        self.cfg = cfg
        xbar_cfg = cfg.xbar_cfg
        col_num = xbar_cfg.col_num
        row_num = xbar_cfg.row_num
        digit_count = xbar_cfg.w_digit_count  # type: ignore[attr-defined]

        # Symbolic organized shape: (*batch, 1, Tc, Tr, 1, Sw, col_num, D, row_num).
        *w_batch, n_logical, k_logical = w_logical_shape
        tr = (n_logical + col_num - 1) // col_num
        tc = (k_logical + row_num - 1) // row_num
        sw = cfg.w_slice_num
        xbar_w_layout_shape = (*w_batch, 1, tc, tr, 1, sw, col_num, digit_count, row_num)

        self.xbar = self._build_xbar(xbar_w_layout_shape)
        xbar = self.xbar

        x_lo, x_hi = xbar.x_range
        self.w_slicer = SimpleSlicer(
            slice_num=cfg.w_slice_num,
            digit_count=xbar.w_digit_count,
            digit_radix=xbar.w_digit_radix,
            encoding=cfg.w_encoding,
        )
        self.x_slicer = SerialSlicer(
            slice_num=cfg.x_slice_num,
            digit_radix=x_hi - x_lo + 1,
        )

        self._w_parallel_size = max(math.prod(w_batch), 1)
        self._n_logical = n_logical
        self._row_tile_num = tr

        prefix = f"{name}." if name else ""
        self.col_accumulator = Accumulator(
            cfg=cfg.col_accumulator_cfg,
            name=f"{prefix}col_accumulator",
            inst_shape=(self._w_parallel_size, sw, tr),
        )
        self.sa_shift_adder = ShiftAdder(
            cfg=cfg.sa_shift_adder_cfg,
            name=f"{prefix}sa_shift_adder",
            inst_shape=(self._w_parallel_size, tr),
        )
        self.sw_shift_adder = ShiftAdder(
            cfg=cfg.sw_shift_adder_cfg,
            name=f"{prefix}sw_shift_adder",
            inst_shape=(self._w_parallel_size, tr),
        )
        self.requantizer = Requantizer(
            cfg=cfg.requantizer_cfg,
            name=f"{prefix}requantizer",
            inst_shape=(self._w_parallel_size, n_logical),
        )

        self._serial_op_num = 0
        self._x_shape_cached: tuple[int, ...] = ()

        self._log_static()

    def extra_repr(self) -> str:
        """One-line summary shown by ``print(model)``."""
        return (
            f"xbar={type(self.xbar).__name__}, "
            f"row_num={self.xbar.row_num}, col_num={self.xbar.col_num}, "
            f"w_value_range={self.w_value_range}, x_value_range={self.x_value_range}"
        )

    def __repr__(self) -> str:
        """Compact repr that hides internals from ``print(model)``."""
        return f"{type(self).__name__}({self.extra_repr()})"

    # --- value-range / rescale ---

    @property
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive algorithm-side weight range — delegated to the slicer."""
        return self.w_slicer.value_range

    @property
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive algorithm-side activation range — delegated to the slicer."""
        return self.x_slicer.value_range

    @property
    def output_rescale_factor(self) -> float:
        """Forwarded from the xbar."""
        return self.xbar.output_rescale_factor

    # --- organize ---

    def _organize_w(self, weight: Tensor) -> Tensor:
        """Map a logical weight tensor into xbar-native layout.

        Args:
            weight: Integer weight tensor of shape ``[..., N, K]``.

        Returns:
            Tensor of shape
            ``[..., M=1, Tc, Tr, Sa=1, Sw, data_num, D, row_num]``.
        """
        col_num = self.xbar.col_num
        row_num = self.xbar.row_num

        # Shape: [..., N, K] -> [..., N, K, Sw, D]
        sliced = self.w_slicer.slice(weight)

        # Shape: [..., N, K, Sw, D] -> [..., Tr, data_num, K, Sw, D]
        tiled = self.chunk_pad_along(sliced, axis=-4, chunk_size=col_num, pad_value=0)
        # Shape: [..., Tr, data_num, K, Sw, D] -> [..., Tr, data_num, Tc, row_num, Sw, D]
        tiled = self.chunk_pad_along(tiled, axis=-3, chunk_size=row_num, pad_value=0)

        # Shape: [..., Tr, data_num, Tc, row_num, Sw, D] -> [..., Tc, Tr, Sw, data_num, D, row_num]
        b = tiled.ndim - 6
        perm = [*range(b), b + 2, b + 0, b + 4, b + 1, b + 5, b + 3]
        arranged = tiled.permute(perm)

        # Shape: [..., Tc, Tr, Sw, data_num, D, row_num] -> [..., Tc, Tr, Sa=1, Sw, data_num, D, row_num]
        arranged = arranged.unsqueeze(b + 2)
        # Shape: [..., Tc, Tr, Sa=1, Sw, data_num, D, row_num] -> [..., M=1, Tc, Tr, Sa=1, Sw, data_num, D, row_num]
        arranged = arranged.unsqueeze(b)
        return arranged

    def _organize_x(self, x: Tensor) -> Tensor:
        """Map a logical activation tensor into xbar-native layout.

        Args:
            x: Integer activation tensor of shape ``[..., M, K]``.

        Returns:
            Tensor of shape ``[..., M, Tc, Tr=1, Sa, Sw=1, row_num]``.
        """
        # Shape: [..., M, K] -> [..., M, K, Sa, digit_num=1]
        sliced = self.x_slicer.slice(x)

        # Shape: [..., M, K, Sa, digit_num=1] -> [..., M, Tc, row_num, Sa, digit_num=1]
        tiled = self.chunk_pad_along(sliced, axis=-3, chunk_size=self.xbar.row_num, pad_value=0)

        # Shape: [..., M, Tc, row_num, Sa, digit_num=1] -> [..., M, Tc, row_num, Sa]
        squeezed = tiled.squeeze(-1)
        # Shape: [..., M, Tc, row_num, Sa] -> [..., M, Tc, Sa, row_num]
        transposed = squeezed.transpose(-2, -1)
        # Shape: [..., M, Tc, Sa, row_num] -> [..., M, Tc, Sa, Sw=1, row_num]
        with_sw = transposed.unsqueeze(-2)
        # Shape: [..., M, Tc, Sa, Sw=1, row_num] -> [..., M, Tc, Tr=1, Sa, Sw=1, row_num]
        x_mapped = with_sw.unsqueeze(-4)
        return x_mapped

    # --- lifecycle ---

    def program(self, weight: Tensor) -> None:
        """Map one logical weight tensor and program the xbar.

        Args:
            weight: Integer weight tensor whose shape matches
                ``self._w_logical_shape``.
        """
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        organized = self._organize_w(weight)
        self.xbar.program(organized)

    @torch.no_grad()
    @torch.compile(dynamic=True)
    def matmul(
        self,
        input: Tensor,
        bias: Tensor | None,
        rescale_multiplier: Tensor,
        rescale_rshift: Tensor,
        output_zero_point: Tensor | None,
    ) -> Tensor:
        """Execute one integer matrix multiply against the programmed weight.

        Args:
            input: Integer activation of shape ``[..., M, K]``.
            bias: Optional integer bias of shape ``[..., N]``.
            rescale_multiplier: Per-output fixed-point multiplier.
            rescale_rshift: Per-output right-shift amount.
            output_zero_point: Optional output zero point.

        Returns:
            Integer output of shape ``[..., M, N]`` in ``input.dtype``.
        """
        n_logical = self._n_logical
        x_dtype = input.dtype

        x = self._organize_x(input)

        if self._x_shape_cached != x.shape:
            self._x_shape_cached = x.shape
            sa_dim = self._x_shape_cached[-3]
            batch_m_prod = math.prod(self._x_shape_cached[:-6])
            self._serial_op_num = batch_m_prod * sa_dim // max(self._w_parallel_size, 1)

        x_slice_radix = self.x_slicer.slice_radix
        w_slice_radix = self.w_slicer.slice_radix

        # Shape: [..., M, Tc, Tr=1, Sa, Sw=1, row_num] -> [..., M, Tc, Tr, Sa, Sw, data_num]
        y = self.xbar.vec_mat_mul(x)
        # Shape: [..., M, Tc, Tr, Sa, Sw, data_num] -> [..., M, Tc, Tr, Sw, data_num]
        y = self.sa_shift_adder.operate(y, x_slice_radix, dim=-3, init_val=None)
        # Shape: [..., M, Tc, Tr, Sw, data_num] -> [..., M, Tc, Tr, data_num]
        y = self.sw_shift_adder.operate(y, w_slice_radix, dim=-2, init_val=None)
        # Shape: [..., M, Tc, Tr, data_num] -> [..., M, Tr, data_num]
        y = self.col_accumulator.operate(y, dim=-3)
        # Shape: [..., M, Tr, data_num] -> [..., M, Tr * data_num] -> [..., M, N]
        y = y.flatten(start_dim=-2)[..., :n_logical]
        if bias is not None:
            y = y + bias

        y = self.requantizer.operate(y, rescale_multiplier, rescale_rshift, output_zero_point)
        return y.to(x_dtype)
