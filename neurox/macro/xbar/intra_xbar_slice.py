"""Intra-xbar slice macro: ``Sw`` gathered inside one xbar (Strategy 2).

See also:
    docs/dev/modules/macro/xbar/intra_xbar_slice.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
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
from neurox.xbar import Xbar

from .base import XbarMacro, XbarMacroConfig


@dataclass(frozen=True)
class IntraXbarSliceMacroConfig(XbarMacroConfig):
    """Configuration for :class:`IntraXbarSliceMacro`.

    Attributes:
        w_slice_num: Per-weight Sw slice count.
        x_slice_num: Per-activation Sa slice count.
        w_encoding: Signed-digit encoding for the weight slicer.
        col_accumulator_cfg: Tc-axis cross-tile accumulator config.
        sa_shift_adder_cfg: Sa-axis intra-xbar shift-adder config.
        sw_shift_adder_cfg: Sw-axis intra-xbar shift-adder config.
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


@XbarMacro.register_key(IntraXbarSliceMacroConfig)
class IntraXbarSliceMacro(XbarMacro):
    """Xbar macro that gathers all slices of one logical weight in one xbar.

    A logical weight's ``Sw`` slices sit in adjacent cols of the same xbar.
    Per-xbar effective capacity is ``(col_num // Sw) * Sw`` cells; the
    remaining ``col_num - (col_num // Sw) * Sw`` cells per xbar are idle.
    """

    def __init__(
        self,
        *,
        cfg: IntraXbarSliceMacroConfig,
        xbar: Xbar,
        name: str = "",
    ) -> None:
        super().__init__(cfg=cfg, xbar=xbar, name=name)

        self.cfg = cfg
        self.xbar = xbar

        if cfg.w_slice_num > xbar.col_num:
            raise ValueError(f"require: w_slice_num ({cfg.w_slice_num}) <= xbar.col_num ({xbar.col_num})")
        self._weights_per_xbar = xbar.col_num // cfg.w_slice_num
        self._used_data_num = self._weights_per_xbar * cfg.w_slice_num
        self._idle_per_xbar = xbar.col_num - self._used_data_num

        self.w_slicer = SimpleSlicer(slice_num=cfg.w_slice_num, encoding=cfg.w_encoding)
        self.x_slicer = SerialSlicer(slice_num=cfg.x_slice_num)

        prefix = f"{name}." if name else ""
        self.col_accumulator = Accumulator(cfg.col_accumulator_cfg, name=f"{prefix}col_accumulator")
        self.sa_shift_adder = ShiftAdder(cfg.sa_shift_adder_cfg, name=f"{prefix}sa_shift_adder")
        self.sw_shift_adder = ShiftAdder(cfg.sw_shift_adder_cfg, name=f"{prefix}sw_shift_adder")
        self.requantizer = Requantizer(cfg.requantizer_cfg, name=f"{prefix}requantizer")

        self._w_parallel_size = 0
        self._serial_op_num = 0
        self._x_shape_cached: tuple[int, ...] = ()

    def extra_repr(self) -> str:
        """One-line summary shown by ``print(model)``."""
        return (
            f"xbar={type(self.xbar).__name__}, "
            f"row_num={self.xbar.row_num}, col_num={self.xbar.col_num}, "
            f"weights_per_xbar={self._weights_per_xbar}, "
            f"w_value_range={self.w_value_range}, x_value_range={self.x_value_range}"
        )

    def __repr__(self) -> str:
        """Compact repr that hides internals from ``print(model)``."""
        return f"{type(self).__name__}({self.extra_repr()})"

    # --- value-range / rescale ---

    @property
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive algorithm-side weight range — delegated to the slicer."""
        return self.w_slicer.value_range(
            digit_count=self.xbar.w_digit_count,
            digit_radix=self.xbar.w_digit_radix,
            digit_range=self.xbar.w_digit_range,
        )

    @property
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive algorithm-side activation range — delegated to the slicer."""
        x_lo, x_hi = self.xbar.x_range
        return self.x_slicer.value_range(
            digit_count=1,
            digit_radix=x_hi - x_lo + 1,
            digit_range=(x_lo, x_hi),
        )

    @property
    def output_rescale_factor(self) -> float:
        """Forwarded from the xbar."""
        return self.xbar.output_rescale_factor

    # --- slice radixes ---

    def _w_slice_radix(self) -> int:
        return self.w_slicer.slice_radix(
            digit_count=self.xbar.w_digit_count,
            digit_radix=self.xbar.w_digit_radix,
            digit_range=self.xbar.w_digit_range,
        )

    def _x_slice_radix(self) -> int:
        x_lo, x_hi = self.xbar.x_range
        return self.x_slicer.slice_radix(
            digit_count=1,
            digit_radix=x_hi - x_lo + 1,
            digit_range=(x_lo, x_hi),
        )

    # --- organize ---

    def _organize_w(self, weight: Tensor) -> Tensor:
        """Map a logical weight tensor into xbar-native layout.

        Args:
            weight: Integer weight tensor of shape ``[..., N, K]``.

        Returns:
            Tensor of shape
            ``[..., M=1, Tc, Tr, Sa=1, data_num, D, row_num]``.
            The trailing ``col_num - (col_num // Sw) * Sw`` cells per
            xbar are zero-padded.
        """
        col_num = self.xbar.col_num
        row_num = self.xbar.row_num
        wpx = self._weights_per_xbar
        used = self._used_data_num
        idle = self._idle_per_xbar

        n_logical = weight.shape[-2]
        # No logical weight may straddle two xbars: pad N up to a multiple of wpx.
        n_padded = ((n_logical + wpx - 1) // wpx) * wpx
        tr = n_padded // wpx

        # Shape: [..., N, K] -> [..., N, K, Sw, D]
        sliced = self.w_slicer.slice(
            weight,
            digit_count=self.xbar.w_digit_count,
            digit_radix=self.xbar.w_digit_radix,
            digit_range=self.xbar.w_digit_range,
        ).values

        # Shape: [..., N, K, Sw, D] -> [..., n_padded, K, Sw, D]
        n_pad = n_padded - n_logical
        if n_pad > 0:
            sliced = F.pad(sliced, (0, 0, 0, 0, 0, 0, 0, n_pad))

        # Shape: [..., n_padded, K, Sw, D] -> [..., Tr, wpx, K, Sw, D]
        unflat = sliced.unflatten(-4, (tr, wpx))

        # Shape: [..., Tr, wpx, K, Sw, D] -> [..., Tr, wpx, Tc, row_num, Sw, D]
        tiled = self.chunk_pad_along(unflat, axis=-3, chunk_size=row_num)

        # Shape: [..., Tr, wpx, Tc, row_num, Sw, D] -> [..., Tc, Tr, wpx, Sw, D, row_num]
        b = tiled.ndim - 6
        perm = [*range(b), b + 2, b + 0, b + 1, b + 4, b + 5, b + 3]
        arranged = tiled.permute(perm)

        # Shape: [..., Tc, Tr, wpx, Sw, D, row_num] -> [..., Tc, Tr, wpx*Sw, D, row_num]
        merged = arranged.flatten(start_dim=b + 2, end_dim=b + 3)
        assert merged.shape[b + 2] == used

        # Shape: [..., Tc, Tr, wpx*Sw, D, row_num] -> [..., Tc, Tr, data_num, D, row_num]
        if idle > 0:
            merged = F.pad(merged, (0, 0, 0, 0, 0, idle))

        # Shape: [..., Tc, Tr, data_num, D, row_num] -> [..., Tc, Tr, Sa=1, data_num, D, row_num]
        merged = merged.unsqueeze(b + 2)
        # Shape: [..., Tc, Tr, Sa=1, data_num, D, row_num] -> [..., M=1, Tc, Tr, Sa=1, data_num, D, row_num]
        merged = merged.unsqueeze(b)
        return merged

    def _organize_x(self, x: Tensor) -> Tensor:
        """Map a logical activation tensor into xbar-native layout.

        Args:
            x: Integer activation tensor of shape ``[..., M, K]``.

        Returns:
            Tensor of shape ``[..., M, Tc, Tr=1, Sa, row_num]``.
        """
        x_lo, x_hi = self.xbar.x_range
        # Shape: [..., M, K] -> [..., M, K, Sa, digit_num=1]
        sliced = self.x_slicer.slice(
            x,
            digit_count=1,
            digit_radix=x_hi - x_lo + 1,
            digit_range=(x_lo, x_hi),
        ).values

        # Shape: [..., M, K, Sa, digit_num=1] -> [..., M, Tc, row_num, Sa, digit_num=1]
        tiled = self.chunk_pad_along(sliced, axis=-3, chunk_size=self.xbar.row_num)

        # Shape: [..., M, Tc, row_num, Sa, digit_num=1] -> [..., M, Tc, row_num, Sa]
        squeezed = tiled.squeeze(-1)
        # Shape: [..., M, Tc, row_num, Sa] -> [..., M, Tc, Sa, row_num]
        transposed = squeezed.transpose(-2, -1)
        # Shape: [..., M, Tc, Sa, row_num] -> [..., M, Tc, Tr=1, Sa, row_num]
        x_mapped = transposed.unsqueeze(-3)
        return x_mapped

    # --- lifecycle ---

    def fabricate(self, weight: Tensor) -> None:
        """Map one weight tensor and program the xbar (re-callable).

        Args:
            weight: Integer weight tensor of shape ``[..., N, K]``.
        """
        organized = self._organize_w(weight)
        self.xbar.fabricate(organized)

        *w_batch, _m, _tc, row_tile_num, _sa = organized.shape[:-3]
        self._w_parallel_size = math.prod(w_batch)
        n_logical = weight.shape[-2]
        self.col_accumulator.fabricate((self._w_parallel_size, row_tile_num))
        self.sw_shift_adder.fabricate((self._w_parallel_size, row_tile_num))
        self.sa_shift_adder.fabricate((self._w_parallel_size, row_tile_num))
        self.requantizer.fabricate((self._w_parallel_size, n_logical))

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
        """Execute one integer matrix multiply.

        Args:
            input: Integer activation of shape ``[..., M, K]``.
            weight: Integer weight of shape ``[..., N, K]``.
            bias: Optional integer bias of shape ``[..., N]``.
            rescale_multiplier: Per-output fixed-point multiplier.
            rescale_rshift: Per-output right-shift amount.
            output_zero_point: Optional output zero point.

        Returns:
            Integer output of shape ``[..., M, N]`` in ``input.dtype``.
        """
        n_logical = weight.shape[-2]
        x_dtype = input.dtype
        wpx = self._weights_per_xbar
        used = self._used_data_num
        sw = self.cfg.w_slice_num

        if self.training:
            self.xbar.fabricate(self._organize_w(weight))

        x = self._organize_x(input)

        if not self.training and self._x_shape_cached != x.shape:
            self._x_shape_cached = x.shape
            sa_dim = self._x_shape_cached[-2]
            batch_m_prod = math.prod(self._x_shape_cached[:-5])
            self._serial_op_num = batch_m_prod * sa_dim // max(self._w_parallel_size, 1)

        x_slice_radix = self._x_slice_radix()
        w_slice_radix = self._w_slice_radix()

        # Shape: [..., M, Tc, Tr=1, Sa, row_num] -> [..., M, Tc, Tr, Sa, data_num]
        y = self.xbar.vec_mat_mul(x)
        # Shape: [..., M, Tc, Tr, Sa, data_num=col_num] -> [..., M, Tc, Tr, Sa, wpx*Sw]
        y = y[..., :used]
        # Shape: [..., M, Tc, Tr, Sa, wpx*Sw] -> [..., M, Tc, Tr, Sa, wpx, Sw]
        y = y.unflatten(-1, (wpx, sw))
        # Shape: [..., M, Tc, Tr, Sa, wpx, Sw] -> [..., M, Tc, Tr, Sa, wpx]
        y = self.sw_shift_adder.operate(y, w_slice_radix, dim=-1)
        # Shape: [..., M, Tc, Tr, Sa, wpx] -> [..., M, Tc, Tr, wpx]
        y = self.sa_shift_adder.operate(y, x_slice_radix, dim=-2)
        # Shape: [..., M, Tc, Tr, wpx] -> [..., M, Tr, wpx]
        y = self.col_accumulator.operate(y, dim=-3)
        # Shape: [..., M, Tr, wpx] -> [..., M, Tr * wpx] -> [..., M, N]
        y = y.flatten(start_dim=-2)[..., :n_logical]
        if bias is not None:
            y = y + bias

        y = self.requantizer.operate(y, rescale_multiplier, rescale_rshift, output_zero_point)
        return y.to(x_dtype)
