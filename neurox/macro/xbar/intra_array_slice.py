"""Intra-array slice macro: ``Sw`` gathered inside one xbar (Strategy 2).

See also:
    docs/dev/modules/macro/xbar/intra_array_slice.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.analog.adc import AdcOperationPoint
from neurox.digital import (
    Accumulator,
    AccumulatorConfig,
    ShiftAdder,
    ShiftAdderConfig,
)
from neurox.mapper.transcoder import Encoding
from neurox.mapper.xbar.slicer import SerialSlicer, SimpleSlicer
from neurox.xbar import Xbar, XbarConfig, XbarPolicy

from .base import XbarMacro, XbarMacroConfig, XbarMacroPolicy


@dataclass(frozen=True)
class IntraArraySliceXbarMacroConfig(XbarMacroConfig):
    """Configuration for :class:`IntraArraySliceXbarMacro`.

    Attributes:
        xbar_config: Owned physical-xbar config.
        w_slice_num: Per-weight Sw slice count.
        x_slice_num: Per-activation Sa slice count.
        w_encoding: Signed-digit encoding for the weight slicer.
        col_accumulator_config: Tc-axis cross-tile accumulator config.
        sa_shift_adder_config: Sa-axis intra-xbar shift-adder config.
        sw_shift_adder_config: Sw-axis intra-xbar shift-adder config.
    """

    xbar_config: XbarConfig
    w_slice_num: int
    x_slice_num: int
    w_encoding: Encoding

    col_accumulator_config: AccumulatorConfig
    sa_shift_adder_config: ShiftAdderConfig
    sw_shift_adder_config: ShiftAdderConfig

    def validate(self) -> None:
        super().validate()
        self._require_pos(self.w_slice_num, "w_slice_num")
        self._require_pos(self.x_slice_num, "x_slice_num")


@dataclass(frozen=True)
class IntraArraySliceXbarMacroPolicy(XbarMacroPolicy):
    """Composite policy for :class:`IntraArraySliceXbarMacro`.

    Attributes:
        xbar: Embedded xbar nonideality policy.
    """

    xbar: XbarPolicy


@XbarMacro.register_key(IntraArraySliceXbarMacroConfig)
class IntraArraySliceXbarMacro(XbarMacro):
    """Xbar macro that gathers all slices of one logical weight in one xbar.

    A logical weight's ``Sw`` slices sit in adjacent cols of the same xbar.
    Per-xbar effective capacity is ``(col_num // Sw) * Sw`` cells; the
    remaining ``col_num - (col_num // Sw) * Sw`` cells per xbar are idle.
    """

    xbar: Xbar
    config: IntraArraySliceXbarMacroConfig

    def __init__(
        self,
        *,
        config: IntraArraySliceXbarMacroConfig,
        policy: IntraArraySliceXbarMacroPolicy,
        name: str,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_xbar: bool,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            name=name,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_xbar=ideal_xbar,
        )
        self.config = config
        xbar_config = config.xbar_config
        col_num = xbar_config.col_num
        row_num = xbar_config.row_num
        if config.w_slice_num > col_num:
            raise ValueError(f"require: w_slice_num ({config.w_slice_num}) <= xbar.col_num ({col_num})")
        self._weights_per_xbar = col_num // config.w_slice_num
        self._used_data_num = self._weights_per_xbar * config.w_slice_num
        self._idle_per_xbar = col_num - self._used_data_num

        # Organized shape: (*batch, M=1, Sa=1, Tc, Tr, col_num, D, row_num).
        # The trailing (col_num, D, row_num) is owned by the xbar.
        *w_batch, n_logical, k_logical = w_logical_shape
        wpx = self._weights_per_xbar
        tr = (n_logical + wpx - 1) // wpx
        tc = (k_logical + row_num - 1) // row_num

        self.xbar = self._build_xbar(
            xbar_config=xbar_config,
            xbar_policy=policy.xbar,
            inst_shape=(*w_batch, 1, 1, tc, tr),
        )
        xbar = self.xbar

        x_lo, x_hi = xbar.x_range
        self.w_slicer = SimpleSlicer(
            slice_num=config.w_slice_num,
            digit_count=xbar.w_digit_count,
            digit_radix=xbar.w_digit_radix,
            encoding=config.w_encoding,
        )
        self.x_slicer = SerialSlicer(
            slice_num=config.x_slice_num,
            digit_radix=x_hi - x_lo + 1,
        )

        self._w_parallel_size = max(math.prod(w_batch), 1)
        self._n_logical = n_logical
        self._row_tile_num = tr

        prefix = f"{name}." if name else ""
        helper_shape = (self._w_parallel_size, tr)
        self.col_accumulator = Accumulator(
            config=config.col_accumulator_config,
            name=f"{prefix}col_accumulator",
            inst_shape=helper_shape,
        )
        self.sa_shift_adder = ShiftAdder(
            config=config.sa_shift_adder_config,
            name=f"{prefix}sa_shift_adder",
            inst_shape=helper_shape,
        )
        self.sw_shift_adder = ShiftAdder(
            config=config.sw_shift_adder_config,
            name=f"{prefix}sw_shift_adder",
            inst_shape=helper_shape,
        )

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

    # --- value-range / ADC surface ---

    @property
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive integer weight range accepted by the macro."""
        return self.w_slicer.value_range

    @property
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive integer activation range accepted by the macro."""
        return self.x_slicer.value_range

    @property
    def adc_mode_num(self) -> int:
        """Number of supported ADC operating points; valid ``adc_mode`` values are ``[0, mode_num)``."""
        return self.xbar.adc_mode_num

    @property
    def adc_max_bits(self) -> int:
        """Maximum supported ``adc_bits`` value."""
        return self.xbar.adc_max_bits

    def adc_rescale_factor(self, adc_operation_point: AdcOperationPoint) -> float:
        """Rescale factor for ``adc_operation_point``; raises ``KeyError`` if uncalibrated."""
        return self.xbar.adc_rescale_factor(adc_operation_point)

    # --- organize ---

    def _organize_w(self, weight: Tensor) -> Tensor:
        """Map a logical weight tensor into xbar-native layout.

        Args:
            weight: Integer weight tensor of shape ``[..., N, K]``.

        Returns:
            Tensor of shape
            ``[..., M=1, Sa=1, Tc, Tr, data_num, D, row_num]``.
            The trailing ``col_num - (col_num // Sw) * Sw`` cells per xbar are zero-padded.
        """
        row_num = self.xbar.row_num
        wpx = self._weights_per_xbar
        idle = self._idle_per_xbar

        n_logical = weight.shape[-2]
        # No logical weight may straddle two xbars: pad N up to a multiple of wpx.
        n_padded = ((n_logical + wpx - 1) // wpx) * wpx
        tr = n_padded // wpx

        # Shape: [..., N, K] -> [..., N, K, Sw, D]
        sliced = self.w_slicer.slice(weight)

        # Shape: [..., N, K, Sw, D] -> [..., n_padded, K, Sw, D]
        n_pad = n_padded - n_logical
        if n_pad > 0:
            sliced = F.pad(sliced, (0, 0, 0, 0, 0, 0, 0, n_pad))

        # Shape: [..., n_padded, K, Sw, D] -> [..., Tr, wpx, K, Sw, D]
        unflat = sliced.unflatten(-4, (tr, wpx))

        # Shape: [..., Tr, wpx, K, Sw, D] -> [..., Tr, wpx, Tc, row_num, Sw, D]
        tiled = self.chunk_pad_along(unflat, axis=-3, chunk_size=row_num, pad_value=0)

        # Shape: [..., Tr, wpx, Tc, row_num, Sw, D] -> [..., Tc, Tr, wpx, Sw, D, row_num]
        b = tiled.ndim - 6
        perm = [*range(b), b + 2, b + 0, b + 1, b + 4, b + 5, b + 3]
        arranged = tiled.permute(perm)

        # Shape: [..., Tc, Tr, wpx, Sw, D, row_num] -> [..., Tc, Tr, wpx*Sw, D, row_num]
        merged = arranged.flatten(start_dim=b + 2, end_dim=b + 3)

        # Shape: [..., Tc, Tr, wpx*Sw, D, row_num] -> [..., Tc, Tr, data_num, D, row_num]
        if idle > 0:
            merged = F.pad(merged, (0, 0, 0, 0, 0, idle))

        # Shape: [..., Tc, Tr, data_num, D, row_num] -> [..., M=1, Sa=1, Tc, Tr, data_num, D, row_num]
        return merged.unsqueeze(b).unsqueeze(b)

    def _organize_x(self, x: Tensor) -> Tensor:
        """Map a logical activation tensor into xbar-native layout.

        Args:
            x: Integer activation tensor of shape ``[..., M, K]``.

        Returns:
            Tensor of shape ``[..., M, Sa, Tc, Tr=1, row_num]``.
        """
        # Shape: [..., M, K] -> [..., M, K, Sa, digit_num=1]
        sliced = self.x_slicer.slice(x)

        # Shape: [..., M, K, Sa, digit_num=1] -> [..., M, Tc, row_num, Sa, digit_num=1]
        tiled = self.chunk_pad_along(sliced, axis=-3, chunk_size=self.xbar.row_num, pad_value=0)

        # Shape: [..., M, Tc, row_num, Sa, digit_num=1] -> [..., M, Tc, row_num, Sa]
        squeezed = tiled.squeeze(-1)
        # Shape: [..., M, Tc, row_num, Sa] -> [..., M, Sa, Tc, row_num]
        b = squeezed.ndim - 4
        permuted = squeezed.permute([*range(b), b + 0, b + 3, b + 1, b + 2])
        # Shape: [..., M, Sa, Tc, row_num] -> [..., M, Sa, Tc, Tr=1, row_num]
        return permuted.unsqueeze(b + 3)

    # --- lifecycle ---

    def program(self, weight: Tensor) -> None:
        """Write the macro's static weight state from one logical weight tensor.

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
    def matmul(self, input: Tensor, *, adc_operation_point: AdcOperationPoint) -> Tensor:
        """Execute one integer matrix multiply against the programmed weight state.

        Matches ``torch.matmul`` semantics (pure matmul, no bias). Bias add
        and requantize live in the operator layer.

        Args:
            input: Integer activation tensor. Shape: ``[..., M, K]``.
            adc_operation_point: Runtime ADC operating point.

        Returns:
            Integer pre-requantize output tensor. Shape: ``[..., M, N]``.
        """
        n_logical = self._n_logical
        wpx = self._weights_per_xbar
        used = self._used_data_num
        sw = self.config.w_slice_num

        x = self._organize_x(input)

        x_slice_radix = self.x_slicer.slice_radix
        w_slice_radix = self.w_slicer.slice_radix

        # Shape: [..., M, Sa, Tc, Tr=1, row_num] -> [..., M, Sa, Tc, Tr, data_num]
        y = self.xbar.vec_mat_mul(x, adc_operation_point=adc_operation_point).to(torch.int64)
        # Shape: [..., M, Sa, Tc, Tr, data_num=col_num] -> [..., M, Sa, Tc, Tr, wpx*Sw]
        y = y[..., :used]
        # Shape: [..., M, Sa, Tc, Tr, wpx*Sw] -> [..., M, Sa, Tc, Tr, wpx, Sw_real]
        y = y.unflatten(-1, (wpx, sw))
        # Shape: [..., M, Sa, Tc, Tr, wpx, Sw_real] -> [..., M, Sa, Tc, Tr, wpx]
        y = self.sw_shift_adder.operate(y, w_slice_radix, dim=-1, init_val=None)
        # Shape: [..., M, Sa, Tc, Tr, wpx] -> [..., M, Tc, Tr, wpx]
        y = self.sa_shift_adder.operate(y, x_slice_radix, dim=-4, init_val=None)
        # Shape: [..., M, Tc, Tr, wpx] -> [..., M, Tr, wpx]
        y = self.col_accumulator.operate(y, dim=-3)
        # Shape: [..., M, Tr, wpx] -> [..., M, N]
        return y.flatten(start_dim=-2)[..., :n_logical]
