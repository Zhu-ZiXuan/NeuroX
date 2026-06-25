"""Inter-array slice macro: ``Sw`` distributed across xbar planes (Strategy 1).

See also:
    docs/reference/macro/xbar/inter_array_slice.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.analog.adc import AdcOperationPoint
from neurox.common.encoding import Encoding
from neurox.digital import (
    Accumulator,
    AccumulatorConfig,
    ShiftAdder,
    ShiftAdderConfig,
)
from neurox.macro.xbar.slicer import SerialSlicer, SimpleSlicer
from neurox.xbar import Xbar, XbarConfig, XbarPolicy

from .base import XbarMacro, XbarMacroConfig, XbarMacroPolicy


@dataclass(frozen=True)
class InterArraySliceXbarMacroConfig(XbarMacroConfig):
    """Configuration for :class:`InterArraySliceXbarMacro`.

    Attributes:
        xbar_config: Owned physical-xbar config.
        w_slice_num: Per-weight Sw slice count.
        x_slice_num: Per-activation Sa slice count.
        w_encoding: Signed-digit encoding for the weight slicer.
        col_accumulator_config: Tc-axis cross-tile accumulator config.
        sa_shift_adder_config: Sa-axis intra-xbar shift-adder config.
        sw_shift_adder_config: Sw-axis cross-xbar shift-adder config.
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
class InterArraySliceXbarMacroPolicy(XbarMacroPolicy):
    """Composite policy for :class:`InterArraySliceXbarMacro`.

    Attributes:
        xbar: Embedded xbar nonideality policy.
    """

    xbar: XbarPolicy


@XbarMacro.register_key(InterArraySliceXbarMacroConfig)
class InterArraySliceXbarMacro(XbarMacro):
    """Xbar macro that distributes weight slices across separate xbar planes.

    One xbar plane holds one ``Sw`` slice index across every logical weight.
    """

    xbar: Xbar
    config: InterArraySliceXbarMacroConfig

    def __init__(
        self,
        *,
        config: InterArraySliceXbarMacroConfig,
        policy: InterArraySliceXbarMacroPolicy,
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

        # Organized shape: (*batch, M=1, Sa=1, Sw, Tc, Tr, col_num, D, row_num).
        # The trailing (col_num, D, row_num) is owned by the xbar.
        *w_batch, n_logical, k_logical = w_logical_shape
        tr = (n_logical + col_num - 1) // col_num
        tc = (k_logical + row_num - 1) // row_num
        sw = config.w_slice_num

        self.xbar = self._build_xbar(
            xbar_config=xbar_config,
            xbar_policy=policy.xbar,
            inst_shape=(*w_batch, 1, 1, sw, tc, tr),
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
        self.col_accumulator = Accumulator(
            config=config.col_accumulator_config,
            name=f"{prefix}col_accumulator",
            inst_shape=(self._w_parallel_size, sw, tr),
        )
        self.sa_shift_adder = ShiftAdder(
            config=config.sa_shift_adder_config,
            name=f"{prefix}sa_shift_adder",
            inst_shape=(self._w_parallel_size, tr),
        )
        self.sw_shift_adder = ShiftAdder(
            config=config.sw_shift_adder_config,
            name=f"{prefix}sw_shift_adder",
            inst_shape=(self._w_parallel_size, tr),
        )

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
        """Number of supported ADC operating points; valid ``adc_mode`` values are ``[0, adc_mode_num)``."""
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
            ``[..., M=1, Sa=1, Sw, Tc, Tr, data_num, D, row_num]``.
        """
        col_num = self.xbar.col_num
        row_num = self.xbar.row_num

        # Shape: [..., N, K] -> [..., N, K, Sw, D]
        sliced = self.w_slicer.slice(weight)

        # Shape: [..., N, K, Sw, D] -> [..., Tr, data_num, K, Sw, D]
        tiled = self.chunk_pad_along(sliced, axis=-4, chunk_size=col_num, pad_value=0)
        # Shape: [..., Tr, data_num, K, Sw, D] -> [..., Tr, data_num, Tc, row_num, Sw, D]
        tiled = self.chunk_pad_along(tiled, axis=-3, chunk_size=row_num, pad_value=0)

        # Shape: [..., Tr, data_num, Tc, row_num, Sw, D] -> [..., Sw, Tc, Tr, data_num, D, row_num]
        b = tiled.ndim - 6
        perm = [*range(b), b + 4, b + 2, b + 0, b + 1, b + 5, b + 3]
        arranged = tiled.permute(perm)

        # Shape: [..., Sw, Tc, Tr, data_num, D, row_num] -> [..., M=1, Sa=1, Sw, Tc, Tr, data_num, D, row_num]
        return arranged.unsqueeze(b).unsqueeze(b)

    def _organize_x(self, x: Tensor) -> Tensor:
        """Map a logical activation tensor into xbar-native layout.

        Args:
            x: Integer activation tensor of shape ``[..., M, K]``.

        Returns:
            Tensor of shape ``[..., M, Sa, Sw=1, Tc, Tr=1, row_num]``.
        """
        # Shape: [..., M, K] -> [..., M, K, Sa, digit_count=1]
        sliced = self.x_slicer.slice(x)

        # Shape: [..., M, K, Sa, digit_count=1] -> [..., M, Tc, row_num, Sa, digit_count=1]
        tiled = self.chunk_pad_along(sliced, axis=-3, chunk_size=self.xbar.row_num, pad_value=0)

        # Shape: [..., M, Tc, row_num, Sa, digit_count=1] -> [..., M, Tc, row_num, Sa]
        squeezed = tiled.squeeze(-1)
        # Shape: [..., M, Tc, row_num, Sa] -> [..., M, Sa, Tc, row_num]
        b = squeezed.ndim - 4
        permuted = squeezed.permute([*range(b), b + 0, b + 3, b + 1, b + 2])
        # Shape: [..., M, Sa, Tc, row_num] -> [..., M, Sa, Sw=1, Tc, Tr=1, row_num]
        return permuted.unsqueeze(b + 2).unsqueeze(b + 4)

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

        x = self._organize_x(input)

        x_slice_radix = self.x_slicer.slice_radix
        w_slice_radix = self.w_slicer.slice_radix

        # Shape: [..., M, Sa, Sw=1, Tc, Tr=1, row_num] -> [..., M, Sa, Sw, Tc, Tr, data_num]
        y = self.xbar.vec_mat_mul(x, adc_operation_point=adc_operation_point).to(torch.int64)
        # Shape: [..., M, Sa, Sw, Tc, Tr, data_num] -> [..., M, Sw, Tc, Tr, data_num]
        y = self.sa_shift_adder.operate(y, x_slice_radix, dim=-5, init_val=None)
        # Shape: [..., M, Sw, Tc, Tr, data_num] -> [..., M, Tc, Tr, data_num]
        y = self.sw_shift_adder.operate(y, w_slice_radix, dim=-4, init_val=None)
        # Shape: [..., M, Tc, Tr, data_num] -> [..., M, Tr, data_num]
        y = self.col_accumulator.operate(y, dim=-3)
        # Shape: [..., M, Tr, data_num] -> [..., M, N]
        return y.flatten(start_dim=-2)[..., :n_logical]
