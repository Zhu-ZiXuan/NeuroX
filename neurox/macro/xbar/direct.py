"""Direct macro: transcode-only.

See also:
    docs/reference/macro/xbar/direct.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.analog.adc import AdcOperationPoint
from neurox.digital import (
    Accumulator,
    AccumulatorConfig,
)
from neurox.mapper.transcoder import Encoding, Transcoder
from neurox.xbar import Xbar, XbarConfig, XbarPolicy

from .base import XbarMacro, XbarMacroConfig, XbarMacroPolicy


@dataclass(frozen=True)
class DirectXbarMacroConfig(XbarMacroConfig):
    """Configuration for :class:`DirectXbarMacro`.

    No slice counts and no shift-adders — the direct macro maps logical
    weights / activations straight onto one xbar's value range.

    Attributes:
        xbar_config: Owned physical-xbar config.
        w_encoding: Signed-digit encoding for the weight transcoder.
        col_accumulator_config: Tc-axis cross-tile accumulator config.
    """

    xbar_config: XbarConfig
    w_encoding: Encoding

    col_accumulator_config: AccumulatorConfig


@dataclass(frozen=True)
class DirectXbarMacroPolicy(XbarMacroPolicy):
    """Composite policy for :class:`DirectXbarMacro`.

    Attributes:
        xbar: Embedded xbar nonideality policy.
    """

    xbar: XbarPolicy


@XbarMacro.register_key(DirectXbarMacroConfig)
class DirectXbarMacro(XbarMacro):
    """Xbar macro with no activation / weight slicing — transcode only.

    The caller must supply integer weight / activation values that already
    fit the xbar's value range (the transcoder's ``value_range`` for ``w``
    and ``xbar.x_range`` for ``x``). Neither this macro nor the underlying
    xbar enforces the range — out-of-range inputs propagate as-is and
    produce undefined results.
    """

    xbar: Xbar
    config: DirectXbarMacroConfig

    def __init__(
        self,
        *,
        config: DirectXbarMacroConfig,
        policy: DirectXbarMacroPolicy,
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

        # Organized shape: (*batch, M=1, Tc, Tr, col_num, D, row_num).
        # The trailing (col_num, D, row_num) is owned by the xbar.
        *w_batch, n_logical, k_logical = w_logical_shape
        tr = (n_logical + col_num - 1) // col_num
        tc = (k_logical + row_num - 1) // row_num

        self.xbar = self._build_xbar(
            xbar_config=xbar_config,
            xbar_policy=policy.xbar,
            inst_shape=(*w_batch, 1, tc, tr),
        )
        xbar = self.xbar

        self.w_transcoder = Transcoder.create(
            encoding=config.w_encoding,
            radix=xbar.w_digit_radix,
            digit_num=xbar.w_digit_count,
        )

        self._w_parallel_size = max(math.prod(w_batch), 1)
        self._n_logical = n_logical
        self._row_tile_num = tr

        prefix = f"{name}." if name else ""
        self.col_accumulator = Accumulator(
            config=config.col_accumulator_config,
            name=f"{prefix}col_accumulator",
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
        return self.w_transcoder.value_range

    @property
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive integer activation range accepted by the macro."""
        return self.xbar.x_range

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
            Tensor of shape ``[..., M=1, Tc, Tr, col_num, D, row_num]``.
        """
        col_num = self.xbar.col_num
        row_num = self.xbar.row_num

        # Shape: [..., N, K] -> [..., N, K, D]
        encoded = self.w_transcoder.encode(weight, dim=-1)

        # Shape: [..., N, K, D] -> [..., Tr, col_num, K, D]
        tiled = self.chunk_pad_along(encoded, axis=-3, chunk_size=col_num, pad_value=0)
        # Shape: [..., Tr, col_num, K, D] -> [..., Tr, col_num, Tc, row_num, D]
        tiled = self.chunk_pad_along(tiled, axis=-2, chunk_size=row_num, pad_value=0)

        # Shape: [..., Tr, col_num, Tc, row_num, D] -> [..., Tc, Tr, col_num, D, row_num]
        b = tiled.ndim - 5
        perm = [*range(b), b + 2, b + 0, b + 1, b + 4, b + 3]
        arranged = tiled.permute(perm)

        # Shape: [..., Tc, Tr, col_num, D, row_num] -> [..., M=1, Tc, Tr, col_num, D, row_num]
        return arranged.unsqueeze(b)

    def _organize_x(self, x: Tensor) -> Tensor:
        """Map a logical activation tensor into xbar-native layout.

        Args:
            x: Integer activation tensor of shape ``[..., M, K]``.

        Returns:
            Tensor of shape ``[..., M, Tc, Tr=1, row_num]``.
        """
        # Shape: [..., M, K] -> [..., M, Tc, row_num]
        tiled = self.chunk_pad_along(x, axis=-1, chunk_size=self.xbar.row_num, pad_value=0)
        # Shape: [..., M, Tc, row_num] -> [..., M, Tc, Tr=1, row_num]
        return tiled.unsqueeze(-2)

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

        x = self._organize_x(input)

        # Shape: [..., M, Tc, Tr=1, row_num] -> [..., M, Tc, Tr, col_num]
        y = self.xbar.vec_mat_mul(x, adc_operation_point=adc_operation_point).to(torch.int64)
        # Shape: [..., M, Tc, Tr, col_num] -> [..., M, Tr, col_num]
        y = self.col_accumulator.operate(y, dim=-3)
        # Shape: [..., M, Tr, col_num] -> [..., M, N]
        return y.flatten(start_dim=-2)[..., :n_logical]
