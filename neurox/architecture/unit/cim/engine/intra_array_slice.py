"""Intra-array slice engine: ``Sw`` gathered inside one xbar (Strategy 2).

See also:
    docs/reference/architecture/unit/cim/engine/intra_array_slice.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.architecture.unit.cim.slicer import SerialSlicer, SimpleSlicer
from neurox.primitive.analog.adc_common import AdcOperationPoint
from neurox.primitive.digital import (
    Accumulator,
    DigitalPolicy,
    SerialAccumulator,
    ShiftAdder,
    ShiftAdderConfig,
)

from .base import CimEngine, CimEngineConfig, CimEnginePolicy


@dataclass(frozen=True)
class IntraArraySliceCimEngineConfig(CimEngineConfig):
    """Configuration for :class:`IntraArraySliceCimEngine`.

    Attributes:
        w_slice_num: Per-weight Sw slice count.
        x_slice_num: Per-activation Sa slice count.
        sa_shift_adder_config: Sa-axis intra-xbar shift-adder config.
        sw_shift_adder_config: Sw-axis intra-xbar shift-adder config.
    """

    w_slice_num: int
    x_slice_num: int

    sa_shift_adder_config: ShiftAdderConfig
    sw_shift_adder_config: ShiftAdderConfig

    def validate(self) -> None:
        super().validate()
        self._require_pos(self.w_slice_num, "w_slice_num")
        self._require_pos(self.x_slice_num, "x_slice_num")


@CimEngine.register_key(IntraArraySliceCimEngineConfig)
class IntraArraySliceCimEngine(CimEngine):
    """CIM engine that gathers all slices of one logical weight in one xbar.

    A logical weight's ``Sw`` slices sit in adjacent cols of the same xbar.
    Per-xbar effective capacity is ``(col_num // Sw) * Sw`` cells; the
    remaining ``col_num - (col_num // Sw) * Sw`` cells per xbar are idle.
    """

    config: IntraArraySliceCimEngineConfig

    def __init__(
        self,
        *,
        config: IntraArraySliceCimEngineConfig,
        policy: CimEnginePolicy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_xbar: bool,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_xbar=ideal_xbar,
        )
        xbar_config = config.cim_macro_config
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

        self._init_engine_backend(
            inst_shape=(*w_batch, 1, 1, tc, tr),
            n_logical=n_logical,
            w_parallel_size=max(math.prod(w_batch), 1),
            row_tile_num=tr,
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
        self._w_value_range = self.w_slicer.value_range
        self._x_value_range = self.x_slicer.value_range

        helper_shape = (self._w_parallel_size, tr)
        self.phase_accumulator = SerialAccumulator(
            config=config.phase_accumulator_config,
            policy=DigitalPolicy(),
            inst_shape=(self._w_parallel_size, tc, tr),
        )
        self.col_accumulator = Accumulator(
            config=config.col_accumulator_config,
            policy=DigitalPolicy(),
            inst_shape=helper_shape,
        )
        self.sa_shift_adder = ShiftAdder(
            config=config.sa_shift_adder_config,
            policy=DigitalPolicy(),
            inst_shape=helper_shape,
        )
        self.sw_shift_adder = ShiftAdder(
            config=config.sw_shift_adder_config,
            policy=DigitalPolicy(),
            inst_shape=helper_shape,
        )

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
        # Shape: [..., M, K] -> [..., M, K, Sa, digit_count=1]
        sliced = self.x_slicer.slice(x)

        # Shape: [..., M, K, Sa, digit_count=1] -> [..., M, Tc, row_num, Sa, digit_count=1]
        tiled = self.chunk_pad_along(sliced, axis=-3, chunk_size=self.xbar.row_num, pad_value=0)

        # Shape: [..., M, Tc, row_num, Sa, digit_count=1] -> [..., M, Tc, row_num, Sa]
        squeezed = tiled.squeeze(-1)
        # Shape: [..., M, Tc, row_num, Sa] -> [..., M, Sa, Tc, row_num]
        b = squeezed.ndim - 4
        permuted = squeezed.permute([*range(b), b + 0, b + 3, b + 1, b + 2])
        # Shape: [..., M, Sa, Tc, row_num] -> [..., M, Sa, Tc, Tr=1, row_num]
        return permuted.unsqueeze(b + 3)

    # --- lifecycle ---

    @torch.no_grad()
    def matmul(self, input: Tensor, *, adc_operation_point: AdcOperationPoint) -> Tensor:
        n_logical = self._n_logical
        wpx = self._weights_per_xbar
        used = self._used_data_num
        sw = self.config.w_slice_num

        # Shape: [..., M, K] -> [..., M, Sa, Tc, Tr=1, row_num]
        x = self._organize_x(input)

        x_slice_radix = self.x_slicer.slice_radix
        w_slice_radix = self.w_slicer.slice_radix

        # Shape: [..., M, Sa, Tc, Tr, row_num] -> [..., P, M, Sa, Tc, Tr, row_num]
        planes = self._unroll_sub_phase(x)
        # *w_batch~ = weight-batch axes materialized by broadcast against the inst grid.
        # Shape: [..., P, M, Sa, Tc, Tr, row_num] -> [..., P, *w_batch~, M, Sa, Tc, Tr, data_num]
        y = self.xbar.vec_mat_mul(planes, adc_operation_point=adc_operation_point).to(torch.int64)
        # Shape: [..., P, *w_batch~, M, Sa, Tc, Tr, data_num] -> [..., *w_batch~, M, Sa, Tc, Tr, data_num]
        y = self.phase_accumulator.operate(y, dim=self._sub_phase_dim)  # -(b+6)
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
