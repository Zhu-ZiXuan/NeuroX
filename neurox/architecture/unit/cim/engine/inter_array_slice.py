"""Inter-array slice engine: ``Sw`` distributed across xbar planes (Strategy 1).

See also:
    docs/reference/architecture/unit/cim/engine/inter_array_slice.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.architecture.unit.cim.slicer import SerialSlicer, SimpleSlicer
from neurox.primitive.digital import (
    Accumulator,
    DigitalPolicy,
    SerialAccumulator,
    ShiftAdder,
    ShiftAdderConfig,
)

from .base import CimEngine, CimEngineConfig, CimEnginePolicy


@dataclass(frozen=True)
class InterArraySliceCimEngineConfig(CimEngineConfig):
    """Configuration for :class:`InterArraySliceCimEngine`.

    Attributes:
        w_slice_num: Per-weight Sw slice count.
        x_slice_num: Per-activation Sa slice count.
        sa_shift_adder_config: Sa-axis intra-xbar shift-adder config.
        sw_shift_adder_config: Sw-axis cross-xbar shift-adder config.
    """

    w_slice_num: int
    x_slice_num: int

    sa_shift_adder_config: ShiftAdderConfig
    sw_shift_adder_config: ShiftAdderConfig

    def validate(self) -> None:
        super().validate()
        self._require_pos(self.w_slice_num, "w_slice_num")
        self._require_pos(self.x_slice_num, "x_slice_num")


@CimEngine.register_key(InterArraySliceCimEngineConfig)
class InterArraySliceCimEngine(CimEngine[InterArraySliceCimEngineConfig]):
    """CIM engine that distributes weight slices across separate xbar planes.

    One xbar plane holds one ``Sw`` slice index across every logical weight.
    """

    def __init__(
        self,
        *,
        config: InterArraySliceCimEngineConfig,
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

        # Organized shape: (*batch, M=1, Sa=1, Sw, Tc, Tr, col_num, D, row_num).
        # The trailing (col_num, D, row_num) is owned by the xbar.
        *w_batch, n_logical, k_logical = w_logical_shape
        tr = (n_logical + col_num - 1) // col_num
        tc = (k_logical + row_num - 1) // row_num
        sw = config.w_slice_num

        self._init_engine_backend(
            inst_shape=(*w_batch, 1, 1, sw, tc, tr),
            n_logical=n_logical,
            k_logical=k_logical,
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

        self.phase_accumulator = SerialAccumulator(
            config=config.phase_accumulator_config,
            policy=DigitalPolicy(),
            inst_shape=(self._w_parallel_size, sw, tc, tr),
        )
        self.col_accumulator = Accumulator(
            config=config.col_accumulator_config,
            policy=DigitalPolicy(),
            inst_shape=(self._w_parallel_size, sw, tr),
        )
        self.sa_shift_adder = ShiftAdder(
            config=config.sa_shift_adder_config,
            policy=DigitalPolicy(),
            inst_shape=(self._w_parallel_size, tr),
        )
        self.sw_shift_adder = ShiftAdder(
            config=config.sw_shift_adder_config,
            policy=DigitalPolicy(),
            inst_shape=(self._w_parallel_size, tr),
        )

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

    @torch.no_grad()
    def matmul(self, input: Tensor, *, adc_mode: int, adc_bits: int) -> Tensor:
        n_logical = self._n_logical

        # Shape: [..., M, K] -> [..., M, Sa, Sw=1, Tc, Tr=1, row_num]
        x = self._organize_x(input)

        x_slice_radix = self.x_slicer.slice_radix
        w_slice_radix = self.w_slicer.slice_radix

        # Shape: [..., M, Sa, Sw, Tc, Tr, row_num] -> [..., P, M, Sa, Sw, Tc, Tr, row_num]
        planes = self._unroll_sub_phase(x)
        # *w_batch~ = weight-batch axes materialized by broadcast against the inst grid.
        # Shape: [..., P, M, Sa, Sw, Tc, Tr, row_num] -> [..., P, *w_batch~, M, Sa, Sw, Tc, Tr, data_num]
        y = self.xbar.vec_mat_mul(planes, adc_mode=adc_mode, adc_bits=adc_bits).to(torch.int64)
        # Shape: [..., P, *w_batch~, M, Sa, Sw, Tc, Tr, data_num] -> [..., *w_batch~, M, Sa, Sw, Tc, Tr, data_num]
        y = self.phase_accumulator.operate(y, dim=self._sub_phase_dim)  # -(b+7)
        # Shape: [..., M, Sa, Sw, Tc, Tr, data_num] -> [..., M, Sw, Tc, Tr, data_num]
        y = self.sa_shift_adder.operate(y, x_slice_radix, dim=-5, init_val=None)
        # Shape: [..., M, Sw, Tc, Tr, data_num] -> [..., M, Tc, Tr, data_num]
        y = self.sw_shift_adder.operate(y, w_slice_radix, dim=-4, init_val=None)
        # Shape: [..., M, Tc, Tr, data_num] -> [..., M, Tr, data_num]
        y = self.col_accumulator.operate(y, dim=-3)
        # Shape: [..., M, Tr, data_num] -> [..., M, N]
        return y.flatten(start_dim=-2)[..., :n_logical]
