"""Direct engine: transcode-only mapping onto the xbar grid.

See also:
    docs/reference/architecture/unit/cim/engine/direct.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.architecture.unit.cim.slicer import DirectSlicer
from neurox.common.encoding import Transcoder
from neurox.primitive.digital import (
    Accumulator,
    DigitalPolicy,
    SerialAccumulator,
)

from .base import CimEngine, CimEngineConfig, CimEnginePolicy

# Largest integer count fp32 represents exactly (2^24): any per-tile dot
# product below this bound survives an fp32 matmul bit-exactly.
_FP32_EXACT_BOUND = 1 << 24


@dataclass(frozen=True)
class DirectCimEngineConfig(CimEngineConfig):
    """Configuration for :class:`DirectCimEngine`.

    No slice counts and no shift-adders — the direct engine maps logical
    weights / activations straight onto one xbar's value range.
    """


@CimEngine.register_key(DirectCimEngineConfig)
class DirectCimEngine(CimEngine[DirectCimEngineConfig]):
    """CIM engine with no activation / weight slicing — transcode only.

    The activation slicing step is the identity :class:`DirectSlicer`
    over the xbar's ``x_range``: inputs are already on the macro's
    per-cycle grid, so ``slice`` only validates the range. Weights are
    transcoded onto the xbar's digit geometry.

    Construction rejects geometries whose worst-case per-tile dot product
    reaches 2^24: below that bound every per-tile partial survives an fp32
    matmul bit-exactly, which keeps the integer MAC engine GPU-capable
    (CUDA has no integer matmul kernel).
    """

    def __init__(
        self,
        *,
        config: DirectCimEngineConfig,
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

        # Organized shape: (*batch, M=1, Tc, Tr, col_num, D, row_num).
        # The trailing (col_num, D, row_num) is owned by the xbar.
        *w_batch, n_logical, k_logical = w_logical_shape
        tr = (n_logical + col_num - 1) // col_num
        tc = (k_logical + row_num - 1) // row_num

        self._init_engine_backend(
            inst_shape=(*w_batch, 1, tc, tr),
            n_logical=n_logical,
            k_logical=k_logical,
            w_parallel_size=max(math.prod(w_batch), 1),
            row_tile_num=tr,
        )
        xbar = self.xbar

        self.w_transcoder = Transcoder.create(
            encoding=config.w_encoding,
            radix=xbar.w_digit_radix,
            digit_count=xbar.w_digit_count,
        )
        self.x_slicer = DirectSlicer(value_range=xbar.x_range)
        self._w_value_range = self.w_transcoder.value_range
        self._x_value_range = self.x_slicer.value_range

        # fp32-exactness bound on the worst-case per-tile dot product.
        x_lo, x_hi = xbar.x_range
        d_lo, d_hi = xbar.w_digit_range
        max_digit_abs = max(abs(d_lo), abs(d_hi))
        max_w_abs = max_digit_abs * sum(xbar.w_digit_radix**k for k in range(xbar.w_digit_count))
        max_x_abs = max(abs(x_lo), abs(x_hi))
        max_tile_dot_abs = row_num * max_w_abs * max_x_abs
        if not (max_tile_dot_abs < _FP32_EXACT_BOUND):
            raise ValueError(
                f"require: row_num * max|w| * max|x| ({max_tile_dot_abs}) < 2^24 "
                "so per-tile dot products survive an fp32 matmul bit-exactly"
            )
        self._max_tile_dot_abs = max_tile_dot_abs

        self.phase_accumulator = SerialAccumulator(
            config=config.phase_accumulator_config,
            policy=DigitalPolicy(),
            inst_shape=(self._w_parallel_size, tc, tr),
        )
        self.col_accumulator = Accumulator(
            config=config.col_accumulator_config,
            policy=DigitalPolicy(),
            inst_shape=(self._w_parallel_size, tr),
        )

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
        # Shape: [..., M, K] -> [..., M, K, 1, 1] -> [..., M, K]
        # Identity slicing: validates the range, changes nothing.
        x = self.x_slicer.slice(x).squeeze(-1).squeeze(-1)
        # Shape: [..., M, K] -> [..., M, Tc, row_num]
        tiled = self.chunk_pad_along(x, axis=-1, chunk_size=self.xbar.row_num, pad_value=0)
        # Shape: [..., M, Tc, row_num] -> [..., M, Tc, Tr=1, row_num]
        return tiled.unsqueeze(-2)

    # --- lifecycle ---

    @torch.no_grad()
    def matmul(self, input: Tensor, *, adc_mode: int, adc_bits: int) -> Tensor:
        n_logical = self._n_logical

        # Shape: [..., M, K] -> [..., M, Tc, Tr=1, row_num]
        x = self._organize_x(input)

        # Shape: [..., M, Tc, Tr, row_num] -> [..., P, M, Tc, Tr, row_num]
        planes = self._unroll_sub_phase(x)
        # *w_batch~ = weight-batch axes materialized by broadcast against the inst grid.
        # Shape: [..., P, M, Tc, Tr, row_num] -> [..., P, *w_batch~, M, Tc, Tr, col_num]
        y = self.xbar.vec_mat_mul(planes, adc_mode=adc_mode, adc_bits=adc_bits).to(torch.int64)
        # Shape: [..., P, *w_batch~, M, Tc, Tr, col_num] -> [..., *w_batch~, M, Tc, Tr, col_num]
        y = self.phase_accumulator.operate(y, dim=self._sub_phase_dim)  # -(b+5)
        # Shape: [..., M, Tc, Tr, col_num] -> [..., M, Tr, col_num]
        y = self.col_accumulator.operate(y, dim=-3)
        # Shape: [..., M, Tr, col_num] -> [..., M, N]
        return y.flatten(start_dim=-2)[..., :n_logical]
