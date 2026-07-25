"""Direct engine: unsliced mapping onto the CIM-macro grid.

See also:
    docs/reference/architecture/unit/cim/engine/direct.md
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from neurox.primitive.digital import (
    Accumulator,
    DigitalPolicy,
    SerialAccumulator,
)

from .base import CimEngine, CimEngineConfig, CimEnginePolicy, _chunk_pad_along


class DirectCimEngineConfig(CimEngineConfig):
    """Configuration for :class:`DirectCimEngine`."""


class DirectCimEnginePolicy(CimEnginePolicy):
    """Policy for :class:`DirectCimEngine`."""


@CimEngine.register_neurox_module(config_type=DirectCimEngineConfig, policy_type=DirectCimEnginePolicy)
class DirectCimEngine(CimEngine[DirectCimEngineConfig, DirectCimEnginePolicy]):
    """CIM engine without activation or weight slicing."""

    def __init__(
        self,
        *,
        config: DirectCimEngineConfig,
        policy: DirectCimEnginePolicy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_macro: bool,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_macro=ideal_macro,
        )
        input_num = config.input_num
        output_num = config.output_num

        *w_batch, n_logical, k_logical = w_logical_shape
        tr = (n_logical + output_num - 1) // output_num
        tc = (k_logical + input_num - 1) // input_num

        self._init_engine_backend(
            inst_shape=(*w_batch, 1, tc, tr),
            n_logical=n_logical,
            k_logical=k_logical,
            w_parallel_size=math.prod(w_batch),
            row_tile_num=tr,
        )
        self._init_data_path_children(tc=tc, tr=tr)

    def _init_data_path_children(self, *, tc: int, tr: int) -> None:
        """Initialize value domains and construct digital accumulators."""
        config = self.config
        cim_macro = self.cim_macro
        self._w_value_range = cim_macro.w_value_range
        self._x_value_range = cim_macro.x_value_range
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

    def _organize_w(self, weight: Tensor) -> Tensor:
        """Map a logical weight tensor into macro-native layout.

        Args:
            weight: Integer weight tensor of shape ``[..., N, K]``.

        Returns:
            Tensor of shape
            ``[..., M=1, Tc, Tr, input_num, output_num]``.
        """
        input_num = self.config.input_num
        output_num = self.config.output_num

        # Shape: [..., N, K] -> [..., Tr, output_num, K]
        tiled = _chunk_pad_along(weight, axis=-2, chunk_size=output_num, pad_value=0)
        # Shape: [..., Tr, output_num, K] -> [..., Tr, output_num, Tc, input_num]
        tiled = _chunk_pad_along(tiled, axis=-1, chunk_size=input_num, pad_value=0)

        # Shape: [..., Tr, output_num, Tc, input_num] -> [..., Tc, Tr, input_num, output_num]
        b = tiled.ndim - 4
        perm = [*range(b), b + 2, b + 0, b + 3, b + 1]
        arranged = tiled.permute(perm)

        # Shape: [..., Tc, Tr, input_num, output_num] -> [..., M=1, Tc, Tr, input_num, output_num]
        return arranged.unsqueeze(b)

    def _organize_x(self, x: Tensor) -> Tensor:
        """Map a logical activation tensor into macro-native layout.

        Args:
            x: Integer activation tensor of shape ``[..., M, K]``.

        Returns:
            Tensor of shape ``[..., M, Tc, Tr=1, input_num]``.
        """
        # Shape: [..., M, K] -> [..., M, Tc, input_num]
        tiled = _chunk_pad_along(x, axis=-1, chunk_size=self.config.input_num, pad_value=0)
        # Shape: [..., M, Tc, input_num] -> [..., M, Tc, Tr=1, input_num]
        return tiled.unsqueeze(-2)

    @torch.no_grad()
    def matmul(self, input: Tensor, *, adc_mode: int, adc_bits: int) -> Tensor:
        n_logical = self._n_logical

        # Shape: [..., M, K] -> [..., M, Tc, Tr=1, input_num]
        x = self._organize_x(input)

        # Shape: [..., M, Tc, Tr, input_num] -> [..., P, M, Tc, Tr, input_num]
        phases = self._unroll_input_phase(x)
        # Weight-batch axes are materialized by broadcast against the instance grid.
        # Shape: [..., P, M, Tc, Tr, input_num] -> [..., P, *w_batch, M, Tc, Tr, output_num]
        y = self.cim_macro.vec_mat_mul(phases, adc_mode=adc_mode, adc_bits=adc_bits).to(torch.int64)
        # Shape: [..., P, *w_batch, M, Tc, Tr, output_num] -> [..., *w_batch, M, Tc, Tr, output_num]
        y = self.phase_accumulator.accumulate(y, dim=self._input_phase_dim)
        # Shape: [..., M, Tc, Tr, output_num] -> [..., M, Tr, output_num]
        y = self.col_accumulator.accumulate(y, dim=-3)
        # Shape: [..., M, Tr, output_num] -> [..., M, N]
        return y.flatten(start_dim=-2)[..., :n_logical]
