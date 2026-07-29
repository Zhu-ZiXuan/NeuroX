"""Composable execution engine for CIM matrix multiplication.

See also:
    docs/internals/architecture/unit/cim/engine/base.md
"""

from __future__ import annotations

import math
from typing import ClassVar

import torch
from torch import Tensor

from neurox.architecture.unit.matmul_mapping import MatmulPlacementPlan, make_matmul_placement_plan
from neurox.common import ConfigBase, ModuleBase, PolicyBase
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy

from .input_activation import (
    InputActivationStage,
    InputActivationStageConfig,
    InputActivationStagePolicy,
)
from .placement import PlacementStage, PlacementStageConfig, PlacementStagePolicy
from .weight_slice import WeightSliceStage, WeightSliceStageConfig, WeightSliceStagePolicy
from .x_slice import XSliceStage, XSliceStageConfig, XSliceStagePolicy


class CimEngineConfig(ConfigBase):
    """Configuration for :class:`CimEngine`.

    Attributes:
        input_num: Logical input ports of each CIM macro.
        output_num: Logical output ports of each CIM macro.
        cim_macro_config: CIM macro configuration.
        placement: Geometric placement configuration.
        input_activation: Max-active input scheduling configuration.
        weight_slice: Weight-slice layout configuration.
        x_slice: Input-slice serialization configuration.
    """

    input_num: int
    output_num: int
    cim_macro_config: CimMacroConfig
    placement: PlacementStageConfig
    input_activation: InputActivationStageConfig
    weight_slice: WeightSliceStageConfig
    x_slice: XSliceStageConfig

    def validate(self) -> None:
        """Validate the engine configuration."""
        self._require_pos(self.input_num, "input_num")
        self._require_pos(self.output_num, "output_num")


class CimEnginePolicy(PolicyBase):
    """Policy for :class:`CimEngine`.

    Attributes:
        cim_macro_policy: Embedded CIM-macro policy.
        placement: Geometric placement policy.
        input_activation: Max-active input scheduling policy.
        weight_slice: Weight-slice layout policy.
        x_slice: Input-slice serialization policy.
    """

    cim_macro_policy: CimMacroPolicy
    placement: PlacementStagePolicy
    input_activation: InputActivationStagePolicy
    weight_slice: WeightSliceStagePolicy
    x_slice: XSliceStagePolicy


class CimEngine(ModuleBase[CimEngineConfig, CimEnginePolicy]):
    """Map one logical matrix multiplication onto CIM macros.

    Args:
        config: Engine configuration.
        policy: Composite engine policy.
        w_logical_shape: Weight shape ``(*prefix, N, K)`` bound to
            :meth:`program`.
        dtype: Tensor dtype used by the CIM macro.
        T__K: Operating temperature.
        ideal_macro: Whether to replace the configured macro with its ideal
            counterpart.
    """

    is_profile_target: ClassVar[bool] = False

    def __init__(
        self,
        *,
        config: CimEngineConfig,
        policy: CimEnginePolicy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_macro: bool,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=())
        if len(w_logical_shape) < 2:
            raise ValueError(f"w_logical_shape must have at least 2 trailing dims (N, K); got {w_logical_shape}")
        self._w_logical_shape = tuple(w_logical_shape)

        *w_batch, n_logical, k_logical = w_logical_shape
        block_output_num, macro_plane_num = config.weight_slice.layout_geometry(output_num=config.output_num)
        plan = make_matmul_placement_plan(
            logical_output_num=n_logical,
            logical_contraction_num=k_logical,
            tile_input_capacity=config.input_num,
            output_block_size=block_output_num,
        )
        self._init_execution_children(
            plan=plan,
            w_batch=tuple(w_batch),
            macro_plane_num=macro_plane_num,
            dtype=dtype,
            T__K=T__K,
            ideal_macro=ideal_macro,
        )

    def _init_execution_children(
        self,
        *,
        plan: MatmulPlacementPlan,
        w_batch: tuple[int, ...],
        macro_plane_num: int,
        dtype: torch.dtype,
        T__K: float,
        ideal_macro: bool,
    ) -> None:
        """Construct the macro and the four paired execution stages."""
        w_parallel_size = math.prod(w_batch)
        input_tile_num = plan.contraction_partition_num
        macro_group_num = plan.block_group_num
        macro_inst_shape = (
            *w_batch,
            1,
            1,
            macro_plane_num,
            input_tile_num,
            macro_group_num,
        )
        self.cim_macro = self._build_cim_macro(
            cim_macro_config=self.config.cim_macro_config,
            cim_macro_policy=self.policy.cim_macro_policy,
            input_num=self.config.input_num,
            output_num=self.config.output_num,
            inst_shape=macro_inst_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_macro=ideal_macro,
        )
        self.placement = PlacementStage(
            config=self.config.placement,
            policy=self.policy.placement,
            plan=plan,
            input_num=self.config.input_num,
            w_batch_rank=len(w_batch),
            w_parallel_size=w_parallel_size,
            macro_plane_num=macro_plane_num,
            macro_inst_rank=len(macro_inst_shape),
        )
        self.input_activation = InputActivationStage(
            config=self.config.input_activation,
            policy=self.policy.input_activation,
            input_block_size=plan.contraction_block_size,
            max_active_num=self.cim_macro.max_active_num,
            w_parallel_size=w_parallel_size,
            macro_plane_num=macro_plane_num,
            input_tile_num=input_tile_num,
            macro_group_num=macro_group_num,
            macro_inst_rank=len(macro_inst_shape),
        )
        self.weight_slice = WeightSliceStage.from_config(
            config=self.config.weight_slice,
            policy=self.policy.weight_slice,
            macro_w_value_range=self.cim_macro.w_value_range,
            output_num=self.config.output_num,
            w_parallel_size=w_parallel_size,
            macro_group_num=macro_group_num,
        )
        self.x_slice = XSliceStage.from_config(
            config=self.config.x_slice,
            policy=self.policy.x_slice,
            macro_x_value_range=self.cim_macro.x_value_range,
            w_parallel_size=w_parallel_size,
            macro_group_num=macro_group_num,
        )

    @classmethod
    def from_config(
        cls,
        *,
        config: CimEngineConfig,
        policy: CimEnginePolicy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_macro: bool,
    ) -> CimEngine:
        """Build an engine from its complete configuration and policy."""
        return cls(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_macro=ideal_macro,
        )

    def _sample_fabricate_mismatch(self) -> None:
        pass

    @property
    def w_value_range(self) -> tuple[int, int]:
        return self.weight_slice.value_range

    @property
    def x_value_range(self) -> tuple[int, int]:
        return self.x_slice.value_range

    @property
    def input_num(self) -> int:
        return self.config.input_num

    @property
    def output_num(self) -> int:
        return self.config.output_num

    @property
    def max_active_num(self) -> int:
        return self.cim_macro.max_active_num

    @property
    def adc_max_bits(self) -> int:
        return self.cim_macro.adc_max_bits

    def rescale_factor(self, *, quantization_mode: int, adc_bits: int | None) -> float:
        """Return the macro output code expressed in ideal-macro codes."""
        return self.cim_macro.rescale_factor(quantization_mode=quantization_mode, adc_bits=adc_bits)

    def _organize_w(self, weight: Tensor) -> Tensor:
        """Map logical weights into the programmed CIM-macro layout."""
        # Shape: [..., N, K] -> [..., N, K, Sw]
        sliced = self.weight_slice.slice(weight)
        # Shape: [..., N, K, Sw] -> [..., D, G, Q, Tc, L, Sw]
        partitioned = self.placement.partition_weight(sliced)
        # Shape: [..., D, G, Q, Tc, L, Sw] -> [..., Sw, Tc, G, D, L, output_num]
        arranged = self.weight_slice.arrange_weight(partitioned)
        # Shape: [..., Sw, Tc, G, D, L, output_num] -> [..., M=1, Sa=1, Sw, Tc, G, input_num, output_num]
        return self.placement.pack_weight(arranged)

    def _organize_x(self, input: Tensor) -> Tensor:
        """Map logical inputs into the CIM-macro execution layout."""
        # Shape: [..., M, K] -> [..., M, K, Sa]
        sliced = self.x_slice.slice(input)
        # Shape: [..., M, K, Sa] -> [..., M, Sa, Sw=1, Tc, G=1, L]
        return self.placement.organize_x(sliced)

    def program(self, weight: Tensor) -> None:
        """Program one integer logical weight tensor.

        Args:
            weight: Weight tensor matching the shape bound at construction.
        """
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        self.cim_macro.program(self._organize_w(weight))

    @torch.no_grad()
    def matmul(self, input: Tensor, *, quantization_mode: int, adc_bits: int | None) -> Tensor:
        """Multiply logical inputs by the programmed weight.

        Args:
            input: Integer activation tensor of shape ``[..., M, K]``.
            quantization_mode: Runtime quantization-mode index.
            adc_bits: Runtime ADC resolution, or ``None`` for the lossless
                oracle.

        Returns:
            Integer tensor of shape ``[..., M, N]``.
        """
        # Shape: [..., M, K] -> [..., M, Sa, Sw=1, Tc, G=1, L]
        organized = self._organize_x(input)
        # Shape: [..., M, Sa, Sw, Tc, G, L] -> [..., M, Sa, Sw, Tc, G, P, L]
        phased = self.input_activation.unroll_input_phases(organized)
        # Shape: [..., M, Sa, Sw, Tc, G, P, L] -> [..., D, P, *w_batch, M, Sa, Sw, Tc, G, input_num]
        code = self.placement.unroll_block_steps(phased)
        # Shape: [..., D, P, *w_batch, M, Sa, Sw, Tc, G, input_num] -> [..., D, P, *w_batch, M, Sa, Sw, Tc, G, output_num]
        code = self.cim_macro.vec_mat_mul(code, quantization_mode=quantization_mode, adc_bits=adc_bits).to(torch.int64)
        # Shape: [..., D, P, *w_batch, M, Sa, Sw, Tc, G, output_num] -> [..., D, *w_batch, M, Sa, Sw, Tc, G, output_num]
        code = self.input_activation.accumulate_phases(code)
        # Shape: [..., D, *w_batch, M, Sa, Sw, Tc, G, output_num] -> [..., D, *w_batch, M, Sa, Sw, G, output_num]
        code = self.placement.accumulate_contraction_tiles(code)
        # Shape: [..., D, *w_batch, M, Sa, Sw, G, output_num] -> [..., D, *w_batch, M, Sa, G, Q]
        code = self.weight_slice.aggregate(code)
        # Shape: [..., D, *w_batch, M, Sa, G, Q] -> [..., D, *w_batch, M, G, Q]
        code = self.x_slice.aggregate(code)
        # Shape: [..., D, *w_batch, M, G, Q] -> [..., *w_batch, M, N]
        return self.placement.restore_output(code)

    def extra_repr(self) -> str:
        return (
            f"cim_macro={type(self.cim_macro).__name__}, "
            f"input_num={self.config.input_num}, output_num={self.config.output_num}, "
            f"w_value_range={self.w_value_range}, x_value_range={self.x_value_range}"
        )

    @staticmethod
    def _build_cim_macro(
        *,
        cim_macro_config: CimMacroConfig,
        cim_macro_policy: CimMacroPolicy,
        input_num: int,
        output_num: int,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_macro: bool,
    ) -> CimMacro:
        """Construct the configured physical or ideal CIM macro."""
        cim_macro = CimMacro.from_config(
            config=cim_macro_config,
            policy=cim_macro_policy,
            input_num=input_num,
            output_num=output_num,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )
        return cim_macro.to_ideal() if ideal_macro else cim_macro
