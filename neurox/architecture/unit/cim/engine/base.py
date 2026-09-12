"""Composable execution engine for CIM matrix multiplication.

See Also:
    docs/reference/architecture/unit/cim/engine/family.md
    docs/system_design/cim_execution.md
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import Tensor

from neurox.architecture.unit.matmul_mapping import MatmulPlacementPlan, make_matmul_placement_plan
from neurox.common.module import ConfigBase, ModuleBase, PolicyBase
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
    cim_macro_config: CimMacroConfig
    placement: PlacementStageConfig
    input_activation: InputActivationStageConfig
    weight_slice: WeightSliceStageConfig
    x_slice: XSliceStageConfig


class CimEnginePolicy(PolicyBase):
    cim_macro_policy: CimMacroPolicy
    placement: PlacementStagePolicy
    input_activation: InputActivationStagePolicy
    weight_slice: WeightSliceStagePolicy
    x_slice: XSliceStagePolicy


_Config = CimEngineConfig
_Policy = CimEnginePolicy


class CimEngine(ModuleBase):
    """Map one logical matrix multiplication onto CIM macros.

    Args:
        w_logical_shape: Weight shape `(N, K)` bound to `program`.
        ideal_macro: Whether to replace the configured macro with its ideal
            counterpart.
    """

    is_profile_target: ClassVar[bool] = False

    config: _Config
    policy: _Policy

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_macro: bool,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=())
        if len(w_logical_shape) != 2:
            raise ValueError(f"w_logical_shape must be (N, K); got {w_logical_shape}")
        self._w_logical_shape = tuple(w_logical_shape)

        n_logical, k_logical = w_logical_shape
        macro_config = config.cim_macro_config
        block_output_num, macro_plane_num = config.weight_slice.layout_geometry(output_num=macro_config.output_num)
        plan = make_matmul_placement_plan(
            logical_output_num=n_logical,
            logical_contraction_num=k_logical,
            tile_input_capacity=macro_config.input_num,
            output_block_size=block_output_num,
        )
        self._init_execution_children(
            plan=plan,
            macro_plane_num=macro_plane_num,
            dtype=dtype,
            T__K=T__K,
            ideal_macro=ideal_macro,
        )

    def latency__ns(
        self,
        *,
        output_plane_num: int,
        adc_active_bits: int | None,
    ) -> float:
        """Return the latency of one logical matrix multiplication.

        Serial work spans the output planes `M` supplied per call, the input
        slices `Sx`, the CIM block slots `D`, and the input phases `P`. One
        macro access serves each `(M, Sx, D, P)` point, so these extents
        multiply the macro latency. Weight slices, contraction
        partitions, and block groups are parallel silicon and do not.

        The digital blocks hold no output-port axis of their own, so each runs
        once per output element the operation it closes delivers: the macro's
        `output_num` ports for the two accumulators, the ports one Sw
        aggregation leaves for the two reconstructions. How often that
        operation happens follows from how each block consumes its reduced
        axis. The phase accumulator folds successive arrivals into one
        register, so it runs once per macro access; the adder-tree and
        positional-sum reductions close their whole axis in a single window,
        so they run once per step the axis completes on.

        Args:
            output_plane_num: Output planes `M` one call unrolls — the only
                extent of the schedule the placement plan does not fix.
                Unrelated to `macro_plane_num`, the Sw weight-slice planes one
                macro instance holds.
            adc_active_bits: Active ADC resolution the macro accesses run at;
                `None` requests the macro's highest available precision.

        Returns:
            Latency of one logical matrix multiplication.
        """
        slice_num = self.x_slice.slice_num
        block_step_num = self.placement.block_step_num
        phase_num = self.input_activation.input_phase_num
        port_num = self.cim_macro.output_num
        aggregated_port_num = self.weight_slice.aggregated_output_num

        phase_accumulate__ns = self.input_activation.phase_accumulator.latency__ns()
        contraction_accumulate__ns = self.placement.contraction_accumulator.latency__ns()
        w_slice_recombine__ns = (
            self.weight_slice.shift_adder.latency__ns() if self.weight_slice.shift_adder is not None else 0.0
        )
        x_slice_recombine__ns = self.x_slice.shift_adder.latency__ns() if self.x_slice.shift_adder is not None else 0.0

        # One block step retires when its phases have been accumulated.
        step_num = output_plane_num * slice_num * block_step_num
        access_num = step_num * phase_num
        total__ns = access_num * self.cim_macro.latency__ns(adc_active_bits=adc_active_bits)
        total__ns += access_num * port_num * phase_accumulate__ns
        total__ns += step_num * port_num * contraction_accumulate__ns
        total__ns += step_num * aggregated_port_num * w_slice_recombine__ns
        # The input slices are reconstructed once every slice has arrived.
        total__ns += output_plane_num * block_step_num * aggregated_port_num * x_slice_recombine__ns
        return total__ns

    def _init_execution_children(
        self,
        *,
        plan: MatmulPlacementPlan,
        macro_plane_num: int,
        dtype: torch.dtype,
        T__K: float,
        ideal_macro: bool,
    ) -> None:
        input_tile_num = plan.contraction_partition_num
        macro_group_num = plan.block_group_num
        macro_inst_shape = (
            1,
            1,
            macro_plane_num,
            input_tile_num,
            macro_group_num,
        )
        self.cim_macro = self._build_cim_macro(
            cim_macro_config=self.config.cim_macro_config,
            cim_macro_policy=self.policy.cim_macro_policy,
            # Shape: [M=1, Sx=1, Sw, Tc, G]
            inst_shape=macro_inst_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_macro=ideal_macro,
        )
        self.placement = PlacementStage(
            config=self.config.placement,
            policy=self.policy.placement,
            plan=plan,
            input_num=self.cim_macro.input_num,
            macro_plane_num=macro_plane_num,
            macro_inst_rank=len(macro_inst_shape),
        )
        self.input_activation = InputActivationStage(
            config=self.config.input_activation,
            policy=self.policy.input_activation,
            input_block_size=plan.contraction_block_size,
            max_active_num=self.cim_macro.max_active_num,
            macro_plane_num=macro_plane_num,
            input_tile_num=input_tile_num,
            macro_group_num=macro_group_num,
            macro_inst_rank=len(macro_inst_shape),
        )
        self.weight_slice = WeightSliceStage.from_config(
            config=self.config.weight_slice,
            policy=self.policy.weight_slice,
            macro_w_value_range=self.cim_macro.w_value_range,
            output_num=self.cim_macro.output_num,
            macro_group_num=macro_group_num,
        )
        self.x_slice = XSliceStage.from_config(
            config=self.config.x_slice,
            policy=self.policy.x_slice,
            macro_x_value_range=self.cim_macro.x_value_range,
            macro_group_num=macro_group_num,
        )

    @classmethod
    def from_config(
        cls,
        *,
        config: _Config,
        policy: _Policy,
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

    @property
    def w_value_range(self) -> tuple[int, int]:
        return self.weight_slice.value_range

    @property
    def x_value_range(self) -> tuple[int, int]:
        return self.x_slice.value_range

    @property
    def input_num(self) -> int:
        return self.cim_macro.input_num

    @property
    def output_num(self) -> int:
        return self.cim_macro.output_num

    @property
    def max_active_num(self) -> int:
        return self.cim_macro.max_active_num

    @property
    def adc_bits(self) -> int:
        return self.cim_macro.adc_bits

    def rescale_factor(
        self,
        *,
        quantization_mode: int,
        adc_active_bits: int | None,
    ) -> float:
        """Return the ideal-macro codes represented by one output code."""
        return self.cim_macro.rescale_factor(quantization_mode=quantization_mode, adc_active_bits=adc_active_bits)

    def _organize_w(self, weight: Tensor) -> Tensor:
        """Map logical weights into the programmed CIM-macro layout."""
        # Shape: [N, K] -> [N, K, Sw]
        sliced = self.weight_slice.slice(weight)
        # Shape: [N, K, Sw] -> [D, G, Q, Tc, L, Sw]
        partitioned = self.placement.partition_weight(sliced)
        # Shape: [D, G, Q, Tc, L, Sw] -> [Sw, Tc, G, D, L, output]
        arranged = self.weight_slice.arrange_weight(partitioned)
        # Shape: [Sw, Tc, G, D, L, output] -> [M=1, Sx=1, Sw, Tc, G, input, output]
        return self.placement.pack_weight(arranged)

    def _organize_x(self, input: Tensor) -> Tensor:
        """Map logical inputs into the CIM-macro execution layout."""
        # Shape: [..., M, K] -> [..., M, K, Sx]
        sliced = self.x_slice.slice(input)
        # Shape: [..., M, K, Sx] -> [..., M, Sx, Sw=1, Tc, G=1, L]
        return self.placement.organize_x(sliced)

    @torch.no_grad()
    def program(self, weight: Tensor) -> None:
        """Program one integer logical weight tensor.

        Args:
            weight: Weight tensor matching the shape bound at construction.
                Shape: `[N, K]`.
        """
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        self.cim_macro.program(self._organize_w(weight))

    @torch.no_grad()
    def matmul(
        self,
        input: Tensor,
        *,
        quantization_mode: int,
        adc_active_bits: int | None,
    ) -> Tensor:
        """Multiply logical inputs by the programmed weight.

        Args:
            input: Integer activation values.
                Shape: `[..., M, K]`.
            quantization_mode: Index selecting the runtime quantization window.
            adc_active_bits: Active ADC resolution; `None` requests the
                macro's highest available precision.

        Returns:
            Integer pre-requantize output tensor.
            Shape: `[..., M, N]`.
        """
        # Shape: [..., M, K] -> [..., M, Sx, Sw=1, Tc, G=1, L]
        organized = self._organize_x(input)
        # Shape: [..., M, Sx, Sw, Tc, G, L] -> [..., M, Sx, Sw, Tc, G, P, L]
        phased = self.input_activation.unroll_input_phases(organized)
        # Shape: [..., M, Sx, Sw, Tc, G, P, L] -> [..., D, P, M, Sx, Sw, Tc, G, input]
        code = self.placement.unroll_block_steps(phased)
        # Shape: [..., D, P, M, Sx, Sw, Tc, G, input] -> [..., D, P, M, Sx, Sw, Tc, G, output]
        code = self.cim_macro.vec_mat_mul(
            code,
            quantization_mode=quantization_mode,
            adc_active_bits=adc_active_bits,
        ).to(torch.int64)
        # Shape: [..., D, P, M, Sx, Sw, Tc, G, output] -> [..., D, M, Sx, Sw, Tc, G, output]
        code = self.input_activation.accumulate_phases(code)
        # Shape: [..., D, M, Sx, Sw, Tc, G, output] -> [..., D, M, Sx, Sw, G, output]
        code = self.placement.accumulate_contraction_tiles(code)
        # Shape: [..., D, M, Sx, Sw, G, output] -> [..., D, M, Sx, G, Q]
        code = self.weight_slice.aggregate(code)
        # Shape: [..., D, M, Sx, G, Q] -> [..., D, M, G, Q]
        code = self.x_slice.aggregate(code)
        # Shape: [..., D, M, G, Q] -> [..., M, N]
        return self.placement.restore_output(code)

    def extra_repr(self) -> str:
        return (
            f"cim_macro={type(self.cim_macro).__name__}, "
            f"input_num={self.input_num}, output_num={self.output_num}, "
            f"w_value_range={self.w_value_range}, x_value_range={self.x_value_range}"
        )

    @staticmethod
    def _build_cim_macro(
        *,
        cim_macro_config: CimMacroConfig,
        cim_macro_policy: CimMacroPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_macro: bool,
    ) -> CimMacro:
        cim_macro = CimMacro.from_config(
            config=cim_macro_config,
            policy=cim_macro_policy,
            # Shape: [*inst_shape]
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )
        return cim_macro.to_ideal() if ideal_macro else cim_macro
