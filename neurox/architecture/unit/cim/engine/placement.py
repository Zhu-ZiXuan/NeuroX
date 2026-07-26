"""Geometric placement and scheduling for CIM-engine matrix multiplication.

See also:
    docs/internals/architecture/unit/cim/engine/placement.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase
from neurox.primitive.digital import Accumulator, AccumulatorConfig, DigitalPolicy, SerialAccumulator


def _chunk_pad_along(
    t: Tensor,
    *,
    axis: int,
    chunk_size: int,
    pad_value: int,
) -> Tensor:
    """Right-pad and split one tensor axis into equal chunks."""
    if chunk_size < 1:
        raise ValueError(f"require: chunk_size ({chunk_size}) >= 1")
    if axis < 0:
        axis += t.ndim
    if not (0 <= axis < t.ndim):
        raise ValueError(f"require: 0 <= axis ({axis}) < ndim ({t.ndim})")
    n = t.size(axis)
    num_chunks = (n + chunk_size - 1) // chunk_size
    pad_amount = num_chunks * chunk_size - n
    if pad_amount > 0:
        # F.pad indexes from the last dim; pad axis only on the high side.
        pad_spec = [0, 0] * (t.ndim - axis - 1) + [0, pad_amount]
        t = F.pad(t, pad_spec, value=pad_value)
    chunks: Tensor = t.unflatten(axis, (num_chunks, chunk_size))
    return chunks


@dataclass(frozen=True, slots=True)
class PlacementPlan:
    """Runtime-derived geometric placement for one logical weight shape.

    Attributes:
        n_logical: Logical output-vector length.
        block_input_num: Input values stored by one logical weight block.
        block_output_num: Output values stored by one logical weight block.
        input_tile_num: Contraction-axis tile count.
        output_block_num: Logical output-block count.
        block_capacity: Maximum logical blocks packed into one macro.
        macro_group_num: Parallel macro-group count.
        block_step_num: Serial output-block step count.
    """

    n_logical: int
    block_input_num: int
    block_output_num: int
    input_tile_num: int
    output_block_num: int
    block_capacity: int
    macro_group_num: int
    block_step_num: int

    @classmethod
    def build(
        cls,
        *,
        n_logical: int,
        k_logical: int,
        input_num: int,
        block_output_num: int,
    ) -> PlacementPlan:
        """Derive balanced input-axis packing from logical and macro geometry."""
        block_input_num = min(k_logical, input_num)
        input_tile_num = -(-k_logical // block_input_num)
        output_block_num = -(-n_logical // block_output_num)
        block_capacity = input_num // block_input_num
        macro_group_num = -(-output_block_num // block_capacity)
        block_step_num = -(-output_block_num // macro_group_num)
        return cls(
            n_logical=n_logical,
            block_input_num=block_input_num,
            block_output_num=block_output_num,
            input_tile_num=input_tile_num,
            output_block_num=output_block_num,
            block_capacity=block_capacity,
            macro_group_num=macro_group_num,
            block_step_num=block_step_num,
        )


class PlacementStageConfig(ConfigBase):
    """Configuration for :class:`PlacementStage`.

    Attributes:
        phase_accumulator_config: P-axis accumulator configuration.
        contraction_accumulator_config: Tc-axis accumulator configuration.
    """

    phase_accumulator_config: AccumulatorConfig
    contraction_accumulator_config: AccumulatorConfig


class PlacementStagePolicy(PolicyBase):
    """Policy for :class:`PlacementStage`."""


class PlacementStage(ModuleBase[PlacementStageConfig, PlacementStagePolicy]):
    """Pair geometric weight placement with input scheduling and aggregation."""

    is_profile_target: ClassVar[bool] = False

    # --- Immutable execution buffers ---

    _active_input_mask: Tensor
    _input_source_index: Tensor

    def __init__(
        self,
        *,
        config: PlacementStageConfig,
        policy: PlacementStagePolicy,
        plan: PlacementPlan,
        input_num: int,
        max_active_num: int,
        w_batch_rank: int,
        w_parallel_size: int,
        macro_plane_num: int,
        macro_inst_rank: int,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=())
        self.plan = plan
        self._input_num = input_num
        self._w_batch_rank = w_batch_rank
        self._macro_inst_rank = macro_inst_rank
        self._input_phase_dim = -(macro_inst_rank + 2)
        self._input_phase_num = -(-plan.block_input_num // max_active_num)

        self._register_input_schedule_buffers(max_active_num=max_active_num)
        self._init_digital_children(
            w_parallel_size=w_parallel_size,
            macro_plane_num=macro_plane_num,
        )

    def _sample_fabricate_mismatch(self) -> None:
        pass

    def _register_input_schedule_buffers(self, *, max_active_num: int) -> None:
        """Register routing metadata for every block step and input phase."""
        plan = self.plan
        rows = torch.arange(self._input_num)
        # Shape: [D] -> [D, 1]
        block_starts = torch.arange(plan.block_step_num).unsqueeze(-1) * plan.block_input_num
        # Shape: [D, input_num]
        local_indices = rows - block_starts
        self.register_buffer(
            "_input_source_index",
            local_indices.clamp(0, plan.block_input_num - 1),
            persistent=False,
        )

        # Shape: [P] -> [1, P, 1]
        phases = torch.arange(self._input_phase_num).view(1, -1, 1)
        # Shape: [D, input_num] -> [D, 1, input_num]
        local_indices = local_indices.unsqueeze(1)
        # Shape: [D, P, input_num]
        self.register_buffer(
            "_active_input_mask",
            (local_indices >= 0) & (local_indices < plan.block_input_num) & (local_indices // max_active_num == phases),
            persistent=False,
        )

    def _init_digital_children(self, *, w_parallel_size: int, macro_plane_num: int) -> None:
        """Construct accumulators at their physical parallel multiplicities."""
        plan = self.plan
        self.phase_accumulator = SerialAccumulator(
            config=self.config.phase_accumulator_config,
            policy=DigitalPolicy(),
            inst_shape=(
                w_parallel_size,
                macro_plane_num,
                plan.input_tile_num,
                plan.macro_group_num,
            ),
        )
        self.contraction_accumulator = Accumulator(
            config=self.config.contraction_accumulator_config,
            policy=DigitalPolicy(),
            inst_shape=(w_parallel_size, macro_plane_num, plan.macro_group_num),
        )

    def partition_weight(self, weight: Tensor) -> Tensor:
        """Partition sliced weights into canonical geometric blocks."""
        plan = self.plan
        # Shape: [..., N, K, Sw] -> [..., B, Q, K, Sw]
        blocks = _chunk_pad_along(
            weight,
            axis=-3,
            chunk_size=plan.block_output_num,
            pad_value=0,
        )
        # Shape: [..., B, Q, K, Sw] -> [..., B, Q, Tc, L, Sw]
        blocks = _chunk_pad_along(
            blocks,
            axis=-2,
            chunk_size=plan.block_input_num,
            pad_value=0,
        )
        # Shape: [..., B, Q, Tc, L, Sw] -> [..., D, G, Q, Tc, L, Sw]
        return _chunk_pad_along(
            blocks,
            axis=-5,
            chunk_size=plan.macro_group_num,
            pad_value=0,
        )

    def pack_weight(self, weight: Tensor) -> Tensor:
        """Pack canonical ``[D,L]`` slots into the macro input axis."""
        # Shape: [..., Sw, Tc, G, D, L, output_num] -> [..., Sw, Tc, G, D*L, output_num]
        packed = weight.flatten(start_dim=-3, end_dim=-2)
        # Shape: [..., Sw, Tc, G, D*L, output_num] -> [..., Sw, Tc, G, input_num, output_num]
        packed = F.pad(packed, (0, 0, 0, self._input_num - packed.shape[-2]))
        b = packed.ndim - 5
        # Shape: [..., Sw, Tc, G, input_num, output_num] -> [..., M=1, Sa=1, Sw, Tc, G, input_num, output_num]
        return packed.unsqueeze(b).unsqueeze(b)

    def organize_x(self, x: Tensor) -> Tensor:
        """Partition sliced inputs into the canonical macro-aligned layout."""
        # Shape: [..., M, K, Sa] -> [..., M, Tc, L, Sa]
        tiled = _chunk_pad_along(
            x,
            axis=-2,
            chunk_size=self.plan.block_input_num,
            pad_value=0,
        )
        # Shape: [..., M, Tc, L, Sa] -> [..., M, Sa, Tc, L]
        b = tiled.ndim - 4
        tiled = tiled.permute([*range(b), b, b + 3, b + 1, b + 2])
        # Shape: [..., M, Sa, Tc, L] -> [..., M, Sa, Sw=1, Tc, G=1, L]
        return tiled.unsqueeze(b + 2).unsqueeze(b + 4)

    def unroll_input_schedule(self, x: Tensor) -> Tensor:
        """Route local input blocks through balanced block steps and phases."""
        fixed_inst_rank = self._macro_inst_rank - self._w_batch_rank
        x_prefix_rank = x.ndim - fixed_inst_rank - 1
        execution_index = max(0, x_prefix_rank - self._w_batch_rank)
        missing_w_batch_rank = max(0, self._w_batch_rank - x_prefix_rank)
        if missing_w_batch_rank:
            # Shape: [*caller, *inst, L] -> [*caller, *missing_w_batch=1, *inst, L]
            x = x.reshape(
                *x.shape[:execution_index],
                *(1,) * missing_w_batch_rank,
                *x.shape[execution_index:],
            )

        # Shape: [..., *span, L] -> [..., *span, D, input_num]
        routed = x[..., self._input_source_index]
        # Shape: [..., *span, D, input_num] -> [..., D, *span, input_num]
        routed = routed.movedim(-2, execution_index)
        # Shape: [..., D, *span, input_num] -> [..., D, P=1, *span, input_num]
        routed = routed.unsqueeze(execution_index + 1)

        plan = self.plan
        # Shape: [D, P, input_num] -> [*caller=1, D, P, *inst=1, input_num]
        mask = self._active_input_mask.reshape(
            *(1,) * execution_index,
            plan.block_step_num,
            self._input_phase_num,
            *(1,) * self._macro_inst_rank,
            self._input_num,
        )
        # Shape: [..., D, P=1, *span, input_num] -> [..., D, P, *span, input_num]
        return torch.where(mask, routed, routed.new_zeros(()))

    def accumulate_phases(self, code: Tensor) -> Tensor:
        """Accumulate P-axis codes produced by one logical input block."""
        return self.phase_accumulator.accumulate(code, dim=self._input_phase_dim)

    def accumulate_contraction_tiles(self, code: Tensor) -> Tensor:
        """Accumulate Tc-axis codes before precision-slice reconstruction."""
        return self.contraction_accumulator.accumulate(code, dim=-3)

    def restore_output(self, code: Tensor) -> Tensor:
        """Restore balanced ``[D,G,Q]`` blocks to logical output order."""
        # Shape: [..., D, *w_batch, M, G, Q] -> [..., *w_batch, M, D, G, Q]
        code = code.movedim(-(self._w_batch_rank + 4), -3)
        # Shape: [..., *w_batch, M, D, G, Q] -> [..., *w_batch, M, N]
        return code.flatten(start_dim=-3)[..., : self.plan.n_logical]
