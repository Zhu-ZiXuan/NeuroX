"""Geometric placement and scheduling for CIM-engine matrix multiplication.

See also:
    docs/internals/architecture/unit/cim/engine/placement.md
"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.architecture.unit.matmul_mapping import (
    MatmulPlacementPlan,
    make_block_slot_routing,
)
from neurox.common import ConfigBase, ModuleBase, PolicyBase
from neurox.primitive.digital import Accumulator, AccumulatorConfig, DigitalPolicy


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


class PlacementStageConfig(ConfigBase):
    """Configuration for :class:`PlacementStage`.

    Attributes:
        contraction_accumulator_config: Tc-axis accumulator configuration.
    """

    contraction_accumulator_config: AccumulatorConfig


class PlacementStagePolicy(PolicyBase):
    """Policy for :class:`PlacementStage`."""


class PlacementStage(ModuleBase[PlacementStageConfig, PlacementStagePolicy]):
    """Pair geometric weight placement with input scheduling and aggregation."""

    is_profile_target: ClassVar[bool] = False

    # --- Immutable execution buffers ---

    _block_slot_mask: Tensor
    _input_source_index: Tensor

    def __init__(
        self,
        *,
        config: PlacementStageConfig,
        policy: PlacementStagePolicy,
        plan: MatmulPlacementPlan,
        input_num: int,
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

        self._register_block_slot_routing_buffers()
        self._init_contraction_accumulator(
            w_parallel_size=w_parallel_size,
            macro_plane_num=macro_plane_num,
        )

    def _sample_fabricate_mismatch(self) -> None:
        pass

    def _register_block_slot_routing_buffers(self) -> None:
        """Register routing metadata for every CIM block step."""
        routing = make_block_slot_routing(placement=self.plan)
        self.register_buffer(
            "_input_source_index",
            routing.gather_index,
            persistent=False,
        )
        self.register_buffer(
            "_block_slot_mask",
            routing.slot_mask,
            persistent=False,
        )

    def _init_contraction_accumulator(self, *, w_parallel_size: int, macro_plane_num: int) -> None:
        """Construct the Tc accumulator at its physical multiplicity."""
        plan = self.plan
        macro_group_num = plan.block_group_num
        self.contraction_accumulator = Accumulator(
            config=self.config.contraction_accumulator_config,
            policy=DigitalPolicy(),
            inst_shape=(w_parallel_size, macro_plane_num, macro_group_num),
        )

    def partition_weight(self, weight: Tensor) -> Tensor:
        """Partition sliced weights into canonical geometric blocks."""
        plan = self.plan
        # Zero-padding after slicing is exact: an all-zero digit vector encodes
        # value 0 in every positional encoding.
        # Shape: [..., N, K, Sw] -> [..., B, Q, K, Sw]
        blocks = _chunk_pad_along(
            weight,
            axis=-3,
            chunk_size=plan.output_block_size,
            pad_value=0,
        )
        # Shape: [..., B, Q, K, Sw] -> [..., B, Q, Tc, L, Sw]
        blocks = _chunk_pad_along(
            blocks,
            axis=-2,
            chunk_size=plan.contraction_block_size,
            pad_value=0,
        )
        # Shape: [..., B, Q, Tc, L, Sw] -> [..., D, G, Q, Tc, L, Sw]
        return _chunk_pad_along(
            blocks,
            axis=-5,
            chunk_size=plan.block_group_num,
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
        # Zero-padding after slicing is exact: an all-zero digit vector encodes
        # value 0 in every positional encoding.
        # Shape: [..., M, K, Sa] -> [..., M, Tc, L, Sa]
        tiled = _chunk_pad_along(
            x,
            axis=-2,
            chunk_size=self.plan.contraction_block_size,
            pad_value=0,
        )
        # Shape: [..., M, Tc, L, Sa] -> [..., M, Sa, Tc, L]
        b = tiled.ndim - 4
        tiled = tiled.permute([*range(b), b, b + 3, b + 1, b + 2])
        # Shape: [..., M, Sa, Tc, L] -> [..., M, Sa, Sw=1, Tc, G=1, L]
        return tiled.unsqueeze(b + 2).unsqueeze(b + 4)

    def unroll_block_steps(self, x: Tensor) -> Tensor:
        """Route phased local inputs through CIM block steps."""
        fixed_inst_rank = self._macro_inst_rank - self._w_batch_rank
        x_prefix_rank = x.ndim - fixed_inst_rank - 2
        execution_index = max(0, x_prefix_rank - self._w_batch_rank)
        missing_w_batch_rank = max(0, self._w_batch_rank - x_prefix_rank)
        if missing_w_batch_rank:
            # Shape: [*caller, *inst, P, L] -> [*caller, *missing_w_batch=1, *inst, P, L]
            x = x.reshape(
                *x.shape[:execution_index],
                *(1,) * missing_w_batch_rank,
                *x.shape[execution_index:],
            )

        # Shape: [..., *span, P, L] -> [..., *span, P, D, input_num]
        routed = x[..., self._input_source_index]
        # Shape: [..., *span, P, D, input_num] -> [..., D, *span, P, input_num]
        routed = routed.movedim(-2, execution_index)
        # Shape: [..., D, *span, P, input_num] -> [..., D, P, *span, input_num]
        routed = routed.movedim(-2, execution_index + 1)

        plan = self.plan
        block_step_num = plan.block_slot_num
        # Shape: [D, input_num] -> [*caller=1, D, P=1, *inst=1, input_num]
        mask = self._block_slot_mask.reshape(
            *(1,) * execution_index,
            block_step_num,
            1,
            *(1,) * self._macro_inst_rank,
            self._input_num,
        )
        # Shape: [..., D, P, *span, input_num]
        return torch.where(mask, routed, routed.new_zeros(()))

    def accumulate_contraction_tiles(self, code: Tensor) -> Tensor:
        """Accumulate Tc-axis codes before precision-slice reconstruction."""
        return self.contraction_accumulator.accumulate(code, dim=-3)

    def restore_output(self, code: Tensor) -> Tensor:
        """Restore balanced ``[D,G,Q]`` blocks to logical output order."""
        # Shape: [..., D, *w_batch, M, G, Q] -> [..., *w_batch, M, D, G, Q]
        code = code.movedim(-(self._w_batch_rank + 4), -3)
        # Shape: [..., *w_batch, M, D, G, Q] -> [..., *w_batch, M, N]
        return code.flatten(start_dim=-3)[..., : self.plan.logical_output_num]
