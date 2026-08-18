"""Geometric weight placement, input routing, and contraction aggregation.

See Also:
    docs/reference/architecture/unit/cim/engine/placement.md
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
    contraction_accumulator_config: AccumulatorConfig
    """Accumulator folding the Tc axis."""


class PlacementStagePolicy(PolicyBase):
    pass


class PlacementStage(ModuleBase[PlacementStageConfig, PlacementStagePolicy]):
    """Pair geometric weight placement with input scheduling and aggregation."""

    is_profile_target: ClassVar[bool] = False

    # === Functional buffers ===

    _block_slot_mask: Tensor  # Shape: [D, input_num]
    _input_source_index: Tensor  # Shape: [D, input_num]

    def __init__(
        self,
        *,
        config: PlacementStageConfig,
        policy: PlacementStagePolicy,
        plan: MatmulPlacementPlan,
        input_num: int,
        macro_plane_num: int,
        macro_inst_rank: int,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=())
        self.plan = plan
        self._input_num = input_num
        self._macro_inst_rank = macro_inst_rank

        self._register_block_slot_routing_buffers()
        self._init_contraction_accumulator(macro_plane_num=macro_plane_num)

    @property
    def block_step_num(self) -> int:
        """Sequential CIM block steps one call unrolls — the D axis."""
        return self.plan.block_slot_num

    def _register_block_slot_routing_buffers(self) -> None:
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

    def _init_contraction_accumulator(self, *, macro_plane_num: int) -> None:
        """Construct the Tc accumulator at its physical multiplicity."""
        plan = self.plan
        macro_group_num = plan.block_group_num
        self.contraction_accumulator = Accumulator(
            config=self.config.contraction_accumulator_config,
            policy=DigitalPolicy(),
            inst_shape=(macro_plane_num, macro_group_num),
        )

    def partition_weight(self, weight: Tensor) -> Tensor:
        """Partition sliced weights into canonical geometric blocks."""
        plan = self.plan
        # Zero-padding after slicing is exact: an all-zero digit vector encodes
        # value 0 in every positional encoding.
        # Shape: [N, K, Sw] -> [B, Q, K, Sw]
        blocks = _chunk_pad_along(
            weight,
            axis=-3,
            chunk_size=plan.output_block_size,
            pad_value=0,
        )
        # Shape: [B, Q, K, Sw] -> [B, Q, Tc, L, Sw]
        blocks = _chunk_pad_along(
            blocks,
            axis=-2,
            chunk_size=plan.contraction_block_size,
            pad_value=0,
        )
        # Shape: [B, Q, Tc, L, Sw] -> [D, G, Q, Tc, L, Sw]
        return _chunk_pad_along(
            blocks,
            axis=-5,
            chunk_size=plan.block_group_num,
            pad_value=0,
        )

    def pack_weight(self, weight: Tensor) -> Tensor:
        """Pack canonical `[D, L]` slots into the macro input axis."""
        # Shape: [Sw, Tc, G, D, L, output_num] -> [Sw, Tc, G, D*L, output_num]
        packed = weight.flatten(start_dim=-3, end_dim=-2)
        # Shape: [Sw, Tc, G, D*L, output_num] -> [Sw, Tc, G, input_num, output_num]
        packed = F.pad(packed, (0, 0, 0, self._input_num - packed.shape[-2]))
        # Shape: [Sw, Tc, G, input_num, output_num] -> [M=1, Sx=1, Sw, Tc, G, input_num, output_num]
        return packed.unsqueeze(0).unsqueeze(0)

    def organize_x(self, x: Tensor) -> Tensor:
        """Partition sliced inputs into the canonical macro-aligned layout."""
        # Zero-padding after slicing is exact: an all-zero digit vector encodes
        # value 0 in every positional encoding.
        # Shape: [..., M, K, Sx] -> [..., M, Tc, L, Sx]
        tiled = _chunk_pad_along(
            x,
            axis=-2,
            chunk_size=self.plan.contraction_block_size,
            pad_value=0,
        )
        # Shape: [..., M, Tc, L, Sx] -> [..., M, Sx, Tc, L]
        b = tiled.ndim - 4
        tiled = tiled.permute([*range(b), b, b + 3, b + 1, b + 2])
        # Shape: [..., M, Sx, Tc, L] -> [..., M, Sx, Sw=1, Tc, G=1, L]
        return tiled.unsqueeze(b + 2).unsqueeze(b + 4)

    def unroll_block_steps(self, x: Tensor) -> Tensor:
        """Route phased local inputs through CIM block steps."""
        inst_rank = self._macro_inst_rank
        # Shape: [..., M, Sx, Sw, Tc, G, P, L] -> [..., M, Sx, Sw, Tc, G, P, D, input_num]
        routed = x[..., self._input_source_index]
        # Shape: [..., M, Sx, Sw, Tc, G, P, D, input_num] -> [..., D, M, Sx, Sw, Tc, G, P, input_num]
        routed = routed.movedim(-2, -(inst_rank + 3))
        # Shape: [..., D, M, Sx, Sw, Tc, G, P, input_num] -> [..., D, P, M, Sx, Sw, Tc, G, input_num]
        routed = routed.movedim(-2, -(inst_rank + 2))

        # Shape: [D, input_num] -> [D, P=1, *inst_shape=1, input_num]
        mask = self._block_slot_mask.reshape(
            self.plan.block_slot_num,
            1,
            *(1,) * inst_rank,
            self._input_num,
        )
        # Shape: [..., D, P, M, Sx, Sw, Tc, G, input_num]
        return torch.where(mask, routed, routed.new_zeros(()))

    def accumulate_contraction_tiles(self, code: Tensor) -> Tensor:
        """Accumulate Tc-axis codes before precision-slice reconstruction."""
        return self.contraction_accumulator.accumulate(code, dim=-3)

    def restore_output(self, code: Tensor) -> Tensor:
        """Restore balanced `[D, G, Q]` blocks to logical output order."""
        # Shape: [..., D, M, G, Q] -> [..., M, D, G, Q]
        code = code.movedim(-4, -3)
        # Shape: [..., M, D, G, Q] -> [..., M, N]
        return code.flatten(start_dim=-3)[..., : self.plan.logical_output_num]
