"""Substrate-independent planning for fixed-capacity matrix multiplication.

See also:
    docs/internals/architecture/unit/matmul_mapping.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class MatmulPlacementPlan:
    """Geometric placement of one logical matrix multiplication."""

    logical_output_num: int
    logical_contraction_num: int
    tile_input_capacity: int
    """Input positions one compute tile offers."""
    output_block_size: int
    """Logical outputs carried by one weight block."""
    contraction_block_size: int
    """Contraction values stored by one weight block."""
    contraction_partition_num: int
    output_block_num: int
    block_group_capacity: int
    """Weight blocks one block group holds at most."""
    block_group_num: int
    block_slot_num: int
    """Input-axis block slots used in each block group."""


@dataclass(frozen=True, slots=True)
class InputActivationPlan:
    """Partition one input block into bounded activation groups."""

    input_block_size: int
    active_input_limit: int
    """Selected values one activation group may hold at most."""
    activation_group_num: int
    """Activation groups needed to cover the block."""


@dataclass(frozen=True, slots=True)
class BlockSlotRouting:
    """Input routing for geometric block slots."""

    gather_index: Tensor
    """Local input index each tile input position reads.
    Shape: `[block_slot, tile_input]`."""
    slot_mask: Tensor
    """Tile input positions belonging to each block slot.
    Shape: `[block_slot, tile_input]`."""


def make_matmul_placement_plan(
    *,
    logical_output_num: int,
    logical_contraction_num: int,
    tile_input_capacity: int,
    output_block_size: int,
) -> MatmulPlacementPlan:
    """Plan geometric partitioning and input-axis weight-block packing."""
    contraction_block_size = min(logical_contraction_num, tile_input_capacity)
    contraction_partition_num = -(-logical_contraction_num // contraction_block_size)
    output_block_num = -(-logical_output_num // output_block_size)
    block_group_capacity = tile_input_capacity // contraction_block_size
    block_group_num = -(-output_block_num // block_group_capacity)
    block_slot_num = -(-output_block_num // block_group_num)
    return MatmulPlacementPlan(
        logical_output_num=logical_output_num,
        logical_contraction_num=logical_contraction_num,
        tile_input_capacity=tile_input_capacity,
        output_block_size=output_block_size,
        contraction_block_size=contraction_block_size,
        contraction_partition_num=contraction_partition_num,
        output_block_num=output_block_num,
        block_group_capacity=block_group_capacity,
        block_group_num=block_group_num,
        block_slot_num=block_slot_num,
    )


def make_input_activation_plan(
    *,
    input_block_size: int,
    active_input_limit: int,
) -> InputActivationPlan:
    """Partition a local input block without assuming an execution schedule."""
    return InputActivationPlan(
        input_block_size=input_block_size,
        active_input_limit=active_input_limit,
        activation_group_num=-(-input_block_size // active_input_limit),
    )


def make_block_slot_routing(
    *,
    placement: MatmulPlacementPlan,
) -> BlockSlotRouting:
    """Map local input indices into geometric block slots."""
    input_positions = torch.arange(placement.tile_input_capacity)
    # Shape: [block_slot] -> [block_slot, 1]
    block_starts = torch.arange(placement.block_slot_num).unsqueeze(-1) * placement.contraction_block_size
    # Shape: [block_slot, tile_input]
    local_indices = input_positions - block_starts
    gather_index = local_indices.clamp(0, placement.contraction_block_size - 1)
    # Shape: [block_slot, tile_input]
    slot_mask = (local_indices >= 0) & (local_indices < placement.contraction_block_size)
    return BlockSlotRouting(
        gather_index=gather_index,
        slot_mask=slot_mask,
    )


def make_activation_group_mask(*, activation: InputActivationPlan) -> Tensor:
    """Select local input positions belonging to each activation group.

    Returns:
        Boolean membership mask of one activation group.
        Shape: `[activation_group, block_input]`.
    """
    input_positions = torch.arange(activation.input_block_size)
    # Shape: [activation_group] -> [activation_group, 1]
    activation_groups = torch.arange(activation.activation_group_num).unsqueeze(-1)
    # Shape: [activation_group, block_input]
    return input_positions // activation.active_input_limit == activation_groups
