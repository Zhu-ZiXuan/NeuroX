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
    """Geometric placement of one logical matrix multiplication.

    Attributes:
        logical_output_num: Number of logical output values.
        logical_contraction_num: Length of the logical contraction dimension.
        tile_input_capacity: Number of input positions in one compute tile.
        output_block_size: Number of logical outputs in one weight block.
        contraction_block_size: Contraction values stored by one weight block.
        contraction_partition_num: Number of contraction partitions.
        output_block_num: Number of logical output blocks.
        block_group_capacity: Maximum weight blocks held by one block group.
        block_group_num: Number of block groups receiving output blocks.
        block_slot_num: Number of input-axis block slots used in each group.
    """

    logical_output_num: int
    logical_contraction_num: int
    tile_input_capacity: int
    output_block_size: int
    contraction_block_size: int
    contraction_partition_num: int
    output_block_num: int
    block_group_capacity: int
    block_group_num: int
    block_slot_num: int


@dataclass(frozen=True, slots=True)
class InputActivationPlan:
    """Partition one input block into bounded activation groups.

    Attributes:
        input_block_size: Number of input values in the block.
        active_input_limit: Maximum selected values in one activation group.
        activation_group_num: Number of activation groups covering the block.
    """

    input_block_size: int
    active_input_limit: int
    activation_group_num: int


@dataclass(frozen=True, slots=True)
class BlockSlotRouting:
    """Input routing for geometric block slots.

    Attributes:
        gather_index: Local input indices with shape
            ``[block_slot, tile_input]``.
        slot_mask: Tile input positions belonging to each block slot, with
            shape ``[block_slot, tile_input]``.
    """

    gather_index: Tensor
    slot_mask: Tensor


def make_matmul_placement_plan(
    *,
    logical_output_num: int,
    logical_contraction_num: int,
    tile_input_capacity: int,
    output_block_size: int,
) -> MatmulPlacementPlan:
    """Plan geometric partitioning and input-axis weight-block packing.

    Args:
        logical_output_num: Number of logical output values.
        logical_contraction_num: Length of the logical contraction dimension.
        tile_input_capacity: Number of input positions in one compute tile.
        output_block_size: Number of logical outputs in one weight block.

    Returns:
        Substrate-independent geometric placement.
    """
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
    """Partition a local input block without assuming an execution schedule.

    Args:
        input_block_size: Number of input values in the block.
        active_input_limit: Maximum selected values in one activation group.

    Returns:
        Activation-group partition independent of geometric placement.
    """
    return InputActivationPlan(
        input_block_size=input_block_size,
        active_input_limit=active_input_limit,
        activation_group_num=-(-input_block_size // active_input_limit),
    )


def make_block_slot_routing(
    *,
    placement: MatmulPlacementPlan,
) -> BlockSlotRouting:
    """Map local input indices into geometric block slots.

    Args:
        placement: Geometric placement defining block slots.

    Returns:
        Gather indices and slot membership over tile input positions.
    """
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

    Args:
        activation: Local activation partition for one input block.

    Returns:
        Boolean mask with shape ``[activation_group, block_input]``.
    """
    input_positions = torch.arange(activation.input_block_size)
    # Shape: [activation_group] -> [activation_group, 1]
    activation_groups = torch.arange(activation.activation_group_num).unsqueeze(-1)
    # Shape: [activation_group, block_input]
    return input_positions // activation.active_input_limit == activation_groups
