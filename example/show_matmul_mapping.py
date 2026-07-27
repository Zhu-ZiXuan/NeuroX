"""Print representative fixed-capacity matrix-multiplication mappings."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from neurox.architecture.unit.matmul_mapping import (
    InputActivationPlan,
    MatmulPlacementPlan,
    make_activation_group_mask,
    make_block_slot_routing,
    make_input_activation_plan,
    make_matmul_placement_plan,
)


@dataclass(frozen=True, slots=True)
class MappingCase:
    """One geometry and activation-limit example.

    Attributes:
        name: Human-readable purpose of the example.
        logical_output_num: Number of logical output values.
        logical_contraction_num: Length of the logical contraction dimension.
        tile_input_capacity: Number of input positions in one compute tile.
        output_block_size: Number of logical outputs in one weight block.
        active_input_limit: Maximum selected inputs in one activation group.
    """

    name: str
    logical_output_num: int
    logical_contraction_num: int
    tile_input_capacity: int
    output_block_size: int
    active_input_limit: int


CASES = (
    MappingCase("exact single block", 2, 8, 8, 2, 8),
    MappingCase("activation groups with exact division", 2, 8, 8, 2, 2),
    MappingCase("activation groups with a short remainder", 2, 10, 10, 2, 3),
    MappingCase("exact contraction partitions", 2, 16, 8, 2, 4),
    MappingCase("padded final contraction partition", 2, 18, 8, 2, 4),
    MappingCase("short block without packing capacity", 6, 5, 8, 2, 3),
    MappingCase("two output blocks packed into one group", 4, 3, 8, 2, 2),
    MappingCase("balanced full block groups", 16, 2, 8, 2, 2),
    MappingCase("balanced block groups with one empty slot", 10, 2, 8, 2, 1),
    MappingCase("output padding and one empty block slot", 10, 3, 8, 4, 2),
    MappingCase("128-input practical-scale example", 176, 25, 128, 16, 16),
)


def _append_inputs(lines: list[str], case: MappingCase) -> None:
    lines.extend(
        [
            "1. Inputs",
            f"   logical_output_num       = {case.logical_output_num}",
            f"   logical_contraction_num  = {case.logical_contraction_num}",
            f"   tile_input_capacity      = {case.tile_input_capacity}",
            f"   output_block_size        = {case.output_block_size}",
            f"   active_input_limit       = {case.active_input_limit}",
        ]
    )


def _append_geometry(lines: list[str], plan: MatmulPlacementPlan) -> None:
    lines.extend(
        [
            "2. Geometric placement",
            f"   contraction_block_size   = {plan.contraction_block_size}",
            f"   contraction_partition_num= {plan.contraction_partition_num}",
            f"   output_block_num         = {plan.output_block_num}",
            f"   block_group_capacity     = {plan.block_group_capacity}",
            f"   block_group_num          = {plan.block_group_num}",
            f"   block_slot_num           = {plan.block_slot_num}",
        ]
    )


def _append_contraction_partitions(lines: list[str], plan: MatmulPlacementPlan) -> None:
    lines.append("3. Contraction partitions")
    for partition_index in range(plan.contraction_partition_num):
        start = partition_index * plan.contraction_block_size
        stop = min(start + plan.contraction_block_size, plan.logical_contraction_num)
        padding = plan.contraction_block_size - (stop - start)
        lines.append(f"   partition {partition_index}: logical input [{start}, {stop}), padding={padding}")


def _append_output_blocks(lines: list[str], plan: MatmulPlacementPlan) -> None:
    lines.append("4. Output-block placement")
    for slot_index in range(plan.block_slot_num):
        input_start = slot_index * plan.contraction_block_size
        input_stop = input_start + plan.contraction_block_size
        for group_index in range(plan.block_group_num):
            block_index = slot_index * plan.block_group_num + group_index
            if block_index >= plan.output_block_num:
                assignment = "padding"
            else:
                output_start = block_index * plan.output_block_size
                output_stop = min(
                    output_start + plan.output_block_size,
                    plan.logical_output_num,
                )
                assignment = f"output block {block_index} -> logical output [{output_start}, {output_stop})"
            lines.append(
                f"   group {group_index}, slot {slot_index}, tile input [{input_start}, {input_stop}): {assignment}"
            )


def _append_activation_groups(lines: list[str], activation: InputActivationPlan) -> None:
    lines.append("5. Local activation groups")
    for group_index in range(activation.activation_group_num):
        start = group_index * activation.active_input_limit
        stop = min(start + activation.active_input_limit, activation.input_block_size)
        lines.append(f"   activation group {group_index}: local input [{start}, {stop})")


def _append_composed_routing(
    lines: list[str],
    plan: MatmulPlacementPlan,
    activation: InputActivationPlan,
) -> None:
    lines.append("6. Consumer-composed input routing")
    routing = make_block_slot_routing(placement=plan)
    activation_mask = make_activation_group_mask(activation=activation)
    for partition_index in range(plan.contraction_partition_num):
        partition_start = partition_index * plan.contraction_block_size
        for slot_index in range(plan.block_slot_num):
            for activation_group_index in range(activation.activation_group_num):
                local_indices = routing.gather_index[slot_index]
                mask = routing.slot_mask[slot_index] & activation_mask[activation_group_index, local_indices]
                tile_positions = mask.nonzero(as_tuple=False).flatten()
                sources: list[str] = []
                for tile_position_tensor in tile_positions:
                    tile_position = int(tile_position_tensor)
                    local_index = int(routing.gather_index[slot_index, tile_position])
                    logical_index = partition_start + local_index
                    source = f"x[{logical_index}]" if logical_index < plan.logical_contraction_num else "padding"
                    sources.append(f"tile[{tile_position}]<-{source}")
                lines.append(
                    f"   partition {partition_index}, slot {slot_index}, "
                    f"activation group {activation_group_index}: {', '.join(sources)}"
                )


def describe_case(case: MappingCase) -> str:
    """Return a step-by-step description of one mapping case.

    Args:
        case: Geometry and activation-limit inputs.

    Returns:
        Multi-line mapping description.
    """
    placement = make_matmul_placement_plan(
        logical_output_num=case.logical_output_num,
        logical_contraction_num=case.logical_contraction_num,
        tile_input_capacity=case.tile_input_capacity,
        output_block_size=case.output_block_size,
    )
    activation = make_input_activation_plan(
        input_block_size=placement.contraction_block_size,
        active_input_limit=case.active_input_limit,
    )

    lines = [f"=== {case.name} ==="]
    _append_inputs(lines, case)
    _append_geometry(lines, placement)
    _append_contraction_partitions(lines, placement)
    _append_output_blocks(lines, placement)
    _append_activation_groups(lines, activation)
    _append_composed_routing(lines, placement, activation)
    return "\n".join(lines)


def main() -> None:
    """Print all representative mapping cases."""
    sys.stdout.write("\n\n".join(describe_case(case) for case in CASES))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
