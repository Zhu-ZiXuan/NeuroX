"""Tests for substrate-independent matrix-multiplication mapping plans."""

from __future__ import annotations

import pytest
import torch

from neurox.architecture.unit import (
    make_activation_group_mask,
    make_block_slot_routing,
    make_input_activation_plan,
    make_matmul_placement_plan,
)


@pytest.mark.parametrize(
    (
        "logical_output_num",
        "logical_contraction_num",
        "tile_input_capacity",
        "output_block_size",
        "expected",
    ),
    [
        pytest.param(2, 8, 8, 2, (8, 1, 1, 1, 1, 1), id="exact-single-block"),
        pytest.param(2, 10, 10, 2, (10, 1, 1, 1, 1, 1), id="nondivisible-activation-only"),
        pytest.param(2, 16, 8, 2, (8, 2, 1, 1, 1, 1), id="exact-contraction-partitions"),
        pytest.param(2, 18, 8, 2, (8, 3, 1, 1, 1, 1), id="padded-contraction-partition"),
        pytest.param(6, 5, 8, 2, (5, 1, 3, 1, 3, 1), id="short-block-without-packing"),
        pytest.param(4, 3, 8, 2, (3, 1, 2, 2, 1, 2), id="two-blocks-in-one-group"),
        pytest.param(16, 2, 8, 2, (2, 1, 8, 4, 2, 4), id="balanced-full-groups"),
        pytest.param(10, 2, 8, 2, (2, 1, 5, 4, 2, 3), id="balanced-padded-group"),
        pytest.param(10, 3, 8, 4, (3, 1, 3, 2, 2, 2), id="output-and-group-padding"),
        pytest.param(176, 25, 128, 16, (25, 1, 11, 5, 3, 4), id="large-balanced-case"),
    ],
)
def test_make_matmul_placement_plan_covers_geometry_cases(
    logical_output_num: int,
    logical_contraction_num: int,
    tile_input_capacity: int,
    output_block_size: int,
    expected: tuple[int, int, int, int, int, int],
) -> None:
    plan = make_matmul_placement_plan(
        logical_output_num=logical_output_num,
        logical_contraction_num=logical_contraction_num,
        tile_input_capacity=tile_input_capacity,
        output_block_size=output_block_size,
    )

    assert (
        plan.contraction_block_size,
        plan.contraction_partition_num,
        plan.output_block_num,
        plan.block_group_capacity,
        plan.block_group_num,
        plan.block_slot_num,
    ) == expected


@pytest.mark.parametrize(
    ("input_block_size", "active_input_limit", "expected_group_num"),
    [
        (8, 8, 1),
        (8, 2, 4),
        (10, 3, 4),
        (3, 2, 2),
        (25, 16, 2),
    ],
)
def test_make_input_activation_plan_covers_activation_cases(
    input_block_size: int,
    active_input_limit: int,
    expected_group_num: int,
) -> None:
    plan = make_input_activation_plan(
        input_block_size=input_block_size,
        active_input_limit=active_input_limit,
    )

    assert plan.activation_group_num == expected_group_num


def test_block_slot_routing_maps_local_inputs_into_each_slot() -> None:
    placement = make_matmul_placement_plan(
        logical_output_num=4,
        logical_contraction_num=3,
        tile_input_capacity=8,
        output_block_size=2,
    )
    routing = make_block_slot_routing(placement=placement)

    assert torch.equal(
        routing.gather_index,
        torch.tensor(
            [
                [0, 1, 2, 2, 2, 2, 2, 2],
                [0, 0, 0, 0, 1, 2, 2, 2],
            ]
        ),
    )
    assert torch.equal(
        routing.slot_mask,
        torch.tensor(
            [
                [True, True, True, False, False, False, False, False],
                [False, False, False, True, True, True, False, False],
            ]
        ),
    )


def test_activation_group_mask_partitions_one_local_input_block() -> None:
    activation = make_input_activation_plan(
        input_block_size=3,
        active_input_limit=2,
    )
    mask = make_activation_group_mask(activation=activation)

    assert torch.equal(
        mask,
        torch.tensor(
            [
                [True, True, False],
                [False, False, True],
            ]
        ),
    )


def test_activation_limit_does_not_change_geometric_placement() -> None:
    placement = make_matmul_placement_plan(
        logical_output_num=176,
        logical_contraction_num=25,
        tile_input_capacity=128,
        output_block_size=16,
    )
    whole_block = make_input_activation_plan(
        input_block_size=placement.contraction_block_size,
        active_input_limit=25,
    )
    split_block = make_input_activation_plan(
        input_block_size=placement.contraction_block_size,
        active_input_limit=16,
    )

    assert whole_block.activation_group_num == 1
    assert split_block.activation_group_num == 2
    assert placement.block_group_num == 3
    assert placement.block_slot_num == 4
