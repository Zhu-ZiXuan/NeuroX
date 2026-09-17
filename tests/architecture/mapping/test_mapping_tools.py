"""Sliced matrix products survive independent tiling and input-slot placement."""

from __future__ import annotations

import pytest
import torch

from neurox.architecture.mapping import InputSlotMerge, OutputSliceTiling, PlaneSliceTiling, Tiling


@pytest.mark.parametrize("tiling_type", [PlaneSliceTiling, OutputSliceTiling])
@pytest.mark.parametrize("merge_enabled", [False, True])
@pytest.mark.parametrize(
    ("input_num", "output_num", "input_capacity", "output_capacity", "slice_num"),
    [(3, 9, 8, 8, 3), (11, 13, 4, 5, 2), (1, 1, 8, 4, 1)],
)
def test_mapping_tools_recover_each_precision_slice(
    tiling_type: type[Tiling],
    merge_enabled: bool,
    input_num: int,
    output_num: int,
    input_capacity: int,
    output_capacity: int,
    slice_num: int,
    device: torch.device,
) -> None:
    tiling = tiling_type(
        input_num=input_num,
        output_num=output_num,
        tile_input_capacity=input_capacity,
        tile_output_capacity=output_capacity,
        w_slice_num=slice_num,
    )
    merge = InputSlotMerge(
        tile_input_num=tiling.tile_input_num,
        logical_tile_output_num=tiling.logical_tile_output_num,
        output_num=tiling.output_num,
        macro_input_num=input_capacity,
        enabled=merge_enabled,
    )
    # Shape: [w_slice, output, input]
    w = (
        torch.arange(slice_num * output_num * input_num, device=device).reshape(slice_num, output_num, input_num) % 5
        - 2
    )
    # Shape: [batch, vector, input]
    x = torch.arange(3 * 4 * input_num, device=device).reshape(3, 4, input_num) % 7 - 3
    # Shape: [batch, vector, w_slice, output]
    expected = (x.cpu().unsqueeze(-3) @ w.cpu().transpose(-1, -2)).transpose(-3, -2).to(device)
    # Shape: [macro_plane, in_tile, macro_group, macro_input, tile_output]
    mapped_w = merge.map_w(tiling.map_w(w))
    # Shape: [batch, vector, macro_plane=1, in_tile, merge_step, macro_group, macro_input]
    mapped_x = merge.map_x(tiling.map_x(x)).unsqueeze(-5)
    # Shape: [batch, vector, macro_plane, in_tile, merge_step, macro_group, tile_output]
    output = (mapped_x.unsqueeze(-1) * mapped_w.unsqueeze(-4)).sum(dim=-2)
    actual = tiling.recover(merge.recover(output))
    torch.testing.assert_close(actual, expected)
    counts = merge.effective_output_num(device=x.device)
    assert torch.count_nonzero(mapped_x.masked_select((counts == 0).unsqueeze(-1))) == 0
    if tiling_type is OutputSliceTiling:
        used_ports = tiling.logical_tile_output_num * slice_num
        assert torch.count_nonzero(mapped_w[..., used_ports:]) == 0


@pytest.mark.parametrize(
    (
        "logical_output_num",
        "logical_input_num",
        "tile_input_capacity",
        "tile_output_num",
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
        pytest.param(33, 1, 32, 1, (1, 1, 33, 32, 2, 17), id="seventeen-plus-sixteen"),
    ],
)
def test_mapping_tools_cover_geometry_cases(
    logical_output_num: int,
    logical_input_num: int,
    tile_input_capacity: int,
    tile_output_num: int,
    expected: tuple[int, int, int, int, int, int],
) -> None:
    tiling = PlaneSliceTiling(
        output_num=logical_output_num,
        input_num=logical_input_num,
        tile_input_capacity=tile_input_capacity,
        tile_output_capacity=tile_output_num,
        w_slice_num=1,
    )

    merge = InputSlotMerge(
        tile_input_num=tiling.tile_input_num,
        logical_tile_output_num=tiling.logical_tile_output_num,
        output_num=tiling.output_num,
        macro_input_num=tile_input_capacity,
        enabled=True,
    )

    assert (
        tiling.tile_input_num,
        tiling.in_tile_num,
        tiling.out_tile_num,
        merge.input_slot_capacity,
        merge.macro_group_num,
        merge.merge_step_num,
    ) == expected
