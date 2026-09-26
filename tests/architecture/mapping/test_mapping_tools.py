"""Sliced matrix products survive tiling and input-slot placement."""

from __future__ import annotations

import pytest
import torch

from neurox.architecture.mapping import InputSlotMerge
from neurox.architecture.mapping.tiler import SimpleTiler


@pytest.mark.parametrize("merge_enabled", [False, True])
@pytest.mark.parametrize(
    ("matrix_input_num", "matrix_output_num", "input_capacity", "output_capacity", "slice_num"),
    [(3, 9, 8, 8, 3), (11, 13, 4, 5, 2), (1, 1, 8, 4, 1), (3, 10, 8, 4, 1)],
)
def test_mapping_tools_recover_each_precision_slice(
    merge_enabled: bool,
    matrix_input_num: int,
    matrix_output_num: int,
    input_capacity: int,
    output_capacity: int,
    slice_num: int,
    device: torch.device,
) -> None:
    tiler = SimpleTiler(
        matrix_input_num=matrix_input_num,
        matrix_output_num=matrix_output_num,
        input_per_tile=min(matrix_input_num, input_capacity),
        output_per_tile=output_capacity,
        w_slice_num=slice_num,
    )
    merge = InputSlotMerge(
        input_per_tile=tiler.input_per_tile,
        output_tile_num=tiler.output_tile_num,
        macro_input_num=input_capacity,
        enabled=merge_enabled,
    )
    # Shape: [w_slice, output, input]
    w = (
        torch.arange(slice_num * matrix_output_num * matrix_input_num, device=device).reshape(
            slice_num, matrix_output_num, matrix_input_num
        )
        % 5
        - 2
    )
    # Shape: [batch, vector, input]
    x = torch.arange(3 * 4 * matrix_input_num, device=device).reshape(3, 4, matrix_input_num) % 7 - 3
    # Shape: [batch, vector, w_slice, output]
    expected = (x.cpu().unsqueeze(-3) @ w.cpu().transpose(-1, -2)).transpose(-3, -2).to(device)
    # Shape: [in_tile, macro_group, macro_input, tile_out]
    mapped_w = merge.map_w(tiler.map_w(w))
    # Shape: [batch, vector, in_tile, merge_step, macro_group, macro_input]
    mapped_x = merge.map_x(tiler.map_x(x))
    # Shape: [batch, vector, in_tile, merge_step, macro_group, tile_out]
    output = (mapped_x.unsqueeze(-1) * mapped_w.unsqueeze(-4)).sum(dim=-2)
    actual = tiler.recover(merge.recover(output))
    torch.testing.assert_close(actual, expected)
    counts = merge.map_effective_output_num(tiler.effective_output_num(device=x.device))
    assert torch.count_nonzero(mapped_x.masked_select((counts == 0).unsqueeze(-1))) == 0
