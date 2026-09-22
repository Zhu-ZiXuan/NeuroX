"""Matrix-tile placement in disjoint, serially selected macro input slots."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class InputSlotMerge:
    """Place consecutive output tiles across macro groups, then input slots.

    Args:
        enabled: Whether multiple output tiles may share a macro's input slots.
    """

    def __init__(
        self,
        *,
        input_per_tile: int,
        output_tile_num: int,
        macro_input_num: int,
        enabled: bool,
    ) -> None:
        self.input_per_tile = input_per_tile
        self.output_tile_num = output_tile_num
        self.macro_input_num = macro_input_num
        self.input_slot_capacity = macro_input_num // input_per_tile if enabled else 1
        self.macro_group_num = -(-output_tile_num // self.input_slot_capacity)
        self.merge_step_num = -(-output_tile_num // self.macro_group_num)

    def map_effective_output_num(self, counts: Tensor) -> Tensor:
        """Arrange tile output counts in the same reuse slots as the weights.

        Args:
            counts: Valid leading output-port counts for each tile.
                Shape: `[out_tile]`.

        Returns:
            Valid output-port counts, with zero for absent tiles.
            Shape: `[merge_step, macro_group]`.
        """
        out_tile_padding = self.merge_step_num * self.macro_group_num - self.output_tile_num
        counts = F.pad(counts, (0, out_tile_padding))
        # Shape: [merge_step*macro_group] -> [merge_step, macro_group]
        grouped: Tensor = counts.unflatten(-1, (self.merge_step_num, self.macro_group_num))
        return grouped

    def map_x(self, x: Tensor) -> Tensor:
        """Route a tile input into independently enabled physical slots.

        Args:
            x: Tile inputs; leading axes may include input tiles and precision slices.
                Shape: `[..., tile_in]`.

        Returns:
            Physical inputs, with empty slots zero-filled.
            Shape: `[..., merge_step, macro_group, macro_input]`.
        """
        # Each step enables its matching input slot, without indirect indexing.
        slots = torch.eye(self.merge_step_num, dtype=torch.bool, device=x.device)
        # Shape: [..., tile_in] -> [..., merge_step, input_slot, tile_in]
        routed = x[..., None, None, :].where(slots[..., None], 0)
        # Shape: [..., merge_step, input_slot, tile_in] -> [..., merge_step, macro_group=1, macro_input]
        routed = routed.flatten(-2)
        routed = F.pad(routed, (0, self.macro_input_num - routed.shape[-1])).unsqueeze(-2)
        # Shape: [merge_step, macro_group]
        tile_indices = torch.arange(self.merge_step_num * self.macro_group_num, device=x.device).view(
            self.merge_step_num, self.macro_group_num
        )
        # Shape: [merge_step, macro_group, macro_input=1]
        selected = (tile_indices < self.output_tile_num).unsqueeze(-1)
        # Shape: [..., merge_step, macro_group=1, macro_input] -> [..., merge_step, macro_group, macro_input]
        return routed.where(selected, 0)

    def map_w(self, w: Tensor) -> Tensor:
        """Place output tiles into physical input slots.

        Args:
            w: Weight tiles; leading axes pass through unchanged.
                Shape: `[..., out_tile, tile_in, tile_out]`.

        Returns:
            Programmed weights.
            Shape: `[..., macro_group, macro_input, tile_out]`.
        """
        out_tile_padding = self.merge_step_num * self.macro_group_num - self.output_tile_num
        padded = F.pad(w, (0, 0, 0, 0, 0, out_tile_padding))
        # Shape: [..., out_tile, tile_in, tile_out] -> [..., merge_step, macro_group, tile_in, tile_out]
        grouped = padded.unflatten(-3, (self.merge_step_num, self.macro_group_num))
        # Shape: [..., merge_step, macro_group, tile_in, tile_out] -> [..., macro_group, merge_step*tile_in, tile_out]
        packed = grouped.transpose(-4, -3).flatten(-3, -2)
        input_padding = self.macro_input_num - packed.shape[-2]
        # Shape: [..., macro_group, merge_step*tile_in, tile_out] -> [..., macro_group, macro_input, tile_out]
        return F.pad(packed, (0, 0, 0, input_padding))

    def recover(self, output: Tensor) -> Tensor:
        """Undo placement and remove absent output tiles.

        Args:
            output: Physical results.
                Shape: `[..., merge_step, macro_group, tile_out]`.

        Returns:
            Results in logical tile order.
            Shape: `[..., out_tile, tile_out]`.
        """
        # Shape: [..., merge_step, macro_group, tile_out] -> [..., out_tile, tile_out]
        return output.flatten(-3, -2)[..., : self.output_tile_num, :]
