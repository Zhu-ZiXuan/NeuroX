"""Matrix-tile placement in disjoint, serially selected macro input slots."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor

from .tiling import _chunk_pad_along


class Merge(ABC):
    """Matched input-slot placement and output recovery transforms."""

    @abstractmethod
    def map_x(self, x: Tensor) -> Tensor:
        """Route a tile input into independently enabled physical slots.

        Args:
            x: Tile inputs; leading axes may include input tiles and precision slices.
                Shape: `[..., tile_input]`.

        Returns:
            Physical inputs, with empty slots zero-filled.
            Shape: `[..., merge_step, macro_group, macro_input]`.
        """
        ...

    @abstractmethod
    def map_w(self, w: Tensor) -> Tensor:
        """Place output tiles into physical input slots.

        Args:
            w: Weight tiles; leading axes pass through unchanged.
                Shape: `[..., out_tile, tile_input, tile_output]`.

        Returns:
            Programmed weights.
            Shape: `[..., macro_group, macro_input, tile_output]`.
        """
        ...

    @abstractmethod
    def recover(self, output: Tensor) -> Tensor:
        """Undo placement and remove absent output tiles.

        Args:
            output: Physical results.
                Shape: `[..., merge_step, macro_group, tile_output]`.

        Returns:
            Results in logical tile order.
            Shape: `[..., out_tile, tile_output]`.
        """
        ...

    @abstractmethod
    def effective_output_num(self, *, device: torch.device) -> Tensor:
        """Return the valid logical output count of each physical access.

        Returns:
            Valid logical output counts.
            Shape: `[merge_step, macro_group]`.
        """
        ...


class InputSlotMerge(Merge):
    """Place consecutive output tiles across macro groups, then input slots.

    Args:
        tile_input_num: Input width of each tile.
        logical_tile_output_num: Number of logical outputs carried by each tile.
        output_num: Logical output count before tile padding.
        macro_input_num: Physical input capacity of one macro.
        enabled: Whether multiple output tiles may share a macro's input slots.
    """

    def __init__(
        self,
        *,
        tile_input_num: int,
        logical_tile_output_num: int,
        output_num: int,
        macro_input_num: int,
        enabled: bool,
    ) -> None:
        self.tile_input_num = tile_input_num
        self.logical_tile_output_num = logical_tile_output_num
        self.output_num = output_num
        self.macro_input_num = macro_input_num
        self.out_tile_num = -(-output_num // logical_tile_output_num)
        self.input_slot_capacity = macro_input_num // tile_input_num if enabled else 1
        self.macro_group_num = -(-self.out_tile_num // self.input_slot_capacity)
        self.merge_step_num = -(-self.out_tile_num // self.macro_group_num)

    def effective_output_num(self, *, device: torch.device) -> Tensor:
        # Shape: [merge_step*macro_group]
        indices = torch.arange(self.merge_step_num * self.macro_group_num, device=device)
        counts = (self.output_num - indices * self.logical_tile_output_num).clamp(0, self.logical_tile_output_num)
        # Shape: [merge_step*macro_group] -> [merge_step, macro_group]
        return counts.view(self.merge_step_num, self.macro_group_num)

    def map_x(self, x: Tensor) -> Tensor:
        # Shape: [macro_input]
        positions = torch.arange(self.macro_input_num, device=x.device)
        # Shape: [merge_step, macro_input=1]
        starts = torch.arange(self.merge_step_num, device=x.device)[:, None] * self.tile_input_num
        # Shape: [macro_input] - [merge_step, macro_input=1] -> [merge_step, macro_input]
        local = positions - starts
        selected = (local >= 0) & (local < self.tile_input_num)
        # Shape: [..., tile_input] -> [..., merge_step, macro_group=1, macro_input]
        routed = x[..., local.clamp(0, self.tile_input_num - 1)].unsqueeze(-2)
        # Shape: [merge_step, macro_input] -> [merge_step, macro_group, macro_input]
        selected = selected.unsqueeze(-2) & (self.effective_output_num(device=x.device) > 0).unsqueeze(-1)
        # Shape: [..., merge_step, macro_group=1, macro_input] -> [..., merge_step, macro_group, macro_input]
        return routed.where(selected, 0)

    def map_w(self, w: Tensor) -> Tensor:
        # Shape: [..., out_tile, tile_input, tile_output] -> [..., merge_step, macro_group, tile_input, tile_output]
        grouped = _chunk_pad_along(w, dim=-3, chunk_size=self.macro_group_num, pad_value=0)
        # Shape: [..., macro_group, merge_step*tile_input, tile_output]
        packed = grouped.transpose(-4, -3).flatten(-3, -2)
        # Shape: [..., macro_group, merge_step*tile_input, tile_output] -> [..., macro_group, macro_input, tile_output]
        return F.pad(packed, (0, 0, 0, self.macro_input_num - packed.shape[-2]))

    def recover(self, output: Tensor) -> Tensor:
        # Shape: [..., merge_step, macro_group, tile_output] -> [..., out_tile, tile_output]
        return output.flatten(-3, -2)[..., : self.out_tile_num, :]
