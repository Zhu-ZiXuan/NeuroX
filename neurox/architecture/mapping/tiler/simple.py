"""Consecutive tiling of flattened weight-slice outputs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.primitive.digital import Accumulator, Summator

from .base import Tiler


class SimpleTiler(Tiler):
    """Flatten weight slices in slice-major order before rectangular tiling.

    A tile may contain outputs from different slices. Padding occurs only at the
    end of the flattened output dimension and the end of the input dimension.

    Supply sliced weights with trailing axes `(w_slice, output, input)` matching
    the configured dimensions. Use `map_x` for the corresponding input tiles and
    `effective_output_num` to distinguish real ports from output padding.
    Recover tile results only after input-phase and input-slice recovery; weight
    slices remain separate in the returned tensor for their subsequent radix
    recovery. This helper owns no hardware or programmed state.

    Args:
        matrix_input_num: Positive logical matrix input width before padding.
        matrix_output_num: Positive logical output width before precision
            slicing.
        w_slice_num: Number of precision slices represented in the logical
            weight layout.
        input_per_tile: Positive input capacity of each rectangular tile.
        output_per_tile: Positive output-port capacity of each tile.
        recovery_circuit: Externally owned recovery circuit, or None for tensor
            arithmetic.
    """

    def __init__(
        self,
        *,
        matrix_input_num: int,
        matrix_output_num: int,
        w_slice_num: int,
        input_per_tile: int,
        output_per_tile: int,
        recovery_circuit: Accumulator | Summator | None = None,
    ) -> None:
        super().__init__(
            matrix_input_num=matrix_input_num,
            matrix_output_num=matrix_output_num,
            input_per_tile=input_per_tile,
            output_per_tile=output_per_tile,
            recovery_circuit=recovery_circuit,
        )
        self.w_slice_num = w_slice_num
        self.flattened_output_num = w_slice_num * matrix_output_num
        self.output_padding = self.output_tile_num * output_per_tile - self.flattened_output_num

    @property
    def output_tile_num(self) -> int:
        return -(-self.flattened_output_num // self.output_per_tile)

    def effective_output_num(self, *, device: torch.device) -> Tensor:
        # Shape: [out_tile]
        starts = torch.arange(self.output_tile_num, device=device) * self.output_per_tile
        return (self.flattened_output_num - starts).clamp(0, self.output_per_tile)

    def map_w(self, w: Tensor) -> Tensor:
        # Shape: [..., w_slice, output, input] -> [..., w_slice*output, input]
        w = w.flatten(-3, -2)
        padded = F.pad(w, (0, self.input_padding, 0, self.output_padding))
        # Shape: [..., padded_output, padded_input] -> [..., out_tile, tile_out, padded_input]
        tiled: Tensor = padded.unflatten(-2, (self.output_tile_num, self.output_per_tile))
        # Shape: [..., out_tile, tile_out, padded_input] -> [..., out_tile, tile_out, in_tile, tile_in]
        tiled = tiled.unflatten(-1, (self.input_tile_num, self.input_per_tile))
        # Shape: [..., out_tile, tile_out, in_tile, tile_in] -> [..., in_tile, out_tile, tile_in, tile_out]
        return tiled.movedim(-2, -4).transpose(-1, -2)

    def _recover_impl(self, output: Tensor) -> Tensor:
        # Shape: [..., out_tile, tile_out] -> [..., w_slice*output]
        output = output.flatten(-2)[..., : self.flattened_output_num]
        # Shape: [..., w_slice*output] -> [..., w_slice, output]
        restored: Tensor = output.unflatten(-1, (self.w_slice_num, self.matrix_output_num))
        return restored
