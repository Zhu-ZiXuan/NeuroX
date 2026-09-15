"""Matrix tiling with self-contained input, weight, and output transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import final

import torch.nn.functional as F
from torch import Tensor


def _chunk_pad_along(t: Tensor, *, dim: int, chunk_size: int, pad_value: int) -> Tensor:
    if chunk_size < 1:
        raise ValueError(f"require: chunk_size ({chunk_size}) >= 1")
    if dim < 0:
        dim += t.ndim
    if not 0 <= dim < t.ndim:
        raise ValueError(f"require: 0 <= dim ({dim}) < ndim ({t.ndim})")
    tile_num = -(-t.shape[dim] // chunk_size)
    padding = tile_num * chunk_size - t.shape[dim]
    if padding:
        t = F.pad(t, [0, 0] * (t.ndim - dim - 1) + [0, padding], value=pad_value)
    # Shape: [..., position, ...] -> [..., tile=tile_num, tile_position=chunk_size, ...]
    t_pad: Tensor = t.unflatten(dim, (tile_num, chunk_size))
    return t_pad


class TilingMode(StrEnum):
    SLICE_PLANES = "slice_planes"
    SLICE_OUTPUTS = "slice_outputs"


class Tiling(ABC):
    """Tile sliced matrices and place their weight slices together.

    Args:
        input_num: Logical matrix input width.
        output_num: Logical matrix output width before slicing.
        tile_input_capacity: Maximum input width of a physical tile.
        tile_output_capacity: Output capacity of a physical tile, including slices.
        w_slice_num: Number of positional slices per logical weight.
    """

    def __init__(
        self,
        *,
        input_num: int,
        output_num: int,
        tile_input_capacity: int,
        tile_output_capacity: int,
        w_slice_num: int,
    ) -> None:
        self.input_num = input_num
        self.output_num = output_num
        self.w_slice_num = w_slice_num
        self.tile_input_num = min(input_num, tile_input_capacity)
        self.tile_output_num = tile_output_capacity
        if self.logical_tile_output_num < 1:
            raise ValueError("tiling must fit at least one complete logical output in a physical tile")
        self.in_tile_num = -(-input_num // self.tile_input_num)
        self.out_tile_num = -(-output_num // self.logical_tile_output_num)

    # === Public API ===

    @final
    def map_x(self, x: Tensor) -> Tensor:
        """Map already sliced inputs into input tiles.

        Args:
            x: Input vectors; leading axes, including x slices, pass through.
                Shape: `[..., input]`.

        Returns:
            Tiled inputs, broadcastable over physical weight planes.
            Shape: `[..., in_tile, tile_input]`.
        """
        # Shape: [..., input] -> [..., in_tile, tile_input]
        return _chunk_pad_along(x, dim=-1, chunk_size=self.tile_input_num, pad_value=0)

    @final
    def map_w(self, w: Tensor) -> Tensor:
        """Tile weights and place their slices within the physical tile capacities.

        Args:
            w: Already sliced logical weights.
                Shape: `[..., w_slice, output, input]`.

        Returns:
            Physically arranged weight tiles with unused positions zero-filled.
            Shape: `[..., macro_plane, in_tile, out_tile, tile_input, tile_output]`.
        """
        # Shape: [..., w_slice, in_tile, out_tile, tile_input, logical_tile_output]
        tiled = self._tile_w(w)
        # Shape: [..., macro_plane, in_tile, out_tile, tile_input, tile_output]
        return self._map_w_impl(tiled)

    @final
    def recover(self, output: Tensor) -> Tensor:
        """Restore logical outputs and independent weight slices from tile results.

        Args:
            output: Tile multiplication results after merge recovery.
                Shape: `[..., macro_plane, in_tile, out_tile, tile_output]`.

        Returns:
            Input-tile partial sums with padding removed and slice positions restored.
            Positional weighting is left to the numerical slicer or reconstruction circuit.
            Shape: `[..., w_slice, output]`.
        """
        # Shape: [..., w_slice, in_tile, out_tile, logical_tile_output]
        restored = self._recover_impl(output)
        # Shape: [..., w_slice, output]
        return self._recover_output(restored)

    # === For subclass to implement or override ===

    @property
    @abstractmethod
    def logical_tile_output_num(self) -> int:
        """Logical output positions in a tile before weight slices consume ports."""
        raise NotImplementedError

    @property
    @abstractmethod
    def macro_plane_num(self) -> int:
        """Physical tile planes used to hold the weight slices."""
        raise NotImplementedError

    @abstractmethod
    def physical_output_count(self, logical_output_num: Tensor) -> Tensor:
        """Convert valid logical output counts to occupied physical ports, preserving shape."""
        raise NotImplementedError

    @abstractmethod
    def _map_w_impl(self, tiled: Tensor) -> Tensor:
        """Arrange weight slices after common logical tiling.

        Args:
            tiled: Padded logical tiles with separate weight slices.
                Shape: `[..., w_slice, in_tile, out_tile, tile_input, logical_tile_output]`.

        Returns:
            Physical tiles with unused output ports zero-filled.
            Shape: `[..., macro_plane, in_tile, out_tile, tile_input, tile_output]`.
        """
        raise NotImplementedError

    @abstractmethod
    def _recover_impl(self, output: Tensor) -> Tensor:
        """Undo slice placement before common input-tile reduction and output cropping.

        Args:
            output: Physical tile results.
                Shape: `[..., macro_plane, in_tile, out_tile, tile_output]`.

        Returns:
            Logical tile results with weight slices separated and port padding removed.
            Shape: `[..., w_slice, in_tile, out_tile, logical_tile_output]`.
        """
        raise NotImplementedError

    # === Internal helpers ===

    @final
    def _tile_w(self, w: Tensor) -> Tensor:
        # Shape: [..., output, input] -> [..., out_tile, logical_tile_output, input]
        tiled = _chunk_pad_along(w, dim=-2, chunk_size=self.logical_tile_output_num, pad_value=0)
        # Shape: [..., out_tile, logical_tile_output, in_tile, tile_input]
        tiled = _chunk_pad_along(tiled, dim=-1, chunk_size=self.tile_input_num, pad_value=0)
        # Shape: [..., in_tile, out_tile, tile_input, logical_tile_output]
        return tiled.movedim(-2, -4).transpose(-1, -2)

    @final
    def _recover_output(self, output: Tensor) -> Tensor:
        # Shape: [..., in_tile, out_tile, logical_tile_output] -> [..., output]
        return output.sum(dim=-3, dtype=output.dtype).flatten(-2)[..., : self.output_num]


class PlaneSliceTiling(Tiling):
    """Place each weight slice in a separate plane of rectangular tiles."""

    @property
    def logical_tile_output_num(self) -> int:
        return self.tile_output_num

    @property
    def macro_plane_num(self) -> int:
        return self.w_slice_num

    def _map_w_impl(self, tiled: Tensor) -> Tensor:
        return tiled

    def _recover_impl(self, output: Tensor) -> Tensor:
        return output

    def physical_output_count(self, logical_output_num: Tensor) -> Tensor:
        return logical_output_num


class OutputSliceTiling(Tiling):
    """Keep all slices of a logical output in adjacent ports of the same tile."""

    @property
    def logical_tile_output_num(self) -> int:
        return self.tile_output_num // self.w_slice_num

    @property
    def macro_plane_num(self) -> int:
        return 1

    def _map_w_impl(self, tiled: Tensor) -> Tensor:
        # Shape: [..., in_tile, out_tile, tile_input, logical_tile_output*w_slice]
        packed = tiled.movedim(-5, -1).flatten(-2)
        # Shape: [..., macro_plane=1, in_tile, out_tile, tile_input, tile_output]
        return F.pad(packed, (0, self.tile_output_num - packed.shape[-1])).unsqueeze(-5)

    def _recover_impl(self, output: Tensor) -> Tensor:
        # Shape: [..., macro_plane=1, in_tile, out_tile, tile_output] -> [..., in_tile, out_tile, tile_output]
        output = output.squeeze(-4)
        # Shape: [..., in_tile, out_tile, logical_tile_output, w_slice]
        sliced = output[..., : self.logical_tile_output_num * self.w_slice_num].unflatten(
            -1, (self.logical_tile_output_num, self.w_slice_num)
        )
        # Shape: [..., w_slice, in_tile, out_tile, logical_tile_output]
        out: Tensor = sliced.movedim(-1, -4)
        return out

    def physical_output_count(self, logical_output_num: Tensor) -> Tensor:
        return logical_output_num * self.w_slice_num
