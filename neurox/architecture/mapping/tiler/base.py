"""Sliced matrix placement and input-tile result aggregation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import final

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.primitive.digital import Accumulator, Summator


class Tiler(ABC):
    """Map sliced weights to a two-dimensional tile grid and recover their results.

    Subclasses determine weight placement, valid output counts and output-layout
    recovery. The base partitions inputs and sums input-tile contributions
    before restoring the weight-slice axis. Input slices pass through the input
    mapping; their recovery precedes tile recovery.

    Construction binds an ordinary reference to the owner's recovery circuit,
    either a serial accumulator or a parallel summator. An absent reference
    selects tensor summation without register wrap or circuit cost. The tiler
    owns no modules or registered buffers.

    Args:
        matrix_output_num: Original matrix output width before slicing.
        output_per_tile: Output capacity of a tile, including weight slices.
    """

    def __init__(
        self,
        *,
        matrix_input_num: int,
        matrix_output_num: int,
        input_per_tile: int,
        output_per_tile: int,
        recovery_circuit: Accumulator | Summator | None = None,
    ) -> None:
        self.matrix_input_num = matrix_input_num
        self.matrix_output_num = matrix_output_num
        self.input_per_tile = input_per_tile
        self.output_per_tile = output_per_tile
        self.recovery_circuit = recovery_circuit

        self.input_tile_num = -(-matrix_input_num // input_per_tile)
        self.input_padding = self.input_tile_num * input_per_tile - matrix_input_num

    # === Public API ===

    @final
    def map_x(self, x: Tensor) -> Tensor:
        """Partition inputs while preserving leading axes, including input slices.

        Args:
            x: Input vectors.
                Shape: `[..., input]`.

        Returns:
            Consecutive input tiles with trailing padding.
            Shape: `[..., in_tile, tile_in]`.
        """
        padded = F.pad(x, (0, self.input_padding))
        # Shape: [..., input] -> [..., in_tile, tile_in]
        tiled: Tensor = padded.unflatten(-1, (self.input_tile_num, self.input_per_tile))
        return tiled

    @final
    def recover(self, output: Tensor) -> Tensor:
        """Sum input-tile contributions, then restore outputs and weight slices.

        Args:
            output: Tile results after phase, input-slice and merge recovery.
                Shape: `[..., in_tile, out_tile, tile_out]`.

        Returns:
            Independent weight-slice results for subsequent radix summation.
            Shape: `[..., w_slice, output]`.
        """
        if self.recovery_circuit is not None:
            # Shape: [out_tile, tile_out]
            enable = torch.arange(self.output_per_tile, device=output.device) < self.effective_output_num(
                device=output.device
            ).unsqueeze(-1)
            if isinstance(self.recovery_circuit, Accumulator):
                # Shape: [..., in_tile, out_tile, tile_out] -> [..., out_tile, tile_out]
                output = self.recovery_circuit.accumulate(output, dim=-3, enable=enable)
            else:
                # Shape: [..., in_tile, out_tile, tile_out] -> [..., out_tile, tile_out]
                output = self.recovery_circuit.sum(output, dim=-3, enable=enable)
        else:
            # Shape: [..., in_tile, out_tile, tile_out] -> [..., out_tile, tile_out]
            output = output.sum(dim=-3, dtype=output.dtype)
        # Shape: [..., out_tile, tile_out] -> [..., w_slice, output]
        return self._recover_impl(output)

    # === For subclass to implement or override ===

    @property
    @abstractmethod
    def output_tile_num(self) -> int:
        """Number of output tiles after weight-slice placement."""
        raise NotImplementedError

    @abstractmethod
    def effective_output_num(self, *, device: torch.device) -> Tensor:
        """Return the number of valid leading output ports in each tile.

        Returns:
            Counts corresponding to the placement returned by `map_w`.
            Shape: `[out_tile]`.
        """
        raise NotImplementedError

    @abstractmethod
    def map_w(self, w: Tensor) -> Tensor:
        """Place weight slices in a grid with trailing input and output padding.

        Args:
            w: Sliced logical weights.
                Shape: `[..., w_slice, output, input]`.

        Returns:
            Weights arranged for tile multiplication.
            Shape: `[..., in_tile, out_tile, tile_in, tile_out]`.
        """
        raise NotImplementedError

    @abstractmethod
    def _recover_impl(self, output: Tensor) -> Tensor:
        """Undo output placement and remove padding after input-tile summation.

        Args:
            output: Summed tile results.
                Shape: `[..., out_tile, tile_out]`.

        Returns:
            Restored logical outputs with separate weight slices.
            Shape: `[..., w_slice, output]`.
        """
        raise NotImplementedError
