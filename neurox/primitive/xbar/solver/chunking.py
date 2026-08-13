"""Memory-bounded chunking helpers for broadcast-leading dimensions.

See also:
    docs/internals/primitive/xbar/solver/chunking.md
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Generic, NamedTuple, TypeVar

import torch
from torch import Tensor

from neurox.common import walk_tensor_fields

SnapT = TypeVar("SnapT")
MeasureT = TypeVar("MeasureT")


class ChunkSpec(NamedTuple):
    """Chunk coordinates and global indices."""

    multi_coords: tuple[Tensor, ...]
    """Chunk coordinates along each leading dimension, one tensor per
    dimension, each possibly carrying repeated tail-padding coordinates.
    Shape: `[solve_size]`."""
    solve_size: int
    """Number of positions passed to the solver."""
    valid_size: int
    """Number of real positions before tail padding."""
    flat_global_idx: Tensor
    """Flat indices of the valid positions in the complete leading shape.
    Shape: `[valid_size]`."""


def iter_chunks(
    *,
    leading: tuple[int, ...],
    chunk_size: int,
    device: torch.device,
) -> Iterator[ChunkSpec]:
    """Partition a broadcast-leading shape into contiguous chunks.

    Args:
        leading: Broadcast-leading shape.
        chunk_size: Solver leading size. A positive value pads every
            chunk to this exact size; a non-positive value yields one
            unpadded chunk.
        device: Device for coordinate and index tensors.

    Yields:
        Chunk coordinates and global indices.
    """
    total = math.prod(leading) if leading else 1
    solve_size = chunk_size if chunk_size > 0 and leading else total
    step = min(max(solve_size, 1), total)

    for start in range(0, total, step):
        end = min(start + step, total)
        flat_valid = torch.arange(start, end, device=device, dtype=torch.long)
        valid_size = end - start
        if valid_size < solve_size:
            padding = flat_valid[-1].expand(solve_size - valid_size)
            flat_solve = torch.cat((flat_valid, padding))
        else:
            flat_solve = flat_valid
        multi_coords = torch.unravel_index(flat_solve, leading) if leading else ()
        yield ChunkSpec(
            multi_coords=tuple(multi_coords),
            solve_size=solve_size,
            valid_size=valid_size,
            flat_global_idx=flat_valid,
        )


def slice_snap(
    snap: SnapT,
    *,
    coords: tuple[Tensor, ...],
    leading: tuple[int, ...],
) -> SnapT:
    """Select one chunk's positions from every tensor field of a snap.

    The dataclass-walking form of `slice_tensor`: dataclass-valued fields
    recurse and every other field carries through untouched.

    Args:
        snap: Frozen snap dataclass of tensor fields, each carrying `leading`
            in front of its own trailing block.
        coords: Chunk coordinates, one tensor per leading axis.
            Shape: `[solve_size]`.
        leading: Broadcast-leading shape the coordinates index.

    Returns:
        New snap of the same type, its tensor fields at
        `[solve_size, *trailing]`.
    """
    if not coords:
        return snap
    return walk_tensor_fields(snap, lambda t: slice_tensor(t, coords=coords, leading=leading))


def slice_tensor(
    t: Tensor,
    *,
    coords: tuple[Tensor, ...],
    leading: tuple[int, ...],
) -> Tensor:
    """Select the chunk's positions from one tensor without materialising broadcasts.

    The first `len(leading)` axes give way to the single chunk axis the
    coordinates carry, and whatever stands behind them is this tensor's own
    trailing block. The result stores one value per (chunk position, real
    trailing position).

    Args:
        t: Tensor at `[*leading, *trailing]`.
        coords: Chunk coordinates, one tensor per leading axis. An empty tuple
            is the whole-leading call, which returns `t` itself.
            Shape: `[solve_size]`.
        leading: Broadcast-leading shape the coordinates index.

    Returns:
        Sliced tensor at `[solve_size, *trailing]`, or at `[1, *trailing]`
        when every leading axis is stride 0 and the tensor broadcasts against
        the chunk.

    Raises:
        ValueError: The tensor is of lower rank than the leading, so it has
            no trailing block to keep.
    """
    if not coords:
        return t

    lead_rank = len(leading)
    if t.ndim < lead_rank:
        raise ValueError(f"require: a tensor of rank >= len(leading) ({lead_rank}); got shape {tuple(t.shape)}")

    # --- 1: collapse trailing axes the tensor only broadcasts over ---

    view = t
    trailing_shape = tuple(t.shape[lead_rank:])
    collapsed = False
    for axis in range(lead_rank, t.ndim):
        if view.size(axis) > 1 and view.stride(axis) == 0:
            view = view.narrow(axis, 0, 1)
            collapsed = True

    # --- 2 + 3: index the leading axes, dropping broadcast ones ---

    index: list[int | Tensor] = [0 if view.stride(axis) == 0 else coords[axis] for axis in range(lead_rank)]
    sliced = view[tuple(index)]
    if not any(isinstance(entry, Tensor) for entry in index):
        # Every leading axis held one value for the whole call, so no
        # coordinate survived. Keep a size-1 chunk axis: every sliced operand
        # then presents the same leading rank, whatever it broadcasts over.
        sliced = sliced.unsqueeze(0)

    # --- 4: re-expand the collapsed trailing axes ---

    if collapsed:
        sliced = sliced.expand(*sliced.shape[: sliced.ndim - len(trailing_shape)], *trailing_shape)
    return sliced


class MeasureFold(Generic[MeasureT]):
    """Full-leading buffer the per-chunk measurements are written into.

    Allocating, writing and reading back are valid in that order alone. Every
    tensor field of a per-chunk measurement carries exactly one leading axis —
    the chunk axis — so the trailing shape of the first chunk fixes each
    output; dataclass-valued fields recurse and a non-tensor field is carried
    through, which is how an absent optional field stays absent.

    Args:
        first: First chunk's measurement, its tensor fields at
            `[solve_size, *trailing]`. It fixes the buffers only; it is
            written like every other chunk.
        b_total: Number of leading positions the whole call covers.
    """

    def __init__(self, first: MeasureT, *, b_total: int) -> None:
        def allocate(value: Tensor) -> Tensor:
            # Shape: [solve_size, *trailing] -> [b_total, *trailing]
            return torch.empty(b_total, *value.shape[1:], dtype=value.dtype, device=value.device)

        self._folded = walk_tensor_fields(first, allocate)

    def write(self, chunk: MeasureT, *, flat_idx: Tensor) -> None:
        """Write one chunk's tensor fields into their global positions in place.

        Buffers and chunk are walked in lockstep in field order, so every
        chunk of a call must share field PRESENCE — the same fields hold a
        tensor and the same fields hold `None`. A shared type alone is not
        enough: one chunk filling an optional field another left absent shifts
        the two walks apart and scatters into the wrong buffer.

        Args:
            chunk: This chunk's measurement, its tensor fields at
                `[solve_size, *trailing]`; the tail-padding positions past
                `flat_idx` are dropped.
            flat_idx: Flat indices of the chunk's valid positions in the
                unraveled leading.
                Shape: `[valid_size]`.
        """
        targets: list[Tensor] = []

        def _record(t: Tensor) -> Tensor:
            targets.append(t)
            return t

        walk_tensor_fields(self._folded, _record)

        valid = flat_idx.numel()
        positions = iter(targets)

        def _scatter(value: Tensor) -> Tensor:
            next(positions)[flat_idx] = value[:valid]
            return value

        walk_tensor_fields(chunk, _scatter)

    def result(self, *, leading: tuple[int, ...]) -> MeasureT:
        """Read the filled buffers back with the leading the chunk axis ravelled.

        Args:
            leading: Full broadcast leading.

        Returns:
            Measurement of the same type, its tensor fields at
            `[*leading, *trailing]`.
        """

        def unflatten(value: Tensor) -> Tensor:
            # Shape: [b_total, *trailing] -> [*leading, *trailing]
            return value.reshape(*leading, *value.shape[1:])

        return walk_tensor_fields(self._folded, unflatten)
