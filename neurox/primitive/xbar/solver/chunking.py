"""Memory-bounded chunking helpers for broadcast-leading dimensions."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import NamedTuple

import torch
from torch import Tensor


def classify_leading_positions(
    x_shape: tuple[int, ...],
    g_shape: tuple[int, ...],
    leading_rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Partition leading positions into A (x-real, g-placeholder) and B (g-real or matched).

    Args:
        x_shape: shape of x AFTER ``unsqueeze(-2)`` (broadcast-aligned with g).
        g_shape: shape of ``rram.g__uS``.
        leading_rank: number of leading positions (``len(full_shape) - 2``).

    Returns:
        ``(A_positions, B_positions)`` — tuples of position indices in
        ``[0, leading_rank)``. Positions where both sides are size 1 are
        omitted (broadcast result size 1, no chunking needed).
    """
    full_rank = leading_rank + 2
    x_pad = (1,) * (full_rank - len(x_shape)) + tuple(x_shape)
    g_pad = (1,) * (full_rank - len(g_shape)) + tuple(g_shape)
    a_positions: list[int] = []
    b_positions: list[int] = []
    for i in range(leading_rank):
        xd = x_pad[i]
        gd = g_pad[i]
        if xd > 1 and gd == 1:
            a_positions.append(i)
        elif gd > 1:
            b_positions.append(i)
    return tuple(a_positions), tuple(b_positions)


class ChunkSpec(NamedTuple):
    """Chunk coordinates and global indices.

    Attributes:
        multi_coords: Coordinate tensor per leading dimension. Its length is
            ``solve_size`` and may include repeated padding coordinates.
        solve_size: Number of positions passed to the solver.
        valid_size: Number of real positions before tail padding.
        flat_global_idx: Flat indices of the valid positions in the complete
            leading shape.
    """

    multi_coords: tuple[Tensor, ...]
    solve_size: int
    valid_size: int
    flat_global_idx: Tensor


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


def reassemble_chunks(
    chunks: list[Tensor],
    chunk_global_indices: list[Tensor],
    leading: tuple[int, ...],
    trailing: tuple[int, ...],
) -> Tensor:
    """Cat per-chunk outputs and scatter back to ``(*leading, *trailing)``.

    Args:
        chunks: list of chunk tensors of shape ``(chunk_size, *trailing)``;
            when ``leading == ()`` the single chunk has shape ``(*trailing,)``
            directly (no leading axis to prepend).
        chunk_global_indices: matching list of 1-D flat indices into the
            unraveled ``leading`` for each chunk position.
        leading: full broadcast leading.
        trailing: trailing dims after leading.

    Returns:
        Tensor of shape ``(*leading, *trailing)``.
    """
    if not leading:
        return chunks[0]
    all_chunks = torch.cat(chunks, dim=0)
    flat_idx = torch.cat(chunk_global_indices, dim=0)
    b_total = math.prod(leading)
    target = torch.empty(b_total, *trailing, dtype=all_chunks.dtype, device=all_chunks.device)
    target[flat_idx] = all_chunks
    return target.reshape(*leading, *trailing)
