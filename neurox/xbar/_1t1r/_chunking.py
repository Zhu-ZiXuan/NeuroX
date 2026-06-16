"""Chunking helpers for :class:`CircuitCore1T1R.cim_read`.

:func:`iter_chunks` partitions the broadcast leading (``prod(leading)``
instances) into contiguous C-order slices of at most ``solve_chunk_size``
instances. Each slice's flat indices unravel to a ``multi_coords`` tuple
indexed by leading position — exactly what advanced indexing on the broadcast
view needs — and the flat index is the canonical global index reassembly
scatters back by. A single ``solve_chunk_size`` budget bounds per-chunk
peak memory directly, independent of which leading axes are serial or inst.

:func:`classify_leading_positions` is separate: it splits leading into the
**A subset** (``x`` real, ``g`` placeholder — the serial x-batch / M / Sa
dims) and the **B subset** (``g`` real — the parallel Sw / Tc / Tr inst
dims). That split is not used for chunking; ``cim_read`` uses the A subset
to count the serial per-op latency multiplicity.
"""

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
            # g-real or matched (both > 1) — per the macro / xbar A/B
            # axis taxonomy these are inst positions (parallel hardware),
            # always B.
            b_positions.append(i)
        # both 1 → degenerate, omit
    return tuple(a_positions), tuple(b_positions)


class ChunkSpec(NamedTuple):
    """Per-chunk dispatch payload."""

    multi_coords: tuple[Tensor, ...]  # one per leading position, shape (chunk_size,)
    chunk_size: int  # length of this contiguous broadcast-leading slice (end - start, <= solve_chunk_size)
    flat_global_idx: Tensor  # shape (chunk_size,), index into the unraveled leading


def iter_chunks(
    *,
    leading: tuple[int, ...],
    chunk_size: int,
    device: torch.device,
) -> Iterator[ChunkSpec]:
    """Yield ``ChunkSpec`` partitioning the broadcast leading into pieces of
    at most ``chunk_size`` instances.

    The full leading carries ``prod(leading)`` instances. They are split into
    contiguous C-order slices of at most ``chunk_size``; each slice's flat
    indices unravel to the per-position ``multi_coords`` tuple that advanced
    indexing on the broadcast view needs, and the flat index is itself the
    canonical global index used for reassembly. ``chunk_size <= 0`` puts the
    whole leading in one chunk. Instances in a chunk are independent (every
    instance is solved once), so the contiguous-slice partition is purely a
    memory-bounding choice — ``chunk_size`` is the per-chunk leading, i.e. the
    peak-memory budget, regardless of which leading axes are serial or inst.
    """
    total = math.prod(leading) if leading else 1
    c = chunk_size if chunk_size > 0 else total
    c = min(max(c, 1), total)

    for start in range(0, total, c):
        end = min(start + c, total)
        flat = torch.arange(start, end, device=device, dtype=torch.long)
        multi_coords = torch.unravel_index(flat, leading) if leading else ()
        yield ChunkSpec(
            multi_coords=tuple(multi_coords),
            chunk_size=end - start,
            flat_global_idx=flat,
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
    # Degenerate leading: iter_chunks yields exactly one chunk that
    # already carries the full payload at trailing shape — no cat /
    # scatter is well-defined on 0-D leading.
    if not leading:
        return chunks[0]
    all_chunks = torch.cat(chunks, dim=0)
    flat_idx = torch.cat(chunk_global_indices, dim=0)
    b_total = math.prod(leading)
    target = torch.empty(b_total, *trailing, dtype=all_chunks.dtype, device=all_chunks.device)
    target[flat_idx] = all_chunks
    return target.reshape(*leading, *trailing)
