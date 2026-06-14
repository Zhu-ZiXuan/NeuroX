"""Chunking helpers for :class:`CircuitCore1T1R.cim_read`.

Given the broadcast ``full_shape`` of ``rram.g__uS`` against ``x``,
we partition leading positions into:

* **A subset** — positions where ``x`` is real (size > 1) and ``g`` is
  a size-1 placeholder. These are the x-batch / M / Sa dims, chunked by
  ``solve_chunk_size_x``.
* **B subset** — positions where ``g`` is real, or both sides are real
  (matched Tc-style positions). These are the Sw / Tc / Tr inst dims,
  chunked by ``solve_chunk_size_inst``.

The chunk loop is a nested A-outer / B-inner enumeration. Each
``(chunk_a, chunk_b)`` pair produces a Cartesian-product chunk of size
``chunk_a_size * chunk_b_size`` and an aligned ``multi_coords`` tuple
indexed by leading position — exactly what advanced indexing on the
broadcast view needs.

Reassembly uses scatter back to the canonical position computed from
the global flat index in ``leading`` space.
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
    chunk_size: int  # chunk_x_size * chunk_b_size
    flat_global_idx: Tensor  # shape (chunk_size,), index into the unraveled leading


def iter_chunks(
    *,
    leading: tuple[int, ...],
    a_positions: tuple[int, ...],
    b_positions: tuple[int, ...],
    chunk_size_x: int,
    chunk_size_inst: int,
    device: torch.device,
) -> Iterator[ChunkSpec]:
    """Yield ``ChunkSpec`` per (A_chunk, B_chunk) iteration.

    Within a chunk, A coord varies slowly and B coord varies fast, so
    the chunk's positions correspond to flat indices
    ``a * B_B + b`` where ``a`` ranges over the A chunk and ``b`` over
    the B chunk — same canonical ordering as
    ``torch.unravel_index(arange, leading)``.
    """
    a_leading = tuple(leading[p] for p in a_positions)
    b_leading = tuple(leading[p] for p in b_positions)
    b_a = math.prod(a_leading) if a_leading else 1
    b_b = math.prod(b_leading) if b_leading else 1
    cx = chunk_size_x if chunk_size_x > 0 else b_a
    ci = chunk_size_inst if chunk_size_inst > 0 else b_b
    cx = min(max(cx, 1), b_a)
    ci = min(max(ci, 1), b_b)

    for a_start in range(0, b_a, cx):
        a_end = min(a_start + cx, b_a)
        a_size = a_end - a_start
        a_flat = torch.arange(a_start, a_end, device=device, dtype=torch.long)
        a_multi = torch.unravel_index(a_flat, a_leading) if a_leading else ()

        for b_start in range(0, b_b, ci):
            b_end = min(b_start + ci, b_b)
            b_size = b_end - b_start
            b_flat = torch.arange(b_start, b_end, device=device, dtype=torch.long)
            b_multi = torch.unravel_index(b_flat, b_leading) if b_leading else ()

            chunk_size = a_size * b_size

            # Build multi_coords aligned to leading positions
            zero = torch.zeros(chunk_size, dtype=torch.long, device=device)
            multi_coords: list[Tensor] = [zero] * len(leading)

            for i, p in enumerate(a_positions):
                a_grid = a_multi[i].unsqueeze(1).expand(a_size, b_size).reshape(-1)
                multi_coords[p] = a_grid
            for j, p in enumerate(b_positions):
                b_grid = b_multi[j].unsqueeze(0).expand(a_size, b_size).reshape(-1)
                multi_coords[p] = b_grid

            # Flat global index in leading space: standard C-order
            strides = _row_major_strides(leading)
            flat_global = zero.clone()
            for p in range(len(leading)):
                flat_global = flat_global + multi_coords[p] * strides[p]

            yield ChunkSpec(
                multi_coords=tuple(multi_coords),
                chunk_size=chunk_size,
                flat_global_idx=flat_global,
            )


def _row_major_strides(leading: tuple[int, ...]) -> list[int]:
    """Row-major (C-order) flat strides for the given leading."""
    if not leading:
        return []
    strides = [1] * len(leading)
    for d in range(len(leading) - 2, -1, -1):
        strides[d] = strides[d + 1] * leading[d + 1]
    return strides


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
