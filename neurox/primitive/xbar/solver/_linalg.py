"""Batched linear solves and boundary inverse blocks for wire networks."""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.common.loop import run_scan, run_scan_without_output

type _TensorTuple2 = tuple[Tensor, Tensor]
type _TensorTuple4 = tuple[Tensor, Tensor, Tensor, Tensor]
type _TensorTuple6 = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


def solve_2x2(matrix: _TensorTuple4, rhs: _TensorTuple2) -> _TensorTuple2:
    """Solve a batch of 2×2 systems from their scalar components.

    Args:
        matrix: Entries `(A[0, 0], A[0, 1], A[1, 0], A[1, 1])`.
            Shape: `[...]` per tuple component.
        rhs: Entries `(b[0], b[1])`.
            Shape: `[...]` per tuple component.

    Returns:
        The solution components `(x[0], x[1])`.
        Shape: `[...]` per tuple component.
    """
    m_00, m_01, m_10, m_11 = matrix
    r_0, r_1 = rhs
    det = m_00 * m_11 - m_01 * m_10
    return (
        (m_11 * r_0 - m_01 * r_1) / det,
        (m_00 * r_1 - m_10 * r_0) / det,
    )


def boundary_inverse_block_tridiagonal_2x2(
    diag: _TensorTuple4,
    *,
    off_diag: tuple[float, float],
    dim: int,
) -> _TensorTuple4:
    """Return the leading diagonal block of a block-tridiagonal inverse.

    Every sub- and super-diagonal block is the same constant diagonal matrix
    `U = diag(off_diag)`. Eliminating blocks from the trailing boundary forms
    `S[k] = D[k] - U @ inv(S[k + 1]) @ U`; the requested inverse block is
    `inv(S[0])`.

    CUDA sweeps benefit from contiguous slices `component.select(dim, k)`,
    since each step processes every independent system at one block index.
    Unit stride on the recurrence axis alone does not provide this layout.
    Strided inputs remain supported, and this function does not repack them.

    Args:
        diag: Main-block entries `(D[k, 0, 0], D[k, 0, 1], D[k, 1, 0], D[k, 1, 1])`.
            Shape: `[..., N, ...]`.
        off_diag: The two diagonal entries of the shared off-block.
        dim: Positive or negative index of the recurrence axis of length `N`.

    Returns:
        Entries `(A_inv[0, 0], A_inv[0, 1], A_inv[1, 0], A_inv[1, 1])`
        of the leading diagonal inverse block. The recurrence axis `dim`
        is retained with length one, matching the input axis order.
        Shape: `[..., 1, ...]` per tuple component.
    """
    diag_00, diag_01, diag_10, diag_11 = diag
    u_0, u_1 = off_diag

    w_00 = u_0 * u_0
    w_01 = u_0 * u_1
    w_10 = u_1 * u_0
    w_11 = u_1 * u_1

    block_num = diag_00.shape[dim]

    def body_fn(state: _TensorTuple4, index: Tensor) -> _TensorTuple4:
        a_00, a_01, a_10, a_11 = state
        k = index.view(1)

        def select(t: Tensor) -> Tensor:
            return t.index_select(dim, k)

        # Shape: [..., N, ...] -> [..., 1, ...]
        m_00 = select(diag_00) - w_00 * a_00
        m_01 = select(diag_01) - w_01 * a_01
        m_10 = select(diag_10) - w_10 * a_10
        m_11 = select(diag_11) - w_11 * a_11
        inv_det = 1.0 / (m_00 * m_11 - m_01 * m_10)
        a_00 = m_11 * inv_det
        a_01 = -m_01 * inv_det
        a_10 = -m_10 * inv_det
        a_11 = m_00 * inv_det
        # Shape: [..., 1, ...]
        return a_00, a_01, a_10, a_11

    # A zero inverse represents the virtual block beyond the trailing boundary,
    # so the first loop update reduces directly to inv(D[N - 1]).
    # Shape: [..., N, ...] -> [..., 1, ...]
    init_state = (
        torch.zeros_like(diag_00.narrow(dim, 0, 1), memory_format=torch.contiguous_format),
        torch.zeros_like(diag_01.narrow(dim, 0, 1), memory_format=torch.contiguous_format),
        torch.zeros_like(diag_10.narrow(dim, 0, 1), memory_format=torch.contiguous_format),
        torch.zeros_like(diag_11.narrow(dim, 0, 1), memory_format=torch.contiguous_format),
    )

    # Reverse the small index sequence rather than materializing flipped coefficients.
    indices = torch.arange(block_num - 1, -1, -1, device=diag_00.device)

    # Shape: [..., 1, ...]
    return run_scan_without_output(init_state=init_state, xs=indices, body_fn=body_fn, device=diag_00.device)


def solve_block_tridiagonal_2x2(
    diag: _TensorTuple4,
    rhs: _TensorTuple2,
    *,
    off_diag: tuple[float, float],
    dim: int,
) -> _TensorTuple2:
    """Solve batched 2×2 block-tridiagonal systems with ONE constant off-block.

    Every sub- and super-diagonal block is the same constant diagonal matrix
    `U = diag(off_diag)`, which reduces the block Thomas recurrence to one
    explicit 2×2 inverse plus multiply-adds per step. The off-diagonal blocks
    are never materialized and the boundary slots need no special casing. `N`
    is the number of block rows, each block being 2×2.

    Forward and reverse scans retain only one block in their carry. Reduced
    coefficients and solutions are collected as outputs; their storage grows
    with `N`. The two solution components are returned in ascending row order.

    CUDA sweeps benefit when `component.select(dim, k)` is contiguous across
    the remaining axes, for both `diag` and `rhs`. Unit stride on the recurrence
    axis alone does not provide this layout. Strided inputs remain supported,
    and this function does not repack them. Collected histories use contiguous
    slices for each block index.

    Args:
        diag: Main-block entries `(D[k, 0, 0], D[k, 0, 1], D[k, 1, 0], D[k, 1, 1])`.
            Shape: `[..., N, ...]`.
        rhs: Right-hand-side entries `(b[k, 0], b[k, 1])`.
            Shape: `[..., N, ...]`.
        off_diag: The two diagonal entries of the shared off-block, i.e.
            `U = diag(off_diag[0], off_diag[1])`.
        dim: Positive or negative index of the recurrence axis of length `N`.

    Returns:
        The two solution components, retaining the input axis order.
        Shape: `[..., N, ...]`.
    """
    diag_00, diag_01, diag_10, diag_11 = diag
    rhs_0, rhs_1 = rhs
    u_0, u_1 = off_diag

    # Cross weights of U · M · U — entry (i, j) picks up u_i · u_j.
    w_00 = u_0 * u_0
    w_01 = u_0 * u_1
    w_10 = u_1 * u_0
    w_11 = u_1 * u_1

    block_num = rhs_0.shape[dim]

    # --- Forward elimination ---

    def forward_elimination_step(state: _TensorTuple6, row: _TensorTuple6) -> tuple[_TensorTuple6, _TensorTuple6]:
        a_00_prev, a_01_prev, a_10_prev, a_11_prev, e_0_prev, e_1_prev = state
        m_00, m_01, m_10, m_11, b_0, b_1 = row

        m_00 = m_00 - w_00 * a_00_prev
        m_01 = m_01 - w_01 * a_01_prev
        m_10 = m_10 - w_10 * a_10_prev
        m_11 = m_11 - w_11 * a_11_prev
        b_0 = b_0 - u_0 * e_0_prev
        b_1 = b_1 - u_1 * e_1_prev

        inv_det = 1.0 / (m_00 * m_11 - m_01 * m_10)
        a_00 = m_11 * inv_det
        a_01 = -m_01 * inv_det
        a_10 = -m_10 * inv_det
        a_11 = m_00 * inv_det
        e_0 = a_00 * b_0 + a_01 * b_1
        e_1 = a_10 * b_0 + a_11 * b_1

        next_state = (a_00, a_01, a_10, a_11, e_0, e_1)
        # Native scan forbids carry/output aliasing; clone only the current row.
        output = (a_00.clone(), a_01.clone(), a_10.clone(), a_11.clone(), e_0.clone(), e_1.clone())
        return next_state, output

    # A_{-1} = 0 and e_{-1} = 0 represent the boundary before the first row.
    # Only these six current-block components enter the scan carry.
    # Canonical strides also match subsequent carries for singleton batch axes.
    # Shape: [...]
    init_state = (
        torch.zeros_like(diag_00.select(dim, 0), memory_format=torch.contiguous_format),
        torch.zeros_like(diag_01.select(dim, 0), memory_format=torch.contiguous_format),
        torch.zeros_like(diag_10.select(dim, 0), memory_format=torch.contiguous_format),
        torch.zeros_like(diag_11.select(dim, 0), memory_format=torch.contiguous_format),
        torch.zeros_like(rhs_0.select(dim, 0), memory_format=torch.contiguous_format),
        torch.zeros_like(rhs_1.select(dim, 0), memory_format=torch.contiguous_format),
    )

    # Shape: [..., N, ...] per output component.
    _, reduced = run_scan(
        init_state=init_state,
        xs=(diag_00, diag_01, diag_10, diag_11, rhs_0, rhs_1),
        body_fn=forward_elimination_step,
        output_template=init_state,
        dim=dim,
    )
    a_00_seq, a_01_seq, a_10_seq, a_11_seq, e_0_seq, e_1_seq = reduced

    # --- Back substitution ---

    def back_substitution_step(state: _TensorTuple2, index: Tensor) -> tuple[_TensorTuple2, _TensorTuple2]:
        x_0_next, x_1_next = state
        k = index.view(1)

        def select(t: Tensor) -> Tensor:
            return t.index_select(dim, k).squeeze(dim)

        s_0 = u_0 * x_0_next
        s_1 = u_1 * x_1_next
        # Shape: [..., N, ...] -> [...]
        x_0 = select(e_0_seq) - (select(a_00_seq) * s_0 + select(a_01_seq) * s_1)
        x_1 = select(e_1_seq) - (select(a_10_seq) * s_0 + select(a_11_seq) * s_1)
        return (x_0, x_1), (x_0.clone(), x_1.clone())

    # x_N = 0 makes the last-row update x_{N-1} = e_{N-1}, including N = 1.
    # Shape: [...]
    init_solution = (
        torch.zeros_like(rhs_0.select(dim, 0), memory_format=torch.contiguous_format),
        torch.zeros_like(rhs_1.select(dim, 0), memory_format=torch.contiguous_format),
    )

    # Reverse only the small index sequence, leaving all six histories in place.
    # The adapter restores outputs to ascending row order.
    _, solution = run_scan(
        init_state=init_solution,
        xs=torch.arange(block_num, device=rhs_0.device),
        body_fn=back_substitution_step,
        output_template=init_solution,
        reverse=True,
    )
    x_0_seq, x_1_seq = solution
    return x_0_seq.movedim(0, dim), x_1_seq.movedim(0, dim)


def solve_tridiagonal(
    sub: Tensor,
    diag: Tensor,
    sup: Tensor,
    rhs: Tensor,
    dim: int,
) -> Tensor:
    """Solve batched tridiagonal systems via the Thomas algorithm.

    Solves `A x = rhs` along `dim`, `A` being tridiagonal with
    `(sub, diag, sup)`; `N` is the system length along `dim`.

    Args:
        sub: Sub-diagonal coefficients; the entry at index 0 is unused.
            Shape: `[..., N, ...]`.
        diag: Main diagonal coefficients.
            Shape: `[..., N, ...]`.
        sup: Super-diagonal coefficients; the entry at the last index is
            unused.
            Shape: `[..., N, ...]`.
        rhs: Right-hand-side vectors.
            Shape: `[..., N, ...]`.
        dim: The dimension of length `N`.

    Returns:
        Solution tensor.
        Shape: `[..., N, ...]`.
    """
    block_num = rhs.shape[dim]
    if block_num == 1:
        return rhs / diag

    # Forward sweep.
    d_list: list[Tensor] = [diag.select(dim, 0)]
    r_list: list[Tensor] = [rhs.select(dim, 0)]
    for i in range(1, block_num):
        w = sub.select(dim, i) / d_list[i - 1]
        d_list.append(diag.select(dim, i) - w * sup.select(dim, i - 1))
        r_list.append(rhs.select(dim, i) - w * r_list[i - 1])

    # Back substitution — build the solution list right-to-left.
    x_list: list[Tensor] = [r_list[block_num - 1] / d_list[block_num - 1]]
    for i in range(block_num - 2, -1, -1):
        x_list.append((r_list[i] - sup.select(dim, i) * x_list[-1]) / d_list[i])
    x_list.reverse()

    return torch.stack(x_list, dim=dim)
