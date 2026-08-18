"""Numerical linear-algebra helpers for crossbar IR-drop simulation.

Three properties hold across every routine here. A sweep accumulates its
intermediates into Python lists and stacks once instead of writing in place,
so the unrolled loop traces cleanly and its per-step elementwise work fuses.
Every 2×2 block product and block solve runs in closed form, a batched GEMM or
LU kernel costing far more per launch on tens of millions of tiny blocks than
the arithmetic it performs. And no sweep pivots: each assumes its modified
diagonal block stays non-singular, well-posedness belonging to the Newton
formulation that builds the system.
"""

from __future__ import annotations

import torch
from torch import Tensor


def block_matmul(p: Tensor, q: Tensor) -> Tensor:
    """Compute batched `p @ q` with a closed-form 2×2 path.

    For `B == 2` the product uses elementwise multiply-adds; other block sizes
    fall back to the matmul. `B` is the block dimension, `K` the number of
    right-hand-side columns.

    Args:
        p: Left block operand.
            Shape: `[..., B, B]`.
        q: Right block operand.
            Shape: `[..., B, K]`.

    Returns:
        Block product.
        Shape: `[..., B, K]`.
    """
    if p.shape[-1] == 2 and p.shape[-2] == 2:
        # Shape: [..., B, K] -> [..., K]
        q0 = q[..., 0, :]
        # Shape: [..., B, K] -> [..., K]
        q1 = q[..., 1, :]
        out0 = p[..., 0, 0].unsqueeze(-1) * q0 + p[..., 0, 1].unsqueeze(-1) * q1
        out1 = p[..., 1, 0].unsqueeze(-1) * q0 + p[..., 1, 1].unsqueeze(-1) * q1
        # Shape: [..., K] -> [..., B, K]
        return torch.stack((out0, out1), dim=-2)
    return p @ q


def block_solve(m: Tensor, rhs: Tensor) -> Tensor:
    """Solve batched `m x = rhs` with a closed-form 2×2 path.

    For `B == 2` the solve uses the adjugate/determinant formula; other block
    sizes fall back to `torch.linalg.solve`. `B` is the block dimension, `K`
    the number of right-hand-side columns.

    Args:
        m: Block system matrix.
            Shape: `[..., B, B]`.
        rhs: Right-hand-side columns.
            Shape: `[..., B, K]`.

    Returns:
        Solution columns.
        Shape: `[..., B, K]`.
    """
    if m.shape[-1] == 2 and m.shape[-2] == 2:
        # Shape: [..., B, B] -> [..., 1]
        a = m[..., 0, 0].unsqueeze(-1)
        # Shape: [..., B, B] -> [..., 1]
        b = m[..., 0, 1].unsqueeze(-1)
        # Shape: [..., B, B] -> [..., 1]
        c = m[..., 1, 0].unsqueeze(-1)
        # Shape: [..., B, B] -> [..., 1]
        d = m[..., 1, 1].unsqueeze(-1)
        det = a * d - b * c
        # Shape: [..., B, K] -> [..., K]
        r0 = rhs[..., 0, :]
        # Shape: [..., B, K] -> [..., K]
        r1 = rhs[..., 1, :]
        x0 = (d * r0 - b * r1) / det
        x1 = (a * r1 - c * r0) / det
        # Shape: [..., K] -> [..., B, K]
        return torch.stack((x0, x1), dim=-2)
    return torch.linalg.solve(m, rhs)


def solve_block_tridiagonal(
    sub: Tensor,
    diag: Tensor,
    sup: Tensor,
    rhs: Tensor,
) -> Tensor:
    """Solve batched block-tridiagonal systems via the block Thomas algorithm.

    Solves `A x = rhs` where `A` is block-tridiagonal with `N` block rows of
    `B`×`B` blocks, any `B` running through the same path. The shape
    convention is fixed: the `N` axis is third-to-last for the block tensors
    and second-to-last for `rhs`, so a caller whose layout differs transposes
    at the call site — another axis order mis-indexes the sweep silently
    rather than raising. At `B = 1` the result agrees with `solve_tridiagonal`
    to round-off alone, the 1×1 solve differing arithmetically from a scalar
    division.

    Args:
        sub: Sub-diagonal blocks coupling row `k` to row `k-1`; the entry at
            `k = 0` is unused.
            Shape: `[..., N, B, B]`.
        diag: Main diagonal blocks.
            Shape: `[..., N, B, B]`.
        sup: Super-diagonal blocks coupling row `k` to row `k+1`; the entry at
            `k = N-1` is unused.
            Shape: `[..., N, B, B]`.
        rhs: Right-hand-side vectors.
            Shape: `[..., N, B]`.

    Returns:
        Solution tensor.
        Shape: `[..., N, B]`.
    """
    block_num = rhs.shape[-2]
    if block_num == 1:
        # One block is one plain solve; the block-row axis of extent one stays,
        # so the return keeps the `[..., N, B]` rank whatever `block_num` is.
        # Shape: [..., N=1, B] -> [..., B, 1] -> [..., N=1, B]
        return block_solve(diag[..., 0, :, :], rhs[..., 0, :].unsqueeze(-1)).squeeze(-1).unsqueeze(-2)

    # Forward sweep: C_k = M_k⁻¹ · sup_k, d_k = M_k⁻¹ · (rhs - sub · d_{k-1}).
    m_0 = diag[..., 0, :, :]
    rhs_0 = rhs[..., 0, :].unsqueeze(-1)
    # Stack [sup, rhs] as RHS columns so each step runs one solve instead of two.
    sol_0 = block_solve(m_0, torch.cat((sup[..., 0, :, :], rhs_0), dim=-1))
    c_list: list[Tensor] = [sol_0[..., :-1]]
    d_list: list[Tensor] = [sol_0[..., -1:]]
    for k in range(1, block_num):
        sub_k = sub[..., k, :, :]
        diag_k = diag[..., k, :, :]
        sup_k = sup[..., k, :, :]
        rhs_k = rhs[..., k, :].unsqueeze(-1)
        m_k = diag_k - block_matmul(sub_k, c_list[k - 1])
        sol_k = block_solve(m_k, torch.cat((sup_k, rhs_k - block_matmul(sub_k, d_list[k - 1])), dim=-1))
        c_list.append(sol_k[..., :-1])
        d_list.append(sol_k[..., -1:])

    # Back substitution — build the solution list right-to-left.
    x_list: list[Tensor] = [d_list[block_num - 1].squeeze(-1)]
    for k in range(block_num - 2, -1, -1):
        x_list.append((d_list[k] - block_matmul(c_list[k], x_list[-1].unsqueeze(-1))).squeeze(-1))
    x_list.reverse()

    return torch.stack(x_list, dim=-2)


def solve_block_tridiagonal_2x2_uniform(
    diag: Tensor,
    rhs: Tensor,
    *,
    off_block: tuple[float, float],
) -> Tensor:
    """Solve batched 2×2 block-tridiagonal systems with ONE constant off-block.

    Specialization of `solve_block_tridiagonal` to every sub- and
    super-diagonal block being the same constant diagonal matrix
    `U = diag(off_block)`, which collapses the block Thomas recurrence to one
    explicit 2×2 inverse plus multiply-adds per step. The off-diagonal blocks
    are never materialized and the boundary slots need no special casing. `N`
    is the number of block rows, each block being 2×2.

    Args:
        diag: Main diagonal blocks.
            Shape: `[..., N, 2, 2]`.
        rhs: Right-hand-side vectors.
            Shape: `[..., N, 2]`.
        off_block: The two diagonal entries of the shared off-block, i.e.
            `U = diag(off_block[0], off_block[1])`.

    Returns:
        Solution tensor.
        Shape: `[..., N, 2]`.
    """
    u_0, u_1 = off_block
    # Cross weights of U · M · U — entry (i, j) picks up u_i · u_j.
    w_00, w_01, w_10, w_11 = u_0 * u_0, u_0 * u_1, u_1 * u_0, u_1 * u_1

    # Shape: [..., N, 2, 2] -> [..., N] each
    d_00, d_01, d_10, d_11 = diag[..., 0, 0], diag[..., 0, 1], diag[..., 1, 0], diag[..., 1, 1]
    # Shape: [..., N, 2] -> [..., N] each
    r_0, r_1 = rhs[..., 0], rhs[..., 1]
    block_num = rhs.shape[-2]

    # Forward sweep: keep the inverted reduced block A_k = M_k⁻¹ and the
    # reduced right-hand side e_k = A_k · (rhs_k - U · e_{k-1}).
    a_00_list: list[Tensor] = []
    a_01_list: list[Tensor] = []
    a_10_list: list[Tensor] = []
    a_11_list: list[Tensor] = []
    e_0_list: list[Tensor] = []
    e_1_list: list[Tensor] = []
    for k in range(block_num):
        # Shape: [..., N] -> [...]
        m_00, m_01 = d_00.select(-1, k), d_01.select(-1, k)
        m_10, m_11 = d_10.select(-1, k), d_11.select(-1, k)
        b_0, b_1 = r_0.select(-1, k), r_1.select(-1, k)
        if k > 0:
            m_00 = m_00 - w_00 * a_00_list[k - 1]
            m_01 = m_01 - w_01 * a_01_list[k - 1]
            m_10 = m_10 - w_10 * a_10_list[k - 1]
            m_11 = m_11 - w_11 * a_11_list[k - 1]
            b_0 = b_0 - u_0 * e_0_list[k - 1]
            b_1 = b_1 - u_1 * e_1_list[k - 1]
        inv_det = 1.0 / (m_00 * m_11 - m_01 * m_10)
        a_00, a_01 = m_11 * inv_det, -m_01 * inv_det
        a_10, a_11 = -m_10 * inv_det, m_00 * inv_det
        a_00_list.append(a_00)
        a_01_list.append(a_01)
        a_10_list.append(a_10)
        a_11_list.append(a_11)
        e_0_list.append(a_00 * b_0 + a_01 * b_1)
        e_1_list.append(a_10 * b_0 + a_11 * b_1)

    # Back substitution: x_k = e_k - A_k · (U · x_{k+1}).
    x_0, x_1 = e_0_list[block_num - 1], e_1_list[block_num - 1]
    x_0_list: list[Tensor] = [x_0]
    x_1_list: list[Tensor] = [x_1]
    for k in range(block_num - 2, -1, -1):
        s_0, s_1 = u_0 * x_0, u_1 * x_1
        x_0 = e_0_list[k] - (a_00_list[k] * s_0 + a_01_list[k] * s_1)
        x_1 = e_1_list[k] - (a_10_list[k] * s_0 + a_11_list[k] * s_1)
        x_0_list.append(x_0)
        x_1_list.append(x_1)
    x_0_list.reverse()
    x_1_list.reverse()

    # Shape: [...] * N -> [..., N] -> [..., N, 2]
    return torch.stack((torch.stack(x_0_list, dim=-1), torch.stack(x_1_list, dim=-1)), dim=-1)


def solve_block_tridiagonal_dense(
    sub: Tensor,
    diag: Tensor,
    sup: Tensor,
    rhs: Tensor,
) -> Tensor:
    """Solve batched block-tridiagonal systems by densifying to one `N*B` square solve.

    Same input/output contract as `solve_block_tridiagonal`: `N` block rows of
    `B`×`B` blocks.

    Args:
        sub: Sub-diagonal blocks; `sub[0]` is zeroed during assembly.
            Shape: `[..., N, B, B]`.
        diag: Main diagonal blocks.
            Shape: `[..., N, B, B]`.
        sup: Super-diagonal blocks; `sup[-1]` is zeroed during assembly.
            Shape: `[..., N, B, B]`.
        rhs: Right-hand-side vectors.
            Shape: `[..., N, B]`.

    Returns:
        Solution tensor.
        Shape: `[..., N, B]`.
    """
    block_num = diag.shape[-3]
    block_size = diag.shape[-1]

    if block_num == 1:
        # Shape: [..., N=1, B] -> [..., B, 1] -> [..., N=1, B]
        return torch.linalg.solve(diag[..., 0, :, :], rhs[..., 0, :].unsqueeze(-1)).squeeze(-1).unsqueeze(-2)

    flat_size = block_num * block_size
    device = diag.device
    dtype = diag.dtype

    # Zero the unused boundary blocks so they do not pollute the assembled
    # dense matrix.
    sub_clean = torch.cat([torch.zeros_like(sub[..., :1, :, :]), sub[..., 1:, :, :]], dim=-3)
    sup_clean = torch.cat([sup[..., :-1, :, :], torch.zeros_like(sup[..., -1:, :, :])], dim=-3)

    # Placement matrices over the block-row index.
    #   diag_shift places block k on the main diagonal at (k·B, k·B).
    #   sub_shift puts a 1 at (k, k-1) for k>=1 → block k lands at (k·B, (k-1)·B).
    #   sup_shift puts a 1 at (k, k+1) for k<=N-2 → block k lands at (k·B, (k+1)·B).
    diag_shift = torch.eye(block_num, device=device, dtype=dtype)
    off_ones = torch.ones(block_num - 1, device=device, dtype=dtype)
    sub_shift = torch.diag_embed(off_ones, offset=-1)
    sup_shift = torch.diag_embed(off_ones, offset=+1)

    # `(k, i)` and `(m, j)` become the dense row and column indices.
    # Shape: [..., N, B, B] -> [..., N, B, N, B] -> [..., N*B, N*B]
    diag_part = torch.einsum("...kij,km->...kimj", diag, diag_shift).flatten(-4, -3).flatten(-2, -1)
    # Shape: [..., N, B, B] -> [..., N, B, N, B] -> [..., N*B, N*B]
    sub_part = torch.einsum("...kij,km->...kimj", sub_clean, sub_shift).flatten(-4, -3).flatten(-2, -1)
    # Shape: [..., N, B, B] -> [..., N, B, N, B] -> [..., N*B, N*B]
    sup_part = torch.einsum("...kij,km->...kimj", sup_clean, sup_shift).flatten(-4, -3).flatten(-2, -1)

    dense_matrix = diag_part + sub_part + sup_part

    # Shape: [..., N, B] -> [..., N*B, 1]
    rhs_flat = rhs.reshape(*rhs.shape[:-2], flat_size).unsqueeze(-1)
    # Shape: [..., N*B, 1] -> [..., N*B]
    x_flat = torch.linalg.solve(dense_matrix, rhs_flat).squeeze(-1)
    # Shape: [..., N*B] -> [..., N, B]
    return x_flat.view(*rhs.shape)


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
