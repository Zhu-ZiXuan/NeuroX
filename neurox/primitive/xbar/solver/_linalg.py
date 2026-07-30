"""Numerical linear-algebra helpers for crossbar IR-drop simulation.

See also:
    docs/reference/primitive/xbar/solver/nested.md
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def elementwise_diff(fn: Callable[..., Tensor], *, wrt: str, **kwargs: Tensor) -> Tensor:
    """Per-element derivative of an element-wise function via autograd.

    Args:
        fn: An element-wise tensor function.
        wrt: Keyword argument name to differentiate with respect to.
        **kwargs: All keyword arguments forwarded to ``fn``.

    Returns:
        Tensor with the same shape as ``kwargs[wrt]``; element ``k``
        is ``∂fn[k] / ∂kwargs[wrt][k]``.
    """
    with torch.enable_grad():
        x = kwargs[wrt].detach().clone().requires_grad_(True)
        kwargs[wrt] = x
        y = fn(**kwargs)
        (dy_dx,) = torch.autograd.grad(y.sum(), x)
    return dy_dx


def block_matmul(p: Tensor, q: Tensor) -> Tensor:
    """Compute batched ``p @ q`` with a closed-form 2×2 path.

    For ``B == 2`` the product uses elementwise multiply-adds. Other block
    sizes use ``p @ q``.

    Args:
        p: Left block operand.
            Shape: ``[..., B, B]``.
        q: Right block operand.
            Shape: ``[..., B, K]``.

    Returns:
        Block product.
        Shape: ``[..., B, K]``.
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
    """Solve batched ``m x = rhs`` with a closed-form 2×2 path.

    For ``B == 2`` the solve uses the adjugate/determinant formula. Other
    block sizes use ``torch.linalg.solve``.

    Args:
        m: Block system matrix.
            Shape: ``[..., B, B]``.
        rhs: Right-hand-side columns.
            Shape: ``[..., B, K]``.

    Returns:
        Solution columns.
        Shape: ``[..., B, K]``.
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

    Solves ``A x = rhs`` where ``A`` is block-tridiagonal with ``B × B``
    blocks. Shape convention is fixed (no ``dim`` arg) — the N axis is
    always third-to-last for the block tensors and second-to-last for
    ``rhs``. Transpose at the call site if your data is laid out
    differently.

    The block at row ``k`` has:

      * sub-diagonal block ``sub[..., k, :, :]`` (coupling to row ``k-1``);
        the entry at ``k = 0`` is unused
      * main diagonal block ``diag[..., k, :, :]``
      * super-diagonal block ``sup[..., k, :, :]`` (coupling to row ``k+1``);
        the entry at ``k = N-1`` is unused

    Args:
        sub: Sub-diagonal blocks.
            Shape: ``[..., N, B, B]``.
        diag: Main diagonal blocks.
            Shape: ``[..., N, B, B]``.
        sup: Super-diagonal blocks.
            Shape: ``[..., N, B, B]``.
        rhs: Right-hand-side vectors.
            Shape: ``[..., N, B]``.

    Returns:
        Solution tensor.
        Shape: ``[..., N, B]``.
    """
    n = rhs.shape[-2]
    if n == 1:
        return block_solve(diag[..., 0, :, :], rhs[..., 0, :].unsqueeze(-1)).squeeze(-1)

    # Forward sweep: C_k = M_k⁻¹ · sup_k, d_k = M_k⁻¹ · (rhs - sub · d_{k-1}).
    m_0 = diag[..., 0, :, :]
    rhs_0 = rhs[..., 0, :].unsqueeze(-1)
    # Stack [sup, rhs] as RHS columns so we do one solve per step instead of two.
    sol_0 = block_solve(m_0, torch.cat((sup[..., 0, :, :], rhs_0), dim=-1))
    c_list: list[Tensor] = [sol_0[..., :-1]]
    d_list: list[Tensor] = [sol_0[..., -1:]]
    for k in range(1, n):
        sub_k = sub[..., k, :, :]
        diag_k = diag[..., k, :, :]
        sup_k = sup[..., k, :, :]
        rhs_k = rhs[..., k, :].unsqueeze(-1)
        m_k = diag_k - block_matmul(sub_k, c_list[k - 1])
        sol_k = block_solve(m_k, torch.cat((sup_k, rhs_k - block_matmul(sub_k, d_list[k - 1])), dim=-1))
        c_list.append(sol_k[..., :-1])
        d_list.append(sol_k[..., -1:])

    # Back substitution — build the solution list right-to-left.
    x_list: list[Tensor] = [d_list[n - 1].squeeze(-1)]
    for k in range(n - 2, -1, -1):
        x_list.append((d_list[k] - block_matmul(c_list[k], x_list[-1].unsqueeze(-1))).squeeze(-1))
    x_list.reverse()

    return torch.stack(x_list, dim=-2)


def solve_block_tridiagonal_dense(
    sub: Tensor,
    diag: Tensor,
    sup: Tensor,
    rhs: Tensor,
) -> Tensor:
    """Solve batched block-tridiagonal systems by densifying to one ``N*B`` square solve.

    Same input/output contract as :func:`solve_block_tridiagonal`.

    Args:
        sub: Sub-diagonal blocks. ``sub[0]`` is ignored (zeroed during
            assembly).
            Shape: ``[..., N, B, B]``.
        diag: Main diagonal blocks.
            Shape: ``[..., N, B, B]``.
        sup: Super-diagonal blocks. ``sup[-1]`` is ignored.
            Shape: ``[..., N, B, B]``.
        rhs: Right-hand-side vectors.
            Shape: ``[..., N, B]``.

    Returns:
        Solution tensor.
        Shape: ``[..., N, B]``.
    """
    n = diag.shape[-3]
    b = diag.shape[-1]

    if n == 1:
        return torch.linalg.solve(diag[..., 0, :, :], rhs[..., 0, :].unsqueeze(-1)).squeeze(-1)

    nb = n * b
    device = diag.device
    dtype = diag.dtype

    # Zero out the un-used boundary blocks so they don't pollute the
    # assembled dense matrix.
    sub_clean = torch.cat([torch.zeros_like(sub[..., :1, :, :]), sub[..., 1:, :, :]], dim=-3)
    sup_clean = torch.cat([sup[..., :-1, :, :], torch.zeros_like(sup[..., -1:, :, :])], dim=-3)

    # Shift matrices for block placement.
    #   eye_n places block k on the main diagonal at (k·B, k·B).
    #   sub_shift puts a 1 at (k, k-1) for k>=1 → block k lands at (k·B, (k-1)·B).
    #   sup_shift puts a 1 at (k, k+1) for k<=n-2 → block k lands at (k·B, (k+1)·B).
    eye_n = torch.eye(n, device=device, dtype=dtype)
    ones_n1 = torch.ones(n - 1, device=device, dtype=dtype)
    sub_shift = torch.diag_embed(ones_n1, offset=-1)
    sup_shift = torch.diag_embed(ones_n1, offset=+1)

    # ``(k, i)`` and ``(m, j)`` become the dense row and column indices.
    # Shape: [..., N, B, B] -> [..., N, B, N, B] -> [..., N*B, N*B]
    diag_part = torch.einsum("...kij,km->...kimj", diag, eye_n).flatten(-4, -3).flatten(-2, -1)
    # Shape: [..., N, B, B] -> [..., N, B, N, B] -> [..., N*B, N*B]
    sub_part = torch.einsum("...kij,km->...kimj", sub_clean, sub_shift).flatten(-4, -3).flatten(-2, -1)
    # Shape: [..., N, B, B] -> [..., N, B, N, B] -> [..., N*B, N*B]
    sup_part = torch.einsum("...kij,km->...kimj", sup_clean, sup_shift).flatten(-4, -3).flatten(-2, -1)

    dense_matrix = diag_part + sub_part + sup_part

    # Shape: [..., N, B] -> [..., N*B, 1]
    rhs_flat = rhs.reshape(*rhs.shape[:-2], nb).unsqueeze(-1)
    # Shape: [..., N*B, 1] -> [..., N*B]
    x_flat = torch.linalg.solve(dense_matrix, rhs_flat).squeeze(-1)
    # Shape: [..., N*B] -> [..., N, B]
    return x_flat.view(*rhs.shape)


def _pcr_validity_mask(n: int, stride: int, dim: int, ndim: int, device: torch.device) -> Tensor:
    """1 where the shifted position has a valid in-range neighbour, 0 at boundary.

    Every axis other than ``dim`` is size 1, so the mask broadcasts against
    tensors of the full shape, ready to multiply or select against.

    Returns:
        Boundary-validity mask.
        Shape: ``[..., N, ...]``.
    """
    indices = torch.arange(n, device=device)
    valid = indices >= stride if stride > 0 else indices < n + stride
    shape = [1] * ndim
    shape[dim] = n
    return valid.view(shape)


def _pcr_shift_zero(t: Tensor, stride: int, dim: int) -> Tensor:
    """Shift ``t`` along ``dim`` so position k gets the value at position ``k - stride``.

    A ``torch.roll`` + a broadcast multiplicative mask that zeros wraparound
    positions.
    """
    if abs(stride) >= t.shape[dim]:
        return torch.zeros_like(t)
    shifted = torch.roll(t, shifts=stride, dims=dim)
    mask = _pcr_validity_mask(t.shape[dim], stride, dim, t.ndim, t.device)
    return shifted * mask


def _pcr_shift_identity(t: Tensor, stride: int, dim: int) -> Tensor:
    """Same as :func:`_pcr_shift_zero` but the boundary fill is the B×B identity.

    Used for the diagonal tensor: out-of-range neighbours produce
    ``-sub · I = -sub`` which is then multiplied by the zero-padded
    ``sub_l`` / ``rhs_l`` giving zero contribution at the boundary.
    """
    b = t.shape[-1]
    eye = torch.eye(b, dtype=t.dtype, device=t.device).expand_as(t)
    if abs(stride) >= t.shape[dim]:
        return eye
    shifted = torch.roll(t, shifts=stride, dims=dim)
    mask = _pcr_validity_mask(t.shape[dim], stride, dim, t.ndim, t.device)
    return torch.where(mask, shifted, eye)


def solve_block_tridiagonal_pcr(
    sub: Tensor,
    diag: Tensor,
    sup: Tensor,
    rhs: Tensor,
) -> Tensor:
    """Solve batched block-tridiagonal systems by Parallel Cyclic Reduction.

    Uses the same tensor contract as :func:`solve_block_tridiagonal`.
    The implementation has no pivoting and is intended for diagonally-dominant systems.
    """
    n = diag.shape[-3]
    block_dim = -3
    vec_dim = -2

    if n == 1:
        return block_solve(diag[..., 0, :, :], rhs[..., 0, :].unsqueeze(-1)).squeeze(-1)

    a, b, c, r = sub, diag, sup, rhs

    stride = 1
    while stride < n:
        a_l = _pcr_shift_zero(a, stride, block_dim)
        b_l = _pcr_shift_identity(b, stride, block_dim)
        c_l = _pcr_shift_zero(c, stride, block_dim)
        r_l = _pcr_shift_zero(r, stride, vec_dim)
        a_r = _pcr_shift_zero(a, -stride, block_dim)
        b_r = _pcr_shift_identity(b, -stride, block_dim)
        c_r = _pcr_shift_zero(c, -stride, block_dim)
        r_r = _pcr_shift_zero(r, -stride, vec_dim)

        # α = -a · b_l⁻¹  via  α · b_l = -a  →  b_l.T · α.T = -a.T
        alpha = block_solve(b_l.transpose(-1, -2), -a.transpose(-1, -2)).transpose(-1, -2)
        beta = block_solve(b_r.transpose(-1, -2), -c.transpose(-1, -2)).transpose(-1, -2)

        new_a = block_matmul(alpha, a_l)
        new_c = block_matmul(beta, c_r)
        new_b = b + block_matmul(alpha, c_l) + block_matmul(beta, a_r)
        new_r = (
            r.unsqueeze(-1) + block_matmul(alpha, r_l.unsqueeze(-1)) + block_matmul(beta, r_r.unsqueeze(-1))
        ).squeeze(-1)

        a, b, c, r = new_a, new_b, new_c, new_r
        stride *= 2

    return block_solve(b, r.unsqueeze(-1)).squeeze(-1)


def solve_tridiagonal(
    sub: Tensor,
    diag: Tensor,
    sup: Tensor,
    rhs: Tensor,
    dim: int,
) -> Tensor:
    """Solve batched tridiagonal systems via the Thomas algorithm.

    Solves ``A x = rhs`` along ``dim``; ``A`` is tridiagonal with
    ``(sub, diag, sup)``. The entry of ``sub`` at index 0 and the
    entry of ``sup`` at the last index are unused boundary slots
    (the algorithm ignores them).

    Args:
        sub: Sub-diagonal coefficients.
            Shape: ``[..., N, ...]``.
        diag: Main diagonal coefficients.
            Shape: ``[..., N, ...]``.
        sup: Super-diagonal coefficients.
            Shape: ``[..., N, ...]``.
        rhs: Right-hand-side vectors.
            Shape: ``[..., N, ...]``.
        dim: The dimension of length ``N``.

    Returns:
        Solution tensor.
        Shape: ``[..., N, ...]``.
    """
    N = rhs.shape[dim]
    if N == 1:
        return rhs / diag

    # Forward sweep.
    d_list: list[Tensor] = [diag.select(dim, 0)]
    r_list: list[Tensor] = [rhs.select(dim, 0)]
    for i in range(1, N):
        w = sub.select(dim, i) / d_list[i - 1]
        d_list.append(diag.select(dim, i) - w * sup.select(dim, i - 1))
        r_list.append(rhs.select(dim, i) - w * r_list[i - 1])

    # Back substitution — build the solution list right-to-left.
    x_list: list[Tensor] = [r_list[N - 1] / d_list[N - 1]]
    for i in range(N - 2, -1, -1):
        x_list.append((r_list[i] - sup.select(dim, i) * x_list[-1]) / d_list[i])
    x_list.reverse()

    return torch.stack(x_list, dim=dim)
