"""Shared circuit-solver math utilities for crossbar IR-drop simulation.

This module owns purely numerical helpers (tridiagonal Thomas algorithm,
block-tridiagonal Thomas, KCL residual builders for column / row wire
ladders). Solver class bases live with their topology family (e.g.
:mod:`neurox.xbar._1t1r.solver`) — there is no generic ``Solver``
abstraction here because device / boundary signatures differ per
topology.

See also:
    docs/modules/xbar/solver.md
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
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

    Each ``B × B`` block-inverse step uses ``torch.linalg.solve``; for
    ``B = 1`` the algorithm reduces to scalar Thomas (no overhead beyond
    the extra rank).

    Args:
        sub: Sub-diagonal blocks. Shape ``[..., N, B, B]``.
        diag: Main diagonal blocks. Same shape as ``sub``.
        sup: Super-diagonal blocks. Same shape as ``sub``.
        rhs: Right-hand-side vectors. Shape ``[..., N, B]``.

    Returns:
        Solution tensor with the same shape as ``rhs``.
    """
    n = rhs.shape[-2]
    if n == 1:
        # Single block: just one B×B solve.
        return torch.linalg.solve(diag[..., 0, :, :], rhs[..., 0, :].unsqueeze(-1)).squeeze(-1)

    # Forward sweep — keep C_{k} = M_k⁻¹ · sup_k and d_k = M_k⁻¹ · (rhs - sub · d_{k-1})
    # in lists; never write in-place so ``torch.compile`` can fuse.
    m_0 = diag[..., 0, :, :]
    rhs_0 = rhs[..., 0, :].unsqueeze(-1)
    # Stack [sup, rhs] as RHS columns so we do one solve per step instead of two.
    sol_0 = torch.linalg.solve(m_0, torch.cat((sup[..., 0, :, :], rhs_0), dim=-1))
    c_list: list[Tensor] = [sol_0[..., :-1]]
    d_list: list[Tensor] = [sol_0[..., -1:]]
    for k in range(1, n):
        sub_k = sub[..., k, :, :]
        diag_k = diag[..., k, :, :]
        sup_k = sup[..., k, :, :]
        rhs_k = rhs[..., k, :].unsqueeze(-1)
        m_k = diag_k - sub_k @ c_list[k - 1]
        sol_k = torch.linalg.solve(m_k, torch.cat((sup_k, rhs_k - sub_k @ d_list[k - 1]), dim=-1))
        c_list.append(sol_k[..., :-1])
        d_list.append(sol_k[..., -1:])

    # Back substitution — build the solution list right-to-left.
    x_list: list[Tensor] = [d_list[n - 1].squeeze(-1)]
    for k in range(n - 2, -1, -1):
        x_list.append((d_list[k] - c_list[k] @ x_list[-1].unsqueeze(-1)).squeeze(-1))
    x_list.reverse()

    return torch.stack(x_list, dim=-2)


def solve_block_tridiagonal_dense(
    sub: Tensor,
    diag: Tensor,
    sup: Tensor,
    rhs: Tensor,
) -> Tensor:
    """Solve batched block-tridiagonal systems by densifying to a ``[NB, NB]`` solve.

    Same input/output contract as :func:`solve_block_tridiagonal`. Trades
    asymptotic work (``O((NB)³)`` flops vs ``O(N · B³)``) for **graph
    flatness**: the entire path is a fixed-shape sequence of three
    ``einsum`` placements + one ``torch.linalg.solve`` — no Python loop
    over ``N``, no list mutation, no per-step intermediate. dynamo sees
    O(1) nodes regardless of N.

    On GPU, dense ``NB × NB`` solves at ``NB ≤ 256`` (i.e. N ≤ 128 for
    B=2) are dominated by kernel launch + memory bandwidth, not flops,
    so this beats Thomas in eager runtime as well. Memory scales as
    ``O(N² · B²)`` per (batch, inst); at N≈1024 this becomes the
    limiting factor — use ``solve_block_tridiagonal`` (Thomas) for very
    large N where memory matters more than depth.

    Args:
        sub: Sub-diagonal blocks. Shape ``[..., N, B, B]``. ``sub[0]`` is
            taken as ignored (zeroed during assembly).
        diag: Main diagonal blocks. Same shape.
        sup: Super-diagonal blocks. Same shape. ``sup[-1]`` is taken as
            ignored.
        rhs: Right-hand-side vectors. Shape ``[..., N, B]``.

    Returns:
        Solution tensor with the same shape as ``rhs``.
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

    # Place blocks via einsum: '...kij,km->...kimj'. The output shape
    # [..., k=n, i=B, m=n, j=B] is then flattened to [..., n·B, n·B] so
    # that index (k, i) → row k·B+i and (m, j) → col m·B+j.
    diag_part = torch.einsum("...kij,km->...kimj", diag, eye_n).flatten(-4, -3).flatten(-2, -1)
    sub_part = torch.einsum("...kij,km->...kimj", sub_clean, sub_shift).flatten(-4, -3).flatten(-2, -1)
    sup_part = torch.einsum("...kij,km->...kimj", sup_clean, sup_shift).flatten(-4, -3).flatten(-2, -1)

    dense_matrix = diag_part + sub_part + sup_part

    rhs_flat = rhs.reshape(*rhs.shape[:-2], nb).unsqueeze(-1)
    x_flat = torch.linalg.solve(dense_matrix, rhs_flat).squeeze(-1)
    return x_flat.view(*rhs.shape)


def _pcr_validity_mask(n: int, stride: int, dim: int, ndim: int, device: torch.device) -> Tensor:
    """1 where the shifted position has a valid in-range neighbour, 0 at boundary.

    Returns a broadcastable bool tensor with 1s on ``dim`` of size N and
    1s everywhere else, ready to multiply / where against tensors of full
    shape.
    """
    indices = torch.arange(n, device=device)
    valid = indices >= stride if stride > 0 else indices < n + stride
    shape = [1] * ndim
    shape[dim] = n
    return valid.view(shape)


def _pcr_shift_zero(t: Tensor, stride: int, dim: int) -> Tensor:
    """Shift ``t`` along ``dim`` so position k gets the value at position ``k - stride``.

    Implemented as one ``torch.roll`` + a broadcast multiplicative mask
    that zeros wraparound positions — both are graph-friendlier than
    ``narrow + zeros + cat`` for the unrolled PCR loop.
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
    """Solve batched block-tridiagonal systems via Parallel Cyclic Reduction.

    Same input/output contract as :func:`solve_block_tridiagonal` (block
    Thomas), but the unrolled dynamo graph has depth ``ceil(log2 N)``
    instead of ``N``. Each PCR step is a fixed-shape batched B×B matrix
    kernel — no Python list mutation, no ``torch.compile`` graph break.

    Algorithm sketch (block size B, system size N):

    1. At stride ``s = 1, 2, 4, ..., < N``, compute for every k::

           α_k = -sub_k · diag_{k-s}⁻¹      (zero where k - s < 0)
           β_k = -sup_k · diag_{k+s}⁻¹      (zero where k + s ≥ N)
           sub_k  ← α_k · sub_{k-s}
           sup_k  ← β_k · sup_{k+s}
           diag_k ← diag_k + α_k · sup_{k-s} + β_k · sub_{k+s}
           rhs_k  ← rhs_k + α_k · rhs_{k-s} + β_k · rhs_{k+s}

       Out-of-range ``diag_{...}`` is padded with the identity, all other
       out-of-range tensors with zero. Each step halves the sub/sup
       neighbour distance, so after ``ceil(log2 N)`` steps the off-diagonals
       are zero.

    2. The decoupled diagonal system ``diag_k · x_k = rhs_k`` is solved as
       one batched B×B inverse.

    Numerical stability: PCR has the same forward error as Thomas for
    diagonally-dominant systems (the regime our Newton Jacobian sits in),
    but without partial pivoting it is slightly more sensitive in
    ill-conditioned corners. Verified against Thomas + dense LU in
    :file:`tests/test_block_tridiagonal_pcr.py` (rtol 1e-10 fp64, 1e-4 fp32).
    """
    n = diag.shape[-3]
    block_dim = -3
    vec_dim = -2

    if n == 1:
        return torch.linalg.solve(diag[..., 0, :, :], rhs[..., 0, :].unsqueeze(-1)).squeeze(-1)

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
        alpha = torch.linalg.solve(b_l.transpose(-1, -2), -a.transpose(-1, -2)).transpose(-1, -2)
        beta = torch.linalg.solve(b_r.transpose(-1, -2), -c.transpose(-1, -2)).transpose(-1, -2)

        new_a = alpha @ a_l
        new_c = beta @ c_r
        new_b = b + alpha @ c_l + beta @ a_r
        new_r = (r.unsqueeze(-1) + alpha @ r_l.unsqueeze(-1) + beta @ r_r.unsqueeze(-1)).squeeze(-1)

        a, b, c, r = new_a, new_b, new_c, new_r
        stride *= 2

    return torch.linalg.solve(b, r.unsqueeze(-1)).squeeze(-1)


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
        sub: Sub-diagonal coefficients, same shape as ``rhs``.
        diag: Main diagonal coefficients, same shape as ``rhs``.
        sup: Super-diagonal coefficients, same shape as ``rhs``.
        rhs: Right-hand-side vectors, shape ``[..., N, ...]``.
        dim: The dimension of length ``N``.

    Returns:
        Solution tensor with the same shape as ``rhs``.
    """
    N = rhs.shape[dim]
    if N == 1:
        return rhs / diag

    # Forward sweep — collect updated diagonal and rhs slices into lists
    # without any in-place writes; ``torch.compile`` is responsible for
    # fusing the per-step elementwise ops.
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


def col_wire_kcl_residual(
    v_node: Tensor,
    v_drive: Tensor,
    segment_g: Tensor,
    i_inject: Tensor,
) -> Tensor:
    """KCL residual at every node of a column-oriented wire.

    The wire runs along ``dim=-1`` (the ``row_num`` axis); the driver
    sits at index 0. Per-segment conductances ``segment_g[k]`` map index
    0 to the driver-to-first segment and index ``k`` to the
    ``node-(k-1) → node-k`` segment.

    Args:
        v_node: Wire node voltages [V]. Shape
            ``[..., col_num, row_num]``.
        v_drive: Driver voltage [V]. Shape ``[..., col_num, 1]``.
        segment_g: Per-segment conductance [uS]. Shape ``(row_num,)``.
        i_inject: Cell current drawn at each node [uA].

    Returns:
        KCL residual tensor [uA], same shape as ``v_node``.
    """
    dim = -1
    num_row = v_node.shape[dim]
    shape_broadcast = [1] * v_node.ndim
    shape_broadcast[dim] = num_row
    g_to_left = segment_g.view(shape_broadcast)
    # g_to_right[k] = segment_g[k+1] for k <= N-2; 0 at k = N-1 (no right segment).
    g_to_right = F.pad(segment_g[1:], (0, 1)).view(shape_broadcast)

    # dv_to_left[k] = v_node[k] - v_node[k-1] for k >= 1; v_node[0] - v_drive at k = 0.
    v_with_drive = torch.cat((v_drive, v_node), dim=dim)
    dv_to_left = torch.diff(v_with_drive, dim=dim)
    # dv_to_right[k] = v_node[k] - v_node[k+1] for k <= N-2; 0 at k = N-1.
    dv_to_right = F.pad(-torch.diff(v_node, dim=dim), (0, 1))

    return i_inject + dv_to_left * g_to_left + dv_to_right * g_to_right


def row_wire_kcl_residual(
    v_node: Tensor,
    v_drive: Tensor,
    segment_g: Tensor,
    i_inject: Tensor,
) -> Tensor:
    """KCL residual at every node of a row-oriented wire.

    Same structure as :func:`col_wire_kcl_residual` along ``dim=-2``.

    Args:
        v_node: Wire node voltages [V]. Shape
            ``[..., col_num, row_num]``.
        v_drive: Driver voltage [V]. Shape ``[..., 1, row_num]``.
        segment_g: Per-segment conductance [uS]. Shape ``(col_num,)``.
        i_inject: Cell current drawn at each node [uA].

    Returns:
        KCL residual tensor [uA], same shape as ``v_node``.
    """
    dim = -2
    num_col = v_node.shape[dim]
    shape_broadcast = [1] * v_node.ndim
    shape_broadcast[dim] = num_col
    g_to_left = segment_g.view(shape_broadcast)
    g_to_right = F.pad(segment_g[1:], (0, 1)).view(shape_broadcast)

    v_with_drive = torch.cat((v_drive, v_node), dim=dim)
    dv_to_left = torch.diff(v_with_drive, dim=dim)
    dv_to_right = F.pad(-torch.diff(v_node, dim=dim), (0, 0, 0, 1))

    return i_inject + dv_to_left * g_to_left + dv_to_right * g_to_right


def col_driver_current(
    v_node: Tensor,
    v_drive: Tensor,
    segment_g: Tensor,
) -> Tensor:
    """Net current from a column-oriented-wire driver into the wire [uA].

    Args:
        v_node: Wire node voltages [V]. Shape
            ``[..., col_num, row_num]``.
        v_drive: Driver voltage [V]. Shape ``[..., col_num, 1]``.
        segment_g: Per-segment conductance [uS] — only
            ``segment_g[0]`` is read.

    Returns:
        Driver current [uA]. Shape ``[..., col_num]``.
    """
    dim = -1
    return (v_drive.squeeze(dim) - v_node.select(dim, 0)) * segment_g[0]


def row_driver_current(
    v_node: Tensor,
    v_drive: Tensor,
    segment_g: Tensor,
) -> Tensor:
    """Net current from a row-oriented-wire driver into the wire [uA].

    Args:
        v_node: Wire node voltages [V]. Shape
            ``[..., col_num, row_num]``.
        v_drive: Driver voltage [V]. Shape ``[..., 1, row_num]``.
        segment_g: Per-segment conductance [uS] — only
            ``segment_g[0]`` is read.

    Returns:
        Driver current [uA]. Shape ``[..., row_num]``.
    """
    dim = -2
    return (v_drive.squeeze(dim) - v_node.select(dim, 0)) * segment_g[0]
