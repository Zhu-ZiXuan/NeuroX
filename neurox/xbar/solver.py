"""Shared circuit-solver math utilities for crossbar IR-drop simulation.

This module owns purely numerical helpers (tridiagonal Thomas algorithm,
block-tridiagonal Thomas, KCL residual builders for column / row wire
ladders). Solver class bases live with their topology family (e.g.
:mod:`neurox.xbar._1t1r.solver`) — there is no generic ``Solver``
abstraction here because device / boundary signatures differ per
topology.

See also:
    docs/dev/modules/xbar/solver.md
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
