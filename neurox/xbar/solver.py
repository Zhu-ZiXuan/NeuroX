"""Shared circuit-solver utilities for crossbar IR-drop simulation.

See also:
    docs/dev/modules/xbar/solver.md
"""

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
    entry of ``sup`` at the last index are unused placeholders.

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
    """KCL residual at every node of a column-oriented (BL) wire.

    The BL wire runs along ``dim=-1`` (the ``row_num`` axis); the
    driver sits at index 0. Per-segment conductances ``segment_g[k]``
    map index 0 to the driver-to-first segment and index ``k`` to the
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
    """KCL residual at every node of a row-oriented (SL / WL) wire.

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
    """Net current from each BL driver into the wire [uA].

    Args:
        v_node: BL wire node voltages [V]. Shape
            ``[..., col_num, row_num]``.
        v_drive: BL driver voltage [V]. Shape ``[..., col_num, 1]``.
        segment_g: BL per-segment conductance [uS] — only
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
    """Net current from each SL / WL driver into the wire [uA].

    Args:
        v_node: SL wire node voltages [V]. Shape
            ``[..., col_num, row_num]``.
        v_drive: SL driver voltage [V]. Shape ``[..., 1, row_num]``.
        segment_g: SL per-segment conductance [uS] — only
            ``segment_g[0]`` is read.

    Returns:
        Driver current [uA]. Shape ``[..., row_num]``.
    """
    dim = -2
    return (v_drive.squeeze(dim) - v_node.select(dim, 0)) * segment_g[0]
