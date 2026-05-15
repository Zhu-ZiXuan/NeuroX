"""Shared circuit-solver utilities for crossbar IR-drop simulation.

This module provides primitives used by the xbar solver:

Thomas tridiagonal algorithm (``solve_tridiagonal``):
    Solves batched tridiagonal linear systems ``A x = rhs`` via forward
    elimination followed by back substitution.  The implementation
    accumulates intermediate scalings into Python lists rather than
    writing back into the input tensors.  This avoids in-place mutations,
    which would break ``torch.compile`` graph capture.  Inductor fuses the
    per-step elementwise operations across loop iterations automatically.

KCL residuals (``col_wire_kcl_residual``, ``row_wire_kcl_residual``):
    Evaluate Kirchhoff's Current Law imbalance at every node of a
    column-oriented (BL, dim=-1) or row-oriented (SL/WL, dim=-2) wire.
    Both helpers consume a per-segment conductance tensor produced by
    :meth:`Wire.fabricate`; index 0 is the driver-to-first segment and
    indices ``1..N-1`` are the inter-node segments.  This naturally
    supports non-uniform pitches without changing the call site.

Driver-port currents (``col_driver_current``, ``row_driver_current``):
    Return the net current flowing out of the driver into each wire,
    computed as ``(v_drive - v_node[0]) * segment_g[0]`` — i.e. the
    Ohmic drop across the driver-to-first segment.

Autograd-based elementwise differentiation (``elementwise_diff``):
    Obtains ``∂f/∂x`` for an elementwise device function ``f`` by
    running a single PyTorch autograd backward pass.  Because every
    output element depends only on the corresponding input element,
    ``∂(Σ y)/∂x[k] = ∂y[k]/∂x[k]``, so no Jacobian materialisation
    is required.  This lets device models (RRAM, NMOS) expose only their
    forward I-V equation; the solver derives conductances automatically.

The tridiagonal wire Jacobian templates (the self-conductance
diagonal and the negated inter-segment off-diagonal) are
**solver-specific** views of :class:`~neurox.device.Wire` state and
are built inside :class:`~neurox.xbar.NewtonRaphsonSolver1T1R`
from ``segment_g__uS`` (see ``temp/wire.md``).
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


def elementwise_diff(fn: Callable[..., Tensor], *, wrt: str, **kwargs: Tensor) -> Tensor:
    """Per-element derivative of an elementwise device function via autograd.

    Computes ``∂fn[k] / ∂kwargs[wrt][k]`` for every element ``k`` in a
    single backward pass.  Because ``fn`` is elementwise, ``∂(Σ y)/∂x[k]
    = ∂y[k]/∂x[k]``, so ``grad(y.sum(), x)`` gives the full per-element
    Jacobian diagonal without materialising any off-diagonal terms.

    Device models (RRAM, NMOS) only need to implement the forward I-V
    equation; the solver calls this helper to derive differential
    conductances without requiring closed-form derivatives in the model.

    Note: Not compatible inside a ``@torch.compile`` region — call only
    from eager code or use a closed-form derivative instead (see
    ``_cell_diff_conductance__uS`` in ``xbar_1t1r_ideal_switch``).

    Args:
        fn: An elementwise tensor function, e.g. ``rram.i__uA``.
        wrt: Keyword argument name to differentiate with respect to.
        **kwargs: All keyword arguments forwarded to ``fn``.

    Returns:
        Tensor with the same shape as ``kwargs[wrt]``, where element
        ``[k]`` equals ``∂fn[k] / ∂kwargs[wrt][k]``.
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

    Solves ``A x = rhs`` along ``dim``, where ``A`` is tridiagonal with
    sub-diagonal ``sub``, main diagonal ``diag``, and super-diagonal
    ``sup``.  All four tensors share the same shape.  The entry of
    ``sub`` at index 0 along ``dim`` and the entry of ``sup`` at the
    last index are unused placeholders and may hold any value.

    Implementation: forward elimination accumulates the modified diagonal
    and RHS slices into Python lists (no in-place writes), then back
    substitution builds the solution list right-to-left and stacks.
    List-based accumulation avoids tensor mutation, which is incompatible
    with ``torch.compile`` graph tracing.  Inductor fuses the per-step
    elementwise ops across the unrolled loop at compile time.

    Args:
        sub: Sub-diagonal coefficients, same shape as ``rhs``.
        diag: Main diagonal coefficients, same shape as ``rhs``.
        sup: Super-diagonal coefficients, same shape as ``rhs``.
        rhs: Right-hand-side vectors, shape ``[..., N, ...]``.
        dim: The dimension of length ``N`` along which each system
            is laid out.

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

    The BL wire runs along ``dim=-1`` (the ``row_num`` axis); the driver
    sits at index 0 of that axis.  The residual at each node is the
    net current *leaving* the node (positive: excess; negative: deficit).
    At equilibrium the residual is zero.

    With per-segment conductances ``segment_g[0..N-1]`` (index 0 is
    the driver-to-first segment), the residual at node ``k`` is

    * ``k = 0``:           ``i_inject + (v_0   - v_drv) g_s[0] + (v_0   - v_1) g_s[1]``.
    * ``1 <= k <= N - 2``: ``i_inject + (v_k - v_{k-1}) g_s[k] + (v_k - v_{k+1}) g_s[k+1]``.
    * ``k = N - 1``:       ``i_inject + (v_{N-1} - v_{N-2}) g_s[N-1]`` (open far end).

    Args:
        v_node: Wire node voltages [V]. Shape ``[..., col_num, row_num]``.
        v_drive: Driver voltage [V].  Shape ``[..., col_num, 1]``.
        segment_g: Per-segment conductance [uS] for one BL line;
            shape ``(row_num,)``.  ``segment_g[0]`` is the driver
            segment, ``segment_g[k]`` for ``k >= 1`` is the
            node-``(k-1)`` → node-``k`` segment.  Owned by
            :class:`~neurox.device.Wire`.
        i_inject: Cell current drawn at each node [uA], same shape as
            ``v_node``.

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
    """KCL residual at every node of a row-oriented (SL/WL) wire.

    Analogous to :func:`col_wire_kcl_residual` but the wire runs along
    ``dim=-2`` (the ``col_num`` axis); the driver sits at index 0 of
    that axis.

    Args:
        v_node: Wire node voltages [V]. Shape ``[..., col_num, row_num]``.
        v_drive: Driver voltage [V].  Shape ``[..., 1, row_num]``.
        segment_g: Per-segment conductance [uS] for one SL line;
            shape ``(col_num,)``.  ``segment_g[0]`` is the driver
            segment, ``segment_g[k]`` for ``k >= 1`` is the
            node-``(k-1)`` → node-``k`` segment.  Owned by
            :class:`~neurox.device.Wire`.
        i_inject: Cell current drawn at each node [uA], same shape as
            ``v_node``.

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
    """Net current flowing from each BL driver into the wire [uA].

    Computed as the Ohmic drop across the driver-to-first segment:
    ``(v_drive - v_node[..., 0]) * segment_g[0]``.

    Args:
        v_node: BL wire node voltages [V]. Shape ``[..., col_num, row_num]``.
        v_drive: BL driver voltage [V]. Shape ``[..., col_num, 1]``.
        segment_g: BL per-segment conductance [uS] (only
            ``segment_g[0]``, the driver segment, is read).

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
    """Net current flowing from each SL/WL driver into the wire [uA].

    Analogous to :func:`col_driver_current` but on the row axis
    (``dim=-2``).

    Args:
        v_node: SL wire node voltages [V]. Shape ``[..., col_num, row_num]``.
        v_drive: SL driver voltage [V]. Shape ``[..., 1, row_num]``.
        segment_g: SL per-segment conductance [uS] (only
            ``segment_g[0]`` is read).

    Returns:
        Driver current [uA]. Shape ``[..., row_num]``.
    """
    dim = -2
    return (v_drive.squeeze(dim) - v_node.select(dim, 0)) * segment_g[0]
