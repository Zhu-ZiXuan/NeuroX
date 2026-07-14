"""Crossbar wire-ladder KCL residual and driver-current builders.

These builders own the wire-ladder layout convention: the wire axis
(column = ``dim=-1``, row = ``dim=-2``), the driver node at index 0, the
per-segment conductance-to-left / conductance-to-right mapping, and the KCL
sign convention ``r = i_inject + Δv_left · g_left + Δv_right · g_right``.

See also:
    docs/reference/primitive/xbar/solver/README.md
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


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
        v_drive: Drive voltage [V]. Shape ``[..., col_num, 1]``.
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
        v_drive: Drive voltage [V]. Shape ``[..., 1, row_num]``.
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
        v_drive: Drive voltage [V]. Shape ``[..., col_num, 1]``.
        segment_g: Per-segment conductance [uS] — only
            ``segment_g[0]`` is read.

    Returns:
        Drive current [uA]. Shape ``[..., col_num]``.
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
        v_drive: Drive voltage [V]. Shape ``[..., 1, row_num]``.
        segment_g: Per-segment conductance [uS] — only
            ``segment_g[0]`` is read.

    Returns:
        Drive current [uA]. Shape ``[..., row_num]``.
    """
    dim = -2
    return (v_drive.squeeze(dim) - v_node.select(dim, 0)) * segment_g[0]
