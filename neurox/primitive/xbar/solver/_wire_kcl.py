"""Wire-ladder KCL residuals and driver-current calculations.

Every wire is a UNIFORM ladder: one lattice link joins each pair of adjacent
nodes and the same link joins the driver to the node at index 0, so a whole
rail is described by one scalar conductance. The only distinguished node is
the ladder's open end at the far index, which has no link onward.

See also:
    docs/reference/primitive/xbar/solver/nested.md
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def col_wire_kcl_residual(
    v_node: Tensor,
    v_drive: Tensor,
    segment_g: float,
    i_inject: Tensor,
) -> Tensor:
    """KCL residual at every node of a column-oriented wire.

    The wire runs along ``dim=-1`` and its driver hangs off index 0.

    Args:
        v_node: Wire node voltages [V].
            Shape: ``[..., col_num, row_num]``.
        v_drive: Drive voltage [V].
            Shape: ``[..., col_num, 1]``.
        segment_g: Conductance of one lattice link [uS].
        i_inject: Cell current drawn at each node [uA].
            Shape: ``[..., col_num, row_num]``.

    Returns:
        KCL residual tensor [uA].
        Shape: ``[..., col_num, row_num]``.
    """
    dim = -1
    # dv_to_left[k] = v_node[k] - v_node[k-1] for k >= 1; v_node[0] - v_drive at k = 0.
    # Shape: [..., col_num, wire_point]
    v_with_drive = torch.cat((v_drive, v_node), dim=dim)
    # Shape: [..., col_num, wire_point] -> [..., col_num, row_num]
    dv_to_left = torch.diff(v_with_drive, dim=dim)
    # dv_to_right[k] = v_node[k] - v_node[k+1] for k <= N-2; the open end at
    # k = N-1 has no link onward, so its difference is padded away.
    # Shape: [..., col_num, row_num]
    dv_to_right = F.pad(-torch.diff(v_node, dim=dim), (0, 1))

    return i_inject + (dv_to_left + dv_to_right) * segment_g


def row_wire_kcl_residual(
    v_node: Tensor,
    v_drive: Tensor,
    segment_g: float,
    i_inject: Tensor,
) -> Tensor:
    """KCL residual at every node of a row-oriented wire.

    Same structure as :func:`col_wire_kcl_residual` along ``dim=-2``.

    Args:
        v_node: Wire node voltages [V].
            Shape: ``[..., col_num, row_num]``.
        v_drive: Drive voltage [V].
            Shape: ``[..., 1, row_num]``.
        segment_g: Conductance of one lattice link [uS].
        i_inject: Cell current drawn at each node [uA].
            Shape: ``[..., col_num, row_num]``.

    Returns:
        KCL residual tensor [uA].
        Shape: ``[..., col_num, row_num]``.
    """
    dim = -2
    # Shape: [..., wire_point, row_num]
    v_with_drive = torch.cat((v_drive, v_node), dim=dim)
    # Shape: [..., wire_point, row_num] -> [..., col_num, row_num]
    dv_to_left = torch.diff(v_with_drive, dim=dim)
    # Shape: [..., col_num, row_num]
    dv_to_right = F.pad(-torch.diff(v_node, dim=dim), (0, 0, 0, 1))

    return i_inject + (dv_to_left + dv_to_right) * segment_g


def col_driver_current(
    v_node: Tensor,
    v_drive: Tensor,
    segment_g: float,
) -> Tensor:
    """Net current from a column-oriented-wire driver into the wire [uA].

    Args:
        v_node: Wire node voltages [V].
            Shape: ``[..., col_num, row_num]``.
        v_drive: Drive voltage [V].
            Shape: ``[..., col_num, 1]``.
        segment_g: Conductance of one lattice link [uS] — the driver reaches
            node 0 through exactly one of them.

    Returns:
        Drive current [uA].
        Shape: ``[..., col_num]``.
    """
    dim = -1
    # Shape: [..., col_num, 1] -> [..., col_num]
    return (v_drive.squeeze(dim) - v_node.select(dim, 0)) * segment_g


def row_driver_current(
    v_node: Tensor,
    v_drive: Tensor,
    segment_g: float,
) -> Tensor:
    """Net current from a row-oriented-wire driver into the wire [uA].

    Args:
        v_node: Wire node voltages [V].
            Shape: ``[..., col_num, row_num]``.
        v_drive: Drive voltage [V].
            Shape: ``[..., 1, row_num]``.
        segment_g: Conductance of one lattice link [uS] — the driver reaches
            node 0 through exactly one of them.

    Returns:
        Drive current [uA].
        Shape: ``[..., row_num]``.
    """
    dim = -2
    # Shape: [..., 1, row_num] -> [..., row_num]
    return (v_drive.squeeze(dim) - v_node.select(dim, 0)) * segment_g
