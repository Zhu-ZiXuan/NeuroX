"""Wire-ladder KCL residuals, node self-conductances, and driver currents.

Every wire is a UNIFORM ladder: one lattice link joins each pair of adjacent
nodes and the same link joins the driver to the node at index 0, so a whole
rail is described by one scalar conductance. The only distinguished node is
the ladder's open end at the far index, which has no link onward.

The wire axis is a parameter rather than a per-orientation function: `dim` is
the NEGATIVE index of the axis the ladder runs along, which is what makes the
trailing-pad width exact. Shapes below are written for a generic wire axis;
the present call sites pass `dim=-1` over cell grids at `[..., col, row]`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def f_kcl__uA(
    v_node__V: Tensor,
    v_drive__V: Tensor,
    segment_g__uS: float,
    i_inject__uA: Tensor,
    *,
    dim: int,
) -> Tensor:
    """KCL residual at every node of one wire ladder.

    Args:
        v_node__V: Wire node voltages.
            Shape: `[..., node, ...]`.
        v_drive__V: Drive voltage, of extent 1 along `dim`.
            Shape: `[..., node=1, ...]`.
        segment_g__uS: Conductance of one lattice link.
        i_inject__uA: Cell current drawn at each node.
            Shape: `[..., node, ...]`.
        dim: Negative index of the axis the wire runs along.

    Returns:
        KCL residual tensor.
        Shape: `[..., node, ...]`.
    """
    # `F.pad` reads its argument from the last axis backwards, two entries per
    # axis: a negative `dim` sits behind exactly `-dim - 1` untouched axes and
    # takes the single trailing pad itself.
    pad = (0, 0) * (-dim - 1) + (0, 1)
    # dv_to_left__V[k] = v_node__V[k] - v_node__V[k-1] for k >= 1;
    # v_node__V[0] - v_drive__V at k = 0.
    # Shape: [..., point, ...]
    v_with_drive__V = torch.cat((v_drive__V, v_node__V), dim=dim)
    # Shape: [..., point, ...] -> [..., node, ...]
    dv_to_left__V = torch.diff(v_with_drive__V, dim=dim)
    # dv_to_right__V[k] = v_node__V[k] - v_node__V[k+1] for k <= N-2; the open
    # end at k = N-1 has no link onward, so its difference is padded away.
    # Shape: [..., node, ...]
    dv_to_right__V = F.pad(-torch.diff(v_node__V, dim=dim), pad)

    return i_inject__uA + (dv_to_left__V + dv_to_right__V) * segment_g__uS


def g_self__uS(
    g_cell_eff__uS: Tensor,
    segment_g__uS: float,
    *,
    dim: int,
) -> Tensor:
    """Self-conductance of every node of one wire ladder.

    A node's own conductance is its cell branch plus every rail link attached
    to it. An interior node has two — the one back towards the driver and the
    one onward — while the far index is the ladder's open end and has only the
    first. Index 0 is NOT distinguished: its driver-side link is one standard
    lattice pitch like any other.

    Args:
        g_cell_eff__uS: Per-node cell branch derivative.
            Shape: `[..., node, ...]`.
        segment_g__uS: Conductance of one lattice link.
        dim: Negative index of the axis the wire runs along.

    Returns:
        Per-node self-conductance.
        Shape: `[..., node, ...]`.
    """
    node_num = g_cell_eff__uS.shape[dim]
    # Shape: [..., node, ...]
    return torch.cat(
        (
            torch.narrow(g_cell_eff__uS, dim, 0, node_num - 1) + 2.0 * segment_g__uS,
            torch.narrow(g_cell_eff__uS, dim, node_num - 1, 1) + segment_g__uS,
        ),
        dim=dim,
    )


def i_drive__uA(
    v_node__V: Tensor,
    v_drive__V: Tensor,
    segment_g__uS: float,
    *,
    dim: int,
) -> Tensor:
    """Net current from one wire ladder's driver into the wire.

    Args:
        v_node__V: Wire node voltages.
            Shape: `[..., node, ...]`.
        v_drive__V: Drive voltage, of extent 1 along `dim`.
            Shape: `[..., node=1, ...]`.
        segment_g__uS: Conductance of one lattice link — the driver reaches
            node 0 through exactly one of them.
        dim: Negative index of the axis the wire runs along.

    Returns:
        Drive current, the wire axis dropped.
        Shape: `[...]`.
    """
    # Shape: [..., node=1, ...] -> [...]
    return (v_drive__V.squeeze(dim) - v_node__V.select(dim, 0)) * segment_g__uS
