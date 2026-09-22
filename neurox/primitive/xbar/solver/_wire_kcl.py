"""Uniform wire-ladder KCL with a driven first node and an open far end."""

from __future__ import annotations

import torch
from torch import Tensor


def f_kcl__uA(
    *,
    v_node__V: Tensor,
    v_port__V: Tensor,
    segment_g__uS: float,
    i_inject__uA: Tensor,
    dim: int,
) -> Tensor:
    """KCL residual at every node of one wire ladder.

    Args:
        v_node__V: Wire node voltages.
            Shape: `[..., node, ...]`.
        v_port__V: Array-port voltage, of extent 1 along `dim`.
            Shape: `[..., node=1, ...]`.
        segment_g__uS: Conductance of one lattice link.
        i_inject__uA: Cell current drawn at each node.
            Shape: `[..., node, ...]`.
        dim: Negative index of the axis the wire runs along.

    Returns:
        KCL residual tensor.
        Shape: `[..., node, ...]`.
    """
    node_num = v_node__V.shape[dim]
    node_index_shape = [1] * v_node__V.ndim
    node_index_shape[dim] = node_num
    node_index = torch.arange(node_num, device=v_node__V.device).reshape(node_index_shape)

    # Roll supplies interior neighbors; masking removes its periodic boundary links.
    # Shape: [..., node, ...]
    dv_to_prev__V = v_node__V - torch.roll(v_node__V, shifts=1, dims=dim)
    dv_to_prev__V = torch.where(node_index == 0, v_node__V - v_port__V, dv_to_prev__V)
    # Shape: [..., node, ...]
    dv_to_next__V = v_node__V - torch.roll(v_node__V, shifts=-1, dims=dim)
    dv_to_next__V = torch.where(node_index == node_num - 1, 0.0, dv_to_next__V)

    return i_inject__uA + (dv_to_prev__V + dv_to_next__V) * segment_g__uS


def dfkcl_dvnode__uS(
    g_cell_eff__uS: Tensor,
    *,
    segment_g__uS: float,
    dim: int,
) -> Tensor:
    """Differentiate each node's KCL residual with respect to its own voltage.

    Other node and driver voltages stay fixed. Each attached wire link adds
    its conductance to the cell-current derivative. The open far end has one
    link; all other nodes have two, including the driver link at index zero.

    Args:
        g_cell_eff__uS: Signed derivative of the cell current leaving each node
            with respect to that node's voltage.
            Shape: `[..., node, ...]`.
        segment_g__uS: Conductance of one lattice link.
        dim: Negative index of the axis the wire runs along.

    Returns:
        Diagonal entries of the node KCL Jacobian.
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


def f_kcl_roundoff__uA(
    *,
    v_node__V: Tensor,
    v_port__V: Tensor,
    segment_g__uS: float,
    dim: int,
) -> Tensor:
    """Estimate each node's wire-current uncertainty from voltage rounding.

    Only voltages attached to a node contribute to its scale. The allowance
    uses the node dtype's precision and positive segment conductance.

    Args:
        v_node__V: Wire node voltages.
            Shape: `[..., node, ...]`.
        v_port__V: Array-port voltage, broadcastable with nodes and of extent 1
            along `dim`.
            Shape: `[..., node=1, ...]`.
        dim: Negative index of the axis the wire runs along.

    Returns:
        Local wire-rounding current allowances.
        Shape: `[..., node, ...]`.
    """
    node_num = v_node__V.shape[dim]
    node_index_shape = [1] * v_node__V.ndim
    node_index_shape[dim] = node_num
    node_index = torch.arange(node_num, device=v_node__V.device).reshape(node_index_shape)

    # Replace rolled endpoints with the actual driver and open-end boundaries.
    # Shape: [..., node, ...]
    v_abs__V = v_node__V.abs()
    v_left__V = torch.where(node_index == 0, v_port__V.abs(), torch.roll(v_abs__V, shifts=1, dims=dim))
    v_right__V = torch.where(node_index == node_num - 1, 0.0, torch.roll(v_abs__V, shifts=-1, dims=dim))
    v_scale__V = torch.maximum(v_abs__V, torch.maximum(v_left__V, v_right__V))
    return (2.0 * torch.finfo(v_node__V.dtype).eps * segment_g__uS) * v_scale__V
