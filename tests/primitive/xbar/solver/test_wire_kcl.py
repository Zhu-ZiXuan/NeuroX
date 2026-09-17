"""Wire-ladder KCL equations, axis handling, and roundoff estimates."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from neurox.primitive.xbar.solver._wire_kcl import f_kcl__uA, f_kcl_roundoff__uA


def _dense_kcl__uA(
    v_node__V: Tensor,
    v_port__V: Tensor,
    segment_g__uS: float,
    i_inject__uA: Tensor,
    *,
    dim: int,
) -> Tensor:
    node_num = v_node__V.shape[dim]
    conductance = v_node__V.new_zeros(node_num, node_num)
    conductance.diagonal().fill_(2.0 * segment_g__uS)
    conductance[-1, -1] = segment_g__uS
    if node_num > 1:
        index = torch.arange(node_num - 1)
        conductance[index, index + 1] = -segment_g__uS
        conductance[index + 1, index] = -segment_g__uS
    moved_node__V = v_node__V.movedim(dim, -1)
    moved_inject__uA = i_inject__uA.movedim(dim, -1)
    moved_port__V = v_port__V.movedim(dim, -1)
    residual__uA = moved_inject__uA + moved_node__V @ conductance.mT
    residual__uA[..., 0] = residual__uA[..., 0] - moved_port__V[..., 0] * segment_g__uS
    return residual__uA.movedim(-1, dim)


@pytest.mark.parametrize(("shape", "dim"), [((2, 4), -1), ((2, 4, 3), -2), ((2, 1, 3), -2)])
def test_kcl_matches_dense_ladder_for_any_negative_wire_axis(shape: tuple[int, ...], dim: int) -> None:
    generator = torch.Generator().manual_seed(7)
    v_node__V = torch.randn(shape, generator=generator, dtype=torch.float64)
    port_shape = list(shape)
    port_shape[dim] = 1
    v_port__V = torch.randn(port_shape, generator=generator, dtype=torch.float64)
    i_inject__uA = torch.randn(shape, generator=generator, dtype=torch.float64)
    segment_g__uS = 2.5

    actual = f_kcl__uA(v_node__V, v_port__V, segment_g__uS, i_inject__uA, dim=dim)
    expected = _dense_kcl__uA(v_node__V, v_port__V, segment_g__uS, i_inject__uA, dim=dim)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("shape", "dim"), [((2, 5), -1), ((2, 5, 3), -2), ((2, 1, 3), -2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_roundoff_covers_voltage_representation_errors(
    device: torch.device, shape: tuple[int, ...], dim: int, dtype: torch.dtype
) -> None:
    generator = torch.Generator(device=device).manual_seed(7)
    v_node = torch.randn(shape, generator=generator, dtype=dtype, device=device)
    port_shape = list(shape)
    port_shape[dim] = 1
    v_port = torch.randn(port_shape, generator=generator, dtype=dtype, device=device)
    g = 2.5
    allowance = f_kcl_roundoff__uA(v_node, v_port, g, dim=dim)

    # The linear wire operator propagates independent half-ulp voltage errors.
    # An alternating sign pattern exercises reinforcement at interior nodes.
    node_shape = [1] * len(shape)
    node_shape[dim] = shape[dim]
    signs = (2 * (torch.arange(shape[dim], device=device) % 2) - 1).reshape(node_shape)
    dv_node = 0.5 * torch.finfo(dtype).eps * v_node.abs() * signs
    dv_port = 0.5 * torch.finfo(dtype).eps * v_port.abs()
    error = _dense_kcl__uA(dv_node, dv_port, g, torch.zeros_like(v_node), dim=dim)
    assert (error.abs() <= allowance).all()
    torch.testing.assert_close(f_kcl_roundoff__uA(-2 * v_node, -2 * v_port, 4 * g, dim=dim), 8 * allowance)


def test_roundoff_uses_only_attached_neighbors_and_driver(device: torch.device) -> None:
    v_node = torch.ones(1, 5, dtype=torch.float64, device=device)
    # Two driver states broadcast over the same node grid.
    v_port = v_node.new_tensor([[1.0], [4.0]])
    baseline = f_kcl_roundoff__uA(v_node, v_port, 2.5, dim=-1)
    torch.testing.assert_close(baseline[1, 1:], baseline[0, 1:])
    assert baseline[1, 0] > baseline[0, 0]

    for changed_node in range(5):
        perturbed = v_node.clone()
        perturbed[..., changed_node] = 8.0
        actual = f_kcl_roundoff__uA(perturbed, v_port, 2.5, dim=-1)
        affected = (torch.arange(5, device=device) - changed_node).abs() <= 1
        assert torch.equal(actual > baseline, affected.expand_as(actual))
