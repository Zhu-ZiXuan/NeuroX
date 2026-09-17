"""Tests for the shared dataclass tensor-field traversal."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common.dataclass_mixin import map_paired_tensor_fields, map_single_tensor_fields, visit_tensor_fields


@dataclass(frozen=True)
class _Leaf:
    value: Tensor | None


@dataclass(frozen=True)
class _Branch:
    child: _Leaf


@dataclass(frozen=True)
class _Tree:
    value: Tensor
    child: _Leaf
    label: str


@dataclass(frozen=True)
class _ExtendedTree(_Tree):
    extra: Tensor


def test_map_transforms_nested_fields_without_mutating_the_input(device: torch.device) -> None:
    tree = _ExtendedTree(
        torch.zeros(2, 3, device=device),
        _Leaf(torch.ones(2, device=device)),
        "kept",
        torch.full((3,), 2.0, device=device),
    )

    def fn(tensor: Tensor) -> Tensor:
        return tensor + 1

    result = map_single_tensor_fields(fn, tree)

    torch.testing.assert_close(result.value, torch.ones(2, 3, device=device))
    torch.testing.assert_close(result.child.value, torch.full((2,), 2.0, device=device))
    torch.testing.assert_close(result.extra, torch.full((3,), 3.0, device=device))
    assert result.label == "kept"
    torch.testing.assert_close(tree.value, torch.zeros(2, 3, device=device))
    torch.testing.assert_close(tree.child.value, torch.ones(2, device=device))
    torch.testing.assert_close(tree.extra, torch.full((3,), 2.0, device=device))


def test_map_combines_corresponding_fields_from_two_trees(device: torch.device) -> None:
    left = _Tree(torch.ones(2, 3, device=device), _Leaf(torch.full((2,), 2.0, device=device)), "left")
    right = _Tree(torch.full((2, 3), 3.0, device=device), _Leaf(torch.full((2,), 4.0, device=device)), "right")

    result = map_paired_tensor_fields(torch.add, left, right)

    torch.testing.assert_close(result.value, torch.full((2, 3), 4.0, device=device))
    torch.testing.assert_close(result.child.value, torch.full((2,), 6.0, device=device))
    assert result.label == "left"


def test_single_inspection_visits_nested_fields_without_reconstruction(device: torch.device) -> None:
    tree = _Tree(torch.ones(2, 3, device=device), _Leaf(torch.ones(2, device=device)), "kept")
    visited: list[Tensor] = []

    def inspect(tensor: Tensor) -> None:
        visited.append(tensor)

    visit_tensor_fields(inspect, tree)
    assert len(visited) == 2
    torch.testing.assert_close(visited[0], tree.value)
    torch.testing.assert_close(visited[1], tree.child.value)


def test_map_accepts_tensors_only_in_nested_fields(device: torch.device) -> None:
    tree = _Branch(_Leaf(torch.tensor([1.0, 2.0], device=device)))
    result = map_single_tensor_fields(torch.neg, tree)
    torch.testing.assert_close(result.child.value, torch.tensor([-1.0, -2.0], device=device))
    paired = map_paired_tensor_fields(torch.add, tree, tree)
    torch.testing.assert_close(paired.child.value, torch.tensor([2.0, 4.0], device=device))
    visited: list[Tensor] = []
    visit_tensor_fields(visited.append, tree)
    assert len(visited) == 1
    torch.testing.assert_close(visited[0], tree.child.value)
