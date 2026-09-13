"""Tests for the shared dataclass tensor-field traversal."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
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
    tree = _Tree(torch.zeros(2, 3, device=device), _Leaf(torch.ones(2, device=device)), "kept")

    def fn(tensor: Tensor) -> Tensor:
        return tensor + 1

    result = map_single_tensor_fields(fn, tree)

    torch.testing.assert_close(result.value, torch.ones(2, 3, device=device))
    torch.testing.assert_close(result.child.value, torch.full((2,), 2.0, device=device))
    assert result.label == "kept"
    torch.testing.assert_close(tree.value, torch.zeros(2, 3, device=device))
    torch.testing.assert_close(tree.child.value, torch.ones(2, device=device))


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

    result = visit_tensor_fields(inspect, tree)
    assert result is None
    assert len(visited) == 2
    torch.testing.assert_close(visited[0], tree.value)
    torch.testing.assert_close(visited[1], tree.child.value)


@pytest.mark.parametrize("tree", [_Leaf(None), _Branch(_Leaf(None))])
def test_maps_preserve_empty_trees_and_visit_skips_them(tree: _Leaf | _Branch) -> None:
    def inspect(tensor: Tensor) -> None:
        raise AssertionError("empty trees must not invoke the callback")

    for result in (map_single_tensor_fields(torch.clone, tree), map_paired_tensor_fields(torch.add, tree, tree)):
        assert type(result) is type(tree)
        leaf = result.child if isinstance(result, _Branch) else result
        assert leaf.value is None
    assert visit_tensor_fields(inspect, tree) is None


def test_map_accepts_tensors_only_in_nested_fields(device: torch.device) -> None:
    tree = _Branch(_Leaf(torch.empty(0, device=device)))
    result = map_single_tensor_fields(torch.clone, tree)
    torch.testing.assert_close(result.child.value, tree.child.value)
    paired = map_paired_tensor_fields(torch.add, tree, tree)
    torch.testing.assert_close(paired.child.value, tree.child.value)

    def inspect(tensor: Tensor) -> None:
        assert tensor.numel() == 0

    assert visit_tensor_fields(inspect, tree) is None


def test_map_preserves_the_concrete_subclass(device: torch.device) -> None:
    tree = _ExtendedTree(torch.zeros(2, device=device), _Leaf(None), "kept", torch.ones(3, device=device))
    result = map_single_tensor_fields(torch.neg, tree)

    assert type(result) is _ExtendedTree
    assert result.child.value is None
    assert result.label == "kept"
    torch.testing.assert_close(result.extra, -torch.ones(3, device=device))
