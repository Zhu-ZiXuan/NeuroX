"""Tests for the shared dataclass tensor-field traversal."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor

from neurox.common.tensor_dataclass_mixin import walk_paired_tensor_fields, walk_single_tensor_fields


def test_walk_single_tensor_fields_rejects_a_non_dataclass_instance() -> None:
    with pytest.raises(TypeError, match=r"walk_single_tensor_fields\(\) requires a dataclass instance"):
        walk_single_tensor_fields(torch.clone, object())


@dataclass(frozen=True)
class _Leaf:
    value: Tensor | None


@dataclass(frozen=True)
class _Tree:
    value: Tensor
    child: _Leaf
    label: str


@dataclass(frozen=True)
class _ExtendedTree(_Tree):
    extra: Tensor


def test_walk_transforms_nested_fields_without_mutating_the_input(device: torch.device) -> None:
    tree = _Tree(torch.zeros(2, 3, device=device), _Leaf(torch.ones(2, device=device)), "kept")

    def fn(tensor: Tensor) -> Tensor:
        return tensor + 1

    result = walk_single_tensor_fields(fn, tree)

    torch.testing.assert_close(result.value, torch.ones(2, 3, device=device))
    torch.testing.assert_close(result.child.value, torch.full((2,), 2.0, device=device))
    assert result.label == "kept"
    torch.testing.assert_close(tree.value, torch.zeros(2, 3, device=device))
    torch.testing.assert_close(tree.child.value, torch.ones(2, device=device))


def test_walk_combines_corresponding_fields_from_two_trees(device: torch.device) -> None:
    left = _Tree(torch.ones(2, 3, device=device), _Leaf(torch.full((2,), 2.0, device=device)), "left")
    right = _Tree(torch.full((2, 3), 3.0, device=device), _Leaf(torch.full((2,), 4.0, device=device)), "right")

    result = walk_paired_tensor_fields(torch.add, left, right)

    torch.testing.assert_close(result.value, torch.full((2, 3), 4.0, device=device))
    torch.testing.assert_close(result.child.value, torch.full((2,), 6.0, device=device))
    assert result.label == "left"


def test_walk_rejects_incompatible_parallel_structures(device: torch.device) -> None:
    left = _Tree(torch.ones(2, device=device), _Leaf(None), "left")
    right = _Tree(torch.ones(2, device=device), _Leaf(torch.ones(2, device=device)), "right")

    with pytest.raises(TypeError, match="same concrete dataclass structure"):
        walk_paired_tensor_fields(torch.add, left, right)


def test_walk_preserves_absent_tensor_fields() -> None:
    result = walk_single_tensor_fields(torch.clone, _Leaf(None))
    assert result.value is None


def test_walk_preserves_the_concrete_subclass(device: torch.device) -> None:
    tree = _ExtendedTree(torch.zeros(2, device=device), _Leaf(None), "kept", torch.ones(3, device=device))
    result = walk_single_tensor_fields(torch.neg, tree)

    assert type(result) is _ExtendedTree
    assert result.child.value is None
    assert result.label == "kept"
    torch.testing.assert_close(result.extra, -torch.ones(3, device=device))


def test_walk_callback_captures_shared_tensor_indices(device: torch.device) -> None:
    values = torch.arange(6.0, device=device).reshape(2, 3)
    tree = _Tree(values, _Leaf(values + 10), "kept")
    index = torch.tensor([2, 0], device=device)

    def fn(tensor: Tensor) -> Tensor:
        return tensor.index_select(-1, index)

    result = walk_single_tensor_fields(fn, tree)

    torch.testing.assert_close(result.value, values[:, [2, 0]])
    torch.testing.assert_close(result.child.value, (values + 10)[:, [2, 0]])
    assert result.label == "kept"


def test_fullgraph_preserves_nested_callback_results(device: torch.device) -> None:
    values = torch.arange(6.0, device=device).reshape(2, 3)
    factor = torch.tensor(2.0, device=device)

    def apply(values: Tensor, factor: Tensor) -> tuple[Tensor, Tensor | None]:
        tree = _Tree(values, _Leaf(values.sum(-1)), "kept")

        def fn(tensor: Tensor) -> Tensor:
            return tensor * factor

        result = walk_single_tensor_fields(fn, tree)
        return result.value, result.child.value

    with torch._dynamo.config.patch(disable=False):
        actual = torch.compile(apply, backend="inductor", fullgraph=True)(values, factor)

    torch.testing.assert_close(actual, (2.0 * values, 2.0 * values.sum(-1)))


def test_fullgraph_combines_two_nested_trees(device: torch.device) -> None:
    left = torch.arange(6.0, device=device).reshape(2, 3)
    right = left + 10

    def apply(left: Tensor, right: Tensor) -> tuple[Tensor, Tensor | None]:
        left_tree = _Tree(left, _Leaf(left.sum(-1)), "kept")
        right_tree = _Tree(right, _Leaf(right.sum(-1)), "ignored")
        result = walk_paired_tensor_fields(torch.add, left_tree, right_tree)
        return result.value, result.child.value

    with torch._dynamo.config.patch(disable=False):
        actual = torch.compile(apply, backend="inductor", fullgraph=True)(left, right)

    torch.testing.assert_close(actual, (left + right, left.sum(-1) + right.sum(-1)))
