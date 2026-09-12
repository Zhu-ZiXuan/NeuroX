"""Opt-in PyTree registration preserves inherited fields and nested structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
import torch
from torch import Tensor
from torch.utils import _pytree as pytree

from neurox.common.module import DcopBase
from neurox.common.pytree_dataclass_mixin import PyTreeDataClassMixin
from neurox.common.recorder import RecordBase
from neurox.common.tensor_dataclass_mixin import TensorDataClassMixin


class _Value(TensorDataClassMixin, PyTreeDataClassMixin):
    value: Tensor


class _Node(_Value):
    index: int
    child: _Value | None


class _Plain(TensorDataClassMixin):
    value: Tensor


class _Dcop(DcopBase):
    value: Tensor


class _Record(RecordBase):
    value: Tensor


@pytest.mark.parametrize(
    "bases",
    [(TensorDataClassMixin, PyTreeDataClassMixin), (PyTreeDataClassMixin, TensorDataClassMixin)],
    ids=["construction-first", "registration-first"],
)
def test_mixin_order_preserves_parent_and_child_fields(bases: tuple[type, type]) -> None:
    class _Parent(*bases):
        parent: int

    class _Child(_Parent):
        child: int

    leaves, spec = pytree.tree_flatten(_Child(parent=1, child=2))
    assert leaves == [1, 2]
    restored = pytree.tree_unflatten(leaves, spec)
    assert type(restored) is _Child
    assert restored.parent == 1
    assert restored.child == 2


def test_registration_preserves_an_independent_mutable_dataclass_contract() -> None:
    class _MutableDataClassMixin:
        def __init_subclass__(cls) -> None:
            dataclass(cls)
            super().__init_subclass__()

    class _Metadata(PyTreeDataClassMixin, _MutableDataClassMixin):
        name: str
        count: int = 0

    node = _Metadata(name="sample")
    node.count = 2
    leaves, spec = pytree.tree_flatten(node)
    assert leaves == ["sample", 2]
    assert pytree.tree_unflatten(leaves, spec) == node


@pytest.mark.parametrize("nested", [False, True])
def test_inherited_fields_and_optional_children_round_trip(device: torch.device, nested: bool) -> None:
    value = torch.arange(3, device=device, dtype=torch.float64)
    child = _Value(value=value + 1) if nested else None
    node = _Node(value=value, index=2, child=child)

    leaves, spec = pytree.tree_flatten(node)
    assert len(leaves) == (3 if nested else 2)
    assert leaves[0] is value
    assert leaves[1] == 2
    if child is not None:
        assert leaves[2] is child.value

    restored = pytree.tree_unflatten(leaves, spec)
    assert type(restored) is _Node
    assert restored.value is value
    assert restored.index == node.index
    if child is not None:
        assert type(restored.child) is _Value
        assert restored.child.value is child.value
    else:
        assert restored.child is None


@pytest.mark.parametrize("node_type", [_Plain, _Dcop, _Record])
def test_other_dataclass_families_remain_opaque(
    device: torch.device, node_type: type[_Plain | _Dcop | _Record]
) -> None:
    node = node_type(value=torch.ones(2, device=device))
    leaves, spec = pytree.tree_flatten(node)

    assert len(leaves) == 1
    assert leaves[0] is node
    assert pytree.tree_unflatten(leaves, spec) is node


def test_registered_dataclass_transform_is_fullgraph_safe(device: torch.device) -> None:
    value = torch.arange(3, device=device, dtype=torch.float64)
    node = _Node(value=value, index=2, child=_Value(value=value * 2))

    def increment(value: Tensor | int) -> Tensor | int:
        return value + 1

    def transform(node: _Node) -> _Node:
        return cast(_Node, pytree.tree_map(increment, node))

    expected = transform(node)
    with torch._dynamo.config.patch(disable=False):
        actual = torch.compile(transform, backend="inductor", fullgraph=True)(node)

    actual_leaves, actual_spec = pytree.tree_flatten(actual)
    expected_leaves, expected_spec = pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    torch.testing.assert_close(actual_leaves, expected_leaves)
    torch.testing.assert_close(node.value, value)
