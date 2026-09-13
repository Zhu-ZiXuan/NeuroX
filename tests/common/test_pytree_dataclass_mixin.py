"""Opt-in PyTree registration preserves inherited fields and nested structure."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from torch.utils import _pytree as pytree

from neurox.common.dataclass_mixin import PyTreeDataClassMixin, TensorDataClassMixin


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
