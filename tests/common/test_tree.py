"""Module-root discovery and subtree naming."""

from __future__ import annotations

import torch.nn as nn

from neurox.common.module import ConfigBase, ModuleBase, PolicyBase, neurox_roots


class _Config(ConfigBase):
    pass


class _Policy(PolicyBase):
    pass


class _Node(ModuleBase):
    """NeuroX module that may hold NeuroX children."""

    def __init__(self, *children: nn.Module) -> None:
        super().__init__(config=_Config(), policy=_Policy(), inst_shape=())
        self.children_ = nn.ModuleList(children)


def test_a_neurox_module_is_its_own_root() -> None:
    node = _Node()
    assert neurox_roots(node) == [node]


def test_descent_stops_at_the_first_neurox_module() -> None:
    inner = _Node()
    outer = _Node(inner)
    assert neurox_roots(nn.Sequential(outer)) == [outer]


def test_a_plain_container_may_hold_several_roots() -> None:
    first = _Node()
    second = _Node()

    class _Host(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(2, 2)
            self.first = first
            self.block = nn.Sequential(nn.ReLU(), second)

    assert neurox_roots(_Host()) == [first, second]


def test_a_module_bound_under_two_parents_is_one_root() -> None:
    shared = _Node()

    class _Host(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = nn.Sequential(shared)
            self.right = nn.Sequential(shared)

    assert neurox_roots(_Host()) == [shared]


def test_a_tree_without_neurox_modules_has_no_root() -> None:
    assert neurox_roots(nn.Sequential(nn.Linear(2, 2), nn.ReLU())) == []


def test_a_neurox_module_stamps_its_own_subtree() -> None:
    inner = _Node()
    outer = _Node(nn.Sequential(inner))
    outer.stamp_names()
    assert outer.qualified_name == ""
    assert inner.qualified_name == "children_.0.0"
