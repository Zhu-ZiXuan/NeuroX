"""Module traversal preserves ownership boundaries, lifecycle order, and subtree isolation."""

from __future__ import annotations

import torch.nn as nn

from neurox import fabricate, set_temperature
from neurox.common.module import ConfigBase, NonProfileModule, PolicyBase, neurox_roots


class _Config(ConfigBase):
    pass


class _Policy(PolicyBase):
    pass


class _Node(NonProfileModule):
    def __init__(self, name: str, events: list[tuple[str, float]], *children: nn.Module) -> None:
        super().__init__(config=_Config(), policy=_Policy(), inst_shape=())
        self.name = name
        self.events = events
        self.children_ = nn.ModuleList(children)

    def _sample_fabrication_variation(self) -> None:
        self.events.append((self.name, self.T__K))

    def _on_temperature_changed(self) -> None:
        self.events.append((self.name, self.T__K))


def test_roots_cross_plain_containers_stop_at_neurox_nodes_and_deduplicate() -> None:
    events: list[tuple[str, float]] = []
    inner = _Node("inner", events)
    first = _Node("first", events, inner)
    second = _Node("second", events)
    model = nn.ModuleList([nn.Linear(2, 2), nn.Sequential(first), nn.ModuleList([first, nn.Sequential(second)])])
    assert neurox_roots(model) == [first, second]


def test_fabrication_visits_each_owned_child_once_in_preorder() -> None:
    events: list[tuple[str, float]] = []
    shared = _Node("shared", events)
    last = _Node("last", events)
    parent = _Node("parent", events, nn.Sequential(shared), nn.ModuleList([shared, last]))
    fabricate(nn.Sequential(parent))
    assert [name for name, _ in events] == ["parent", "shared", "last"]


def test_temperature_updates_current_children_in_order_without_changing_other_subtrees() -> None:
    events: list[tuple[str, float]] = []
    child = _Node("child", events)
    parent = _Node("parent", events, nn.Sequential(child))
    sibling = _Node("sibling", events)
    model = nn.ModuleList([nn.Sequential(parent), sibling])

    set_temperature(model, 350.0)
    assert events == [("parent", 350.0), ("child", 350.0), ("sibling", 350.0)]

    events.clear()
    late = _Node("late", events)
    parent.children_.append(nn.Sequential(late))
    parent.set_temperature(320.0)
    assert events == [("parent", 320.0), ("child", 320.0), ("late", 320.0)]
    assert sibling.T__K == 350.0
