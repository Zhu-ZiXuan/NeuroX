"""Module traversal preserves ownership boundaries, lifecycle order, and subtree isolation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from neurox import Profiler, fabricate, set_profile_leading_rank, set_temperature
from neurox.common.module import (
    ConfigBase,
    NonProfileModule,
    PolicyBase,
    ProfileModule,
    neurox_profile_children,
    neurox_profile_roots,
    neurox_roots,
)


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


class _ProfileNode(ProfileModule):
    def __init__(self) -> None:
        super().__init__(config=_Config(), policy=_Policy(), inst_shape=())

    @property
    def _area_per_inst__um2(self) -> float:
        return 0.0

    @property
    def _leakage_per_inst__uW(self) -> float:
        return 0.0

    def emit(self, energy: Tensor) -> None:
        self._record_dynamic_energy(energy)


def test_roots_cross_plain_containers_stop_at_neurox_nodes_and_deduplicate() -> None:
    events: list[tuple[str, float]] = []
    inner = _Node("inner", events)
    first = _Node("first", events, inner)
    second = _Node("second", events)
    model = nn.ModuleList([nn.Linear(2, 2), nn.Sequential(first), nn.ModuleList([first, nn.Sequential(second)])])
    assert list(neurox_roots(model)) == [("1.0", first), ("2.1.0", second)]


def test_profile_children_cross_non_profile_owners_and_stop_before_nested_profile_nodes() -> None:
    shared = _ProfileNode()
    shared.inner = _ProfileNode()
    sibling = _ProfileNode()
    parent = _ProfileNode()
    parent.bridge = _Node("bridge", [], nn.Sequential(shared))
    parent.other = nn.ModuleList([shared, nn.Sequential(sibling)])

    assert list(neurox_profile_children(parent)) == [
        ("bridge.children_.0.0", shared),
        ("other.1.0", sibling),
    ]
    assert list(neurox_profile_roots(parent)) == [("", parent)]
    container = nn.ModuleList([nn.Sequential(parent), _Node("owner", [], parent)])
    assert list(neurox_profile_roots(container)) == [("0.0", parent)]


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


@pytest.mark.parametrize(("parent_rank", "sibling_rank"), [(2, 1), (3, 0)])
def test_profile_rank_crosses_containers_and_non_profile_nodes_with_subtree_overrides(
    parent_rank: int, sibling_rank: int
) -> None:
    child = _ProfileNode()
    parent = _ProfileNode()
    parent.bridge = _Node("bridge", [], nn.Sequential(child))
    sibling = _ProfileNode()
    model = nn.ModuleDict({"parent": parent, "sibling": sibling})
    energy = torch.arange(30.0).reshape(2, 3, 5)

    set_profile_leading_rank(model, sibling_rank)
    parent.set_profile_leading_rank(parent_rank)
    profiler = Profiler(concat_dim=0)
    profiler.collect_static_data(model)
    with profiler:
        parent.emit(energy)
        child.emit(energy)
        sibling.emit(energy)
    parent_energy = energy if parent_rank == 3 else energy.sum(dim=2)
    sibling_energy = energy.sum(dim=(1, 2)) if sibling_rank else energy.sum().reshape(1)
    torch.testing.assert_close(profiler.result["parent"].dynamic_energy__fJ, parent_energy, check_dtype=False)
    torch.testing.assert_close(
        profiler.result[child.qualified_name].dynamic_energy__fJ, parent_energy, check_dtype=False
    )
    torch.testing.assert_close(profiler.result["sibling"].dynamic_energy__fJ, sibling_energy, check_dtype=False)

    late = _ProfileNode()
    parent.bridge.children_.append(late)
    set_profile_leading_rank(model, 1)
    updated = Profiler(concat_dim=0)
    updated.collect_static_data(model)
    with updated:
        parent.emit(energy)
        child.emit(energy)
        late.emit(energy)
        sibling.emit(energy)
    for item in updated.result.values():
        torch.testing.assert_close(item.dynamic_energy__fJ, energy.sum(dim=(1, 2)), check_dtype=False)
