"""Tests for what a module tree tells about the NeuroX modules it holds: names and roots.

Both answers are the tree's alone. A module never knows its own name, so
``stamp_names`` walks the assembled model once and hands every profile-capable
module the name that walk produced — the name a record later carries. The root
walk stops descending at the first NeuroX module, so a root covers its own
NeuroX children instead of listing them beside it. Placement is read from the
tree rather than declared at construction: the same module is named for where it
is bound, and is a root or an inner node depending on where it is bound.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neurox import stamp_names
from neurox.common import ConfigBase, ModuleBase, PolicyBase, neurox_roots


class _Config(ConfigBase):
    pass


class _Policy(PolicyBase):
    pass


class _Node(ModuleBase[_Config, _Policy]):
    """NeuroX module that may hold NeuroX children."""

    def __init__(self, *children: nn.Module) -> None:
        super().__init__(config=_Config(), policy=_Policy(), inst_shape=())
        self.children_ = nn.ModuleList(children)

    def _sample_fabricate_mismatch(self) -> None:
        pass

    def run(self) -> None:
        """The emitting shape of a forward: bill one operation, profiled or not."""
        self._record_dynamic_energy(torch.tensor(1.0))


class _Owner(nn.Module):
    """A plain container binding one NeuroX child under a role name."""

    def __init__(self, node: nn.Module) -> None:
        super().__init__()
        self.leaf = node


# === Roots ===


def test_a_neurox_module_is_its_own_root() -> None:
    node = _Node()
    assert neurox_roots(node) == [node]


def test_descent_stops_at_the_first_neurox_module() -> None:
    inner = _Node()
    outer = _Node(inner)
    assert neurox_roots(nn.Sequential(outer)) == [outer]


def test_a_plain_container_may_hold_several_roots() -> None:
    first, second = _Node(), _Node()

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


# === Stamps ===


def test_a_stamp_is_the_walks_own_name() -> None:
    """The name is the attribute path the model's traversal composes, verbatim."""
    inner = _Node()
    model = _Owner(_Node(inner))
    stamp_names(model)
    assert model.leaf.qualified_name == "leaf"
    assert inner.qualified_name == "leaf.children_.0"


def test_the_stamped_model_names_itself_with_the_empty_string() -> None:
    """``named_modules`` names a root ``""``; the stamp keeps that name as it is."""
    node = _Node()
    stamp_names(node)
    assert node.qualified_name == ""


def test_stamping_against_another_root_overwrites_the_earlier_name() -> None:
    """A stamp holds one tree's answer, so the last walk to name a module wins."""
    node = _Node()
    stamp_names(_Owner(node))
    assert node.qualified_name == "leaf"
    stamp_names(node)
    assert node.qualified_name == ""


def test_a_restamp_renames_a_model_rewired_after_it_was_named() -> None:
    """The probe-install shape: swap a child, walk again, and names follow the tree."""
    model = _Owner(_Node())
    stamp_names(model)
    probe = _Node()
    model.leaf = probe
    stamp_names(model)
    assert probe.qualified_name == "leaf"


def test_one_instance_at_two_locations_is_an_error() -> None:
    """A physical module sits in one place, so a shared instance is refused, not picked from."""
    shared = _Node()

    class _Host(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = shared
            self.right = shared

    with pytest.raises(ValueError, match="bound at both"):
        stamp_names(_Host())


def test_an_unstamped_module_refuses_to_name_itself() -> None:
    """No tree has named it yet, and a module never invents a name of its own."""
    node = _Node()
    with pytest.raises(RuntimeError, match="carries no name stamp") as error:
        _ = node.qualified_name
    assert "stamp_names" in str(error.value)


def test_a_plain_forward_needs_no_stamp() -> None:
    """Naming is for collecting and reporting energy; running the model is neither."""
    _Node().run()  # must not raise
