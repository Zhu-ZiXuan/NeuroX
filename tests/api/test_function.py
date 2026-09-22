"""Public module-tree naming and binding checks."""

from __future__ import annotations

import pytest
import torch.nn as nn

from neurox import check_unique_binding, stamp_names
from neurox.common.module import ConfigBase, NonProfileModule, PolicyBase


class _Config(ConfigBase):
    pass


class _Policy(PolicyBase):
    pass


class _Node(NonProfileModule):
    def __init__(self, *children: nn.Module) -> None:
        super().__init__(config=_Config(), policy=_Policy(), inst_shape=())
        self.children_ = nn.ModuleList(children)


class _Owner(nn.Module):
    def __init__(self, node: nn.Module) -> None:
        super().__init__()
        self.leaf = node


def test_stamping_tracks_nested_paths_rewiring_and_a_new_root() -> None:
    inner = _Node()
    model = _Owner(_Node(inner))
    stamp_names(model)
    assert model.leaf.qualified_name == "leaf"
    assert inner.qualified_name == "leaf.children_.0"

    model.leaf = inner
    stamp_names(model)
    assert inner.qualified_name == "leaf"

    stamp_names(inner)
    assert inner.qualified_name == ""


def test_duplicate_bindings_can_be_checked_before_stamping() -> None:
    shared = _Node()

    class _Host(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = shared
            self.right = shared

    with pytest.raises(ValueError, match="bound at both"):
        check_unique_binding(_Host())
