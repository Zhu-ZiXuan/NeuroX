"""User-facing lifecycle and module-tree functions."""

from __future__ import annotations

import torch.nn as nn

from neurox.common.module import ConfigBase, ModuleBase, PolicyBase, neurox_children

type NeuroxModule = ModuleBase[ConfigBase, PolicyBase]


def check_unique_binding(model: nn.Module) -> None:
    """Check that each NeuroX module occupies one path in `model`.

    Raises:
        ValueError: One module instance is bound at two paths.
    """
    locations: dict[NeuroxModule, str] = {}
    for relative_name, module in model.named_modules(remove_duplicate=False):
        if not isinstance(module, ModuleBase):
            continue
        if module in locations:
            raise ValueError(
                f"{type(module).__name__} is bound at both {locations[module]!r} and {relative_name!r}; "
                "one physical instance holds one location, so bind a separate instance per site"
            )
        locations[module] = relative_name


def fabricate(root: nn.Module) -> None:
    """Resample fabrication variation across every outermost NeuroX subtree."""
    if isinstance(root, ModuleBase):
        root.fabricate()
        return

    for _, module in neurox_children(root):
        module.fabricate()


def stamp_names(model: nn.Module) -> None:
    """Stamp every NeuroX module of `model` with its hierarchical name.

    A module never knows its own name: the name is a property of the tree that
    holds it, and only a walk from a root can hand it out. Stamping again
    overwrites existing names, allowing a rewired model to be renamed.

    Raises:
        ValueError: One module instance sits at two locations of `model`.
    """
    check_unique_binding(model)
    if isinstance(model, ModuleBase):
        model.stamp_names()
        return

    for qualified_name, root in neurox_children(model):
        root.stamp_names(qualified_name=qualified_name)
