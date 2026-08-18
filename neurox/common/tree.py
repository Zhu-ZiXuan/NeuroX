"""Names and roots a module tree gives the NeuroX modules it holds."""

from __future__ import annotations

import torch.nn as nn

from .module import ConfigBase, ModuleBase, PolicyBase
from .profile_mixin import ProfileMixin


def stamp_names(model: nn.Module) -> None:
    """Stamp every profile-capable module of `model` with its hierarchical name.

    A module never knows its own name: the name is a property of the tree that
    holds it, and only a walk from a root can hand it out. This pass performs
    that walk once, after the model is assembled. Stamping again — a second
    call, on the same root or another one — overwrites, which is how a rewired
    model is renamed.

    Args:
        model: Tree to name against; its own name is the empty string, as
            `nn.Module.named_modules` names it.

    Raises:
        ValueError: One module instance sits at two locations of `model`. A
            physical module has one place in the hardware, so a tied or shared
            instance must be one instance per site.
    """
    stamped: dict[ProfileMixin, str] = {}
    for name, module in model.named_modules(remove_duplicate=False):
        if not isinstance(module, ProfileMixin):
            continue
        if module in stamped:
            raise ValueError(
                f"{type(module).__name__} is bound at both {stamped[module]!r} and {name!r}; "
                "one physical instance holds one location, so bind a separate instance per site"
            )
        stamped[module] = name
        module.qualified_name = name


def neurox_roots(model: nn.Module) -> list[ModuleBase[ConfigBase, PolicyBase]]:
    """Collect the outermost NeuroX modules `model` holds.

    The walk stops descending at the first `ModuleBase` it meets, so a root
    covers its own NeuroX children instead of listing them beside it. A `model`
    that is itself a NeuroX module is the single root; a plain container or a
    third-party wrapper may hold several. Roots are deduplicated by identity, so
    a module bound under two parents — a tied or shared layer — is reported
    once, at its first appearance.

    Returns:
        The outermost NeuroX modules, in child order.
    """
    if isinstance(model, ModuleBase):
        return [model]
    # dict keys preserve child order; nn.Module hashes by identity.
    roots: dict[ModuleBase[ConfigBase, PolicyBase], None] = {}
    for child in model.children():
        for root in neurox_roots(child):
            roots.setdefault(root, None)
    return list(roots)
