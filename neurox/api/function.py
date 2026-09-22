"""User-facing lifecycle and module-tree functions."""

from __future__ import annotations

import torch.nn as nn

from neurox.common.module import (
    ModuleBase,
    neurox_named_modules,
    neurox_profile_roots,
    neurox_roots,
)


def check_unique_binding(model: nn.Module) -> None:
    """Check that each NeuroX module occupies one path in `model`.

    Raises:
        ValueError: One module instance is bound at two paths.
    """
    locations: dict[ModuleBase, str] = {}
    for relative_name, module in neurox_named_modules(model):
        if module in locations:
            raise ValueError(
                f"{type(module).__name__} is bound at both {locations[module]!r} and {relative_name!r}; "
                "one physical instance holds one location, so bind a separate instance per site"
            )
        locations[module] = relative_name


def fabricate(root: nn.Module) -> None:
    """Resample fabrication variation across every outermost NeuroX subtree."""
    for _, module in neurox_roots(root):
        module.fabricate()


def set_temperature(model: nn.Module, T__K: float) -> None:
    """Set temperature across the NeuroX subtrees of a PyTorch model.

    Each subtree uses `ModuleBase.set_temperature`; plain containers are
    traversed without acquiring temperature state. Call between executions.
    Fabrication and programming remain explicitly triggered by the caller.

    Raises:
        ValueError: A subtree rejects the temperature or a derived parameter.
    """
    for _, module in neurox_roots(model):
        module.set_temperature(T__K)


def set_profile_leading_rank(model: nn.Module, leading_rank: int) -> None:
    """Set retained observation axes across the profile subtrees of a model.

    Each subtree uses `ProfileModule.set_profile_leading_rank`. Plain containers
    and non-profile owners are traversed without acquiring profiling state.
    Call before profiling and keep the layout fixed while accumulating data.
    """
    if leading_rank < 0:
        raise ValueError("profile leading rank must be non-negative")
    for _, module in neurox_profile_roots(model):
        module.set_profile_leading_rank(leading_rank)


def stamp_names(model: nn.Module) -> None:
    """Stamp every NeuroX module of `model` with its hierarchical name.

    Stamping again overwrites existing names after the model is rewired.

    Raises:
        ValueError: One module instance sits at two locations of `model`.
    """
    check_unique_binding(model)
    for qualified_name, root in neurox_roots(model):
        root.stamp_names(qualified_name=qualified_name)
