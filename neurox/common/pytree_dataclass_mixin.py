"""Automatic PyTree registration for dataclass hosts."""

from __future__ import annotations

import torch


class PyTreeDataClassMixin:
    """Register dataclass hosts as PyTrees during cooperative class initialization.

    The host's dataclass transformation must complete before registration.
    Class-initialization hooks must delegate through `super`; a class decorator
    runs after these hooks and cannot supply the required transformation.

    Registration happens at class definition, before instances enter traced
    code. PyTorch exposes the constructor fields as children and records which
    optional fields are None in the tree structure. Nested dataclass types must
    also be registered to expose their fields to PyTree traversal.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        torch.export.register_dataclass(cls)
