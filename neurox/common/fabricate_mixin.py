"""Auto-cascading `fabricate()` for static manufacturing-variation sampling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import torch.nn as nn


class FabricateMixin(ABC):
    """Add fabrication traversal to an `nn.Module`.

    A host must also inherit `torch.nn.Module` and register its fabricable
    children as `nn.Module` children, which is where the traversal looks.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if not issubclass(cls, nn.Module):
            raise TypeError(f"{cls.__qualname__} must also inherit torch.nn.Module")

    def fabricate(self) -> None:
        """Resample static variation on this module and its descendants.

        Traversal is pre-order: the host hook runs once before each direct
        fabricable child recursively receives the same call.
        """
        self._sample_fabricate_mismatch()
        for child in self._fabricable_children():
            child.fabricate()

    @abstractmethod
    def _sample_fabricate_mismatch(self) -> None:
        """Resample this node's own static state."""
        raise NotImplementedError

    def _fabricable_children(self) -> Iterator[FabricateMixin]:
        """Iterate direct fabricable children."""
        assert isinstance(self, nn.Module)
        for child in self.children():
            yield from self._walk_standard_container(child)

    @classmethod
    def _walk_standard_container(cls, module: nn.Module) -> Iterator[FabricateMixin]:
        """Yield fabricable nodes through PyTorch standard containers."""
        if isinstance(module, FabricateMixin):
            yield module
            return
        if isinstance(module, nn.ModuleList | nn.ModuleDict | nn.Sequential):
            for child in module.children():
                yield from cls._walk_standard_container(child)
