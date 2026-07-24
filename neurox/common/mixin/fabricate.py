"""Auto-cascading ``fabricate()`` for static manufacturing-variation sampling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import torch.nn as nn


class FabricateMixin(ABC):
    """Add fabrication traversal to an ``nn.Module``.

    Host requirements:
        - Also inherit :class:`torch.nn.Module`.
        - Implement :meth:`_sample_fabricate_mismatch` for local static state.
        - Register fabricable children as ``nn.Module`` children.
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
            if isinstance(child, FabricateMixin):
                yield child
            elif isinstance(child, nn.ModuleList):
                for sub in child:
                    if isinstance(sub, FabricateMixin):
                        yield sub
            elif isinstance(child, nn.ModuleDict):
                for sub in child.values():
                    if isinstance(sub, FabricateMixin):
                        yield sub
