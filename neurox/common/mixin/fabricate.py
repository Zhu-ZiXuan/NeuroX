"""Auto-cascading `fabricate()` for static manufacturing variation sampling.

See also:
    docs/dev/modules/common/fabricate.md
    docs/dev/architecture/fabrication_lifecycle.md
"""

from __future__ import annotations

from collections.abc import Iterator

import torch.nn as nn


class FabricateMixin:
    """Mixin granting automatic pre-order `fabricate()` cascade.

    Host must also inherit `nn.Module` (for `self.children()`). Host's
    `__init__` assigns `self._inst_shape: tuple[int, ...]`, encoding the
    per-instance multiplicity at this layer. Subclasses override
    `_sample_fabricate_mismatch` to (re)sample their own static manufacturing
    variation; the default is a no-op for cascade-only nodes.

    `fabricate()` runs pre-order: self first, then each `FabricateMixin`
    child. `nn.ModuleList` / `nn.ModuleDict` containers are transparently
    expanded; non-FabricateMixin children are skipped.
    """

    _inst_shape: tuple[int, ...]

    def fabricate(self) -> None:
        """Re-sample static manufacturing variation across self and descendants."""
        self._sample_fabricate_mismatch()
        for child in self._fabricable_children():
            child.fabricate()

    def _sample_fabricate_mismatch(self) -> None:
        """Override to resample own static state. Default no-op for cascade-only nodes."""
        return

    def _fabricable_children(self) -> Iterator[FabricateMixin]:
        """Iterate direct `FabricateMixin` children, expanding `ModuleList` / `ModuleDict`."""
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
