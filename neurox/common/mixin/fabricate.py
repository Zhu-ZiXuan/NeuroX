"""Auto-cascading ``fabricate()`` for static manufacturing-variation sampling.

See also:
    docs/internals/common/mixin/fabricate.md
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator

import torch.nn as nn


class FabricateMixin:
    """Grant a host an automatic pre-order ``fabricate()`` cascade.

    A host inherits this to get static manufacturing-variation sampling for
    free: the inherited ``fabricate()`` resamples the host's own static state,
    then recurses into every ``FabricateMixin`` descendant in one pre-order
    pass. A subclass implements only its own per-layer sampling step. The mixin
    samples static mismatch only; it never writes a programmed weight.

    Host requirements:
        - Inherit ``nn.Module`` alongside this mixin; the cascade walks
          ``self.children()``.
        - Implement ``_sample_fabricate_mismatch`` to resample this node's own
          static mismatch; a container that owns no static state implements it
          as an explicit no-op.
        - Assign ``self._inst_shape: tuple[int, ...]`` in ``__init__``,
          encoding the per-instance multiplicity at this layer. The mixin
          reads it but never assigns it.
        - Hold a fabricable submodule as a registered child — directly or
          inside an ``nn.ModuleList`` / ``nn.ModuleDict``. One kept in a plain
          attribute falls outside ``self.children()`` and is never reached.
    """

    _inst_shape: tuple[int, ...]

    def fabricate(self) -> None:
        """Re-sample static manufacturing variation across self and descendants.

        Runs pre-order — self first via ``_sample_fabricate_mismatch``, then
        each fabricable child. Children in ``nn.ModuleList`` / ``nn.ModuleDict``
        are reached transparently; non-``FabricateMixin`` children are skipped.
        """
        self._sample_fabricate_mismatch()
        for child in self._fabricable_children():
            child.fabricate()

    @abstractmethod
    def _sample_fabricate_mismatch(self) -> None:
        """Resample this node's own static state."""
        raise NotImplementedError

    def _fabricable_children(self) -> Iterator[FabricateMixin]:
        """Iterate direct ``FabricateMixin`` children.

        Returns:
            Each direct child that is a :class:`FabricateMixin`;
            ``nn.ModuleList`` / ``nn.ModuleDict`` containers are
            transparently expanded so their members yield directly.
        """
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
