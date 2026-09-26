"""Shared module runtime preparation for offline tools."""

from __future__ import annotations

import torch

from neurox.api.function import stamp_names
from neurox.common.module import ModuleBase

__all__ = ["prepare_module"]


def prepare_module[ModuleT: ModuleBase](
    module: ModuleT,
    *,
    device: torch.device,
) -> ModuleT:
    """Prepare a constructed hardware tree in place for an offline experiment.

    Move registered state to `device`, select evaluation mode, fabricate, and
    stamp names in that order. Return the same module object. Set temperature
    before calling when fabrication should use a non-default temperature.
    Fabrication consumes the caller's PyTorch RNG for enabled variation.

    Program weights afterwards: moving registered buffers does not migrate old
    programmed attributes, and this helper does not preserve or replay
    programming. Children held as unregistered external collaborators must be
    prepared by their own owner. Naming rejects duplicate NeuroX bindings in the
    assembled tree.

    Args:
        module: Constructed hardware tree to mutate and return.
        device: Device to which registered tensor sources are moved before
            fabrication.

    Returns:
        The same module object, moved, fabricated, set to evaluation mode, and
        named.
    """
    module = module.to(device)
    module.eval()
    module.fabricate()
    stamp_names(module)
    return module
