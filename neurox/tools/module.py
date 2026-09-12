"""Shared module runtime preparation for offline tools."""

from __future__ import annotations

import torch

from neurox.api.function import stamp_names
from neurox.common.module import ConfigBase, ModuleBase, PolicyBase

__all__ = ["prepare_module"]


def prepare_module[ModuleT: ModuleBase[ConfigBase, PolicyBase]](
    module: ModuleT,
    *,
    device: torch.device,
) -> ModuleT:
    """Move a standalone tree, select evaluation mode, fabricate, and stamp names.

    Call on a constructed tree before programming. Fabrication uses the
    caller's torch RNG state for the enabled manufacturing variations.
    """
    module = module.to(device)
    module.eval()
    module.fabricate()
    stamp_names(module)
    return module
