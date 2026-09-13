"""Standard physical-macro construction for validation campaigns."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from neurox.api.module_from_file import cim_macro_from_file
from neurox.common.module import DEFAULT_T__K
from neurox.primitive.macro.cim import CimMacro
from neurox.tools.module import prepare_module

logger = logging.getLogger(__name__)


def build_macro(
    config_path: Path,
    policy_path: Path,
    *,
    inst_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    T__K: float = DEFAULT_T__K,
) -> CimMacro:
    """Load, build, fabricate, and name one validation macro."""
    macro = cim_macro_from_file(
        config_files=(config_path,),
        policy_files=(policy_path,),
        config_section="cim_macro",
        policy_section="policy",
        inst_shape=inst_shape,
        dtype=dtype,
    )
    macro.set_temperature(T__K)
    macro = prepare_module(macro, device=device)
    logger.info("built %s from %s + %s", type(macro).__name__, config_path, policy_path)
    return macro
