"""CIM-macro configuration and preparation for calibration tools."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from neurox.api.factory import cim_macro_from_file
from neurox.common.validate_mixin import ValidateMixin
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, IdealCimMacro
from neurox.tools.config import resolve_relative_path
from neurox.tools.module import prepare_module

__all__ = [
    "MacroSection",
    "build_ideal_twin",
    "build_physical_macro",
    "load_macro_config",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MacroSection(ValidateMixin):
    config_files: tuple[Path, ...]
    config_section: str
    policy_file: Path
    policy_section: str

    def __post_init__(self) -> None:
        self._require_non_empty(self.config_files, "[macro].config_files")


def load_macro_config(section: MacroSection, *, base: Path) -> CimMacroConfig:
    """Load the macro config that owns the physical port geometry."""
    config_paths = [resolve_relative_path(file, base) for file in section.config_files]
    return CimMacroConfig.from_file(*config_paths, section=section.config_section)


def build_physical_macro(
    section: MacroSection,
    *,
    base: Path,
    device: torch.device,
    inst_shape: tuple[int, ...],
) -> CimMacro:
    """Build the physical macro selected by a calibration run config."""
    config_paths = [resolve_relative_path(file, base) for file in section.config_files]
    policy_path = resolve_relative_path(section.policy_file, base)
    macro = cim_macro_from_file(
        config_files=config_paths,
        policy_files=(policy_path,),
        config_section=section.config_section,
        policy_section=section.policy_section,
        inst_shape=inst_shape,
        dtype=torch.float32,
    )
    macro = prepare_module(macro, device=device)
    logger.info(
        "built %s from %s [%s] + %s [%s]",
        type(macro).__name__,
        " <- ".join(str(path) for path in config_paths),
        section.config_section,
        policy_path,
        section.policy_section,
    )
    return macro


def build_ideal_twin(
    macro: CimMacro,
    *,
    device: torch.device,
) -> IdealCimMacro:
    """Build the ideal twin; weights remain independently programmed."""
    ideal = macro.to_ideal().to(device)
    ideal.eval()
    return ideal
