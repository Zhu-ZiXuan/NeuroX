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
    """Load the hardware configuration for inspecting macro geometry.

    `base` is the run configuration file. Relative paths in
    `section.config_files` resolve against its parent, and earlier selected
    sections take precedence. This function validates configuration but
    constructs no hardware and performs no device placement, fabrication, or
    programming.

    Args:
        section: Configuration paths and table names for the selected macro.
        base: Run configuration file whose parent resolves relative paths.

    Returns:
        Validated macro configuration without a constructed hardware instance.
    """
    config_paths = [resolve_relative_path(file, base) for file in section.config_files]
    return CimMacroConfig.from_file(*config_paths, section=section.config_section)


def build_physical_macro(
    section: MacroSection,
    *,
    base: Path,
    device: torch.device,
    inst_shape: tuple[int, ...],
) -> CimMacro:
    """Construct and prepare the configured macro without programming weights.

    Resolve config and policy paths against `base.parent`, retaining absolute
    paths. Load sections in the declared order, construct in float32, then move
    to `device`, select evaluation mode, fabricate, and stamp names. `base` is
    the run configuration file, not its directory. The caller supplies the
    subsequent programming and input stimuli; enabled fabrication uses the
    current RNG state.

    Args:
        section: Hardware and policy file bindings for the run.
        base: Run configuration file whose parent resolves relative paths.
        device: Destination device before fabrication.
        inst_shape: Physical instance axes; extents must be positive. Singleton
            axes may reserve broadcast positions.

    Returns:
        A float32 macro prepared on the requested device, without programmed
        weights.
    """
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
    """Create an unprogrammed ideal counterpart on the requested device.

    The macro's `to_ideal` supplies geometry, value domains, mode scales,
    temperature, and local static costs. This helper selects evaluation mode and
    device placement; it does not copy weights, profile data, or physical
    children. Program the twin with the same logical weights as the physical
    experiment before comparing it.

    Args:
        macro: Physical macro supplying logical geometry and calibration.
        device: Device for the new unprogrammed ideal macro.

    Returns:
        A new ideal macro on the requested device in evaluation mode, without
        programmed weights.
    """
    ideal = macro.to_ideal().to(device)
    ideal.eval()
    return ideal
