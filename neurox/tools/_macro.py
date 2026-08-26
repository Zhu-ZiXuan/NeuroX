"""Shared macro construction and legal-input layout helpers for offline tools."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from neurox.common import ValidateMixin
from neurox.primitive import T_ROOM__K
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy, IdealCimMacro

from ._config import resolve_relative_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MacroSection(ValidateMixin):
    input_num: int
    output_num: int
    config_files: tuple[Path, ...]
    config_section: str
    policy_file: Path
    policy_section: str

    def __post_init__(self) -> None:
        self._require_pos(self.input_num, "[macro].input_num")
        self._require_pos(self.output_num, "[macro].output_num")
        self._require_non_empty(self.config_files, "[macro].config_files")


def build_physical_macro(
    section: MacroSection,
    *,
    base: Path,
    device: torch.device,
    inst_shape: tuple[int, ...],
) -> CimMacro[CimMacroConfig, CimMacroPolicy]:
    """Build, fabricate, and eval-freeze the configured physical macro."""
    config_paths = [resolve_relative_path(file, base) for file in section.config_files]
    policy_path = resolve_relative_path(section.policy_file, base)
    config = CimMacroConfig.from_file(*config_paths, section=section.config_section)
    policy = CimMacroPolicy.from_file(policy_path, section=section.policy_section)
    macro = CimMacro.from_config(
        config=config,
        policy=policy,
        input_num=section.input_num,
        output_num=section.output_num,
        inst_shape=inst_shape,
        dtype=torch.float32,
        T__K=T_ROOM__K,
    )
    macro = macro.to(device)
    macro.eval()
    macro.fabricate()
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
    macro: CimMacro[CimMacroConfig, CimMacroPolicy],
    *,
    device: torch.device,
) -> IdealCimMacro:
    """Build the lossless ideal twin; weights remain independently programmed."""
    ideal = macro.to_ideal().to(device)
    ideal.eval()
    return ideal


def unroll_active_positions(
    x: Tensor,
    *,
    input_num: int,
    max_active_num: int,
    inst_rank: int,
) -> Tensor:
    """Partition input positions into legal caller-side conversion planes."""
    plane_num = -(-input_num // max_active_num)
    plane_of_input = torch.arange(input_num, device=x.device) // max_active_num
    mask = plane_of_input == torch.arange(plane_num, device=x.device).unsqueeze(-1)
    mask = mask.reshape(plane_num, *(1,) * inst_rank, input_num)
    return torch.where(mask, x.unsqueeze(-(inst_rank + 2)), x.new_zeros(()))
