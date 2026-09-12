"""Construct module families from independent config and policy file sets.

Each file set follows `SerializeMixin.from_file`: the first file wins a
conflict, and a section selects a table within each file. Relative paths
are interpreted from the working directory. Config and policy construction
validate their fields; the family's `from_config` validates their dispatch
pair and constructs the module tree.

The returned tree retains its constructor state. Device placement, training
mode, fabrication, programming, and naming belong to the caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from neurox.architecture.unit.cim import CimUnit, CimUnitConfig, CimUnitPolicy
from neurox.primitive.analog.current_adc import Iadc, IadcConfig, IadcPolicy
from neurox.primitive.analog.diff_voltage_adc import DiffVadc, DiffVadcConfig, DiffVadcPolicy
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy


def cim_macro_from_file(
    *,
    config_files: Sequence[Path],
    policy_files: Sequence[Path],
    inst_shape: tuple[int, ...],
    dtype: torch.dtype,
    T__K: float,
    config_section: str | None = None,
    policy_section: str | None = None,
) -> CimMacro:
    """Load and validate a file pair, then construct its registered CIM macro."""
    config = CimMacroConfig.from_file(*config_files, section=config_section)
    policy = CimMacroPolicy.from_file(*policy_files, section=policy_section)
    return CimMacro.from_config(
        config=config,
        policy=policy,
        inst_shape=inst_shape,
        dtype=dtype,
        T__K=T__K,
    )


def cim_unit_from_file(
    *,
    config_files: Sequence[Path],
    policy_files: Sequence[Path],
    w_logical_shape: tuple[int, ...],
    dtype: torch.dtype,
    T__K: float,
    ideal_macro: bool,
    config_section: str | None = None,
    policy_section: str | None = None,
) -> CimUnit:
    """Load and validate a file pair, then construct its registered CIM unit.

    Args:
        w_logical_shape: Logical weight shape `(..., N, K)` bound to programming.
        ideal_macro: Replace the configured CIM macro with its ideal model
            during owned construction.
    """
    config = CimUnitConfig.from_file(*config_files, section=config_section)
    policy = CimUnitPolicy.from_file(*policy_files, section=policy_section)
    return CimUnit.from_config(
        config=config,
        policy=policy,
        w_logical_shape=w_logical_shape,
        dtype=dtype,
        T__K=T__K,
        ideal_macro=ideal_macro,
    )


def iadc_from_file(
    *,
    config_files: Sequence[Path],
    policy_files: Sequence[Path],
    inst_shape: tuple[int, ...],
    dtype: torch.dtype,
    T__K: float,
    config_section: str | None = None,
    policy_section: str | None = None,
) -> Iadc:
    """Load and validate a file pair, then construct its registered current ADC."""
    config = IadcConfig.from_file(*config_files, section=config_section)
    policy = IadcPolicy.from_file(*policy_files, section=policy_section)
    return Iadc.from_config(
        config=config,
        policy=policy,
        inst_shape=inst_shape,
        dtype=dtype,
        T__K=T__K,
    )


def diff_vadc_from_file(
    *,
    config_files: Sequence[Path],
    policy_files: Sequence[Path],
    inst_shape: tuple[int, ...],
    dtype: torch.dtype,
    T__K: float,
    config_section: str | None = None,
    policy_section: str | None = None,
) -> DiffVadc:
    """Load and validate a file pair, then construct its registered differential voltage ADC."""
    config = DiffVadcConfig.from_file(*config_files, section=config_section)
    policy = DiffVadcPolicy.from_file(*policy_files, section=policy_section)
    return DiffVadc.from_config(
        config=config,
        policy=policy,
        inst_shape=inst_shape,
        dtype=dtype,
        T__K=T__K,
    )
