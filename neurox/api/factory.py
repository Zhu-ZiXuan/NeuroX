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

from neurox.architecture.unit.conv2d import Conv2dUnit, Conv2dUnitConfig, Conv2dUnitPolicy
from neurox.architecture.unit.linear import LinearUnit, LinearUnitConfig, LinearUnitPolicy
from neurox.primitive.analog.current_adc import Iadc, IadcConfig, IadcPolicy
from neurox.primitive.analog.diff_voltage_adc import DiffVadc, DiffVadcConfig, DiffVadcPolicy
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy


def iadc_from_file(
    *,
    config_files: Sequence[Path],
    policy_files: Sequence[Path],
    inst_shape: tuple[int, ...],
    dtype: torch.dtype,
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
    )


def diff_vadc_from_file(
    *,
    config_files: Sequence[Path],
    policy_files: Sequence[Path],
    inst_shape: tuple[int, ...],
    dtype: torch.dtype,
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
    )


def cim_macro_from_file(
    *,
    config_files: Sequence[Path],
    policy_files: Sequence[Path],
    inst_shape: tuple[int, ...],
    dtype: torch.dtype,
    config_section: str | None = None,
    policy_section: str | None = None,
    to_ideal: bool = False,
) -> CimMacro:
    """Load and validate a file pair, then construct its registered CIM macro."""
    config = CimMacroConfig.from_file(*config_files, section=config_section)
    policy = CimMacroPolicy.from_file(*policy_files, section=policy_section)
    module = CimMacro.from_config(
        config=config,
        policy=policy,
        inst_shape=inst_shape,
        dtype=dtype,
    )
    return module.to_ideal() if to_ideal else module


def linear_unit_from_file(
    *,
    config_files: Sequence[Path],
    policy_files: Sequence[Path],
    w_logical_shape: tuple[int, ...],
    dtype: torch.dtype,
    config_section: str | None = None,
    policy_section: str | None = None,
    to_ideal: bool = False,
) -> LinearUnit:
    """Load a config-policy pair and construct its registered linear implementation."""
    config = LinearUnitConfig.from_file(*config_files, section=config_section)
    policy = LinearUnitPolicy.from_file(*policy_files, section=policy_section)
    module = LinearUnit.from_config(
        config=config,
        policy=policy,
        w_logical_shape=w_logical_shape,
        dtype=dtype,
    )
    return module.to_ideal() if to_ideal else module


def conv2d_unit_from_file(
    *,
    config_files: Sequence[Path],
    policy_files: Sequence[Path],
    w_logical_shape: tuple[int, ...],
    dtype: torch.dtype,
    config_section: str | None = None,
    policy_section: str | None = None,
    to_ideal: bool = False,
) -> Conv2dUnit:
    """Load a config-policy pair and construct its registered conv2d implementation."""
    config = Conv2dUnitConfig.from_file(*config_files, section=config_section)
    policy = Conv2dUnitPolicy.from_file(*policy_files, section=policy_section)
    module = Conv2dUnit.from_config(
        config=config,
        policy=policy,
        w_logical_shape=w_logical_shape,
        dtype=dtype,
    )
    return module.to_ideal() if to_ideal else module
