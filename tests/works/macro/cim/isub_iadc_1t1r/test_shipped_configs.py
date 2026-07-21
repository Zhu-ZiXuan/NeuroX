"""Shipped-artifact parse + build test — the one sanctioned config-file test.

Asserts the scheme's shipped artifacts parse and build: ``params/default.toml``
reflects through the registry into :class:`IsubIadc1t1rCimMacroConfig` and the
module builds + fabricates, ``params/cell_linear.toml`` loads as
:class:`XbarCell1t1rLinearConfig`, and ``policy/all_off.toml`` loads with every
nonideality toggle off. No numeric assertions, no forward pass.
"""

from __future__ import annotations

import dataclasses
import importlib.resources
from pathlib import Path

import torch

from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy
from neurox.primitive.xbar.cell import XbarCell1t1rLinearConfig
from neurox.works.macro.cim.isub_iadc_1t1r.macro import (
    IsubIadc1t1rCimMacro,
    IsubIadc1t1rCimMacroConfig,
    IsubIadc1t1rCimMacroPolicy,
)

_PKG_DIR = Path(str(importlib.resources.files("neurox.works.macro.cim.isub_iadc_1t1r")))


def _assert_all_toggles_false(obj: object, path: str = "policy") -> None:
    """Every bool field in the (nested) policy dataclass tree is False."""
    for field in dataclasses.fields(obj):  # type: ignore[arg-type]
        value = getattr(obj, field.name)
        if isinstance(value, bool):
            assert value is False, f"{path}.{field.name} is on in the all-off policy"
        elif dataclasses.is_dataclass(value):
            _assert_all_toggles_false(value, f"{path}.{field.name}")


def test_default_config_parses_and_builds() -> None:
    """``params/default.toml`` registry-dispatches to the scheme config and builds."""
    config = CimMacroConfig.from_file(_PKG_DIR / "params" / "default.toml", section="cim_macro")
    assert isinstance(config, IsubIadc1t1rCimMacroConfig)
    policy = CimMacroPolicy.from_file(_PKG_DIR / "policy" / "all_off.toml", section="policy")
    assert isinstance(policy, IsubIadc1t1rCimMacroPolicy)
    xbar = CimMacro.from_config(
        config=config,
        policy=policy,
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    assert isinstance(xbar, IsubIadc1t1rCimMacro)
    xbar.fabricate()


def test_cell_linear_config_parses() -> None:
    """``params/cell_linear.toml`` loads as the linear cell config."""
    cell = XbarCell1t1rLinearConfig.from_file(_PKG_DIR / "params" / "cell_linear.toml", section="cell_config")
    assert isinstance(cell, XbarCell1t1rLinearConfig)


def test_all_off_policy_every_toggle_false() -> None:
    """``policy/all_off.toml`` loads with every nonideality toggle off."""
    policy = CimMacroPolicy.from_file(_PKG_DIR / "policy" / "all_off.toml", section="policy")
    assert isinstance(policy, IsubIadc1t1rCimMacroPolicy)
    _assert_all_toggles_false(policy)
