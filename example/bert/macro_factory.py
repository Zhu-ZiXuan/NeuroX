"""Local macro factory for the BERT example. Self-contained, no shared code.

Loads the immutable circuit config and the mutable nonideality policy from
their own TOML files, then instantiates one macro per layer.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from pathlib import Path

import torch

import works.offset_1t1r  # noqa: F401  # registers the Offset1T1RCimMacro kind
from neurox.architecture.unit.cim import CimUnit, CimUnitConfig, CimUnitPolicy
from neurox.architecture.unit.matmul import QuantMatMul
from neurox.primitive.physical_constant import T_ROOM__K

_CIRCUIT_DTYPE = torch.float32


@cache
def read_macro_config(config_path: Path) -> CimUnitConfig:
    return CimUnitConfig.from_file(config_path, section="cim_unit")


@cache
def read_macro_policy(policy_path: Path) -> CimUnitPolicy:
    return CimUnitPolicy.from_file(policy_path, section="policy")


def build_macro_factory(
    config_path: Path,
    policy_path: Path,
    *,
    ideal_xbar: bool,
) -> Callable[..., QuantMatMul]:
    """Return ``(name, w_logical_shape) → macro`` for the given config + policy TOMLs.

    The circuit design lives in ``config_path`` (section ``[cim_unit]``); the
    nonideality switches live in ``policy_path`` (section ``[policy]``).
    ``ideal_xbar=True`` swaps the physical xbar for its ideal twin (only
    meaningful when the config carries a physical xbar).
    """

    def factory(*, name: str, w_logical_shape: tuple[int, ...]) -> QuantMatMul:
        return CimUnit.from_config(
            config=read_macro_config(config_path),
            policy=read_macro_policy(policy_path),
            name=name,
            w_logical_shape=w_logical_shape,
            dtype=_CIRCUIT_DTYPE,
            T__K=T_ROOM__K,
            ideal_xbar=ideal_xbar,
        )

    return factory
