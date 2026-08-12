"""Local macro factory for the LeNet example. Self-contained, no shared code.

Loads the immutable circuit config and the mutable nonideality policy from
their own TOML files, then instantiates one macro per layer.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from pathlib import Path

import torch

from neurox.architecture.unit.cim import CimUnit, CimUnitConfig, CimUnitPolicy
from neurox.primitive import T_ROOM__K

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
    ideal_macro: bool,
) -> Callable[..., CimUnit[CimUnitConfig, CimUnitPolicy]]:
    """Return ``(w_logical_shape) → macro`` for the given config + policy TOMLs.

    The circuit design lives in ``config_path`` (section ``[cim_unit]``); the
    nonideality switches live in ``policy_path`` (section ``[policy]``).
    ``ideal_macro=True`` swaps the configured macro for its ideal twin via
    ``to_ideal()`` — the faithful reference of a physical macro, and the macro
    itself when the config already carries an ideal one.
    """

    def factory(*, w_logical_shape: tuple[int, ...]) -> CimUnit[CimUnitConfig, CimUnitPolicy]:
        return CimUnit.from_config(
            config=read_macro_config(config_path),
            policy=read_macro_policy(policy_path),
            w_logical_shape=w_logical_shape,
            dtype=_CIRCUIT_DTYPE,
            T__K=T_ROOM__K,
            ideal_macro=ideal_macro,
        )

    return factory
