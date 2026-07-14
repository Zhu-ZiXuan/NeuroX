"""Root bases for analog-primitive modules and their config/policy dataclasses.

See also:
    docs/internals/primitive/analog/base.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from neurox.common import ConfigBase, ModuleBase, PolicyBase


@dataclass(frozen=True)
class AnalogConfig(ConfigBase):
    """Root config for analog primitives — empty marker, no shared PPA fields."""


@dataclass(frozen=True)
class AnalogPolicy(PolicyBase):
    """Root policy for analog primitives — empty marker."""


ConfigT = TypeVar("ConfigT", bound=AnalogConfig)
PolicyT = TypeVar("PolicyT", bound=AnalogPolicy)


class AnalogBase(ModuleBase[ConfigT, PolicyT]):
    """Common base for analog primitive modules."""
