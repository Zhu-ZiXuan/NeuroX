"""Root bases for analog-primitive modules and their config/policy dataclasses.

See also:
    docs/internals/primitive/analog/base.md
"""

from __future__ import annotations

from abc import ABC
from typing import TypeVar

from neurox.common import ConfigBase, ModuleBase, PolicyBase


class AnalogConfig(ConfigBase, ABC):
    """Root config for analog primitives — empty marker, no shared PPA fields."""


class AnalogPolicy(PolicyBase, ABC):
    """Root policy for analog primitives — empty marker."""


ConfigT = TypeVar("ConfigT", bound=AnalogConfig)
PolicyT = TypeVar("PolicyT", bound=AnalogPolicy)


class AnalogBase(ModuleBase[ConfigT, PolicyT], ABC):
    """Common base for analog primitive modules."""
