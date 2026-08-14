"""Root bases for analog-primitive modules and their config/policy dataclasses.

See Also:
    docs/internals/primitive/analog/base.md
"""

from __future__ import annotations

from abc import ABC

from neurox.common import ConfigBase, ModuleBase, PolicyBase


class AnalogConfig(ConfigBase, ABC):
    """Root config for analog primitives — empty marker, no shared PPA fields."""


class AnalogPolicy(PolicyBase, ABC):
    """Root policy for analog primitives — empty marker."""


class AnalogBase[ConfigT: AnalogConfig, PolicyT: AnalogPolicy](ModuleBase[ConfigT, PolicyT], ABC):
    """Common base for analog primitive modules."""
