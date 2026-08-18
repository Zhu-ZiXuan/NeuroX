"""Root bases for analog-primitive modules and their config/policy dataclasses.

See Also:
    docs/system_design/construction.md
"""

from __future__ import annotations

from abc import ABC

from neurox.common import ConfigBase, ModuleBase, PolicyBase


class AnalogConfig(ConfigBase, ABC):
    """Root config for analog primitives — empty marker, no shared PPA fields."""


class AnalogPolicy(PolicyBase, ABC):
    """Root policy for analog primitives — empty marker."""


class AnalogBase[ConfigT: AnalogConfig, PolicyT: AnalogPolicy](ModuleBase[ConfigT, PolicyT], ABC):
    """Common base for analog primitive modules.

    The family shares a config and policy type plus one construction shape:
    every concrete analog block accepts `dtype` and the operating temperature
    `T__K` beside the config, policy, and instance shape, whether or not it
    uses them, so a single call shape builds any member. Everything else — the
    value path, its parameters, and its static PPA fields — is the block's own.
    """
