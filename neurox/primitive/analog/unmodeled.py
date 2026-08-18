"""Unmodeled circuit block — static PPA seat only, no functional model.

See Also:
    docs/reference/primitive/analog/unmodeled.md
"""

import torch

from .base import AnalogBase, AnalogConfig, AnalogPolicy


class UnmodeledBlockConfig(AnalogConfig):
    """Immutable configuration for `UnmodeledBlock`."""

    area_per_inst__um2: float
    leakage_per_inst__uW: float
    """Carries the block's whole standing bias power."""

    def validate(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class UnmodeledBlockPolicy(AnalogPolicy):
    """Abstract marker for UnmodeledBlock nonideality policy — no sources."""


class UnmodeledBlock(AnalogBase[UnmodeledBlockConfig, UnmodeledBlockPolicy]):
    """Circuit block represented only by static area and leakage."""

    def __init__(
        self,
        *,
        config: UnmodeledBlockConfig,
        policy: UnmodeledBlockPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW
