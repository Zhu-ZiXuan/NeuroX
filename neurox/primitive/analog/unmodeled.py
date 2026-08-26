"""Circuit block represented by lumped PPA.

See Also:
    docs/reference/primitive/analog/unmodeled.md
"""

import torch
from torch import Tensor

from .base import AnalogBase, AnalogConfig, AnalogPolicy


class UnmodeledBlockConfig(AnalogConfig):
    area_per_inst__um2: float
    leakage_per_inst__uW: float
    """Carries the block's whole standing bias power."""
    energy_per_op__fJ: float
    """Flat dynamic energy of one modeled operation."""

    def validate(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")


class UnmodeledBlockPolicy(AnalogPolicy):
    pass


class UnmodeledBlock(AnalogBase[UnmodeledBlockConfig, UnmodeledBlockPolicy]):
    """Circuit block represented by flat per-instance and per-operation PPA."""

    _energy_per_op__fJ: Tensor

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
        self.register_buffer(
            "_energy_per_op__fJ",
            torch.tensor(config.energy_per_op__fJ, dtype=dtype),
            persistent=False,
        )

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    def execute(self, shape: tuple[int, ...]) -> None:
        """Record one operation at every position in `shape`."""
        if self._is_dynamic_energy_profile_active():
            self._record_dynamic_energy(self._energy_per_op__fJ.expand(shape))
