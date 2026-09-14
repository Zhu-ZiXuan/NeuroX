"""Circuit block represented by lumped PPA.

See Also:
    docs/reference/primitive/analog/unmodeled.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.common.module import ConfigBase, ModuleBase, PolicyBase


class UnmodeledBlockConfig(ConfigBase):
    area_per_inst__um2: float
    leakage_per_inst__uW: float
    """Carries the block's whole standing bias power."""
    energy_per_op__fJ: float
    """Flat dynamic energy of one modeled operation."""

    def validate(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")


class UnmodeledBlockPolicy(PolicyBase):
    pass


_Config = UnmodeledBlockConfig
_Policy = UnmodeledBlockPolicy


class UnmodeledBlock(ModuleBase):
    """Circuit block represented by flat per-instance and per-operation PPA."""

    config: _Config
    policy: _Policy

    _energy_per_op__fJ: Tensor

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._register_nonpersistent_buffer(
            "_energy_per_op__fJ",
            torch.tensor(config.energy_per_op__fJ, dtype=dtype),
        )

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    @torch.no_grad()
    def execute(self, shape: tuple[int, ...], *, enable: Tensor | None = None) -> None:
        """Record enabled operations in `shape`; `None` enables every position."""
        if self._is_dynamic_energy_profile_active():
            energy = self._energy_per_op__fJ.expand(shape)
            if enable is not None:
                energy = energy.where(enable, 0)
            self._record_dynamic_energy(energy)
