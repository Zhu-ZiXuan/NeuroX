"""Circuit block represented by lumped PPA.

See Also:
    docs/reference/primitive/analog/unmodeled.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.common.module import ConfigBase, PolicyBase, ProfileModule


class UnmodeledBlockConfig(ConfigBase):
    # === Static PPA ===

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    # === Dynamic energy ===

    energy_per_op__fJ: float
    """Flat dynamic energy of one modeled operation."""

    def validate(self) -> None:
        super().validate()

        # --- Static PPA ---

        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")

        # --- Dynamic energy ---

        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")


class UnmodeledBlockPolicy(PolicyBase):
    pass


_Config = UnmodeledBlockConfig
_Policy = UnmodeledBlockPolicy


class UnmodeledBlock(ProfileModule):
    """Represent configured hardware costs without an electrical model.

    Static area and leakage scale with `inst_shape`. Call `execute` with the
    full layout of actual operations to record flat dynamic costs. Instance
    extents are not inserted into that layout automatically. No fabrication or
    programming is required, and execution returns no signal or timing value.

    Args:
        config: Hardware configuration.
        policy: Run policy matching `config`.
        inst_shape: Positive physical instance extents; singletons allow
            broadcasting.
        dtype: Retained argument; dynamic-energy records use float32.
    """

    config: _Config
    policy: _Policy

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    @torch.no_grad()
    def execute(self, shape: tuple[int, ...], *, enable: Tensor | None = None) -> None:
        """Record a flat cost at every enabled operation position.

        The method has no effect outside an active profiler. Its layout
        describes real operations, so callers must not invoke it once per
        numerical iteration.

        Args:
            shape: Complete observation and internal-work layout, including
                physical multiplicity and temporal reuse exactly once. Trailing
                work axes are reduced according to the configured profile
                leading rank.
            enable: Boolean event mask broadcastable to `shape`; `None` enables
                all positions. Disabled events cost zero.
        """
        if self._is_profiler_active():
            # Mask presence specializes during tracing; only masked costs depend on its device.
            if enable is None:
                energy__fJ = torch.full((), self.config.energy_per_op__fJ, dtype=torch.float32)
            else:
                energy__fJ = enable.to(dtype=torch.float32) * self.config.energy_per_op__fJ
            self._record_dynamic_energy(energy__fJ.expand(shape))
