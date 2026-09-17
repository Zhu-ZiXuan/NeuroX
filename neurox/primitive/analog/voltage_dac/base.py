"""Abstract base class for voltage-domain DAC models.

See Also:
    docs/reference/primitive/analog/voltage_dac/family.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import final

import torch
from torch import Tensor

from neurox.common.module import ConfigBase, PolicyBase, ProfileModule
from neurox.common.registry_mixin import RegistryMixin


class VdacConfig(ConfigBase, ABC):
    # === Static PPA ===

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    # === Required by base class ===

    def validate(self) -> None:
        super().validate()

        # --- Static PPA ---

        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class VdacPolicy(PolicyBase, ABC):
    pass


_Config = VdacConfig
_Policy = VdacPolicy


class Vdac(ProfileModule, RegistryMixin[_Config, _Policy], ABC):
    """Base class for voltage-domain DAC implementations.

    A converter reports no duration; conversion settles within an externally
    scheduled window.
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
        del dtype
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

    # === Public API ===

    @classmethod
    def from_config(
        cls,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> Vdac:
        """Build the implementation registered for the config-policy pair.

        Returns:
            Registered voltage-DAC implementation.
        """
        impl = cls._lookup_impl(config=config, policy=policy)
        return impl(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
        )

    @torch.no_grad()
    def convert(self, code: Tensor) -> Tensor:
        """Convert integer digital codes to analog output voltages.

        Args:
            code: Integer input codes in `[0, code_max]`.

        Returns:
            Analog output voltage [V], one value per `code` element. Dynamic
            energy is emitted through the profiler side channel.
        """
        return self._convert_impl(code)

    # === Required by base class ===

    @property
    @final
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    @final
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    # === For subclass to implement or override ===

    @property
    @abstractmethod
    def code_max(self) -> int:
        """Largest code the converter accepts; valid codes lie in `[0, code_max]`."""
        raise NotImplementedError

    @abstractmethod
    def _convert_impl(self, code: Tensor) -> Tensor:
        """Convert inputs according to the `convert` contract."""
        raise NotImplementedError
