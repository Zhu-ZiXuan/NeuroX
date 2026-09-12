"""Abstract base class for voltage-domain DAC models.

See Also:
    docs/reference/primitive/analog/voltage_dac/family.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from neurox.common.module import ConfigBase, ModuleBase, PolicyBase
from neurox.common.registry_mixin import RegistryMixin


class VdacConfig(ConfigBase, ABC):
    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class VdacPolicy(PolicyBase, ABC):
    pass


_Config = VdacConfig
_Policy = VdacPolicy


class Vdac(
    ModuleBase,
    RegistryMixin["_Config", "_Policy", "Vdac"],
    ABC,
):
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
        T__K: float,
    ) -> None:
        del dtype, T__K
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

    @classmethod
    def from_config(
        cls,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
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
            T__K=T__K,
        )

    @property
    @abstractmethod
    def code_max(self) -> int:
        """Largest code the converter accepts; valid codes lie in `[0, code_max]`."""
        raise NotImplementedError

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

    @abstractmethod
    def _convert_impl(self, code: Tensor) -> Tensor:
        """Convert inputs according to the `convert` contract."""
        raise NotImplementedError
