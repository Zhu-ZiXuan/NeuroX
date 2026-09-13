"""Abstract base class for current-domain DAC models.

See Also:
    docs/reference/primitive/analog/current_dac/family.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from neurox.common.module import ConfigBase, ModuleBase, PolicyBase
from neurox.common.registry_mixin import RegistryMixin


class IdacConfig(ConfigBase, ABC):
    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class IdacPolicy(PolicyBase, ABC):
    pass


_Config = IdacConfig
_Policy = IdacPolicy


class Idac(
    ModuleBase,
    RegistryMixin["_Config", "_Policy", "Idac"],
    ABC,
):
    """Base class for current-domain DAC implementations.

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

    @classmethod
    def from_config(
        cls,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> Idac:
        """Build the implementation registered for the config-policy pair.

        Returns:
            Registered current-DAC implementation.
        """
        impl = cls._lookup_impl(config=config, policy=policy)
        return impl(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
        )

    @property
    @abstractmethod
    def code_max(self) -> int:
        """Largest code the converter accepts; valid codes lie in `[0, code_max]`."""
        raise NotImplementedError

    @torch.no_grad()
    def convert(self, code: Tensor) -> Tensor:
        """Convert integer digital codes to analog output currents.

        Args:
            code: Integer input codes in `[0, code_max]`.

        Returns:
            Analog output current [uA], one value per `code` element. Dynamic
            energy is emitted through the profiler side channel.
        """
        return self._convert_impl(code)

    @abstractmethod
    def _convert_impl(self, code: Tensor) -> Tensor:
        """Convert inputs according to the `convert` contract."""
        raise NotImplementedError
