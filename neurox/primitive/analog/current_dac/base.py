"""Abstract base class for current-domain DAC models.

See Also:
    docs/reference/primitive/analog/current_dac/family.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import final

import torch
from torch import Tensor

from neurox.common.module import ConfigBase, PolicyBase, ProfileModule
from neurox.common.registry_mixin import RegistryMixin


class IdacConfig(ConfigBase, base_only=True):
    # === Static PPA ===

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    # === Required by base class ===

    def validate(self) -> None:
        super().validate()

        # --- Static PPA ---

        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class IdacPolicy(PolicyBase, base_only=True):
    pass


_Config = IdacConfig
_Policy = IdacPolicy


class Idac(ProfileModule, RegistryMixin[_Config, _Policy], ABC, base_only=True):
    """Base class for current-domain DAC implementations.

    A converter reports no duration; conversion settles within an externally
    scheduled window.

    Implement `code_max` and `_convert_impl`, then register the concrete
    config-policy pair on this family. The public `convert` wrapper disables
    gradient tracking and delegates conversion; it does not validate or clip
    codes. The implementation must preserve the caller's element layout, perform
    its modeled sampling, and emit its own dynamic energy when profiling is
    active. Static per-instance costs come from the family configuration.

    Args:
        config: Hardware configuration.
        policy: Run policy matching `config`.
        inst_shape: Positive physical instance extents; singletons allow
            broadcasting.
        dtype: Electrical tensor dtype.
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

    # === Public API ===

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
        """Convert integer codes and record the modeled switching cost.

        Return one analog value for each input element using placed tensor
        sources. Callers supply codes in `[0, code_max]`; define any additional
        restrictions on the concrete class. The wrapper supplies no clipping,
        energy submission, or sampling, so the implementation owns those modeled
        operations. Do not mutate input codes or record another component's
        costs.

        Args:
            code: Integer input codes in the inclusive range from zero to
                code_max.

        Returns:
            Output current for each input code on the placed sources' device.
        """
        raise NotImplementedError
