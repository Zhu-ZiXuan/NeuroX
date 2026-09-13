"""Abstract bases for the CimUnit family.

See Also:
    docs/reference/architecture/unit/family.md
    docs/system_design/cim_execution.md
"""

from __future__ import annotations

from abc import ABC

import torch
from torch import Tensor

from neurox.architecture.unit.base import UnitBase
from neurox.common.module import ConfigBase, ModuleBase, PolicyBase
from neurox.common.registry_mixin import RegistryMixin

from .engine import CimEngine, CimEngineConfig, CimEnginePolicy


class CimUnitConfig(ConfigBase, ABC):
    area_per_inst__um2: float
    """Unit-local peripheral silicon area, excluding every child module."""
    leakage_per_inst__uW: float
    """Unit-local peripheral static leakage, excluding every child module."""

    # === Required by base class ===

    def validate(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class CimUnitPolicy(PolicyBase, ABC):
    pass


class CimUnit(
    ModuleBase,
    RegistryMixin["CimUnitConfig", "CimUnitPolicy", "CimUnit"],
    UnitBase,
    ABC,
):
    """Config-dispatched base for CIM compute units.

    A unit is an exact integer architecture specification: every quantity it
    exchanges is an integer code, the analog domain staying closed below it
    inside the macro.

    Args:
        w_logical_shape: Logical weight shape `(..., N, K)` bound to `program(...)`.
        ideal_macro: Whether to replace the configured CIM macro with its ideal model.
    """

    config: CimUnitConfig
    policy: CimUnitPolicy

    def __init__(
        self,
        *,
        config: CimUnitConfig,
        policy: CimUnitPolicy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        ideal_macro: bool,
    ) -> None:
        ModuleBase.__init__(self, config=config, policy=policy, inst_shape=())
        if len(w_logical_shape) < 2:
            raise ValueError(f"w_logical_shape must have at least 2 trailing dims (N, K); got {w_logical_shape}")
        self._w_logical_shape = tuple(w_logical_shape)

    # === Public API ===

    @classmethod
    def from_config(
        cls,
        *,
        config: CimUnitConfig,
        policy: CimUnitPolicy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        ideal_macro: bool,
    ) -> CimUnit:
        """Build the concrete implementation registered for the config-policy pair."""
        impl = cls._lookup_impl(config=config, policy=policy)
        return impl(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            ideal_macro=ideal_macro,
        )


class EngineBackedCimUnitConfig(CimUnitConfig, ABC):
    engine: CimEngineConfig


class EngineBackedCimUnitPolicy(CimUnitPolicy, ABC):
    engine: CimEnginePolicy
    """Engine policy matching `config.engine`."""


class EngineBackedCimUnit(CimUnit, ABC):
    """CIM unit backed by the engine selected by `config.engine`."""

    config: EngineBackedCimUnitConfig
    policy: EngineBackedCimUnitPolicy

    def __init__(
        self,
        *,
        config: EngineBackedCimUnitConfig,
        policy: EngineBackedCimUnitPolicy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        ideal_macro: bool,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            ideal_macro=ideal_macro,
        )
        self._init_engine_child(dtype=dtype, ideal_macro=ideal_macro)

    # === Required by base class ===

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    @property
    def w_value_range(self) -> tuple[int, int]:
        return self.engine.w_value_range

    @property
    def x_value_range(self) -> tuple[int, int]:
        return self.engine.x_value_range

    @property
    def adc_bits(self) -> int:
        return self.engine.adc_bits

    def rescale_factor(
        self,
        *,
        quantization_mode: int,
        adc_active_bits: int | None,
    ) -> float:
        return self.engine.rescale_factor(quantization_mode=quantization_mode, adc_active_bits=adc_active_bits)

    def _matmul(
        self,
        input: Tensor,
        *,
        quantization_mode: int,
        adc_active_bits: int | None,
    ) -> Tensor:
        return self.engine.matmul(input, quantization_mode=quantization_mode, adc_active_bits=adc_active_bits)

    # === For subclass to implement or override ===

    def _engine_w_logical_shape(self) -> tuple[int, ...]:
        """Logical weight shape handed to the engine; defaults to the unit's own."""
        return self._w_logical_shape

    # === Tools for subclass and internal use ===

    def _init_engine_child(self, *, dtype: torch.dtype, ideal_macro: bool) -> None:
        self.engine = CimEngine.from_config(
            config=self.config.engine,
            policy=self.policy.engine,
            w_logical_shape=self._engine_w_logical_shape(),
            dtype=dtype,
            ideal_macro=ideal_macro,
        )
