"""Abstract bases for the CimUnit family.

See also:
    docs/internals/architecture/unit/cim/base.md
"""

from __future__ import annotations

from abc import ABC
from typing import Generic, TypeVar

import torch
from torch import Tensor

from neurox.architecture.unit.base import UnitBase
from neurox.common import ConfigBase, ModuleBase, PolicyBase
from neurox.common.mixin import RegistryMixin

from .engine import CimEngine, CimEngineConfig, CimEnginePolicy


class CimUnitConfig(ConfigBase, ABC):
    """Abstract config root for the :class:`CimUnit` registry.

    Attributes:
        area_per_inst__um2: Unit-local peripheral silicon area per instance.
        leakage_per_inst__uW: Unit-local peripheral static leakage per instance.
    """

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate(self) -> None:
        """Run all ``validate_*`` checks."""
        self.validate_ppa()

    def validate_ppa(self) -> None:
        """Require non-negative unit-local PPA fields."""
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class CimUnitPolicy(PolicyBase, ABC):
    """Abstract marker base for CimUnit-family nonideality policies."""


ConfigT = TypeVar("ConfigT", bound=CimUnitConfig)
PolicyT = TypeVar("PolicyT", bound=CimUnitPolicy)


class CimUnit(
    ModuleBase[ConfigT, PolicyT],
    RegistryMixin[type["CimUnitConfig"], "CimUnit"],
    UnitBase,
    Generic[ConfigT, PolicyT],
    ABC,
):
    """Config-dispatched base for CIM compute units.

    Args:
        config: Concrete configuration dataclass.
        policy: Composite nonideality policy.
        w_logical_shape: Logical weight shape ``(*prefix, N, K)`` bound to ``program(...)``.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
        ideal_xbar: Whether to replace the configured xbar with its ideal model.
    """

    def __init__(
        self,
        *,
        config: ConfigT,
        policy: PolicyT,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_xbar: bool,
    ) -> None:
        ModuleBase.__init__(self, config=config, policy=policy, inst_shape=())
        if len(w_logical_shape) < 2:
            raise ValueError(f"w_logical_shape must have at least 2 trailing dims (N, K); got {w_logical_shape}")
        self._w_logical_shape = tuple(w_logical_shape)

    @classmethod
    def from_config(
        cls,
        *,
        config: CimUnitConfig,
        policy: CimUnitPolicy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_xbar: bool,
    ) -> CimUnit:
        """Build the concrete impl registered for ``type(config)``."""
        impl = cls._lookup_impl(type(config))
        return impl(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_xbar=ideal_xbar,
        )

    def _sample_fabricate_mismatch(self) -> None:
        pass


class EngineBackedCimUnitConfig(CimUnitConfig, ABC):
    """Abstract config base for engine-backed CIM units.

    Attributes:
        engine: Execution-engine configuration.
    """

    engine: CimEngineConfig


class EngineBackedCimUnitPolicy(CimUnitPolicy, ABC):
    """Abstract policy base for engine-backed CIM units.

    Attributes:
        engine: Execution-engine policy matching ``config.engine``.
    """

    engine: CimEnginePolicy


EbConfigT = TypeVar("EbConfigT", bound=EngineBackedCimUnitConfig)
EbPolicyT = TypeVar("EbPolicyT", bound=EngineBackedCimUnitPolicy)


class EngineBackedCimUnit(CimUnit[EbConfigT, EbPolicyT], Generic[EbConfigT, EbPolicyT], ABC):
    """CIM unit backed by the engine selected by ``config.engine``."""

    def __init__(
        self,
        *,
        config: EbConfigT,
        policy: EbPolicyT,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_xbar: bool,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_xbar=ideal_xbar,
        )
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW
        self._init_engine_child(dtype=dtype, T__K=T__K, ideal_xbar=ideal_xbar)

    def _init_engine_child(self, *, dtype: torch.dtype, T__K: float, ideal_xbar: bool) -> None:
        """Construct the configured execution engine."""
        self.engine = CimEngine.from_config(
            config=self.config.engine,
            policy=self.policy.engine,
            w_logical_shape=self._engine_w_logical_shape(),
            dtype=dtype,
            T__K=T__K,
            ideal_xbar=ideal_xbar,
        )

    def _engine_w_logical_shape(self) -> tuple[int, ...]:
        """Logical weight shape handed to the engine; defaults to the unit's own."""
        return self._w_logical_shape

    @property
    def w_value_range(self) -> tuple[int, int]:
        return self.engine.w_value_range

    @property
    def x_value_range(self) -> tuple[int, int]:
        return self.engine.x_value_range

    @property
    def adc_mode_num(self) -> int:
        return self.engine.adc_mode_num

    @property
    def adc_max_bits(self) -> int:
        return self.engine.adc_max_bits

    def adc_rescale_factor(self, *, adc_mode: int, adc_bits: int) -> float:
        return self.engine.adc_rescale_factor(adc_mode=adc_mode, adc_bits=adc_bits)

    def _matmul(self, input: Tensor, *, adc_mode: int, adc_bits: int) -> Tensor:
        return self.engine.matmul(input, adc_mode=adc_mode, adc_bits=adc_bits)
