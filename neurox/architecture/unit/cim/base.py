"""Abstract bases for the CimUnit family.

See also:
    docs/reference/architecture/unit/cim/README.md
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from torch import Tensor

from neurox.architecture.unit.base import UnitBase
from neurox.common import ConfigBase, ModuleBase, PolicyBase
from neurox.common.mixin import RegistryMixin
from neurox.primitive.macro.cim import CimMacroPolicy

from .engine import CimEngine, CimEngineConfig, CimEnginePolicy


@dataclass(frozen=True)
class CimUnitConfig(ConfigBase, ABC):
    """Abstract config root for the :class:`CimUnit` registry.

    Attributes:
        area_per_inst__um2: Unit-local peripheral silicon area per instance;
            excludes children, which self-report their own PPA.
        leakage_per_inst__uW: Unit-local peripheral static leakage per instance;
            excludes children, which self-report their own PPA.
    """

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Run all ``validate_*`` checks."""
        self.validate_ppa()

    def validate_ppa(self) -> None:
        """Require non-negative unit-local PPA fields."""
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


@dataclass(frozen=True)
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
    """Abstract root of the config-dispatched CimUnit family.

    The value-range / ADC surface and the protected lowering machinery
    come from :class:`UnitBase`; the operator surface comes from the
    ``UnitBase``-derived operator ABC mixed in by each concrete leaf;
    ``fabricate`` is satisfied by the ``FabricateMixin`` cascade.

    Args:
        config: Concrete configuration dataclass.
        policy: Composite nonideality policy.
        w_logical_shape: Logical weight shape ``(*prefix, N, K)`` bound to ``program(...)``.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
        ideal_xbar: Hint accepted for API uniformity. Consumed by
            xbar-using subclasses (swaps the physical xbar for its ideal
            twin); degenerate members ignore it.
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
        self._macro_dtype = dtype
        self._macro_T__K = T__K
        self._ideal_xbar = ideal_xbar

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
        pass  # container: child mismatch is sampled through the cascade


@dataclass(frozen=True)
class EngineBackedCimUnitConfig(CimUnitConfig, ABC):
    """Abstract config base for engine-backed CIM units.

    Attributes:
        engine: Nested engine config; its concrete type selects the
            execution variant through the ``_neurox_class`` discriminator.
    """

    engine: CimEngineConfig


@dataclass(frozen=True)
class EngineBackedCimUnitPolicy(CimUnitPolicy, ABC):
    """Abstract policy base for engine-backed CIM units.

    Attributes:
        cim_macro_policy: Embedded xbar nonideality policy, forwarded to the
            engine's composite policy.
    """

    cim_macro_policy: CimMacroPolicy


EbConfigT = TypeVar("EbConfigT", bound=EngineBackedCimUnitConfig)
EbPolicyT = TypeVar("EbPolicyT", bound=EngineBackedCimUnitPolicy)


class EngineBackedCimUnit(CimUnit[EbConfigT, EbPolicyT], Generic[EbConfigT, EbPolicyT], ABC):
    """Unregistered intermediate: a CIM unit delegating execution to an owned engine.

    Builds the :class:`CimEngine` selected by ``config.engine`` and
    delegates the whole execution surface to it; concrete subclasses add
    only their operator's ``program`` mapping.
    """

    engine: CimEngine

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
        self.engine = CimEngine.from_config(
            config=config.engine,
            policy=CimEnginePolicy(cim_macro_policy=policy.cim_macro_policy),
            w_logical_shape=self._engine_w_logical_shape(),
            dtype=dtype,
            T__K=T__K,
            ideal_xbar=ideal_xbar,
        )

    def _engine_w_logical_shape(self) -> tuple[int, ...]:
        """Logical weight shape handed to the engine; defaults to the unit's own."""
        return self._w_logical_shape

    # --- delegation to the engine ---

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
