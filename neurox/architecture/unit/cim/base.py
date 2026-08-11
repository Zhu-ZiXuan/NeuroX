"""Abstract bases for the CimUnit family.

See also:
    docs/internals/architecture/unit/cim/base.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class CimUnitPolicy(PolicyBase, ABC):
    """Abstract marker base for CimUnit-family nonideality policies."""


ConfigT = TypeVar("ConfigT", bound=CimUnitConfig)
PolicyT = TypeVar("PolicyT", bound=CimUnitPolicy)


class CimUnit(
    ModuleBase[ConfigT, PolicyT],
    RegistryMixin["CimUnitConfig", "CimUnitPolicy", "CimUnit"],
    UnitBase,
    Generic[ConfigT, PolicyT],
    ABC,
):
    """Config-dispatched base for CIM compute units.

    Args:
        config: Concrete configuration dataclass.
        policy: Composite nonideality policy.
        w_logical_shape: Logical weight shape ``(..., N, K)`` bound to ``program(...)``.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
        ideal_macro: Whether to replace the configured CIM macro with its ideal model.
    """

    def __init__(
        self,
        *,
        config: ConfigT,
        policy: PolicyT,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_macro: bool,
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
        ideal_macro: bool,
    ) -> CimUnit:
        """Build the concrete impl registered for the config-policy pair."""
        impl = cls._lookup_neurox_module(config=config, policy=policy)
        return impl(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_macro=ideal_macro,
        )

    @abstractmethod
    def latency__ns(self, input_shape: tuple[int, ...], *, adc_bits: int | None) -> float:
        """Duration of one operator call [ns].

        The unit is the boundary where a runtime shape enters the timing line:
        every extent below it is fixed by the placement plan, and the one that
        is not — a convolution's output-position count ``M = H_out * W_out`` —
        follows from the input resolution alone. A unit that lowers to a
        single plane reads nothing out of the shape. Below the unit the mode
        knob travels on, since the readout runs as long as its resolution
        takes.

        Args:
            input_shape: Layout of the operand the unit's operator receives.
            adc_bits: Conversion resolution [bits], or ``None`` for the
                lossless oracle.

        Returns:
            Duration of one call through this unit.
        """
        raise NotImplementedError

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
        ideal_macro: bool,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_macro=ideal_macro,
        )
        self._init_engine_child(dtype=dtype, T__K=T__K, ideal_macro=ideal_macro)

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    def _init_engine_child(self, *, dtype: torch.dtype, T__K: float, ideal_macro: bool) -> None:
        """Construct the configured execution engine."""
        self.engine = CimEngine.from_config(
            config=self.config.engine,
            policy=self.policy.engine,
            w_logical_shape=self._engine_w_logical_shape(),
            dtype=dtype,
            T__K=T__K,
            ideal_macro=ideal_macro,
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
    def adc_max_bits(self) -> int:
        return self.engine.adc_max_bits

    def rescale_factor(self, *, quantization_mode: int, adc_bits: int | None) -> float:
        return self.engine.rescale_factor(quantization_mode=quantization_mode, adc_bits=adc_bits)

    def _matmul(self, input: Tensor, *, quantization_mode: int, adc_bits: int | None) -> Tensor:
        if input.dtype.is_floating_point or input.dtype.is_complex or input.dtype == torch.bool:
            raise TypeError(f"CIM execution requires an integer input tensor; got dtype {input.dtype}")
        return self.engine.matmul(input, quantization_mode=quantization_mode, adc_bits=adc_bits)
