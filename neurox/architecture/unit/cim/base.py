"""Abstract bases for the CimUnit family.

See Also:
    docs/reference/architecture/unit/family.md
    docs/system_design/cim_execution.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from neurox.architecture.unit.base import UnitBase
from neurox.common import ConfigBase, ModuleBase, PolicyBase, RegistryMixin

from .engine import CimEngine, CimEngineConfig, CimEnginePolicy


class CimUnitConfig(ConfigBase, ABC):
    """Abstract config root for the `CimUnit` registry."""

    area_per_inst__um2: float
    """Unit-local peripheral silicon area, excluding every child module."""
    leakage_per_inst__uW: float
    """Unit-local peripheral static leakage, excluding every child module."""

    def validate(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class CimUnitPolicy(PolicyBase, ABC):
    """Abstract marker base for CimUnit-family nonideality policies."""


class CimUnit[ConfigT: CimUnitConfig, PolicyT: CimUnitPolicy](
    ModuleBase[ConfigT, PolicyT],
    RegistryMixin["CimUnitConfig", "CimUnitPolicy", "CimUnit[CimUnitConfig, CimUnitPolicy]"],
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
    ) -> CimUnit[CimUnitConfig, CimUnitPolicy]:
        """Build the concrete implementation registered for the config-policy pair."""
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
        """Duration of one operator call.

        The unit is the boundary where a runtime shape enters the timing line:
        every extent below it is fixed by the placement plan, and the one that
        is not — a convolution's output-position count `M = H_out * W_out` —
        follows from the input resolution alone. A unit that lowers to a
        single plane reads nothing out of the shape.

        Args:
            input_shape: Layout of the operand the unit's operator receives.
            adc_bits: Conversion resolution, or `None` for the lossless oracle.
        """
        raise NotImplementedError

    def _sample_fabricate_mismatch(self) -> None:
        pass


class EngineBackedCimUnitConfig(CimUnitConfig, ABC):
    """Abstract config base for engine-backed CIM units."""

    engine: CimEngineConfig


class EngineBackedCimUnitPolicy(CimUnitPolicy, ABC):
    """Abstract policy base for engine-backed CIM units."""

    engine: CimEnginePolicy
    """Engine policy matching `config.engine`."""


class EngineBackedCimUnit[ConfigT: EngineBackedCimUnitConfig, PolicyT: EngineBackedCimUnitPolicy](
    CimUnit[ConfigT, PolicyT], ABC
):
    """CIM unit backed by the engine selected by `config.engine`."""

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
