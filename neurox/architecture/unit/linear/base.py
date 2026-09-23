"""LinearUnit operator interface.

See Also:
    docs/reference/architecture/unit/family.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, final

import torch
from torch import Tensor

from neurox.architecture.unit.base import UnitBase, UnitConfig
from neurox.common.module import PolicyBase
from neurox.common.registry_mixin import RegistryMixin

if TYPE_CHECKING:
    from .ideal import IdealLinearUnit


class LinearUnitConfig(UnitConfig, base_only=True):
    pass


class LinearUnitPolicy(PolicyBase, base_only=True):
    pass


_Config = LinearUnitConfig
_Policy = LinearUnitPolicy


class LinearUnit(RegistryMixin[_Config, _Policy], UnitBase, ABC, base_only=True):
    """Interface for an integer `torch.nn.functional.linear` replacement.

    Construction initializes the common unit and retains the logical weight
    shape and dtype used by `to_ideal`. Implementations initialize any additional
    implementation base explicitly after this constructor returns.

    `w_logical_shape` accepts `weight.shape` and is stored as a fixed-length
    `(output, input)` tuple. A different number of axes raises `ValueError`.
    One basic operation for latency and profiling is one VMM.
    """

    config: _Config
    policy: _Policy

    # === Programmed state ===

    _int_bias: Tensor | None = None  # Shape: [output]

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        if len(w_logical_shape) != 2:
            raise ValueError(f"linear w_logical_shape must have 2 axes; got {w_logical_shape}")
        UnitBase.__init__(self, config=config, policy=policy)
        self._w_logical_shape = w_logical_shape
        self._dtype = dtype

    # === Public API ===

    @classmethod
    def from_config(
        cls,
        *,
        config: _Config,
        policy: _Policy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> LinearUnit:
        """Construct the linear implementation registered for the config-policy pair."""
        impl = cls._lookup_impl(config=config, policy=policy)
        return impl(config=config, policy=policy, w_logical_shape=w_logical_shape, dtype=dtype)

    @final
    def to_ideal(self) -> IdealLinearUnit:
        """Construct a fresh ideal linear unit with the same logical parameters.

        Preserve value ranges, weight shape, dtype, temperature and unit-local
        static PPA. Weights, bias and child circuits are not copied.
        """
        from .ideal import IdealLinearUnit, IdealLinearUnitConfig, IdealLinearUnitPolicy

        config = IdealLinearUnitConfig(
            x_value_range=self.x_value_range,
            w_value_range=self.w_value_range,
            area_per_inst__um2=self._area_per_inst__um2,
            leakage_per_inst__uW=self._leakage_per_inst__uW,
        )
        ideal = IdealLinearUnit(
            config=config,
            policy=IdealLinearUnitPolicy(),
            w_logical_shape=self._w_logical_shape,
            dtype=self._dtype,
        )
        ideal.set_temperature(self.T__K)
        return ideal

    @final
    @torch.no_grad()
    @torch.compile(dynamic=False, fullgraph=True)
    def linear(
        self,
        input: Tensor,
        *,
        quantization_mode: int,
        adc_active_bits: int | None,
    ) -> Tensor:
        """Execute one integer linear operator against the programmed state.

        Profiling requires `set_profile_leading_rank(input.ndim - 1)` on the
        assembled unit before execution. Every leading position retains its
        own energy and one-vector duration; an unbatched vector uses rank zero.

        Args:
            input: Integer activation values.
                Shape: `[*leading, K]`.
            quantization_mode: Index selecting the runtime quantization window.
            adc_active_bits: Active ADC resolution; `None` requests the
                unit's highest available precision.

        Returns:
            Integer pre-requantize output tensor. Leading dimensions are
            preserved, exactly as `torch.nn.functional.linear`.
            Shape: `[*leading, N]`.

        Raises:
            ValueError: The input has no feature axis, its feature width does
                not match the programmed matrix, or profiling is active with
                a configured rank different from the input-leading rank.
        """
        if input.ndim < 1 or input.shape[-1] != self._w_logical_shape[-1]:
            raise ValueError(f"linear() expects input [..., {self._w_logical_shape[-1]}]; got {tuple(input.shape)}")
        # Profiler presence and tensor rank are fixed while tracing each variant.
        profiling = self._is_profiler_active()
        if profiling:
            self._check_profile_leading_rank(input.ndim - 1)
        output = self._linear_impl(input, quantization_mode=quantization_mode, adc_active_bits=adc_active_bits)
        if profiling:
            latency__ns = self.latency__ns(input.shape, adc_active_bits=adc_active_bits)
            latency = torch.tensor(latency__ns, dtype=torch.float64)
            self._record_latency(latency.expand(input.shape[:-1]))
        return output

    # === For subclass to implement or override ===

    @abstractmethod
    def _linear_impl(self, input: Tensor, *, quantization_mode: int, adc_active_bits: int | None) -> Tensor:
        """Compute the linear output, including the programmed bias."""
        raise NotImplementedError

    @abstractmethod
    def program(self, weight: Tensor, *, bias: Tensor | None = None) -> None:
        """Write the unit's static weight state and optional integer bias.

        Args:
            weight: Integer weight values.
                Shape: `[N, K]`.
            bias: Per-channel integer bias added in the int64 accumulation
                domain; `None` clears any programmed bias.
                Shape: `[N]`.
        """
        raise NotImplementedError

    # === Tools for subclass and internal use ===

    @final
    def _program_int_bias(self, bias: Tensor | None, *, channels: int) -> None:
        """Store a per-output bias in the int64 accumulation domain."""
        if bias is not None and tuple(bias.shape) != (channels,):
            raise ValueError(f"require: bias.shape ({tuple(bias.shape)}) == ({channels},)")
        self._int_bias = None if bias is None else bias.long()
