"""Abstract base class for single-ended current-domain ADC models.

See Also:
    docs/reference/primitive/analog/current_adc/family.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import final

import torch
from torch import Tensor

from neurox.common.module import ConfigBase, PolicyBase, ProfileModule
from neurox.common.registry_mixin import RegistryMixin
from neurox.common.torch_compat import torch_assert_async
from neurox.primitive.analog.adc_probe import AdcProber


class IadcConfig(ConfigBase, base_only=True):
    # === Resolution ===

    bits: int
    """Physical maximum conversion resolution, from 1 to 31 bits."""

    # === Static PPA ===

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    # === Required by base class ===

    def validate(self) -> None:
        super().validate()

        # --- Resolution ---

        self._require_in_closed_interval(self.bits, "bits", 1, 31)

        # --- Static PPA ---

        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class IadcPolicy(PolicyBase, base_only=True):
    pass


_Config = IadcConfig
_Policy = IadcPolicy


class Iadc(ProfileModule, RegistryMixin[_Config, _Policy], ABC, base_only=True):
    """Base class for single-ended current ADCs with injected references.

    Callers select and supply reference taps for every conversion. Each
    implementation defines and validates its tap count independently of the
    requested resolution; converters receive no mode identifier.
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
    ) -> Iadc:
        """Build the implementation registered for the config-policy pair."""
        impl = cls._lookup_impl(config=config, policy=policy)
        return impl(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
        )

    @property
    @final
    def bits(self) -> int:
        return self.config.bits

    @torch.no_grad()
    def convert(
        self,
        i_in__uA: Tensor,
        i_refs__uA: Tensor,
        *,
        active_bits: int,
        enable: Tensor | None = None,
    ) -> Tensor:
        """Digitise a single-ended magnitude current into an unsigned integer code.

        Args:
            i_in__uA: Non-negative magnitude current.
            i_refs__uA: Reference ladder with the taps on the last axis and the
                leading dims right-broadcasting against `i_in__uA`.
                Shape: `[..., tap]`.
            active_bits: Active conversion resolution in `[1, bits]`.
            enable: Broadcastable conversion enables; disabled conversions
                return zero and emit no conversion energy. `None` enables all conversions.

        Returns:
            Unsigned code values stored as `int32`, one per `i_in__uA` element, in
            the range `unsigned_range` reports for `active_bits`. For a deterministic
            converter the code at `active_bits` is the full-width code
            right-shifted by `bits - active_bits`. Dynamic energy is emitted
            through the profiler side channel.

        Raises:
            ValueError: The active bit count or output shape is invalid.
            RuntimeError: An output code lies outside the active-bit range.
        """
        self._check_active_bits(active_bits)
        code, energy__fJ = self._convert_impl(
            i_in__uA,
            i_refs__uA,
            active_bits=active_bits,
            record_energy=self._is_profiler_active(),
        )
        code = code.int()
        if code.shape != i_in__uA.shape:
            raise ValueError("ADC implementation must return one code per input element")
        min_code, max_code = self.unsigned_range(active_bits)
        torch_assert_async(((code >= min_code) & (code <= max_code)).all(), "ADC output code outside active-bit range")
        if enable is not None:
            code = code.where(enable, 0)
            if energy__fJ is not None:
                energy__fJ = energy__fJ.where(enable, 0)
        if energy__fJ is not None:
            self._record_dynamic_energy(energy__fJ)
        self._record_input(i_in__uA)
        return code

    @final
    def unsigned_range(self, active_bits: int) -> tuple[int, int]:
        """Return the inclusive full unsigned range at `active_bits`."""
        self._check_active_bits(active_bits)
        return 0, (1 << active_bits) - 1

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

    @abstractmethod
    def latency__ns(self, *, active_bits: int) -> float:
        """Duration of one `convert` call.

        Args:
            active_bits: Active conversion resolution in `[1, bits]`.
        """
        raise NotImplementedError

    @abstractmethod
    def _convert_impl(
        self,
        i_in__uA: Tensor,
        i_refs__uA: Tensor,
        *,
        active_bits: int,
        record_energy: bool,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute conversion outputs according to the `convert` contract.

        Args:
            record_energy: Whether to compute dynamic energy.

        Returns:
            Output codes and per-output dynamic energy [fJ]. Energy is `None`
            when not requested or when the implementation owns no energy.
            The caller submits the energy and observation records.
        """
        raise NotImplementedError

    # === Tools for subclass and internal use ===

    @final
    def _record_input(self, i_in__uA: Tensor) -> None:
        prober = AdcProber.current()
        if prober is not None:
            prober.submit_current(i_in__uA=i_in__uA)

    @final
    def _check_active_bits(self, active_bits: int) -> None:
        if not (1 <= active_bits <= self.bits):
            raise ValueError(f"require: active_bits ({active_bits}) in [1, bits ({self.bits})]")
