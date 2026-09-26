"""Abstract base class for differential voltage-domain ADC models.

See Also:
    docs/reference/primitive/analog/diff_voltage_adc/family.md
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


class DiffVadcConfig(ConfigBase, base_only=True):
    # === Resolution ===

    bits: int
    """Physical output bit width."""

    # === Static PPA ===

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    # === Required by base class ===

    def validate(self) -> None:
        super().validate()

        # --- Resolution ---

        self._require_in_closed_interval(self.bits, "bits", lower=1, upper=31)

        # --- Static PPA ---

        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class DiffVadcPolicy(PolicyBase, base_only=True):
    pass


_Config = DiffVadcConfig
_Policy = DiffVadcPolicy


class DiffVadc(ProfileModule, RegistryMixin[_Config, _Policy], ABC, base_only=True):
    """Base for differential voltage ADCs with injected reference taps.

    Implement `_convert_impl` and `latency__ns`, and register the config-policy
    pair. Define topology-specific reference checks in `_validate_runtime_args`.
    Initialize sources and fabricated state through the physical-module lifecycle.

    Keep the final `convert` wrapper: it checks code layout and range and records
    returned energy and inputs. The hook must not submit energy again and must
    remain traceable. Place and fabricate outside conversion.

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
    ) -> DiffVadc:
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

    @final
    @torch.no_grad()
    @torch.compile(dynamic=False, fullgraph=True)
    def convert(
        self,
        *,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        v_refs__V: Tensor,
        active_bits: int,
    ) -> Tensor:
        """Digitise a differential analog voltage into a raw unsigned code.

        Args:
            v_pos__V: Positive-side analog input voltages for all conversion
                positions.
            v_neg__V: Negative-side analog input voltage, at the same shape as
                `v_pos__V`.
            v_refs__V: Injected reference taps, with the taps on the last axis.
                Shape: `[..., tap]`.
            active_bits: Active conversion resolution in `[1, bits]`.

        Returns:
            Raw unsigned code values stored as `int32`, one per `v_pos__V`
            element, in the range `unsigned_range` reports for `active_bits`.
            For offset-binary codes, recover the signed value as `(code -
            zero_offset(active_bits)) · rescale_factor` with a positive
            `rescale_factor`.

        Raises:
            ValueError: The active bit count, reference arguments, or output
                shape is invalid.
            RuntimeError: An output code lies outside the active-bit range.
        """
        self._check_active_bits(active_bits)
        self._validate_runtime_args(v_refs__V)
        code, energy__fJ = self._convert_impl(
            v_pos__V=v_pos__V,
            v_neg__V=v_neg__V,
            v_refs__V=v_refs__V,
            active_bits=active_bits,
            record_energy=self._is_profiler_active(),
        )
        code = code.int()
        if code.shape != v_pos__V.shape:
            raise ValueError("ADC implementation must return one code per input element")
        min_code, max_code = self.unsigned_range(active_bits)
        torch_assert_async(((code >= min_code) & (code <= max_code)).all(), "ADC output code outside active-bit range")
        if energy__fJ is not None:
            self._record_dynamic_energy(energy__fJ)
        self._record_input(v_pos__V=v_pos__V, v_neg__V=v_neg__V)
        return code

    @final
    def unsigned_range(self, active_bits: int) -> tuple[int, int]:
        """Return the inclusive full offset-binary range at `active_bits`."""
        self._check_active_bits(active_bits)
        return 0, (1 << active_bits) - 1

    @final
    def zero_offset(self, active_bits: int) -> int:
        """Return the raw analog-zero code to subtract before scaling."""
        self._check_active_bits(active_bits)
        return 1 << (active_bits - 1)

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

    def _validate_runtime_args(self, v_refs__V: Tensor) -> None:
        """Check the reference layout required by this converter.

        The default accepts any reference tensor. Override for topology-specific
        requirements such as the trailing tap count; raise `ValueError` for an
        invalid layout. This hook runs inside compiled conversion, so keep shape
        checks static and avoid Python decisions based on tensor values.

        Args:
            v_refs__V: Injected reference tensor whose trailing tap layout must
                fit the topology.
        """

    @abstractmethod
    def _convert_impl(
        self,
        *,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        v_refs__V: Tensor,
        active_bits: int,
        record_energy: bool,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute conversion outputs according to the `convert` contract.

        Args:
            v_pos__V: Positive-side input voltages.
            v_neg__V: Negative-side voltages with the same shape as the positive
                side.
            v_refs__V: Injected taps with the concrete converter reference
                layout.
            active_bits: Requested active resolution in the inclusive range from
                one to bits.
            record_energy: Whether to compute dynamic energy.

        Returns:
            A tuple (codes, energy) with one code and optional energy per input.
            Output codes and per-output dynamic energy [fJ]. Energy is `None`
            when not requested or when the implementation owns no energy. The
            caller submits the energy and observation records.
        """
        raise NotImplementedError

    # === Tools for subclass and internal use ===

    @final
    def _record_input(self, *, v_pos__V: Tensor, v_neg__V: Tensor) -> None:
        prober = AdcProber.current()
        if prober is not None:
            prober.submit_diff_voltage(v_pos__V=v_pos__V, v_neg__V=v_neg__V)

    @final
    def _check_active_bits(self, active_bits: int) -> None:
        if not (1 <= active_bits <= self.bits):
            raise ValueError(f"require: active_bits ({active_bits}) in [1, bits ({self.bits})]")
