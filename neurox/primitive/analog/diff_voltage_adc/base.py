"""Abstract base class for differential voltage-domain ADC models.

See Also:
    docs/reference/primitive/analog/diff_voltage_adc/family.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import final

import torch
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase, RegistryMixin
from neurox.primitive.analog.adc_probe import AdcProber, AdcRecord


class DiffVadcRecord(AdcRecord):
    v_pos__V: Tensor
    """Positive-side input voltage the call was handed."""
    v_neg__V: Tensor
    """Negative-side input voltage the call was handed."""

    def input_name(self) -> str:
        return "v_diff__V"

    def input_value(self) -> Tensor:
        return self.v_pos__V - self.v_neg__V


class DiffVadcConfig(ConfigBase, ABC):
    bits: int
    """Physical output bit width."""
    area_per_inst__um2: float
    """Physical area per ADC instance."""
    leakage_per_inst__uW: float
    """Static leakage power per ADC instance."""

    def validate(self) -> None:
        self._require_pos(self.bits, "bits")
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class DiffVadcPolicy(PolicyBase, ABC):
    pass


class DiffVadc[ConfigT: DiffVadcConfig, PolicyT: DiffVadcPolicy](
    ModuleBase[ConfigT, PolicyT],
    RegistryMixin[
        "DiffVadcConfig",
        "DiffVadcPolicy",
        "DiffVadc[DiffVadcConfig, DiffVadcPolicy]",
    ],
    ABC,
):
    """Base class for differential voltage-domain ADC implementations.

    A converter owns its transfer structure and never its reference values:
    every `convert` call carries the taps in. How many taps a call needs is the
    concrete converter's own circuit property, so the base validates no tap
    count.

    A converter is mode-blind: reference selection and mode ranges remain
    outside it, while `convert` receives only the electrical operating point
    and the bit width.

    """

    def __init__(
        self,
        *,
        config: ConfigT,
        policy: PolicyT,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        del dtype, T__K
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

    @property
    @final
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    @final
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    @classmethod
    def from_config(
        cls,
        *,
        config: DiffVadcConfig,
        policy: DiffVadcPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> DiffVadc[DiffVadcConfig, DiffVadcPolicy]:
        """Build the implementation registered for the config-policy pair.

        Returns:
            Registered voltage-ADC implementation.
        """
        impl = cls._lookup_neurox_module(config=config, policy=policy)
        return impl(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

    @property
    @final
    def bits(self) -> int:
        """Physical output bit width."""
        return self.config.bits

    @final
    def _check_active_bits(self, active_bits: int) -> None:
        if not (1 <= active_bits <= self.bits):
            raise ValueError(f"require: active_bits ({active_bits}) in [1, bits ({self.bits})]")

    @abstractmethod
    def latency__ns(self, *, active_bits: int) -> float:
        """Duration of one `convert` call at `active_bits` [ns].

        Args:
            active_bits: Active conversion resolution in `[1, bits]`.
        """
        raise NotImplementedError

    def convert(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        v_refs__V: Tensor,
        active_bits: int,
    ) -> Tensor:
        """Digitise a differential analog voltage into a raw unsigned code.

        Args:
            v_pos__V: Positive-side analog input voltage.
            v_neg__V: Negative-side analog input voltage, at the same shape as
                `v_pos__V`.
            v_refs__V: Injected reference taps, with the taps on the last axis.
                The tap count `n_ref` is the concrete converter's circuit
                property, not a base-level contract.
                Shape: `[..., tap]`.
            active_bits: Active conversion resolution in `[1, bits]`.

        Returns:
            Raw unsigned integer code tensor, one code per `v_pos__V` element,
            in the range `unsigned_range` reports for `active_bits`. For offset-binary
            codes, recover the signed value as
            `(code - zero_offset(active_bits)) · rescale_factor` with a positive
            `rescale_factor`.
        """
        self._check_active_bits(active_bits)
        code = self._convert_impl(
            v_pos__V,
            v_neg__V,
            v_refs__V=v_refs__V,
            active_bits=active_bits,
        )
        if AdcProber.active():
            AdcProber.submit(
                DiffVadcRecord(
                    v_pos__V=v_pos__V,
                    v_neg__V=v_neg__V,
                ),
            )
        return code

    @abstractmethod
    def _convert_impl(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        v_refs__V: Tensor,
        active_bits: int,
    ) -> Tensor:
        """Convert inputs according to the `convert` contract."""
        raise NotImplementedError

    @final
    def unsigned_range(self, active_bits: int) -> tuple[int, int]:
        """Return `(min_code, max_code)` the ADC can emit at `active_bits`.

        Every differential voltage ADC emits the family's full raw
        offset-binary active-bit range.
        """
        self._check_active_bits(active_bits)
        return 0, (1 << active_bits) - 1

    @final
    def zero_offset(self, active_bits: int) -> int:
        """Return the raw code representing analog zero at `active_bits`.

        Subtract this offset before scaling:
        `(code - zero_offset(active_bits)) · rescale_factor`. Sign and offset
        are not folded into the emitted code.
        """
        self._check_active_bits(active_bits)
        return 1 << (active_bits - 1)
