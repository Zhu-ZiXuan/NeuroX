"""Abstract base class for single-ended current-domain ADC models.

See Also:
    docs/reference/primitive/analog/current_adc/family.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import final

import torch
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase, RegistryMixin
from neurox.primitive.analog.adc_probe import AdcProber, AdcRecord


class IadcRecord(AdcRecord):
    i_in__uA: Tensor
    """Input magnitude current the call was handed."""

    def input_name(self) -> str:
        return "i_in__uA"

    def input_value(self) -> Tensor:
        return self.i_in__uA


class IadcConfig(ConfigBase, ABC):
    bits: int
    """Physical maximum conversion resolution."""
    area_per_inst__um2: float
    """Physical area per ADC instance."""
    leakage_per_inst__uW: float
    """Static leakage power per ADC instance."""

    def validate(self) -> None:
        self._require_pos(self.bits, "bits")
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class IadcPolicy(PolicyBase, ABC):
    pass


class Iadc[ConfigT: IadcConfig, PolicyT: IadcPolicy](
    ModuleBase[ConfigT, PolicyT],
    RegistryMixin[
        "IadcConfig",
        "IadcPolicy",
        "Iadc[IadcConfig, IadcPolicy]",
    ],
    ABC,
):
    """Base class for single-ended current ADCs with injected references.

    The base owns the `bits` contract and nothing else about the call. How
    many reference taps a conversion consumes is the concrete converter's own
    circuit property, so it is neither declared nor validated at this level; a
    ladder the leaf cannot use fails inside that leaf.

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

    @classmethod
    def from_config(
        cls,
        *,
        config: IadcConfig,
        policy: IadcPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> Iadc[IadcConfig, IadcPolicy]:
        """Build the implementation registered for the config-policy pair.

        Returns:
            Registered current-ADC implementation.
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
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    @final
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    @property
    @final
    def bits(self) -> int:
        """Physical output bit width."""
        return self.config.bits

    def _check_active_bits(self, active_bits: int) -> None:
        """Require an active resolution this converter supports.

        Raises:
            ValueError: `active_bits` is outside `[1, bits]`.
        """
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
        i_in__uA: Tensor,
        i_refs__uA: Tensor,
        *,
        active_bits: int,
    ) -> Tensor:
        """Digitise a single-ended magnitude current into an unsigned integer code.

        Bit width is ADC-internal: the injected ladder states the converter's
        own wiring and does not follow the requested resolution.

        Args:
            i_in__uA: Non-negative magnitude current.
            i_refs__uA: Reference ladder with the taps on the last axis and the
                leading dims right-broadcasting against `i_in__uA`. The tap
                count `n_ref` is the concrete converter's circuit property, not
                a base-level contract.
                Shape: `[..., n_ref]`.
            active_bits: Active conversion resolution in `[1, bits]`.

        Returns:
            Unsigned integer code tensor, one code per `i_in__uA` element, in
            the range `unsigned_range` reports for `active_bits`. For a deterministic
            converter the code at `active_bits` is the full-width code
            right-shifted by `bits - active_bits`. Dynamic energy is emitted
            through the profiler side channel.

        Raises:
            ValueError: `active_bits` is outside `[1, bits]`.
        """
        self._check_active_bits(active_bits)
        code = self._convert_impl(i_in__uA, i_refs__uA, active_bits=active_bits)
        if AdcProber.active():
            AdcProber.submit(IadcRecord(i_in__uA=i_in__uA))
        return code

    @abstractmethod
    def _convert_impl(
        self,
        i_in__uA: Tensor,
        i_refs__uA: Tensor,
        *,
        active_bits: int,
    ) -> Tensor:
        """Convert inputs according to the `convert` contract."""
        raise NotImplementedError

    @final
    def unsigned_range(self, active_bits: int) -> tuple[int, int]:
        """Return `(min_code, max_code)` the ADC can emit at `active_bits`.

        Every current ADC emits the family's full unsigned active-bit range.
        """
        self._check_active_bits(active_bits)
        return 0, (1 << active_bits) - 1
