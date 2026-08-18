"""Abstract base class for single-ended current-domain ADC models.

See Also:
    docs/reference/primitive/analog/current_adc/family.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from neurox.common import RecordBase, RecorderBase, RegistryMixin
from neurox.primitive.analog.base import AnalogBase, AnalogConfig, AnalogPolicy


class IadcRecord(RecordBase):
    i_in__uA: Tensor
    """Input magnitude current the call was handed.
    Shape: `[...]`."""
    code: Tensor
    """Unsigned integer code the call returned.
    Shape: `[...]`."""
    bits: int
    """Resolution the conversion executed."""


class IadcProber(RecorderBase[IadcRecord]):
    """Capture current-ADC conversion records."""


class IadcConfig(AnalogConfig, ABC):
    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class IadcPolicy(AnalogPolicy, ABC):
    pass


class Iadc[ConfigT: IadcConfig, PolicyT: IadcPolicy](
    AnalogBase[ConfigT, PolicyT],
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
    @abstractmethod
    def max_bits(self) -> int:
        """Physical bit width — the maximum `bits` a `convert` call may request."""
        raise NotImplementedError

    def _check_bits(self, bits: int) -> None:
        """Require a resolution this converter's own bit width supports.

        Raises:
            ValueError: `bits` is outside `[1, max_bits]`.
        """
        if not (1 <= bits <= self.max_bits):
            raise ValueError(f"require: bits ({bits}) in [1, max_bits ({self.max_bits})]")

    @abstractmethod
    def latency__ns(self, *, bits: int) -> float:
        """Duration of one `convert` call at `bits` [ns].

        Args:
            bits: Conversion resolution [bits] in `[1, max_bits]`.
        """
        raise NotImplementedError

    def convert(
        self,
        i_in__uA: Tensor,
        i_refs__uA: Tensor,
        *,
        bits: int,
    ) -> Tensor:
        """Digitise a single-ended magnitude current into an unsigned integer code.

        Bit width is ADC-internal: the injected ladder states the converter's
        own wiring and does not follow the requested resolution.

        Args:
            i_in__uA: Non-negative magnitude current.
                Shape: `[...]`.
            i_refs__uA: Reference ladder with the taps on the last axis and the
                leading dims right-broadcasting against `i_in__uA`. The tap
                count `n_ref` is the concrete converter's circuit property, not
                a base-level contract.
                Shape: `[..., n_ref]`.
            bits: Conversion resolution [bits] in `[1, max_bits]`.

        Returns:
            Unsigned integer code tensor, one code per `i_in__uA` element, in
            the range `unsigned_range` reports for `bits`. For a deterministic
            converter the code at `bits` is the code at `max_bits`
            right-shifted by `max_bits - bits`. Dynamic energy is emitted
            through the profiler side channel.
            Shape: `[...]`.

        Raises:
            ValueError: `bits` is outside `[1, max_bits]`.
        """
        self._check_bits(bits)
        code = self._convert_impl(i_in__uA, i_refs__uA, bits=bits)
        if IadcProber.active():
            IadcProber.submit(
                IadcRecord(
                    i_in__uA=i_in__uA,
                    code=code,
                    bits=bits,
                ),
            )
        return code

    @abstractmethod
    def _convert_impl(
        self,
        i_in__uA: Tensor,
        i_refs__uA: Tensor,
        *,
        bits: int,
    ) -> Tensor:
        """Convert inputs according to the `convert` contract."""
        raise NotImplementedError

    @abstractmethod
    def unsigned_range(self, bits: int) -> tuple[int, int]:
        """Return `(min_code, max_code)` the ADC can emit at `bits`.

        For ADCs whose code count matches `2 ** bits` exactly, this is
        `(0, 2 ** bits - 1)`.
        """
        raise NotImplementedError
