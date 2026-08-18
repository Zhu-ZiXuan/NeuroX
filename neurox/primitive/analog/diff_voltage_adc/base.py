"""Abstract base class for differential voltage-domain ADC models.

See Also:
    docs/reference/primitive/analog/diff_voltage_adc/family.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from neurox.common import RecordBase, RecorderBase, RegistryMixin
from neurox.primitive.analog.base import AnalogBase, AnalogConfig, AnalogPolicy


class DiffVadcRecord(RecordBase):
    v_pos__V: Tensor
    """Positive-side input voltage the call was handed.
    Shape: `[...]`."""
    v_neg__V: Tensor
    """Negative-side input voltage the call was handed.
    Shape: `[...]`."""
    code: Tensor
    """Raw unsigned integer code the call returned.
    Shape: `[...]`."""
    bits: int
    """Resolution the conversion executed."""


class DiffVadcProber(RecorderBase[DiffVadcRecord]):
    """Capture differential-voltage ADC conversion records."""


class DiffVadcConfig(AnalogConfig, ABC):
    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class DiffVadcPolicy(AnalogPolicy, ABC):
    pass


class DiffVadc[ConfigT: DiffVadcConfig, PolicyT: DiffVadcPolicy](
    AnalogBase[ConfigT, PolicyT],
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

    A member that rounds stochastically gates the draw on `self.training`, the
    module's own train / eval state, rather than on a policy source or a
    constructor flag, so `eval()` is what makes any member deterministic.
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
    @abstractmethod
    def max_bits(self) -> int:
        """Physical bit width — the maximum `bits` value."""
        raise NotImplementedError

    @abstractmethod
    def latency__ns(self, *, bits: int) -> float:
        """Duration of one `convert` call at `bits` [ns].

        Args:
            bits: Active conversion resolution [bits].
        """
        raise NotImplementedError

    def convert(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        v_refs__V: Tensor,
        bits: int,
    ) -> Tensor:
        """Digitise a differential analog voltage into a raw unsigned code.

        Args:
            v_pos__V: Positive-side analog input voltage.
                Shape: `[...]`.
            v_neg__V: Negative-side analog input voltage, at the same shape as
                `v_pos__V`.
                Shape: `[...]`.
            v_refs__V: Injected reference taps, with the taps on the last axis.
                The tap count `n_ref` is the concrete converter's circuit
                property, not a base-level contract.
                Shape: `[..., n_ref]`.
            bits: Active conversion resolution [bits].

        Returns:
            Raw unsigned integer code tensor, one code per `v_pos__V` element,
            in the range `unsigned_range` reports for `bits`. For offset-binary
            codes, recover the signed value as
            `(code - zero_offset(bits)) · rescale_factor` with a positive
            `rescale_factor`.
            Shape: `[...]`.
        """
        code = self._convert_impl(
            v_pos__V,
            v_neg__V,
            v_refs__V=v_refs__V,
            bits=bits,
        )
        if DiffVadcProber.active():
            DiffVadcProber.submit(
                DiffVadcRecord(
                    v_pos__V=v_pos__V,
                    v_neg__V=v_neg__V,
                    code=code,
                    bits=bits,
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
        bits: int,
    ) -> Tensor:
        """Convert inputs according to the `convert` contract."""
        raise NotImplementedError

    @abstractmethod
    def unsigned_range(self, bits: int) -> tuple[int, int]:
        """Return `(min_code, max_code)` the ADC can emit at `bits`.

        The code is raw (unsigned / offset-binary), so `min_code` is 0. For
        ADCs whose code count matches `2 ** bits` exactly this is
        `(0, 2 ** bits - 1)`; for ADCs whose code count is not a power of two
        the upper bound reflects the actual realisable code count.
        """
        raise NotImplementedError

    @abstractmethod
    def zero_offset(self, bits: int) -> int:
        """Return the raw code representing analog zero at `bits`.

        Subtract this offset before scaling:
        `(code - zero_offset(bits)) · rescale_factor`. Sign and offset are not
        folded into the emitted code. For a symmetric power-of-two design this
        is `2 ** (bits - 1)`.
        """
        raise NotImplementedError
