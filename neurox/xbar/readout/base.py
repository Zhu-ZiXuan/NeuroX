"""Abstract base class for readout chains.

See also:
    docs/dev/modules/xbar/readout/README.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.analog.adc import AdcOperationPoint
from neurox.common.mixin import FabricateMixin, ProfileMixin, RegistryMixin, ValidateMixin

# ---------------------------------------------------------------------------
# Config (orchestrator-only knobs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class ReadOutConfig(ValidateMixin):
    """Top-level readout configuration.

    Attributes:
        energy_per_op__fJ: Readout-local dynamic overhead per operation [fJ].
        leakage_per_inst__uW: Leakage power per instance [uW].
        area_per_inst__um2: Area per instance [um^2].
        latency_per_op__ns: Fixed latency contribution per operation [ns].
    """

    energy_per_op__fJ: float
    leakage_per_inst__uW: float
    area_per_inst__um2: float
    latency_per_op__ns: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_ppa()

    def validate_ppa(self) -> None:
        self._require_nonneg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_nonneg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_nonneg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_nonneg(self.latency_per_op__ns, "latency_per_op__ns")


# ---------------------------------------------------------------------------
# ReadOut ABC
# ---------------------------------------------------------------------------


class ReadOut(FabricateMixin, nn.Module, ProfileMixin, RegistryMixin[type["ReadOutConfig"], "ReadOut"], ABC):
    """Abstract base class for voltage-domain readout chains."""

    @classmethod
    def from_config(
        cls,
        *,
        cfg: ReadOutConfig,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        data_num: int,
        digit_weights: tuple[float, ...],
    ) -> ReadOut:
        """Build the concrete impl registered for ``type(cfg)``.

        Args:
            data_num: Number of data per reference group.
            digit_weights: Per-digit weight vector, length ``digit_num``.
        """
        impl = cls._lookup_impl(type(cfg))
        return impl(
            cfg=cfg,
            name=name,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
            data_num=data_num,
            digit_weights=digit_weights,
        )

    def __init__(
        self,
        *,
        cfg: ReadOutConfig,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        data_num: int,
        digit_weights: tuple[float, ...],
    ) -> None:
        """Register the instance with :class:`nn.Module` and the profiler.

        Args:
            cfg: Concrete configuration dataclass.
            name: Hierarchical instance name used by the profiler.
            inst_shape: Per-instance fabrication shape ``(*prefix, group_num)``.
            dtype: Tensor dtype for internal buffers.
            T__K: Operating temperature [K].
            data_num: Number of data per reference group.
            digit_weights: Per-digit weight vector, length ``digit_num``.
        """
        del cfg, dtype, T__K, data_num, digit_weights  # captured by the subclass init
        nn.Module.__init__(self)
        ProfileMixin.__init__(self, name)
        self._inst_shape = inst_shape

    @abstractmethod
    def readout(
        self,
        v_data_grouped__V: Tensor,
        v_ref_grouped__V: Tensor,
        *,
        adc_operation_point: AdcOperationPoint,
    ) -> Tensor:
        """Run one VMM through the readout chain.

        Args:
            v_data_grouped__V: Data-path voltages [V]. Shape:
                [..., group_num, data_num, digit_num].
            v_ref_grouped__V: Reference-path voltages [V]. Shape:
                [..., group_num].
            adc_operation_point: Runtime ADC operating point.

        Returns:
            Integer ADC code tensor.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def area_per_inst__um2(self) -> float:
        """Silicon area per instance [um^2]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def leakage_per_inst__uW(self) -> float:
        """Static leakage per instance [uW]."""
        raise NotImplementedError

    @abstractmethod
    def latency_per_op__ns(self, *, adc_operation_point: AdcOperationPoint) -> float:
        """Return the per-VMM pipeline latency [ns].

        Args:
            adc_operation_point: Runtime ADC operating point.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def adc_mode_num(self) -> int:
        """Number of supported ADC operating points; valid ``adc_mode`` values are ``[0, mode_num)``."""
        raise NotImplementedError

    @property
    @abstractmethod
    def adc_max_bits(self) -> int:
        """Maximum supported ``adc_bits`` value."""
        raise NotImplementedError
