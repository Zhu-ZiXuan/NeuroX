"""Abstract base class for readout chains.

See also:
    docs/dev/modules/analog/readout/README.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.registry_dispatch import RegistryDispatchMixin
from neurox.common.validate import ValidateMixin
from neurox.profiler import ProfiledModule

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


class ReadOut(nn.Module, ProfiledModule, RegistryDispatchMixin[type["ReadOutConfig"], "ReadOut"], ABC):
    """Abstract base class for voltage-domain readout chains."""

    @classmethod
    def from_config(
        cls,
        *,
        cfg: ReadOutConfig,
        name: str,
        T__K: float,
        dtype: torch.dtype,
        data_num: int,
        digit_weights: tuple[float, ...],
    ) -> ReadOut:
        """Build the concrete readout implementation for `type(cfg)`.

        Args:
            cfg: Readout configuration.
            name: Profiler/debug name.
            T__K: Operating temperature [K].
            dtype: Tensor dtype for internal buffers.
            data_num: Number of data per reference group.
            digit_weights: Per-digit weight vector, length ``digit_num``.

        Returns:
            Concrete readout implementation registered for `type(cfg)`.
        """
        impl = cls._lookup_impl(type(cfg))
        return impl(
            cfg=cfg,
            name=name,
            T__K=T__K,
            dtype=dtype,
            data_num=data_num,
            digit_weights=digit_weights,
        )

    def __init__(
        self,
        *,
        cfg: ReadOutConfig,
        name: str,
        T__K: float,
        dtype: torch.dtype,
        data_num: int,
        digit_weights: tuple[float, ...],
    ) -> None:
        """Register the instance with :class:`nn.Module` and the profiler.

        Args:
            cfg: Readout configuration.
            name: Profiler/debug name.
            T__K: Operating temperature [K].
            dtype: Tensor dtype for internal buffers.
            data_num: Number of data per reference group.
            digit_weights: Per-digit weight vector, length ``digit_num``.
        """
        del cfg, T__K, dtype, data_num, digit_weights  # captured by the subclass init
        nn.Module.__init__(self)
        ProfiledModule.__init__(self, name)

    @abstractmethod
    def fabricate(self, shape: tuple[int, ...]) -> None:
        """Sample static per-instance state over ``shape`` (re-callable).

        Args:
            shape: Per-instance fabrication shape.
        """
        raise NotImplementedError

    @abstractmethod
    def readout(
        self,
        v_data_grouped__V: Tensor,
        v_ref_grouped__V: Tensor,
        *,
        adc_mode: int,
        adc_bits: int,
    ) -> Tensor:
        """Run one VMM through the readout chain.

        Args:
            v_data_grouped__V: Data-path voltages [V]. Shape:
                [..., group_num, data_num, digit_num].
            v_ref_grouped__V: Reference-path voltages [V]. Shape:
                [..., group_num].
            adc_mode: Runtime ADC operating-point index.
            adc_bits: Runtime ADC bit width.

        Returns:
            Integer ADC code tensor.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def area_per_inst__um2(self) -> float:
        """Aggregated silicon area per readout instance [um^2]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def leakage_per_inst__uW(self) -> float:
        """Aggregated static leakage per readout instance [uW]."""
        raise NotImplementedError

    @abstractmethod
    def latency_per_op__ns(self, *, adc_bits: int) -> float:
        """Return the per-VMM pipeline latency [ns].

        Args:
            adc_bits: Runtime ADC bit width.
        """
        raise NotImplementedError
