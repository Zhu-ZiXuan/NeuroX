"""Abstract base class for readout chains.

See also:
    docs/reference/xbar/readout/README.md
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.analog.adc import AdcOperationPoint
from neurox.common.circuit import CircuitBase, CircuitConfig
from neurox.common.mixin import RegistryMixin

# ---------------------------------------------------------------------------
# Config (orchestrator-only knobs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class ReadOutConfig(CircuitConfig):
    """Top-level readout configuration.

    ``energy_per_op__fJ`` / ``latency_per_op__ns`` describe the readout
    chain's own orch overhead per VMM, not the sum of its children —
    children that emit their own profile events log independently;
    children without a dynamic model contribute nothing extra to the
    event stream.

    Attributes:
        energy_per_op__fJ: Readout-local dynamic overhead per operation [fJ].
        latency_per_op__ns: Readout-local per-VMM orch latency [ns];
            multiplied by the runtime serial-op count at logging time.
    """

    energy_per_op__fJ: float
    latency_per_op__ns: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_ppa()

    def validate_ppa(self) -> None:
        super().validate_ppa()
        self._require_nonneg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_nonneg(self.latency_per_op__ns, "latency_per_op__ns")


@dataclass(frozen=True)
class ReadOutPolicy:
    """Abstract marker base for ReadOut-family nonideality policies."""


# ---------------------------------------------------------------------------
# ReadOut ABC
# ---------------------------------------------------------------------------


class ReadOut(CircuitBase[ReadOutConfig], RegistryMixin[type["ReadOutConfig"], "ReadOut"]):
    """Abstract base class for voltage-domain readout chains.

    Per-VMM total latency is composed at the profiler level: children
    that emit their own profile events (e.g. the inner ADC, switch-cap
    banks) call ``_log_latency`` / ``_log_dynamic_energy`` independently;
    the readout itself emits only the orch / glue contribution from
    ``self.config.latency_per_op__ns``, gated independently of its
    energy overhead.
    """

    @classmethod
    def from_config(
        cls,
        *,
        config: ReadOutConfig,
        policy: ReadOutPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        data_num: int,
        digit_weights: tuple[float, ...],
    ) -> ReadOut:
        """Build the concrete impl registered for ``type(config)``.

        Args:
            data_num: Number of data per reference group.
            digit_weights: Per-digit weight vector, length ``digit_num``.
        """
        impl = cls._lookup_impl(type(config))
        return impl(
            config=config,
            policy=policy,
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
        config: ReadOutConfig,
        policy: ReadOutPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        data_num: int,
        digit_weights: tuple[float, ...],
    ) -> None:
        """Register the instance with :class:`nn.Module` and the profiler.

        Args:
            config: Concrete configuration dataclass.
            policy: Composite nonideality policy.
            name: Hierarchical instance name used by the profiler.
            inst_shape: Per-instance fabrication shape ``(*prefix, group_num)``.
            dtype: Tensor dtype for internal buffers.
            T__K: Operating temperature [K].
            data_num: Number of data per reference group.
            digit_weights: Per-digit weight vector, length ``digit_num``.
        """
        del policy, dtype, T__K, data_num, digit_weights  # captured by the subclass init
        super().__init__(config=config, name=name, inst_shape=inst_shape)

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
    def adc_mode_num(self) -> int:
        """Number of supported ADC operating points; valid ``adc_mode`` values are ``[0, mode_num)``."""
        raise NotImplementedError

    @property
    @abstractmethod
    def adc_max_bits(self) -> int:
        """Maximum supported ``adc_bits`` value."""
        raise NotImplementedError
