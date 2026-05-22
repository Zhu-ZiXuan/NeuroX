"""Row decoder + WL driver wrapper.

See also:
    docs/dev/modules/analog/decoder.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.mixin import FabricateMixin, ProfileMixin, ValidateMixin

from .dac import DAC


@dataclass(frozen=True)
class DecoderConfig(ValidateMixin):
    """Immutable configuration for :class:`Decoder`.

    Attributes:
        n_address_bits: Row-address width = ``log2(numRow)``.
        fanout: Driver fanout (rows per buffer leg).  Affects the
            driver energy term but not the signal value.
        drive_strength__uA: Output drive current per row.  Used for
            the driver-energy estimate.
        bit_serial: When True, per-row codes are expanded into
            bit-cycles (one drive pulse per bit).  When False
            (default), the per-row code is forwarded straight to the
            DAC.
        c_gate__fF: Per-stage gate capacitance.
        v_dd__V: Decoder logic supply.
        t_gate__ns: Per-stage gate delay.
        e_overhead__fJ: Constant per-call overhead.
        leakage_per_inst__uW: Static leakage per instance.
        area_per_inst__um2: Silicon area per instance.
    """

    n_address_bits: int
    fanout: int

    drive_strength__uA: float

    bit_serial: bool

    c_gate__fF: float
    v_dd__V: float
    t_gate__ns: float

    e_overhead__fJ: float

    leakage_per_inst__uW: float
    area_per_inst__um2: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_address()
        self.validate_drive()
        self.validate_energy()
        self.validate_ppa()

    def validate_address(self) -> None:
        self._require_pos(self.n_address_bits, "n_address_bits")
        self._require_pos(self.fanout, "fanout")

    def validate_drive(self) -> None:
        self._require_nonneg(self.drive_strength__uA, "drive_strength__uA")

    def validate_energy(self) -> None:
        self._require_nonneg(self.c_gate__fF, "c_gate__fF")
        self._require_nonneg(self.v_dd__V, "v_dd__V")
        self._require_nonneg(self.t_gate__ns, "t_gate__ns")
        self._require_nonneg(self.e_overhead__fJ, "e_overhead__fJ")

    def validate_ppa(self) -> None:
        self._require_nonneg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_nonneg(self.area_per_inst__um2, "area_per_inst__um2")


class Decoder(FabricateMixin, nn.Module, ProfileMixin):
    """Row decoder + driver, wraps a WL DAC.

    Args:
        cfg: Concrete configuration dataclass.
        name: Hierarchical instance name used by the profiler.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature [K].
    """

    def __init__(
        self,
        *,
        cfg: DecoderConfig,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        nn.Module.__init__(self)
        ProfileMixin.__init__(self, name)
        if cfg.n_address_bits < 1:
            raise ValueError(f"Decoder n_address_bits ({cfg.n_address_bits}) must be >= 1")
        self.cfg = cfg
        self._inst_shape = inst_shape
        self.dtype = dtype
        self.T__K = T__K

        self._t_op__ns = cfg.n_address_bits * cfg.t_gate__ns
        # fF · V² = fJ — no scaling factor needed.
        self._e_per_call__fJ = cfg.n_address_bits * cfg.c_gate__fF * cfg.v_dd__V**2 + cfg.e_overhead__fJ

        self._log_static()

    @property
    def bit_serial(self) -> bool:
        """True when per-row codes are expanded into one-pulse-per-bit cycles."""
        return self.cfg.bit_serial

    @property
    def n_address_bits(self) -> int:
        return self.cfg.n_address_bits

    @property
    def area_per_inst__um2(self) -> float:
        """Silicon area per instance [um^2]."""
        return self.cfg.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        """Static leakage per instance [uW]."""
        return self.cfg.leakage_per_inst__uW

    @property
    def latency_per_op__ns(self) -> float:
        """Latency per op [ns]."""
        return self._t_op__ns

    def drive(self, x_int: Tensor, dac: DAC) -> Tensor:
        """Decode integer per-row codes and drive the DAC.

        Bit-serial mode inserts a new bit-cycle axis at ``dim=-3``,
        one pulse per bit.

        Args:
            x_int: Per-row integer codes, shape ``[..., row_size]``.
            dac: WL DAC the decoder feeds.

        Returns:
            Analog WL drive signal from ``dac.convert``.
        """
        if self.cfg.bit_serial:
            n_bits = max(self.cfg.n_address_bits, 1)
            bit_planes = []
            for k in range(n_bits):
                bit = (x_int >> k) & 1
                bit_planes.append(bit)
            x_int = torch.stack(bit_planes, dim=-2)

        signal = dac.convert(x_int)
        if self._e_per_call__fJ > 0.0:
            self._log_dynamic(self._e_per_call__fJ, self._t_op__ns)
        return signal
