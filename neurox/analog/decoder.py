"""Row decoder + WL driver wrapper.

A real chip's WL path is:

1. The decoder receives an N-bit row address.
2. The decoder asserts one of ``2^N`` row-select lines (or all of
   them in parallel for VMM operation).
3. The driver buffers the row-select to drive the WL load through
   the WL DAC's voltage levels.

Prior to this redesign the xbar wired the WL DAC directly with no
explicit decoder/driver entity, which left:

* Row-address energy unaccounted for.
* The bit-serial input mode (one DAC pulse per input bit) hard-
  baked into the mapper.

The :class:`Decoder` class consumes per-row integer codes from the
macro's Sa loop and (a) feeds them straight to the WL DAC for the
default parallel-multibit case or (b) expands each code into
``log2(N)`` bit-cycles when ``bit_serial`` is on.  Either way the
xbar sees a single ``(signal, energy)`` pair per call.

Latency / energy
----------------

Decoder gate-delay::

    t_op = (n_address_bits) · t_gate__ns

Energy per row access::

    E = (n_address_bits · C_gate__fF · V_DD²)            (fJ)
        + n_address_bits · driver_energy_per_row__fJ
        + E_overhead

When bit_serial is on, the macro multiplies through by the bit-cycle
count itself; we accumulate energy here per single drive call.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .dac import DAC


@dataclass(frozen=True)
class DecoderConfig:
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

    n_address_bits: int = 6
    fanout: int = 4

    drive_strength__uA: float = 1000.0

    bit_serial: bool = False

    c_gate__fF: float = 0.5
    v_dd__V: float = 1.0
    t_gate__ns: float = 0.05

    e_overhead__fJ: float = 0.0

    leakage_per_inst__uW: float = 0.0
    area_per_inst__um2: float = 0.0


class Decoder(nn.Module):
    """Row decoder + driver, wraps a WL DAC.

    Args:
        cfg: Topology configuration.
    """

    def __init__(self, cfg: DecoderConfig, *, name: str = "") -> None:
        super().__init__()
        if cfg.n_address_bits < 1:
            raise ValueError(f"Decoder n_address_bits ({cfg.n_address_bits}) must be >= 1")
        self._neurox_name = name
        self.cfg = cfg

        self._t_op__ns: float = cfg.n_address_bits * cfg.t_gate__ns
        # fF · V² = fJ — no scaling factor needed.
        self._e_per_call__fJ: float = cfg.n_address_bits * cfg.c_gate__fF * cfg.v_dd__V**2 + cfg.e_overhead__fJ

    @property
    def bit_serial(self) -> bool:
        """True when per-row codes are expanded into one-pulse-per-bit cycles."""
        return self.cfg.bit_serial

    @property
    def n_address_bits(self) -> int:
        return self.cfg.n_address_bits

    @property
    def area_per_inst__um2(self) -> float:
        return self.cfg.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        return self.cfg.leakage_per_inst__uW

    @property
    def latency_per_op__ns(self) -> float:
        return self._t_op__ns

    def fabricate(self, shape: tuple[int, ...]) -> None:
        """Default no-op — Decoder has no static fabrication state today.

        The unified ``fabricate(shape)`` signature is kept for
        lifecycle consistency with the other readout-chain modules
        (per ``temp/fabricate.md``).  A future iteration may introduce
        static per-row mismatch sized over ``shape``; until then the
        call has nothing to do.

        Args:
            shape: Reserved for future per-row static mismatch.
                Ignored today.
        """
        return

    def drive(self, x_int: Tensor, dac: DAC) -> Tensor:
        """Decode integer per-row codes and drive the DAC.

        Non-bit-serial mode (default): hands ``x_int`` straight to
        ``dac.convert(x_int)``.

        Bit-serial mode: expands ``x_int`` into ``n_bits`` codes via
        ``torch.bitwise_and`` and concatenates the new bit-cycle
        axis at ``dim=-3`` (the macro's existing Sa axis is
        preserved; bit-cycle is inserted as a finer cycling
        dimension).  The DAC sees codes in ``{0, 1}`` per bit.

        Decoder dynamic energy and the wrapped DAC's energy are
        emitted as profiler side-channel events.

        Args:
            x_int: Per-row integer codes from the macro's Sa loop.
                Shape: ``[..., row_size]`` (or with whatever leading
                axes the macro has).
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
        # Decoder's own per-call energy as a side-channel event.
        if self._e_per_call__fJ > 0.0:
            # ``getattr`` because Decoder is currently name-only (not
            # ``ProfiledModule``); when the project adds it the call
            # short-circuits cleanly.
            log = getattr(self, "_log_dynamic", None)
            if callable(log):
                log(self._e_per_call__fJ, self._t_op__ns)
        return signal
