"""Abstract base class and shared utilities for the ADC family.

Every concrete NeuroX ADC subclass (``GeneralADC``, ``McsSarAdc``,
``PipelineADC``, ``CyclicADC``, ``RampADC``) ships its own
physics-based energy and latency model and
plugs into the same :class:`ADC` interface so the xbar can hold any
of them through one handle.

Design tenets
-------------

* **Pure converter** — an ADC is digitisation only.  BL clamp voltage
  and current-to-voltage conversion live on the upstream
  :class:`neurox.analog.opamp_tia.OpAmpTIA`; column multiplexing on
  :class:`neurox.analog.analog_mux.AnalogMux`; row drive on
  :class:`neurox.analog.decoder.Decoder` + DAC.

* **Runtime multi-mode** — a real chip's ADC exposes several
  ``(reference, precision)`` operating points selected at runtime.
  The base API pins ``mode`` and ``bits`` as **per-call** kwargs on
  :meth:`convert` and :meth:`latency_per_op__ns`.  Single-mode
  subclasses honour the contract by validating ``mode == 0`` and
  ``bits == max_bits`` and ignoring the values otherwise.

* **Floor semantics** — boundaries are placed at ``c · LSB`` (code
  edges), not at ``(c - 0.5) · LSB`` (midpoints).  Stochastic
  rounding adds ``uniform(0, LSB)`` jitter before the floor and is
  unbiased.

* **No driver** — the legacy ``GeneralADC.drive()`` (BL-clamp
  reference voltage) was a leak of OpAmpTIA functionality into the ADC.
  New subclasses do not implement it.  ``GeneralADC`` keeps it for
  backward compatibility but xbars built against the new pipeline
  read the bias voltage from the OpAmpTIA instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

from neurox.profiler import ProfiledModule


@dataclass(frozen=True)
class ADCMode:
    """One precision/range operating mode of a multi-mode ADC.

    Real silicon ADCs share a single physical capacitor / comparator
    array but expose several configurable ``(n_bits, range)`` modes.
    NeuroX models this directly: the highest-precision mode pins the
    physical comparator thresholds; lower-precision modes use a
    sub-sampled subset.

    Attributes:
        n_bits: Number of output code bits — defines the active
            comparator depth.  ``2 ** n_bits`` codes.  Must be
            ``>= 1`` and ``<= n_bits`` of the highest-precision mode
            on the same ADC.
        n_states: Number of distinct ideal-integer states the
            measurement range is calibrated to cover.  Must be
            ``<= 2 ** n_bits``.
        max_signal: Upper edge of the measurement range, in input
            units (uA for current-mode topologies, V for voltage-mode).
            Below the highest-precision mode this typically shrinks
            in lockstep with ``n_states``.
    """

    n_bits: int
    n_states: int
    max_signal: float

    def __post_init__(self) -> None:
        if self.n_bits < 1:
            raise ValueError(f"ADCMode.n_bits ({self.n_bits}) must be >= 1")
        if self.n_states < 2:
            raise ValueError(f"ADCMode.n_states ({self.n_states}) must be >= 2")
        if self.n_states > (1 << self.n_bits):
            raise ValueError(f"ADCMode.n_states ({self.n_states}) exceeds 2**n_bits ({1 << self.n_bits})")
        if not (self.max_signal > 0.0):
            raise ValueError(f"ADCMode.max_signal ({self.max_signal}) must be > 0")

    @property
    def n_codes(self) -> int:
        """Number of distinct output codes — ``2 ** n_bits``."""
        return 1 << self.n_bits

    @property
    def lsb(self) -> float:
        """Bin width — ``max_signal / n_codes``."""
        return self.max_signal / self.n_codes


class ADC(nn.Module, ProfiledModule, ABC):
    """Abstract base class for every NeuroX ADC.

    The ABC is **runtime multi-mode**: ``mode`` selects the operating
    point (reference / range) and ``bits`` selects the active bit
    width.  Single-mode subclasses honour the contract by validating
    ``mode == 0`` and ``bits == max_bits`` and ignoring the values
    otherwise.

    Concrete subclasses must implement:

    * ``convert(v_pos__V, v_neg__V, *, mode, bits) -> (code, energy)``
      — the digitisation kernel.  Code dtype is implementation-
      specific (``int16`` for boundary-bucketize variants,
      ``int32`` for SAR); energy is float and shape matches ``code``.
    * ``latency_per_op__ns(*, bits) -> float`` — per-conversion
      latency at the active bit width.  ``mode`` is intentionally
      absent: every modelled topology has V_ref-independent latency
      (mode shifts the reference, not the SAR loop length or the
      flash settling time).
    * Cost-metric properties: ``area_per_inst__um2``,
      ``leakage_per_inst__uW``.

    Note: the ADC layer **does not** expose an analog-domain
    rescale factor.  Mapping ADC codes back to ideal-integer scale
    is a macro-level concern (see
    :meth:`neurox.xbar.base.Xbar.output_rescale_factor`); the ADC
    only digitises voltages.
    """

    def __init__(self, *, name: str = "") -> None:
        nn.Module.__init__(self)
        ProfiledModule.__init__(self, name)

    @abstractmethod
    def convert(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        mode: int,
        bits: int,
    ) -> Tensor:
        """Digitise a differential analog voltage into an integer code.

        Args:
            v_pos__V: Positive-side analog input voltage [V].  Shape:
                arbitrary.
            v_neg__V: Negative-side analog input voltage [V].  Same
                shape as ``v_pos__V``.
            mode: Runtime operating-point index.  ``[0, n_modes)``.
            bits: Active bit width for this conversion.

        Returns:
            Integer code tensor in ``[0, 2 ** bits - 1]``, same shape
            as ``v_pos__V``.  Dynamic energy and latency are emitted
            through the profiler side channel.
        """
        raise NotImplementedError

    def fabricate(self, shape: tuple[int, ...]) -> None:
        """Sample static per-instance non-idealities and store them.

        Default no-op — single-mode behavioural ADCs (``GeneralADC``,
        Pipeline, Cyclic, Ramp) carry no static state in the current
        model.  SAR variants override to register their per-instance
        capacitor mismatch and comparator offset, sized over the
        requested per-instance prefix ``shape`` (``()`` for a scalar
        instance).

        Args:
            shape: Physical-instance prefix shape ``*P`` for any
                per-instance static mismatch tensors.  Behavioural
                ADCs ignore this; SAR variants use it to size their
                cap-mismatch / comparator-offset buffers.
        """
        return

    @abstractmethod
    def latency_per_op__ns(self, *, bits: int) -> float:
        """Per-conversion latency in [ns] at the active bit width.

        ``mode`` is intentionally not part of the signature: every
        modelled topology has V_ref-independent latency.

        Args:
            bits: Active bit width — required.

        Returns:
            Latency in [ns].
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def area_per_inst__um2(self) -> float:
        """Silicon area per ADC instance in [um^2]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def leakage_per_inst__uW(self) -> float:
        """Static leakage power per ADC instance in [uW]."""
        raise NotImplementedError
