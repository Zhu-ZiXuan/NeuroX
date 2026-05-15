"""Abstract base class and shared types for the readout-chain family.

A ``ReadOut`` bundles every block between the core's per-physical-
column OpAmpTIA output and the final ADC code into one fabricate /
readout lifecycle.  Concrete subclasses (today only
:class:`~neurox.analog.readout.offset_switchcap_mux_adc.OffsetSwitchCapMuxAdcReadOut`)
own the data / ref sample-and-hold banks, the differential transport
MUX, and the differential ADC.

The per-VMM kernel is exposed on the explicit :meth:`ReadOut.readout`
method rather than the conventional ``forward`` / ``__call__`` pair —
the readout is a non-trainable analog chain and never participates in
autograd, so the nn.Module hook plumbing buys nothing and we make the
call site read ``xbar_readout.readout(...)`` for clarity.

Design tenet — leaf modules own all electrical math
---------------------------------------------------
The readout abstraction enforces the project-wide invariant that
**all signal-value changes happen inside leaf circuit modules**; the
readout container only performs shape operations (``unflatten``,
``unsqueeze``, ``expand``) and aggregates energies.  In particular:

* the readout must never multiply voltages by a "weight" tensor
  outside a ``SwitchCap``;
* the readout must never re-derive the per-data weighted sum or ref
  baseline mathematically — those are properties of the SwitchCap
  kernels;
* the readout must never scale the ref leg by any
  encoding-specific geometric constant — the per-data negative leg
  is the ref ``SwitchCap`` output broadcast via
  ``unsqueeze + expand`` to the data lattice.

Upper-layer xbars (today
:class:`~neurox.xbar.Offset1T1RXbar`) are similarly
constrained: they may regroup the core's per-physical-column outputs
into the grouped readout lattice using shape operations only
(``index_select`` of physical columns into data / ref slots and
``unflatten`` to ``(group_num, data_num, digit_num)``), then hand
the readout the **semantically-labelled** grouped tensors.  Both
restrictions ensure the simulated chain matches the silicon chain
block-for-block.

Grouped lattice convention
--------------------------
The readout chain operates on a grouped lattice keyed by the
reference-group structure of the upstream encoding:

* ``*prefix`` — physical-instance prefix (xbar's
  ``w_phys.shape[:-2]``).
* ``*runtime`` — runtime tensor prefix (broadcast of ``*prefix``
  with the upstream activation batch dims).
* ``group_num`` — number of reference groups per physical array
  (one ref column per group).
* ``data_num`` — number of data per group (``ref_group_size``).
* ``digit_num`` — number of digits per data (``w_digit_count``).

``fabricate(shape, *, data_num, digit_weights)`` takes the readout
container's own virtual-instance shape
``shape == (*prefix, group_num)`` and the two **independent**
inputs needed to size its submodules: the integer ``data_num`` for
the data-side bank fan-out, and the 1-D ``digit_weights`` template
(length ``digit_num``) for the per-cap weight pattern.  The two
inputs are kept separate from ``shape`` per the project-wide
``fabricate`` contract — ``shape`` describes the readout's own
instance organisation and never carries information that belongs
to a sub-module's interior structure.  ``readout`` takes the
runtime-prefixed grouped voltages
``v_data_grouped: (*runtime, group_num, data_num, digit_num)`` and
``v_ref_grouped: (*runtime, group_num)`` and returns a
:class:`ReadOutOutput` whose tensors carry
``(*runtime, group_num, data_num)`` for per-data quantities and
``(*runtime, group_num)`` for the ref-bank energy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

from neurox.profiler import ProfiledModule

# ---------------------------------------------------------------------------
# Config (orchestrator-only knobs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class ReadOutConfig:
    """Orchestrator-level configuration for :class:`ReadOut`.

    SwitchCap physics live in :class:`SwitchCapConfig` (one each for
    the data and reference banks); AnalogMux and ADC physics each
    live in their own configs.  This dataclass only carries the
    top-level energy / PPA knobs the readout adds on top of its
    submodules — the readout's effective PPA aggregates these with
    every owned submodule's PPA at runtime (see
    :class:`OffsetSwitchCapMuxAdcReadOut`).

    Attributes:
        energy_per_op__fJ: Per-VMM orchestrator-level dynamic energy
            [fJ], distributed uniformly across the
            ``group_num * data_num`` data outputs so the per-data sum
            yields the configured total exactly.
        leakage_per_inst__uW: Static leakage added by the
            orchestrator-level logic [uW] (does not include the
            submodules' leakage — those are summed in at runtime).
        area_per_inst__um2: Silicon area added by the orchestrator
            [um^2] (same exclusion as leakage).
        latency_per_op__ns: Fixed orchestrator-level pipeline latency
            [ns] — control / handshaking only.  The readout's full
            per-VMM latency aggregates this with the data S/H, ref
            S/H, MUX and (bit-dependent) ADC latencies; see
            :meth:`ReadOut.latency_per_op__ns`.
    """

    energy_per_op__fJ: float
    leakage_per_inst__uW: float
    area_per_inst__um2: float
    latency_per_op__ns: float

    def __post_init__(self) -> None:
        if self.energy_per_op__fJ < 0.0:
            raise ValueError(f"require: energy_per_op__fJ ({self.energy_per_op__fJ}) >= 0")
        if self.leakage_per_inst__uW < 0.0:
            raise ValueError(f"require: leakage_per_inst__uW ({self.leakage_per_inst__uW}) >= 0")
        if self.area_per_inst__um2 < 0.0:
            raise ValueError(f"require: area_per_inst__um2 ({self.area_per_inst__um2}) >= 0")
        if self.latency_per_op__ns < 0.0:
            raise ValueError(f"require: latency_per_op__ns ({self.latency_per_op__ns}) >= 0")


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadOutOutput:
    """Per-VMM readout-chain result keyed by the grouped lattice.

    Per-leg energy fields were removed in the Phase D cleanup — every
    physical leaf inside the chain emits its own profiler event so
    aggregated energy flows through the side channel rather than the
    numerical return path.  All tensors carry the runtime prefix
    ``*runtime`` — the broadcast of the fabricated physical-instance
    prefix ``*prefix`` with the upstream activation batch dims.

    Attributes:
        code: Integer ADC code per data, with the grouped lattice
            preserved.  Shape ``(*runtime, group_num, data_num)``.
        v_pos__V: Post-MUX positive-leg voltage fed to the ADC.
            Shape ``(*runtime, group_num, data_num)``.
        v_neg__V: Post-MUX negative-leg voltage fed to the ADC.
            Shape ``(*runtime, group_num, data_num)``.
    """

    code: Tensor
    v_pos__V: Tensor
    v_neg__V: Tensor


# ---------------------------------------------------------------------------
# ReadOut ABC
# ---------------------------------------------------------------------------


class ReadOut(nn.Module, ProfiledModule, ABC):
    """Abstract base class for the xbar voltage-domain readout chain.

    Concrete subclasses bundle every block between the per-physical-
    column OpAmpTIA output and the final ADC code into one fabricate /
    readout lifecycle.  The contract pins the per-VMM API at
    ``readout(v_data_grouped, v_ref_grouped, *, adc_mode, adc_bits)``
    and returns a :class:`ReadOutOutput`.  Encoding-specific regrouping (e.g.
    data-major-digit-minor → grouped
    ``(group_num, data_num, digit_num)``) is the *upper xbar's*
    responsibility; the readout itself only sees grouped,
    semantically-labelled tensors and may not perform arithmetic
    outside its owned leaf circuit modules.

    Subclasses must implement:

    * ``fabricate(shape, *, data_num, digit_weights)`` — set up
      every owned submodule's static state.  ``shape`` carries the
      readout container's own virtual-instance organisation
      ``(*prefix, group_num)``; ``data_num`` and ``digit_weights``
      are the independent inputs the data-side bank fans out into.
    * ``readout(v_data_grouped, v_ref_grouped, *, adc_mode,
      adc_bits)`` — the per-VMM kernel returning
      :class:`ReadOutOutput`.
    * Aggregated PPA properties (``area_per_inst__um2``,
      ``leakage_per_inst__uW``) and the bit-dependent latency
      method ``latency_per_op__ns(*, adc_bits)``.
    """

    def __init__(self, *, name: str = "") -> None:
        nn.Module.__init__(self)
        ProfiledModule.__init__(self, name)

    @abstractmethod
    def fabricate(
        self,
        shape: tuple[int, ...],
        *,
        data_num: int,
        digit_weights: Tensor,
    ) -> None:
        """Sample static per-instance state across the grouped lattice.

        Args:
            shape: Readout container's own virtual-instance shape
                ``(*prefix, group_num)``.  ``*prefix`` is the
                physical-instance prefix (typically
                ``w_phys.shape[:-2]`` from the upper xbar);
                ``group_num`` is the number of reference groups per
                array.  Must have at least one trailing entry.
            data_num: Number of data per reference group
                (``ref_group_size`` upstream).  Passed independently
                — it sizes the data-side bank's fan-out, not the
                readout's own instance organisation.
            digit_weights: Per-digit weight vector of shape
                ``[digit_num]`` supplied by the encoding-specific
                upper xbar (e.g.
                ``[radix^0, radix^1, ..., radix^(digit_num - 1)]``
                for offset coding).  The readout broadcasts it to
                the ``(*prefix, group_num, data_num, digit_num)``
                data-side cap-ratio tensor internally — the caller
                never sees the high-dim form.
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
    ) -> ReadOutOutput:
        """Run one VMM through the readout chain.

        Args:
            v_data_grouped__V: Per-data per-digit OpAmpTIA output voltages
                [V], regrouped by the upper xbar.  Shape
                ``(*runtime, group_num, data_num, digit_num)``.
            v_ref_grouped__V: Per-reference-column OpAmpTIA output
                voltages [V].  Shape ``(*runtime, group_num)``.
            adc_mode: Runtime ADC operating-point index.
            adc_bits: Runtime ADC bit width.

        Returns:
            :class:`ReadOutOutput`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def area_per_inst__um2(self) -> float:
        """Aggregated silicon area per readout instance [um^2] — orchestrator + every owned submodule."""
        raise NotImplementedError

    @property
    @abstractmethod
    def leakage_per_inst__uW(self) -> float:
        """Aggregated static leakage per readout instance [uW] — orchestrator + every owned submodule."""
        raise NotImplementedError

    @abstractmethod
    def latency_per_op__ns(self, *, adc_bits: int) -> float:
        """Aggregated per-VMM pipeline latency [ns] at the active bit width.

        Unlike ``area`` / ``leakage`` this is a method, not a
        property, because the ADC's latency depends on the active
        ``adc_bits``.  The return aggregates orchestrator + data /
        ref S/H + MUX + ADC latencies for the requested operating
        point.

        Args:
            adc_bits: Active ADC bit width forwarded to the bound
                ADC's :meth:`latency_per_op__ns`.
        """
        raise NotImplementedError
