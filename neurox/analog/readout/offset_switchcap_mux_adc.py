"""Offset-coded readout chain — data S/H + ref S/H + MUX + ADC.

Bundles the four blocks of the offset-coding voltage-domain readout
into a single fabricate / readout lifecycle.  The chain operates on
the grouped lattice produced by the upper xbar (see
:class:`~neurox.analog.readout.base.ReadOut`):

1. ``data_switchcap`` — bottom-plate-sampled cap bank, bank shape
   ``(*prefix, group_num, data_num, digit_num)`` with one bank per
   data and ``digit_num`` weighted caps per bank.  Output is the
   passive charge-share weighted average ``Σ_k C_k V_k / Σ_k C_k``
   per data — the positive ADC leg, shape
   ``(*runtime, group_num, data_num)``.
2. ``ref_switchcap`` — bottom-plate-sampled cap bank, bank shape
   ``(*prefix, group_num, 1)`` with a single unit-weight cap per ref
   column.  With one cap the passive charge-share output equals the
   sampled reference voltage itself (within kT/C and per-cap
   mismatch), shape ``(*runtime, group_num)``.
3. ``analog_mux`` — differential transport with optional gain
   attenuation and CM / DM noise.
4. ``bl_adc`` — differential ADC at the runtime
   ``(adc_mode, adc_bits)`` operating point.

Critical: no algebraic ref scaling, no per-data lookup
------------------------------------------------------
The negative ADC leg is built **purely** by broadcasting the
``ref_switchcap`` output across the per-group data axis::

    v_neg = v_ref_sampled.unsqueeze(-1).expand(*, group_num, data_num)

There is **no** ``index_select`` / per-data lookup and **no**
geometric scaling.  The grouped layout of the inputs already places
every data alongside its matching ref on the same ``group_num`` axis,
so ``unsqueeze + expand`` is the only shape operation needed.
Rationale:

* ``data_switchcap.sample_and_accumulate`` returns the *normalised*
  weighted average ``Σ C_k V_k / Σ C_k`` (the physical bottom-plate
  passive charge-share, not an unnormalised sum).
* ``ref_switchcap`` with one cap returns the sampled ref voltage.
* If every data digit input equals the ref voltage, both legs equal
  the ref voltage and the differential signal is zero — the
  physically correct behaviour.

The offset-coding "digit weighting" is realised inside the
``data_switchcap`` via the per-cap ratios broadcast from
``digit_weights`` at fabricate time; it is **not** an algebraic
post-processing step.  Any "digit-weighted sum" interpretation lives
in the system-level scaling (the macro's ``output_rescale_factor``),
not in this block.

PPA aggregation
---------------
``area_per_inst__um2``, ``leakage_per_inst__uW`` and
``latency_per_op__ns(*, adc_bits)`` aggregate the orchestrator's own
contributions with every owned submodule, weighted by the **per-array
instance count** of each block on the grouped lattice:

* ``data_switchcap`` — ``group_num * data_num`` banks (one bank per
  data column);
* ``ref_switchcap`` — ``group_num`` banks (one bank per reference
  group / column);
* ``analog_mux`` — ``group_num`` instances (one MUX per group);
* ``bl_adc`` — ``group_num`` instances (one ADC per group).

These counts are bound by :meth:`OffsetSwitchCapMuxAdcReadOut.fabricate`
from the runtime lattice ``shape = (*prefix, group_num, data_num)``,
so PPA properties are only valid post-fabricate (the matching call-
site contract for the readout chain).  Aggregating here keeps the
macro PPA accounting consistent with the bundled readout abstraction
— the upper xbar treats the readout as one opaque block.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from neurox.analog.adc import ADC
from neurox.analog.analog_mux import AnalogMux
from neurox.analog.switch_cap import SwitchCap

from .base import ReadOut, ReadOutConfig, ReadOutOutput


class OffsetSwitchCapMuxAdcReadOut(ReadOut):
    """Offset-coded readout: data S/H+WA + ref S/H + AnalogMux + ADC.

    Owned submodules:

    * ``data_switchcap`` — bank shape
      ``(*prefix, group_num, data_num, digit_num)`` with
      ``digit_weights`` broadcast on the ``digit_num`` axis.
    * ``ref_switchcap`` — bank shape ``(*prefix, group_num, 1)`` with
      unit ratios.
    * ``analog_mux`` — differential transport, fabricated over
      ``(*prefix, group_num, 1)`` (one MUX per group with an explicit
      broadcasting axis for the ``data_num`` data per group).
    * ``bl_adc`` — differential ADC, fabricated over
      ``(*prefix, group_num, 1)`` (one ADC per group, trailing ``1``
      broadcasts its per-group static state across the ``data_num``
      data per group at convert time).

    Note: ``__init__`` binds only the configuration object, the
    submodule factories and the shared dtype.  Lattice constants
    (``group_num``, ``data_num``, ``digit_num``) and per-instance
    static state are bound by :meth:`fabricate`, in line with the
    project-wide ``__init__`` / ``fabricate`` split (the constructor
    never sees a weight-derived shape).

    Args:
        cfg: Orchestrator-level configuration (energy / PPA knobs
            this class adds on top of its submodules).
        data_switchcap_factory: Zero-arg factory returning a fresh
            :class:`~neurox.analog.SwitchCap` for the data leg.
        ref_switchcap_factory: Zero-arg factory returning a fresh
            :class:`~neurox.analog.SwitchCap` for the reference leg.
        analog_mux_factory: Zero-arg factory returning a fresh
            :class:`~neurox.analog.AnalogMux`.
        adc_factory: Zero-arg factory returning a fresh
            :class:`~neurox.analog.adc.ADC`.
        dtype: Floating-point dtype used for the fabricated cap-ratio
            tensors and the orchestrator-level energy share tensor.
    """

    def __init__(
        self,
        cfg: ReadOutConfig,
        *,
        name: str = "",
        data_switchcap_factory: Callable[..., SwitchCap],
        ref_switchcap_factory: Callable[..., SwitchCap],
        analog_mux_factory: Callable[..., AnalogMux],
        adc_factory: Callable[..., ADC],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(name=name)

        self.cfg = cfg
        self.dtype = dtype

        # Compose child names from this readout's hierarchical name so
        # the profiler emits ``<readout>.data_switchcap`` etc. events.
        prefix = name + "." if name else ""
        self.data_switchcap = data_switchcap_factory(name=f"{prefix}data_switchcap")
        self.ref_switchcap = ref_switchcap_factory(name=f"{prefix}ref_switchcap")
        self.analog_mux = analog_mux_factory(name=f"{prefix}analog_mux")
        self.bl_adc: ADC = adc_factory(name=f"{prefix}bl_adc")

        self._group_num: float = -float("inf")
        self._data_num: float = -float("inf")

    @property
    def area_per_inst__um2(self) -> float:
        """Aggregated silicon area [um^2] over the grouped lattice.

        Adds the orchestrator's own area to every owned submodule
        scaled by its per-array instance count:

        * ``data_switchcap`` × ``group_num * data_num`` (per-data bank);
        * ``ref_switchcap`` × ``group_num`` (per-group bank);
        * ``analog_mux`` × ``group_num`` (per-group instance);
        * ``bl_adc`` × ``group_num`` (per-group instance).

        Requires :meth:`fabricate` to have run.
        """
        group_num, data_num = self._group_num, self._data_num
        return (
            self.cfg.area_per_inst__um2
            + group_num * data_num * self.data_switchcap.area_per_inst__um2
            + group_num * self.ref_switchcap.area_per_inst__um2
            + group_num * self.analog_mux.area_per_inst__um2
            + group_num * self.bl_adc.area_per_inst__um2
        )

    @property
    def leakage_per_inst__uW(self) -> float:
        """Aggregated static leakage [uW] over the grouped lattice.

        Same multiplicity convention as :attr:`area_per_inst__um2`.
        Requires :meth:`fabricate` to have run.
        """
        group_num, data_num = self._group_num, self._data_num
        return (
            self.cfg.leakage_per_inst__uW
            + group_num * data_num * self.data_switchcap.leakage_per_inst__uW
            + group_num * self.ref_switchcap.leakage_per_inst__uW
            + group_num * self.analog_mux.leakage_per_inst__uW
            + group_num * self.bl_adc.leakage_per_inst__uW
        )

    def latency_per_op__ns(self, *, adc_bits: int) -> float:
        """Aggregated per-VMM latency [ns] at the active ``adc_bits``.

        Sums every block's fixed per-op latency with the bit-
        dependent ADC latency:

        ``orch + data_sc + ref_sc + mux + bl_adc(adc_bits)``

        The data and ref SwitchCaps share the same physical clock
        edge, so their nominal latencies usually match and the sum
        models a worst-case pessimistic schedule.

        Args:
            adc_bits: Active ADC bit width.
        """
        return (
            self.cfg.latency_per_op__ns
            + self.data_switchcap.latency_per_op__ns
            + self.ref_switchcap.latency_per_op__ns
            + self.analog_mux.latency_per_op__ns
            + self.bl_adc.latency_per_op__ns(bits=adc_bits)
        )

    def fabricate(
        self,
        shape: tuple[int, ...],
        *,
        data_num: int,
        digit_weights: Tensor,
    ) -> None:
        """Fan the per-group readout instance shape out into each submodule.

        ``shape == (*prefix, group_num)`` carries the readout
        container's own virtual-instance organisation — one ref
        path, one mux, one ADC per reference group.  ``data_num``
        and ``digit_weights`` are independent inputs that size the
        data-side bank's fan-out; ``digit_num`` is read off
        ``digit_weights.shape[0]``.  Per-submodule shapes:

        * ``data_switchcap``: bank shape
          ``(*prefix, group_num, data_num)`` with the per-cap weight
          template ``digit_weights`` of length ``digit_num``.  The
          SwitchCap's fabricate produces ``c__fF`` at
          ``(*prefix, group_num, data_num, digit_num)``.
        * ``ref_switchcap``: bank shape ``(*prefix, group_num)`` with
          unit weights — one cap per ref column.  The SwitchCap's
          fabricate produces ``c__fF`` at ``(*prefix, group_num, 1)``.
        * ``analog_mux``: ``(*prefix, group_num, 1)`` — one MUX per
          group (no-op fabricate today; the trailing ``1`` keeps the
          broadcasting shape aligned with the runtime
          ``(*runtime, group_num, data_num)`` data flow for future
          per-leg mismatch).
        * ``bl_adc``: ``(*prefix, group_num, 1)`` — one ADC per group;
          the trailing ``1`` lets its static state (cap mismatch,
          comparator offset) broadcast cleanly across the ``data_num``
          data per group at convert time.

        Args:
            shape: Readout instance shape ``(*prefix, group_num)``.
                Must have at least one trailing entry.
            data_num: Number of data per reference group.  Must be
                ``> 0``.  Sizes the data-side bank's fan-out.
            digit_weights: 1-D per-digit weight vector of shape
                ``[digit_num]`` (e.g. ``(radix^0, ..., radix^(D-1))``).
        """
        if data_num <= 0:
            raise ValueError(f"data_num must be > 0, got {data_num}")
        if digit_weights.ndim != 1:
            raise ValueError(f"digit_weights must be 1-D, got shape {tuple(digit_weights.shape)}")
        if len(shape) < 1:
            raise ValueError(f"shape must be (*prefix, group_num), got {shape}")
        readout_shape = shape
        weights = digit_weights.to(self.dtype)
        ones = torch.ones(1, dtype=self.dtype, device=weights.device)

        # (*prefix, group_num): readout-level instance shape (used by
        # ref_switchcap directly and by mux / adc with a trailing
        # data_num=1 broadcast axis).
        per_group_shape = readout_shape
        # (*prefix, group_num, 1): per-group instance plus a trailing
        # broadcast axis so the runtime data lattice
        # (*runtime, group_num, data_num) lines up against the static
        # state without a manual unsqueeze inside each convert kernel.
        per_group_bcast_shape = (*per_group_shape, 1)
        # Cache the trailing lattice counts so the aggregated PPA
        # properties can weight each owned submodule's contribution by
        # its per-array instance count (data_sc bank per data, ref_sc /
        # mux / adc per group).
        self._group_num = readout_shape[-1]
        self._data_num = data_num

        # Data: bank shape (*prefix, group_num, data_num) with the
        # per-cap weight template ``digit_weights`` of length digit_num.
        self.data_switchcap.fabricate(
            (*per_group_shape, self._data_num),
            cap_ratio=weights,
        )

        # Ref: bank shape (*prefix, group_num) with unit weight — one
        # cap per ref column.  Trailing n_caps=1 inside the SwitchCap.
        self.ref_switchcap.fabricate(per_group_shape, cap_ratio=ones)

        self.analog_mux.fabricate(per_group_bcast_shape)
        self.bl_adc.fabricate(per_group_bcast_shape)

    def readout(
        self,
        v_data_grouped__V: Tensor,
        v_ref_grouped__V: Tensor,
        *,
        adc_mode: int,
        adc_bits: int,
    ) -> ReadOutOutput:
        """Run one VMM through ``data S/H -> ref S/H -> MUX -> ADC``.

        Every value-domain operation lives inside an owned leaf
        circuit module; this method itself only does shape ops
        (``unsqueeze``, ``expand``).  Energy from each leaf is
        emitted through the profiler side channel — no per-leg
        return tensors.

        Args:
            v_data_grouped__V: Per-data per-digit OpAmpTIA output voltages
                [V].  Shape
                ``(*runtime, group_num, data_num, digit_num)``.
            v_ref_grouped__V: Per-reference-column OpAmpTIA output
                voltages [V].  Shape ``(*runtime, group_num)``.
            adc_mode: Runtime ADC operating-point index.
            adc_bits: Runtime ADC bit width.

        Returns:
            :class:`ReadOutOutput` with per-data ADC code and the
            post-MUX differential voltages.
        """
        # data S/H reduces the trailing digit_num axis.
        v_pos_pre_mux__V = self.data_switchcap.sample_and_accumulate(v_data_grouped__V)
        data_num = v_pos_pre_mux__V.shape[-1]

        # ref S/H reduces the trailing unit-cap axis.
        v_ref_bank__V = v_ref_grouped__V.unsqueeze(-1)
        v_ref_sampled__V = self.ref_switchcap.sample_and_accumulate(v_ref_bank__V)

        # Build v_neg purely by shape — broadcast the per-group ref
        # sample across the per-group data axis.  No algebraic scaling.
        v_neg_pre_mux__V = v_ref_sampled__V.unsqueeze(-1).expand(*v_ref_sampled__V.shape, data_num)

        v_pos__V, v_neg__V = self.analog_mux.transport(v_pos_pre_mux__V, v_neg_pre_mux__V)

        code = self.bl_adc.convert(
            v_pos__V=v_pos__V,
            v_neg__V=v_neg__V,
            mode=adc_mode,
            bits=adc_bits,
        )

        # Orchestrator-level dynamic energy emits once per VMM as a
        # side-channel event — distributed flat across the whole bank.
        if self.cfg.energy_per_op__fJ > 0.0:
            self._log_dynamic(self.cfg.energy_per_op__fJ, self.cfg.latency_per_op__ns)

        return ReadOutOutput(code=code, v_pos__V=v_pos__V, v_neg__V=v_neg__V)
