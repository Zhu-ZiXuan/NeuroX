"""Simplified ternary-weight crossbar tile — kernel pure array + inline current-mode readout.

The tile composes the kernel :class:`~neurox.primitive.xbar.array.XbarArray1t1r`
pure array (cells, wire parasitics, DC solver) with the boundary drivers it
owns as peers of the array — a 1-bit ON/OFF WL
:class:`~neurox.primitive.analog.voltage_dac.VoltageDac`, a current-aware BL
clamp and an ideal SL drive (both the generic Thevenin
:class:`~neurox.primitive.analog.VoltageDriver`), and a shared boundary
:class:`~neurox.primitive.analog.VoltageReference` — and with four kernel
readout primitives: two :class:`~neurox.primitive.analog.CurrentMirror`
stages, one :class:`~neurox.primitive.analog.CurrentSubtractor`, and the
:class:`~neurox.primitive.analog.SarCurrentAdc` quantizer against a shared
static :class:`~neurox.primitive.analog.CurrentReference` threshold source.
All of the readout is folded directly into
:meth:`IsubIadc1t1rCimMacro.vec_mat_mul`. In the read-out chain of the
provenance paper (Xue et al., JSSC 2020) the BL clamp sits at the CABLC slot,
the two mirror stages at the DSWCT and combiner slots, and the subtractor at
the PN-ISUB slot; here they are the bare generic kernel blocks.

Column layout: each logical ternary weight (one output column) maps to **2
physical columns** — a P (positive) and an N (negative) column — so
``phys_col_num = 2 * col_num``. A ``+1`` weight sets the P cell LRS, ``-1``
the N cell, ``0`` neither. The signed difference and the sign bit are
recovered by the current subtractor; the magnitude is quantized by the SAR
current ADC against the shared CurrentReference mid-point thresholds.

Orchestration (one independent conversion per WL plane; planes arrive
pre-expanded, every leading axis of ``x`` is anonymous broadcast batch): the
WL DAC drives the planes, the array settles once at full leading (FLAT solve,
the existing chunking bounds memory), and the post-solve readout regroups the
BL currents by polarity / MUX lane / CIM-IO with the column-serial axes moved
to the broadcast leading — so every readout block operates at its native
fabricated ``inst_shape`` (static mismatch shared across the serial axes and
the caller's planes, per-call draws at full shape) — then p-mirror ->
n-mirror -> subtractor -> ADC recovers the signed-magnitude codes, reassembled
to the caller's order with primitive trailing ``[col_num]``. No in-macro
accumulation and no phase axis of the macro's own.

Sharing granularity: the boundary clamp and the readout blocks are column-MUX
time-shared, so their fabricated ``inst_shape`` carries the REAL device count
(front-end ``(2, n_lane)``, back-end ``(2, n_io)``, subtractor / ADC
``(n_io,)``). The BL clamp keeps a trailing size-1 broadcast axis and reaches
the flat array solve through the :class:`_LaneGroupedClamp` adapter; the
post-solve blocks need no broadcast axis — their serial columns ride the
leading.

Accounting ownership (the seam / intra-block law): the energy atom is one
rail-to-GND branch, ``E = V_rail * I_branch * t``. Intra-block branches are
self-billed by the owning block: the pure array bills its internal terms
(cell-side DC conduction, wire / cell capacitive cycling) and the subtractor
its three internal replica branches; the ADC bills its per-step switching
constant; the WL DAC and BL clamp bill their interface-node charge as per-op
energy. The array branch at the solver-solved ``V_BL`` node is the single
split branch: the array bills the cell side ``V_BL * I_BL`` and the macro —
the rail owner — bills the clamp side ``(V_DD - V_BL) * I_BL``. The three
inter-stage seam branches (p-mirror output, n-mirror output, subtractor
output delivery) are billed WHOLE by the macro at ``V_DD * I * t_conduct``,
once each. The conduction window ``t_conduct__ns`` of the no-S&H chain is
derived from the ADC per-step latencies. The bare kernel mirrors are pure
transports; their static PPA rolls up into the macro's lumped
``leakage_per_inst__uW`` / ``area_per_inst__um2``. The CurrentReference is
static-only (PPA leakage, no forward path).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.primitive.analog import (
    CurrentMirror,
    CurrentMirrorConfig,
    CurrentMirrorPolicy,
    CurrentReference,
    CurrentReferenceConfig,
    CurrentReferencePolicy,
    CurrentSubtractor,
    CurrentSubtractorConfig,
    CurrentSubtractorPolicy,
    VoltageDriver,
    VoltageDriverConfig,
    VoltageDriverPolicy,
    VoltageDriverSnap,
    VoltageReference,
    VoltageReferenceConfig,
    VoltageReferencePolicy,
)
from neurox.primitive.analog.adc_common import AdcCalibrationRecord, AdcOperationPoint
from neurox.primitive.analog.current_adc import SarCurrentAdc, SarCurrentAdcConfig, SarCurrentAdcPolicy
from neurox.primitive.analog.voltage_dac import VoltageDac, VoltageDacConfig, VoltageDacPolicy
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy
from neurox.primitive.xbar.array import XbarArray1t1r, XbarArray1t1rConfig, XbarArray1t1rPolicy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fixed structural layout of the ternary weight: a single signed digit
# carried by two polarity columns (P, N).
_POLARITY_NUM = 2  # P, N
_W_DIGIT_COUNT = 1  # single ternary digit per weight
_W_DIGIT_RADIX = 2  # positional base (trivial at one digit)
_PHYS_PER_COL = _POLARITY_NUM * _W_DIGIT_COUNT  # 2 physical columns per output


# ---------------------------------------------------------------------------
# BL-clamp lane adapter
# ---------------------------------------------------------------------------


class _LaneGroupedClamp:
    """P/N lane-grouped BL-clamp view satisfying the ``ClampDriver`` role.

    Wraps the macro-owned kernel :class:`~neurox.primitive.analog.VoltageDriver`
    (fabricated at the REAL device count, inst trailing ``(2, n_lane, 1)``)
    and presents the flat per-physical-column contract the pure array
    consumes: :meth:`snapshot` draws at the polarity/lane-grouped view
    ``[..., 2, n_lane, cols_per_lane]`` — the static per-DEVICE offset
    broadcasts onto every column the device time-serves via the trailing
    size-1 axis (P and N independent), while the per-solve thermal noise
    stays per column (a conservative upper bound on thermal diversity) —
    then regroups to the flat physical-column order ``p = 2c + polarity``
    the array solver needs (one ``v_clamp`` per BL). :meth:`solve_clamp`
    delegates unchanged.
    """

    def __init__(self, driver: VoltageDriver, *, n_lane: int) -> None:
        self._driver = driver
        self._n_lane = n_lane

    def snapshot(
        self,
        *,
        v_ref__V: Tensor,
        shape: tuple[int, ...],
        multi_coords: tuple[Tensor, ...] | None,
    ) -> VoltageDriverSnap:
        *lead, phys_col_num = shape
        cols_per_lane = (phys_col_num // _POLARITY_NUM) // self._n_lane
        grouped = self._driver.snapshot(
            v_ref__V=v_ref__V,
            shape=(*lead, _POLARITY_NUM, self._n_lane, cols_per_lane),
            multi_coords=multi_coords,
        )
        return VoltageDriverSnap(
            v_ref__V=grouped.v_ref__V.flatten(-2).movedim(-2, -1).flatten(-2),
            r_out__MOhm=grouped.r_out__MOhm,
        )

    def solve_clamp(
        self,
        i_port__uA: Tensor,
        snap: VoltageDriverSnap,
        *,
        v_clamp_init__V: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        return self._driver.solve_clamp(i_port__uA, snap, v_clamp_init__V=v_clamp_init__V)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class IsubIadc1t1rCimMacroConfig(CimMacroConfig):
    """Configuration for the simplified ternary-weight crossbar tile.

    Every physical / PPA field is required — values live in the TOML, never as
    a code default. The ternary weight structure (1 signed digit, two polarity
    columns) is fixed and not configurable.

    Column-MUX time-sharing: the boundary clamp and the readout circuits are
    NOT replicated per bit-line — they are column-MUX time-multiplexed, so
    their real physical instance count is far smaller than ``phys_col_num``.
    Two structural knobs (``mux_factor``, ``io_col_num``), both in LOGICAL
    columns per polarity, derive every block's genuine shared count from the
    geometry; see :meth:`n_lane` / :meth:`n_io`. Because the count is
    geometry-derived, the per-block ``leakage_per_inst__uW`` /
    ``area_per_inst__um2`` are GENUINE per-circuit values
    (geometry-independent), and total PPA scales physically with the real
    shared count instead of ballooning with the raw column count.

    Attributes:
        mux_factor: Logical columns per front-end MUX lane, per polarity. One
            front-end lane (BL clamp / p-mirror) time-multiplexes
            ``mux_factor`` logical columns of one polarity, so there are
            ``n_lane = col_num // mux_factor`` lanes per polarity and the
            front-end device count is ``2 * n_lane`` (P and N conduct
            simultaneously through separate devices). ``col_num`` must divide
            exactly by ``mux_factor`` (the lane regrouping is an exact
            reshape).
        io_col_num: Logical columns per CIM-IO. The post-mux back-end serves a
            whole CIM-IO: the n-mirror device count is ``2 * n_io``, the
            subtractor and the ADC count ``n_io`` (one per IO, consuming both
            polarities), with ``n_io = col_num // io_col_num``. ``col_num``
            must divide exactly by ``io_col_num``, and ``io_col_num`` by
            ``mux_factor`` (each IO contains whole mux lanes; the lane-to-IO
            regroup is an exact reshape). The shared CurrentReference source
            is one per tile, not per IO.
        adc_calibration: Externally-calibrated ``(adc_mode, adc_bits) ->
            rescale_factor`` records; ``M_ideal ~= code * rescale_factor``.
            Lists the set of ADC operating points the tile supports.
        array_config: Owned kernel pure-array config (cell array + wire
            parasitics + solver). Excludes the boundary drivers / DAC /
            reference, which are macro-owned peers of the array.
        wl_dac_config: WL 1-bit ON/OFF DAC configuration. Its
            ``energy_per_op__fJ`` carries only the driver circuit's own
            conversion energy; the WL load caps (wire + gates) are
            array-billed from the array's own geometry.
        bl_clamp_config: BL current-aware clamp configuration — a generic
            :class:`~neurox.primitive.analog.VoltageDriver`. It does not
            self-hold ``V_BLC``; the macro injects the clamp reference per
            ``snapshot`` from the shared ``clamp_ref_config`` source (tap 0).
            Its ``energy_per_op__fJ`` carries the full ``C * V_DD**2``
            charge/discharge cycle of the per-column CMD interface node.
        sl_driver_config: SL ideal-clamp driver configuration. Like the BL
            clamp it receives its reference per call from ``clamp_ref_config``
            (tap 1), not from its own config.
        clamp_ref_config: Shared :class:`~neurox.primitive.analog.VoltageReference`
            source for the boundary clamps. Sources the two taps consumed in
            :meth:`vec_mat_mul` in order ``[BL-clamp V_BLC, SL-drive V_SL]``:
            tap 0 feeds the BL clamp's reference, tap 1 the SL driver's.
            Snapshotted once per VMM (shared across the chunk loop, preserving
            chunk bit-exactness).
        p_mirror_config: Front-end mirror-stage config — the generic
            :class:`~neurox.primitive.analog.CurrentMirror` applying the
            global ``1/k`` down-scale. A non-reporter pure transport: its
            silicon rolls up into the macro's lumped static PPA; its output
            seam branch is billed whole by the macro.
        n_mirror_config: Back-end mirror-stage config — the combiner
            normalization, same roll-up and seam-billing rules as
            ``p_mirror_config``.
        subtractor_config: Kernel
            :class:`~neurox.primitive.analog.CurrentSubtractor` config (unit
            gain). A reporter: it self-holds its static PPA and self-bills
            its internal replica branches per :meth:`subtract` call. Its
            ``v_rail__V`` must equal the tile's ``v_dd__V`` (pinned by
            :meth:`validate_supply`).
        adc_config: SAR current-ADC quantizer config. Its per-mode
            ``ref_levels__uA`` ladders must equal the CurrentReference tap
            rows mode for mode — the shared CurrentReference source is the
            single source of truth (pinned by
            :meth:`validate_ref_consistency`). Its ``step_latency__ns``
            entries also derive the tile's conduction window
            :attr:`t_conduct__ns`.
        reference_config: Shared
            :class:`~neurox.primitive.analog.CurrentReference` config — the
            static per-mode current-reference tap rows (the mid-point ADC
            threshold ladders, one row per ADC operating mode), the single
            source of truth.
        v_dd__V: Supply-rail voltage [V]. The macro is the rail owner: it
            bills the ``(V_DD - V_BL) * I_BL`` clamp-side split of the array
            branch and the three whole readout seam branches at this rail.
        readout_latency_per_op__ns: Readout-chain per-op latency [ns], logged
            at the macro level scaled by the back-end serial op count (batch
            — including any caller plane serialization — and per-IO
            column-serial factors over the owning stage's devices; the ADC
            adds its own step latency).
        area_per_inst__um2: Macro-owned silicon area per fabricated tile
            instance [um²] — lumped Control infrastructure plus the
            non-reporter blocks (p/n mirrors). Excludes the owned reporter
            children (array / boundary drivers / subtractor / ADC /
            references), which roll up separately.
        leakage_per_inst__uW: Macro-owned static leakage per fabricated tile
            instance [uW] — same lump scope as ``area_per_inst__um2``. The
            value is geometry-DEPENDENT (it folds device counts at this tile's
            ``n_lane`` / ``n_io``); the TOML records the breakdown.
    """

    mux_factor: int
    io_col_num: int

    adc_calibration: tuple[AdcCalibrationRecord, ...]

    array_config: XbarArray1t1rConfig

    # --- Boundary drivers + reference (peers of the array) ---
    wl_dac_config: VoltageDacConfig
    bl_clamp_config: VoltageDriverConfig
    sl_driver_config: VoltageDriverConfig
    clamp_ref_config: VoltageReferenceConfig

    # --- Inline readout-chain blocks (shared analog library) ---
    p_mirror_config: CurrentMirrorConfig
    n_mirror_config: CurrentMirrorConfig
    subtractor_config: CurrentSubtractorConfig
    adc_config: SarCurrentAdcConfig
    reference_config: CurrentReferenceConfig

    # --- Macro-owned accounting (rail owner) ---
    v_dd__V: float
    readout_latency_per_op__ns: float

    @property
    def phys_col_num(self) -> int:
        """Physical column count — ``2 * col_num`` (P/N polarity pair per output)."""
        return _PHYS_PER_COL * self.col_num

    @property
    def t_conduct__ns(self) -> float:
        """Conduction window of one solved WL plane [ns] — ``sum(adc_config.step_latency__ns)``.

        The chain has no sample-and-hold, so the array DC conduction, the
        clamp-side drop, the seam branches, and the subtractor's internal
        branches all conduct for the whole per-conversion sensing window.
        Single-sourced from the ADC per-step latencies, it auto-scales with
        the binary-search depth (``n_bits``).
        """
        return sum(self.adc_config.step_latency__ns)

    def n_lane(self) -> int:
        """Real physical lane count for the per-MUX-lane front-end, per polarity.

        One front-end lane time-multiplexes ``mux_factor`` LOGICAL columns of
        one polarity, so the genuine per-polarity count is
        ``col_num // mux_factor`` (exact divisibility enforced by
        :meth:`validate_sharing`); the front-end device count is
        ``2 * n_lane`` (P and N conduct simultaneously through separate
        devices). This sizes the BL-clamp and p-mirror ``inst_shape``
        trailing axes.
        """
        return self.col_num // self.mux_factor

    def n_io(self) -> int:
        """Real physical IO-sense-lane count for the per-CIM-IO back-end.

        The post-mux back-end serves a whole CIM-IO of ``io_col_num`` logical
        columns, so the genuine count is ``col_num // io_col_num`` (exact
        divisibility enforced by :meth:`validate_sharing`): the n-mirror
        device count is ``2 * n_io``; the subtractor and the ADC count
        ``n_io`` each (one per IO, consuming both polarities). The shared
        CurrentReference source is one per tile, not per IO.
        """
        return self.col_num // self.io_col_num

    def validate(self) -> None:
        super().validate()
        self.validate_sharing()
        self.validate_supply()
        self.validate_accounting()
        self.validate_adc_calibration()
        self.validate_ref_consistency()

    def validate_sharing(self) -> None:
        self._require_pos(self.mux_factor, "mux_factor")
        self._require_pos(self.io_col_num, "io_col_num")
        # The lane / IO regrouping is a reshape, so it needs exact blocking.
        if self.col_num % self.mux_factor != 0:
            raise ValueError(
                f"require: col_num ({self.col_num}) % mux_factor ({self.mux_factor}) == 0 — "
                "the front-end lane regrouping is an exact reshape"
            )
        if self.col_num % self.io_col_num != 0:
            raise ValueError(
                f"require: col_num ({self.col_num}) % io_col_num ({self.io_col_num}) == 0 — "
                "the CIM-IO regrouping is an exact reshape"
            )
        # The physical mux tree of one lane is contained in one CIM-IO, so
        # each IO must hold whole lanes.
        if self.io_col_num % self.mux_factor != 0:
            raise ValueError(
                f"require: io_col_num ({self.io_col_num}) % mux_factor ({self.mux_factor}) == 0 — "
                "the lane-to-IO regroup is an exact reshape (each IO contains whole mux lanes)"
            )

    def validate_supply(self) -> None:
        self._require_non_neg(self.v_dd__V, "v_dd__V")
        # The subtractor's internal replica branches conduct from the tile's
        # supply rail; a diverging copy would silently bill a different rail.
        if self.subtractor_config.v_rail__V != self.v_dd__V:
            raise ValueError(
                f"require: subtractor_config.v_rail__V ({self.subtractor_config.v_rail__V}) == "
                f"v_dd__V ({self.v_dd__V}) — the subtractor's internal branches conduct from the "
                "tile's supply rail"
            )

    def validate_accounting(self) -> None:
        self._require_non_neg(self.readout_latency_per_op__ns, "readout_latency_per_op__ns")

    def validate_adc_calibration(self) -> None:
        if len(self.adc_calibration) == 0:
            raise ValueError("require: adc_calibration must contain at least one entry")
        n_bits = self.adc_config.n_bits
        mode_num = self.adc_config.mode_num
        seen: set[tuple[int, int]] = set()
        for entry in self.adc_calibration:
            key = (entry.adc_mode, entry.adc_bits)
            if key in seen:
                raise ValueError(f"adc_calibration has duplicate (adc_mode, adc_bits)={key}")
            seen.add(key)
            if not (entry.rescale_factor > 0.0):
                raise ValueError(
                    f"require: rescale_factor ({entry.rescale_factor}) > 0 for "
                    f"(adc_mode={entry.adc_mode}, adc_bits={entry.adc_bits})"
                )
            # This single-point ADC digitizes at exactly n_bits; an entry whose
            # adc_bits differs would only surface as a KeyError at lookup time.
            if entry.adc_bits != n_bits:
                raise ValueError(
                    f"require: adc_calibration entry adc_bits ({entry.adc_bits}) == "
                    f"adc_config.n_bits ({n_bits}); got (adc_mode={entry.adc_mode}, adc_bits={entry.adc_bits})"
                )
            # The quantizer bounds-checks adc_mode at convert time; a record
            # outside the ladder set would otherwise only fail at runtime.
            if not (0 <= entry.adc_mode < mode_num):
                raise ValueError(
                    f"require: adc_calibration entry adc_mode ({entry.adc_mode}) in "
                    f"[0, adc_config.mode_num ({mode_num}))"
                )
        # Every ladder mode must be calibrated so `adc_mode_num` (the ladder
        # count) equals the set of operating points the tile can actually run.
        calibrated_modes = {entry.adc_mode for entry in self.adc_calibration}
        missing = sorted(set(range(mode_num)) - calibrated_modes)
        if missing:
            raise ValueError(
                f"require: adc_calibration covers every adc_mode in [0, {mode_num}); missing modes {missing}"
            )

    def validate_ref_consistency(self) -> None:
        """The ADC thresholds must equal the CurrentReference taps, mode for mode.

        The shared :class:`~neurox.primitive.analog.CurrentReference` source is
        the single source of truth for the mid-point threshold ladders; the ADC
        config copy must match its tap rows per mode
        (``ref_levels__uA[m] == i_refs__uA[m]`` for every ``m``, with equal
        mode counts) so the ADC quantizes against the same levels in every
        operating mode. This equality also transitively preserves the
        per-row ``2 ** n_bits - 1`` length guarantee enforced on
        ``adc_config.ref_levels__uA`` (the CurrentReference side validates
        strictly increasing equal-length rows, not the ADC row length).
        """
        if self.adc_config.mode_num != self.reference_config.mode_num:
            raise ValueError(
                "adc_config.ref_levels__uA must equal reference_config.i_refs__uA "
                f"(CurrentReference is the single source of truth); got {self.adc_config.mode_num} "
                f"vs {self.reference_config.mode_num} mode rows"
            )
        for m, (adc_row, ref_row) in enumerate(
            zip(self.adc_config.ref_levels__uA, self.reference_config.i_refs__uA, strict=True)
        ):
            if adc_row != ref_row:
                raise ValueError(
                    f"adc_config.ref_levels__uA[{m}] must equal reference_config.i_refs__uA[{m}] "
                    f"(CurrentReference is the single source of truth); got {adc_row} vs {ref_row}"
                )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IsubIadc1t1rCimMacroPolicy(CimMacroPolicy):
    """Composite nonideality policy for :class:`IsubIadc1t1rCimMacro`.

    One child policy per owned ``ModuleBase`` block. The ``all_off`` preset
    sets every child to its lossless baseline.

    Attributes:
        array: Kernel pure-array nonideality policy (cell toggles plus the
            load-bearing ``solve_chunk_size`` chunking knob).
        wl_dac: WL DAC nonideality policy.
        bl_clamp: BL current-aware clamp nonideality policy.
        sl_driver: SL ideal-driver nonideality policy.
        clamp_ref: Boundary-clamp voltage-reference nonideality policy
            (tolerance / noise) for the shared ``clamp_ref`` source.
        p_mirror: Front-end mirror-stage policy (``mismatch``).
        n_mirror: Back-end mirror-stage policy (``mismatch``), independent of
            the front-end draw.
        subtractor: Current-subtractor policy (``mismatch`` / ``offset``).
        adc: SAR current-ADC quantizer policy.
        reference: Current-reference policy (tolerance / noise).
    """

    array: XbarArray1t1rPolicy
    wl_dac: VoltageDacPolicy
    bl_clamp: VoltageDriverPolicy
    sl_driver: VoltageDriverPolicy
    clamp_ref: VoltageReferencePolicy
    p_mirror: CurrentMirrorPolicy
    n_mirror: CurrentMirrorPolicy
    subtractor: CurrentSubtractorPolicy
    adc: SarCurrentAdcPolicy
    reference: CurrentReferencePolicy


# ---------------------------------------------------------------------------
# Macro
# ---------------------------------------------------------------------------


@CimMacro.register_key(IsubIadc1t1rCimMacroConfig)
class IsubIadc1t1rCimMacro(CimMacro):
    """Simplified ternary-weight 1T1R crossbar tile with an inline kernel readout chain.

    Composes the kernel :class:`~neurox.primitive.xbar.array.XbarArray1t1r`
    pure array with the macro-owned boundary drivers and the kernel
    current-mode readout blocks. Each logical ternary weight occupies 2
    physical columns (P / N); :meth:`program` routes the weight magnitude to
    the polarity column matching its sign, and :meth:`vec_mat_mul` runs one
    independent conversion per WL plane — boundary drive -> array solve ->
    p-mirror -> n-mirror -> subtractor -> ADC — assembling the
    signed-magnitude code tensor with leading order preserved and primitive
    trailing ``[col_num]`` (no in-macro accumulation, no phase axis of the
    macro's own).
    """

    config: IsubIadc1t1rCimMacroConfig
    policy: IsubIadc1t1rCimMacroPolicy

    def __init__(
        self,
        *,
        config: IsubIadc1t1rCimMacroConfig,
        policy: IsubIadc1t1rCimMacroPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW

        prefix = self._inst_shape
        col_num = config.col_num
        self.physical_col_num = _PHYS_PER_COL * col_num
        self.n_lane = config.n_lane()
        self.n_io = config.n_io()

        # --- Pure array: cells + wire parasitics + DC solver ---
        self.core = XbarArray1t1r(
            config=config.array_config,
            policy=policy.array,
            w_layout_shape=(*prefix, self.physical_col_num, config.row_num),
            dtype=dtype,
            T__K=T__K,
        )

        # --- Boundary drivers + reference, peers of the array ---
        # The WL DAC drives one binary level per row. The BL clamp (paper:
        # CABLC) is a column-MUX time-shared front-end lane, one per
        # mux_factor LOGICAL columns per polarity: its fabricated inst_shape
        # carries the REAL device count (2 * n_lane) with a trailing size-1
        # broadcast axis, because it enters the FLAT array solve through the
        # _LaneGroupedClamp adapter. The SL driver clamps every physical
        # column; the clamp reference is a global-scalar source
        # (inst_shape=()) snapshotted once per VMM, its two ordered taps
        # feeding [BL clamp, SL drive] — one reference per tile, mirroring
        # the per-tile CurrentReference shared by the ADCs.
        self.wl_dac = VoltageDac.from_config(
            config=config.wl_dac_config,
            policy=policy.wl_dac,
            inst_shape=(*prefix, config.row_num),
            dtype=dtype,
            T__K=T__K,
        )
        self.bl_clamp = VoltageDriver(
            config=config.bl_clamp_config,
            policy=policy.bl_clamp,
            inst_shape=(*prefix, _POLARITY_NUM, self.n_lane, 1),
            dtype=dtype,
            T__K=T__K,
        )
        self._bl_clamp_lanes = _LaneGroupedClamp(self.bl_clamp, n_lane=self.n_lane)
        self.sl_driver = VoltageDriver(
            config=config.sl_driver_config,
            policy=policy.sl_driver,
            inst_shape=(*prefix, self.physical_col_num),
            dtype=dtype,
            T__K=T__K,
        )
        self.clamp_ref = VoltageReference(
            config=config.clamp_ref_config,
            policy=policy.clamp_ref,
            inst_shape=(),
            dtype=dtype,
            T__K=T__K,
        )

        # --- Readout blocks: column-MUX time-shared, NOT per-column ---
        # Each block is fabricated at its REAL device count. The post-solve
        # readout in vec_mat_mul moves the column-serial axes (mux slot,
        # lanes-per-IO) to the anonymous broadcast leading, so every forward
        # tensor right-aligns with these native inst shapes: static draws
        # are shared across the serial axes and the caller's planes, while
        # per-call draws sample at full shape. P and N polarities carry
        # independent mismatch. The mirrors are non-reporters — their static
        # PPA is the macro's lump — so no _area_per_inst__um2 /
        # _leakage_per_inst__uW is set on them; the subtractor and the ADC
        # are reporters and self-hold theirs.
        self.p_mirror = CurrentMirror(
            config=config.p_mirror_config,
            policy=policy.p_mirror,
            inst_shape=(*prefix, _POLARITY_NUM, self.n_lane),
            dtype=dtype,
            T__K=T__K,
        )
        self.n_mirror = CurrentMirror(
            config=config.n_mirror_config,
            policy=policy.n_mirror,
            inst_shape=(*prefix, _POLARITY_NUM, self.n_io),
            dtype=dtype,
            T__K=T__K,
        )
        self.subtractor = CurrentSubtractor(
            config=config.subtractor_config,
            policy=policy.subtractor,
            inst_shape=(*prefix, self.n_io),
            dtype=dtype,
            T__K=T__K,
        )
        # The ADC's forward tensor carries exactly n_io trailing positions,
        # matching inst_shape, so its internal column-to-lane gather is the
        # identity. Reporter: self-holds its static PPA.
        self.bl_adc = SarCurrentAdc(
            config=config.adc_config,
            policy=policy.adc,
            inst_shape=(*prefix, self.n_io),
            dtype=dtype,
            T__K=T__K,
        )
        # One shared static threshold source per tile (no forward path).
        self.reference = CurrentReference(
            config=config.reference_config,
            policy=policy.reference,
            inst_shape=tuple(prefix),
            dtype=dtype,
            T__K=T__K,
        )

        self._rescale_lut: dict[AdcOperationPoint, float] = {
            AdcOperationPoint(adc_mode=e.adc_mode, adc_bits=e.adc_bits): e.rescale_factor
            for e in config.adc_calibration
        }

    # -----------------------------------------------------------------
    # Value-domain semantics
    # -----------------------------------------------------------------

    @property
    def x_range(self) -> tuple[int, int]:
        """Inclusive binary input range — the WL plane is driven directly."""
        return (0, 1)

    @property
    def w_digit_count(self) -> int:
        """Digits per weight — fixed at 1 (single ternary digit)."""
        return _W_DIGIT_COUNT

    @property
    def w_digit_radix(self) -> int:
        """Positional base of the digit combination — fixed at 2 (trivial at one digit)."""
        return _W_DIGIT_RADIX

    @property
    def w_digit_range(self) -> tuple[int, int]:
        """Inclusive signed per-digit value range.

        Ternary: the single digit selects the P column (``+1`` sets the
        positive cell), the N column (``-1``), or neither (``0``); the
        subtractor nets the two polarities into the signed output.
        """
        return (-1, 1)

    @property
    def adc_mode_num(self) -> int:
        """Number of ADC operating modes — the quantizer's ladder-row count.

        The calibration covers every mode (pinned by
        :meth:`IsubIadc1t1rCimMacroConfig.validate_adc_calibration`), so this
        equals the count of calibrated operating modes.
        """
        return self.config.adc_config.mode_num

    @property
    def adc_max_bits(self) -> int:
        """Maximum ADC magnitude resolution [bits] — the quantizer's ``n_bits``."""
        return self.config.adc_config.n_bits

    def adc_rescale_factor(self, adc_operation_point: AdcOperationPoint) -> float:
        """Rescale factor for ``adc_operation_point``; raises ``KeyError`` if uncalibrated."""
        try:
            return self._rescale_lut[adc_operation_point]
        except KeyError:
            available = sorted((op.adc_mode, op.adc_bits) for op in self._rescale_lut)
            raise KeyError(
                f"{adc_operation_point} not in adc_calibration; available (adc_mode, adc_bits): {available}"
            ) from None

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def program(self, w: Tensor) -> None:
        """Write the cells from one ternary digit tensor.

        Maps each logical ternary weight to its 2 physical cells: ``+1``
        writes the P-column cell LRS, ``-1`` the N-column cell, ``0`` leaves
        both at state ``0`` (HRS). The result is one physical state-index
        tensor delegated to :meth:`XbarArray1t1r.program`.

        Args:
            w: Signed digit tensor whose shape matches
                ``self._w_layout_shape = (*inst_shape, col_num, w_digit_count, row_num)``
                (the size-1 digit axis is kept for the CimMacro contract).
                Entries must lie in :attr:`w_digit_range` = ``(-1, 1)``.
        """
        if tuple(w.shape) != self._w_layout_shape:
            raise ValueError(f"program() expects w.shape {self._w_layout_shape}; got {tuple(w.shape)}")

        # Explicit digit-range check: every entry must lie in w_digit_range
        # = (-1, 1), i.e. {-1, 0, +1}. An out-of-range digit (e.g. +2) would
        # otherwise surface only downstream as a bare cell state_to_g_map
        # IndexError; fail here with a clear message naming w_digit_range.
        lo, hi = self.w_digit_range
        if int(w.min()) < lo or int(w.max()) > hi:
            raise ValueError(
                f"require: program() digits in w_digit_range [{lo}, {hi}] (ternary {{-1, 0, +1}}); "
                f"got min {int(w.min())}, max {int(w.max())}"
            )

        # Magnitude bit and polarity per (col, row) element; the size-1 digit
        # axis is squeezed away.
        mag = w.abs().squeeze(-2)  # {0, 1}
        is_neg = (w < 0).squeeze(-2)
        pwg_state = torch.where(is_neg, torch.zeros_like(mag), mag)
        nwg_state = torch.where(is_neg, mag, torch.zeros_like(mag))

        # Scatter into the physical-column layout: logical column c maps to
        # physical 2c (P) and 2c+1 (N), the inverse of the vec_mat_mul
        # unflatten. Shape: [*inst, col, row] x2 -> [*inst, 2*col, row].
        w_phys = torch.stack((pwg_state, nwg_state), dim=-2).flatten(-3, -2)

        self.core.program(w_phys)

    def vec_mat_mul(self, x: Tensor, *, adc_operation_point: AdcOperationPoint) -> Tensor:
        """Run one independent conversion per WL plane through the array and readout chain.

        Converts the WL DAC once at full leading, settles the array once for
        the batched planes (FLAT solve; the existing chunking bounds memory),
        then regroups the BL currents for the readout chain with the column
        index law ``c = (io * lanes_per_io + lane_in_io) * mux_factor + slot``
        (``lanes_per_io = io_col_num // mux_factor``): polarity split, then
        the mux-slot and lanes-per-IO serial axes move to the broadcast
        leading so p-mirror / n-mirror / subtractor / ADC each operate at
        their native fabricated ``inst_shape`` — static mismatch broadcasts
        over the serial axes and the caller's planes, per-call draws sample
        at full shape. Every leading axis of ``x`` is anonymous broadcast
        batch (the engine's sub-phase axis rides there); every
        ``serial_op_count = numel // inst_count`` accounting picks up the
        plane and column-serial factors automatically.

        Billing: the pure array, the drivers, the subtractor, and the ADC
        self-emit their own events; the macro adds the clamp-side split of
        the array branch (:meth:`_log_array_side_block`) and the three whole
        seam branches plus the readout latency (:meth:`_log_readout_block`),
        all over the derived conduction window
        :attr:`IsubIadc1t1rCimMacroConfig.t_conduct__ns`. The shared
        CurrentReference is static-only (no forward path, no dynamic event).

        Compile-path: yes (via the macro entry). The bottleneck DC solve is
        compiled one layer down inside :meth:`XbarArray1t1r.solve_array`'s
        eager island; tracing the macro ``matmul`` fuses this method's
        reshape / readout math and breaks only at ``solve_array``.

        Args:
            x: Binary WL plane tensor with primitive trailing
                ``[row_num]``; entries in :attr:`x_range` = ``(0, 1)``.
                Rows outside the caller's active window (at most
                :attr:`max_active_rows` live rows per plane) must arrive
                zeroed (WL off). Every leading axis is anonymous
                broadcast batch — the macro never inspects, reorders, or
                reduces leading axes.
            adc_operation_point: Runtime ADC operating point.

        Returns:
            Signed-magnitude code tensor with the same leading order and
            primitive trailing ``[col_num]``: the ADC magnitude negated
            where the subtractor sign bit is set (signed-magnitude
            decoded to a signed integer), per plane. No in-macro
            accumulation and no phase axis of the macro's own.
        """
        cfg = self.config
        t_conduct__ns = cfg.t_conduct__ns

        # --- 1. Boundary drive + array steady state per plane ---
        # The WL DAC converts at the weight-grid full leading (per-instance
        # draws + per-op energy counted once per row per plane). The
        # boundary-clamp reference is snapshotted ONCE per VMM (so the
        # per-read noise is common across every chunk, preserving chunk
        # bit-exactness); its taps are 0-d scalars that broadcast onto any
        # per-column grid: tap 0 = BL clamp V_BLC, tap 1 = SL drive. The BL
        # clamp reaches the array through the lane-grouped adapter (see
        # _LaneGroupedClamp).
        _phys_col_num, row_num = self.core.weight_grid_shape[-2:]
        leading = torch.broadcast_shapes(self.core.weight_grid_shape, x.unsqueeze(-2).shape)[:-2]
        v_wl = self.wl_dac.convert(x.expand(*leading, row_num))
        clamp_taps = self.clamp_ref.v_ref__V(self.clamp_ref.snapshot())  # (2,)
        steady = self.core.solve_array(
            v_wl,
            bl_driver=self._bl_clamp_lanes,
            bl_v_ref__V=clamp_taps[0],
            sl_driver=self.sl_driver,
            sl_v_ref__V=clamp_taps[1],
            t_conduct__ns=t_conduct__ns,
        )
        i_bl = steady.i_bl_port__uA  # [..., 2 * col_num]

        # --- 2. Macro-owned clamp-side split of the array branch ---
        self._log_array_side_block(steady.v_bl_clamp__V, i_bl)

        # --- 3. Post-solve regroup: polarity split, serial axes to leading ---
        # Column index law: c = (io * lanes_per_io + lane_in_io) * mux_factor
        # + slot. Physical 2c = P, 2c+1 = N. Shapes:
        # [..., 2*col] -> [..., col, 2] -> [..., 2, col]
        # -> [..., 2, n_lane, mux] -> [mux, ..., 2, n_lane].
        lanes_per_io = cfg.io_col_num // cfg.mux_factor
        i_lane = (
            i_bl.unflatten(-1, (self.col_num, _POLARITY_NUM))
            .movedim(-1, -2)
            .unflatten(-1, (self.n_lane, cfg.mux_factor))
            .movedim(-1, 0)
        )

        # --- 4. Front-end mirror stage: global 1/k down-scale ---
        i_wdl = self.p_mirror.replicate(i_lane)  # [mux, ..., 2, n_lane]

        # --- 5. Regroup by CIM-IO; back-end mirror stage: normalization ---
        # [mux, ..., 2, n_lane] -> [mux, ..., 2, n_io, lanes_per_io]
        # -> [lpi, mux, ..., 2, n_io].
        i_io = i_wdl.unflatten(-1, (self.n_io, lanes_per_io)).movedim(-1, 0)
        i_dl = self.n_mirror.replicate(i_io)

        # --- 6. Subtractor: sign bit + |I_DL_P - I_DL_N| per logical column ---
        # Self-bills its internal replica branches over the conduction window.
        i_dl_p = i_dl[..., 0, :]  # [lpi, mux, ..., n_io]
        i_dl_n = i_dl[..., 1, :]
        i_sub, sign = self.subtractor.subtract(i_dl_p, i_dl_n, t_conduct__ns=t_conduct__ns)

        # --- 7. ADC: n_bits unsigned magnitude against the Reference thresholds ---
        # The shared CurrentReference is a static source (no forward path);
        # the ADC reads its own config threshold copy (pinned equal to the
        # CurrentReference taps by validate_ref_consistency).
        magnitude = self.bl_adc.convert(i_sub, adc_operation_point=adc_operation_point)

        # --- 8. Macro-owned seam branches + readout latency, once per plane ---
        self._log_readout_block(i_wdl__uA=i_wdl, i_dl__uA=i_dl, i_sub__uA=i_sub)

        # --- 9. Assemble the signed-magnitude codes ---
        sign_factor = 1 - 2 * sign.long()  # +1 (P >= N) / -1 (N dominates)
        codes = sign_factor * magnitude  # [lpi, mux, ..., n_io]
        # Serial axes back to trailing, then flatten by the column index law:
        # [lpi, mux, ..., n_io] -> [..., n_io, lpi, mux] -> [..., col_num].
        return codes.movedim(0, -1).movedim(0, -1).flatten(-3)

    def _log_array_side_block(self, v_bl_clamp__V: Tensor, i_bl__uA: Tensor) -> None:
        """Emit the clamp-side split of the array branch, once per solved WL plane.

        The array branch is the single branch split across owners, at the
        solver-solved ``V_BL`` node: the pure array bills the cell side
        ``V_BL * I_BL`` and the macro — the supply-rail and boundary-clamp
        owner — bills the clamp side ``(V_DD - V_BL) * I_BL`` here, over the
        conduction window, so the two terms together account for the true
        rail draw ``V_DD * I_BL``. Data-dependent; this is also the p-mirror
        INPUT-leg conduction, so the front-end seam bills no input term. The
        clamp feedback's static bias lives in the BL clamp's
        ``leakage_per_inst__uW``; the CMD interface-node charge is the BL
        clamp's per-op ``energy_per_op__fJ``, self-logged at snapshot.

        Args:
            v_bl_clamp__V: Converged BL clamp voltage per physical column,
                shape ``[..., phys_col_num]``.
            i_bl__uA: BL port current per physical column, same shape.
        """
        cfg = self.config
        # Shape: [..., phys_col_num] -> [...]
        e_clamp_drop__fJ = ((cfg.v_dd__V - v_bl_clamp__V) * i_bl__uA).sum(dim=-1) * cfg.t_conduct__ns
        self._log_dynamic_energy(e_clamp_drop__fJ)

    def _log_readout_block(self, *, i_wdl__uA: Tensor, i_dl__uA: Tensor, i_sub__uA: Tensor) -> None:
        """Emit the three whole seam branches + the readout latency, once per WL plane.

        The three inter-stage seam branches — the p-mirror output feeding the
        n-mirror, the n-mirror output feeding the subtractor, and the
        subtractor output delivering ``I_SUB`` to the ADC — each conduct
        rail-to-ground and are billed WHOLE by the composing macro, once
        each: ``E = V_DD * t_conduct * (sum|I_WDL| + sum|I_DL| + sum I_SUB)``.
        The blocks' intra-branch terms are self-billed (the subtractor's
        replica branches, the ADC's switching constant); the p-mirror input
        leg is the clamp-side array-branch split
        (:meth:`_log_array_side_block`). The readout-chain latency is one
        event over the serial op count of the owning back-end stage's
        devices (the subtractor's ``inst_count``, ``n_io`` semantics,
        matching the ADC's denominator).

        Args:
            i_wdl__uA: Front-end mirror output ``[mux, ..., 2, n_lane]``.
            i_dl__uA: Back-end mirror output ``[lpi, mux, ..., 2, n_io]``.
            i_sub__uA: Subtractor output magnitude ``|I_DL_P - I_DL_N|``,
                ``[lpi, mux, ..., n_io]``; also sets, over the subtractor's
                device ``inst_count``, the serial-op latency multiplier.
        """
        cfg = self.config

        # Reduce the macro-internal axes (leading serial mux / lanes-per-IO,
        # trailing polarity / device) so the event tensor keeps the caller's
        # leading batch shape.
        i_seam__uA = (
            i_wdl__uA.abs().sum(dim=(0, -2, -1))
            + i_dl__uA.abs().sum(dim=(0, 1, -2, -1))
            + i_sub__uA.sum(dim=(0, 1, -1))
        )
        self._log_dynamic_energy(cfg.v_dd__V * cfg.t_conduct__ns * i_seam__uA)

        # --- Readout-chain latency: per-op time over the serial op count ---
        # The parallel divisor is the OWNING back-end stage's device count
        # (the subtractor's inst_count, n_io semantics — matching the ADC's
        # denominator), NOT the macro tile inst_count: each back-end device
        # serially serves its io_col_num columns in every plane, so the batch
        # (including the caller's plane serialization) and column-serial
        # factors remain in the multiplier.
        serial_op_count = max(1, i_sub__uA.numel() // max(self.subtractor.inst_count, 1))
        latency__ns = torch.tensor(
            cfg.readout_latency_per_op__ns * serial_op_count,
            device=i_sub__uA.device,
            dtype=i_sub__uA.dtype,
        )
        self._log_latency(latency__ns)
