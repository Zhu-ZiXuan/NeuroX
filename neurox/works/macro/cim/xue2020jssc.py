"""Xue2020 JSSC SINWP 1T1R CIM sub-array — single-file current-mode readout.

Models ONE 256x512 sub-array of the 1-Mb macro of Xue et al. (JSSC 2020). Each of
the ``col_num`` logical signed weights is a sign-magnitude value (sign +
``w_digit_num`` magnitude digits, radix ``w_digit_radix``) carried by
``w_digit_num * 2`` physical cells — a P (PWG) and an N (NWG) cell per digit. A
positive weight programs the PWG digit cells to the magnitude state and the NWG
cells to HRS; a negative weight does the reverse. The signed magnitude and sign
are recovered by the DSWCT digit legs, the SINWP-SC input-radix sample-and-hold
combine, the PN-ISUB subtraction, and the TMCSA current SAR ADC.

Array + solver: the macro composes :class:`~neurox.primitive.xbar.array.XbarArray1t1r`
(a linearized 1T1R cell grid plus BL/SL wire parasitics and the DC solver). The
grouped conductance grid ``[group_size, group_num, P/N, w_digit, row]`` is folded
into the array's flat ``[phys_col, row]`` layout — the array is column-separable,
so each physical column is an independent BL/SL ladder — and the solver returns
the per-column BL port current ``I_DL`` at the position-dependent wire IR drop.
The BL is held near ``V_BLC`` by the CABLC clamp (the array's ``bl_driver``, a
generic Thevenin :class:`~neurox.primitive.analog.VoltageDriver`); the SL is the
grounded drive. Wire resistance is a required field of the nested array config —
a small positive value (the paper gives none), NEVER assumed zero in code; the
solver needs ``R > 0`` (conductance ``g = 1/R``). All ``K`` word-line sub-phases
settle in ONE broadcast array solve (the x-bit axis rides the solve leading); the
macro applies each plane's conduction window post-solve to the whole input-branch
read energy ``V_DD * I_DL`` it bills on the ``cablc`` channel.

Generality: ``w_digit_num >= 1``, ``w_digit_radix >= 2``, and ``input_bit_num >= 1``
are all free. The vectorized readout degenerates cleanly at size-1 digit / bit
axes (``w_digit_num = 1`` is a single P/N digit with no cross-digit sum;
``input_bit_num = 1`` uses the MSB combine ratio with the sample-and-hold leg
off). A ``w_digit_radix > 2`` cell requires a radix-level conductance table in the
cell config (the config author's responsibility).

Composition: the array (cell grid + wire + solver), a 1-bit ON/OFF WL
:class:`~neurox.primitive.analog.voltage_dac.VoltageDac`, the CABLC and SL clamp
seats (generic Thevenin VoltageDriver, ``r_out = 0`` ideal source — the wire IR
drop lives in the array, these are the boundary clamps + static leakage seats),
the TMCSA :class:`~neurox.primitive.analog.current_adc.SarSingleEndedCurrentAdc`
against a shared static :class:`~neurox.primitive.analog.CurrentReference` tap
bank, and two :class:`~neurox.primitive.analog.UnmodeledBlock` seats (control +
PN-ISUB bias/comparator). The DSWCT place-value weighting, the SINWP-SC
input-radix combine, and the PN-ISUB subtraction are pure macro tensor operations
(:meth:`vec_mat_mul`) — linear current combining is KCL, not a circuit block. No
mismatch or noise is modeled anywhere.

Geometry is fully DERIVED: the config never stores a physical column count —
``phys_col_num = col_num * w_digit_num * 2`` and ``io_num = col_num //
mux_factor``. The DSWCT mirror ratios and the SINWP-SC combine ratios are derived
DOWNWARD from the two MSB anchors ``dswct_ratio_msb`` and ``sc_ratio_msb``; for
the paper design (D=2, K=2) the composite BL->I_SUB transfer ``r_d * s_k`` per
(digit d, input-bit k) is the net ``(1/16, 1/8, 1/8, 1/4)`` of Fig. 7.

Energy accounting (branch atom ``E = V * I * t``). The static-energy time base is
``t_cycle`` (the 50 ns measurement period): the macro logs one latency event
``t_cycle * serial`` (serial = ``mux_factor`` column-MUX accesses), and the
profiler's ``leakage_energy = leakage_power * total_latency`` is then the static
energy over the full period. The array's ``latency_per_op__ns`` is 0 (the macro
is the sole latency emitter). The short conduction windows (``t_sample[k]``,
``t_other``) feed ONLY dynamic energy. The whole input branch ``V_DD * I_DL`` is
billed by the macro on the ``cablc`` channel (the macro owns the per-bit conduction
window ``t``); the array bills only its wire / cell capacitive cycling as its own
module row, no conduction. A validation slice map sums the array module row and the
``cablc`` channel into one ``cablc`` slice. The remaining data-dependent draws,
each an additive ``V_DD`` draw (p.204, no double-count), are billed by the macro
on its own channels: the DSWCT output legs ``V_DD * I_WDL`` under ``dswct``; the
SINWP-SC held/live legs under ``sinwp_sc``; the PN-ISUB three-branch conduction
plus its per-op comparator decision under ``pn_isub``; the control per-op constant
under ``control``. The TMCSA self-bills its per-step sensing energy. The
CurrentReference, the two UnmodeledBlock seats, the ADC, and the clamp
VoltageDrivers are static-only.

See also:
    docs/works/macro/cim/xue2020jssc/README.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.primitive.analog import (
    CurrentReference,
    CurrentReferenceConfig,
    CurrentReferencePolicy,
    UnmodeledBlock,
    UnmodeledBlockConfig,
    UnmodeledBlockPolicy,
    VoltageDriver,
    VoltageDriverConfig,
    VoltageDriverPolicy,
)
from neurox.primitive.analog.adc_common import AdcCalibrationRecord
from neurox.primitive.analog.current_adc import (
    SarSingleEndedCurrentAdc,
    SarSingleEndedCurrentAdcConfig,
    SarSingleEndedCurrentAdcPolicy,
)
from neurox.primitive.analog.voltage_dac import VoltageDac, VoltageDacConfig, VoltageDacPolicy
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy
from neurox.primitive.xbar.array import XbarArray1t1r, XbarArray1t1rConfig, XbarArray1t1rPolicy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_POLARITY_NUM = 2  # PWG, NWG per weight digit
_V_SL_DRIVE__V = 0.0  # SL grounded (paper topology: BL -> RRAM -> SL(GND))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Xue2020JsscCimMacroConfig(CimMacroConfig):
    """Configuration for the xue2020jssc SINWP 1T1R CIM sub-array.

    Every physical / PPA field is required — values live in the TOML, never as a
    code default. The physical column count is NEVER stored: it is derived
    (:attr:`phys_col_num`) from ``col_num`` and the weight structure. The DSWCT /
    SINWP-SC ratios are DERIVED from the two MSB anchor fields. The cell, the wire
    parasitics, and the DC solver live inside :attr:`array_config`.

    Attributes:
        col_num: Number of logical signed-weight columns (paper 128).
        row_num: Number of rows (paper 256).
        active_row_num: Row-block size activated per conversion (paper 9); the
            engine, not the macro, serializes across row blocks.
        w_digit_num: Magnitude digits per weight (paper 2); ``>= 1``. A single
            digit (``1``) is one P/N pair with no cross-digit combine.
        w_digit_radix: Positional base of the magnitude digits (paper 2); ``>= 2``.
            ``> 2`` needs a radix-level conductance table in the cell config.
        input_bit_num: Activation bit width K (paper 2); ``>= 1``. K serial
            single-bit WL sub-phases, LSB first. ``1`` runs the live bit alone.
        mux_factor: Logical columns per CIM-IO — the column-MUX depth (paper 32).
            ``col_num`` must divide by it; ``io_num = col_num // mux_factor``. Also
            the serial-access count per conversion (the static-energy latency
            multiplier).
        dswct_ratio_msb: DSWCT MSB-leg mirror ratio anchor (paper 0.5). The
            per-digit ratio ``r_d = dswct_ratio_msb * w_digit_radix**(d - (D-1))``
            (LSB-first d) is derived DOWNWARD from it.
        sc_ratio_msb: SINWP-SC MSB-bit combine ratio anchor (paper 0.5). The
            per-input-bit leg ratio ``s_k = sc_ratio_msb * 2**(k - (K-1))``
            (LSB-first k) is derived DOWNWARD from it; ``input_bit_num = 1`` uses
            the anchor directly with the sample-and-hold leg off.
        t_sample__ns: Sample sub-phase windows [ns], one per SAMPLED input bit
            (length ``input_bit_num - 1``; empty for K=1). Dynamic-energy only.
            The live bit (K-1) conducts in ``t_other`` instead.
        t_settle__ns: Tail non-sensing settle window [ns] — the live-bit settle
            ONLY. The SAR sensing is carried separately by
            ``array_config.solver_config`` timing and the ADC step windows; part of
            ``t_other``, dynamic-energy only.
        t_cycle__ns: Declared operating period [ns] (paper 50 ns = 1/20 MHz); the
            static-energy time base. The macro logs latency ``t_cycle * serial``;
            ``leakage_energy = leakage_power * latency`` is then the static energy
            over the full period. Must be ``>=`` the sum of the conduction windows.
        v_dd__V: Supply-rail voltage [V] (paper 1.0); the rail every channelled
            branch is billed across, including the whole input branch ``V_DD *
            I_DL`` the macro bills on the ``cablc`` channel.
        v_bl_clamp__V: BL clamp reference tap [V] (paper V_BLC ~0.29). Fed to the
            array's ``bl_driver`` as its Thevenin reference; the actual per-cell
            ``V_BL`` droops below it by the wire IR drop the solver computes. Must
            be in ``[0, v_dd__V]``.
        e_control_per_op__fJ: Control per-conversion dynamic energy [fJ] (address
            decode, CMD precharge, timing) billed under the ``control`` channel —
            includes the CMD precharge, so NO CMD capacitance is modeled anywhere.
        e_pn_isub_per_op__fJ: PN-ISUB comparator per-decision energy [fJ]
            (data-independent), billed per column under the ``pn_isub`` channel.
        control_config: Static-PPA seat for the (functionally unmodeled) control
            block.
        pn_isub_config: Static-PPA seat for the PN-ISUB bias/icm + comparator (the
            subtraction itself is macro tensor ops).
        array_config: Nested 1T1R pure-array config — the linearized cell tables
            (the programmable weights, its chord tables encode the g_map), the BL /
            SL / WL wire parasitics (segment + first-segment R > 0, an [uncertain]
            physical estimate), and the DC solver knobs. Its ``latency_per_op__ns``
            should be 0 (the macro is the sole latency emitter).
        wl_dac_config: WL 1-bit ON/OFF DAC config; ``latency_per_op__ns`` should
            be 0 (the WL sub-phase folds into ``t_cycle``).
        cablc_config: CABLC BL-clamp seat = the array's ``bl_driver`` (generic
            Thevenin VoltageDriver, ``r_out__MOhm = 0`` ideal source — the wire IR
            drop is the array's, not the clamp's; ``energy_per_op__fJ = 0``).
        sl_driver_config: SL ideal-clamp seat = the array's ``sl_driver``
            (VoltageDriver, ``r_out__MOhm = 0``).
        adc_config: TMCSA SAR current-ADC config (B-form). The macro builds the
            TMCSA with ``record_latency=False`` so it emits NO latency event — the
            macro is the sole latency emitter (``t_cycle * serial`` already spans
            sensing). The ADC keeps its REAL ``step_latency__ns`` and
            ``t_conduct_per_step__ns`` (its conduction energy is unchanged), and the
            sensing durations feed the read-chain window ``t_other``
            (:attr:`t_other__ns`).
        reference_config: Shared static CurrentReference — the ``[mode, tap]``
            threshold bank; the macro selects one mode row and passes the
            per-instance ladder straight to the ADC.
            ``tap_num == 2**adc_config.bits - 1``.
        adc_calibration: Externally-calibrated ``(adc_mode, adc_bits) ->
            rescale_factor`` records; ``M_ideal ~= code * rescale_factor`` applied
            by the unit layer.
    """

    # --- Weight / input geometry ---
    w_digit_num: int
    w_digit_radix: int
    input_bit_num: int
    mux_factor: int

    # --- Ratio anchors (DSWCT / SINWP-SC ratios derived DOWNWARD from these) ---
    dswct_ratio_msb: float
    sc_ratio_msb: float

    # --- Conduction windows (dynamic-energy only) + static time base ---
    t_sample__ns: tuple[float, ...]
    t_settle__ns: float
    t_cycle__ns: float

    # --- Supply / clamp voltages ---
    v_dd__V: float
    v_bl_clamp__V: float

    # --- Per-op dynamic constants ---
    e_control_per_op__fJ: float
    e_pn_isub_per_op__fJ: float

    # --- Static-PPA seats ---
    control_config: UnmodeledBlockConfig
    pn_isub_config: UnmodeledBlockConfig

    # --- Device-bearing sub-blocks (full nested configs) ---
    array_config: XbarArray1t1rConfig
    wl_dac_config: VoltageDacConfig
    cablc_config: VoltageDriverConfig
    sl_driver_config: VoltageDriverConfig
    adc_config: SarSingleEndedCurrentAdcConfig
    reference_config: CurrentReferenceConfig

    # --- ADC calibration ---
    adc_calibration: tuple[AdcCalibrationRecord, ...]

    # -----------------------------------------------------------------
    # Derived geometry / ratios / windows (never stored)
    # -----------------------------------------------------------------

    @property
    def io_num(self) -> int:
        """Number of CIM-IO sense lanes — ``col_num // mux_factor``."""
        return self.col_num // self.mux_factor

    @property
    def phys_col_num(self) -> int:
        """Physical column count — ``col_num * w_digit_num * 2`` (P/N per digit)."""
        return self.col_num * self.w_digit_num * _POLARITY_NUM

    @property
    def digit_ratios(self) -> tuple[float, ...]:
        """Per-digit DSWCT mirror ratios (LSB-first): ``r_d = dswct_ratio_msb * radix**(d - (D-1))``."""
        d_top = self.w_digit_num - 1
        return tuple(self.dswct_ratio_msb * self.w_digit_radix ** (d - d_top) for d in range(self.w_digit_num))

    @property
    def x_bit_ratios(self) -> tuple[float, ...]:
        """Per-input-bit SINWP-SC combine ratios (LSB-first): ``s_k = sc_ratio_msb * 2**(k - (K-1))``."""
        k_top = self.input_bit_num - 1
        return tuple(self.sc_ratio_msb * 2.0 ** (k - k_top) for k in range(self.input_bit_num))

    @property
    def t_other__ns(self) -> float:
        """Live/tail window [ns] — ``t_settle__ns + sum(adc_config.step_latency__ns)``.

        The live input bit (K-1) has no sample sub-phase of its own; its
        conduction, the SINWP-SC combine, the PN-ISUB, and the TMCSA sensing all
        fall in this tail window. The SAR sensing durations
        (``adc_config.step_latency__ns``) are added here, so the whole read chain
        conducts through sensing while ``t_settle__ns`` stays the pure non-sensing
        settle.
        """
        return self.t_settle__ns + sum(self.adc_config.step_latency__ns)

    @property
    def window_array__ns(self) -> tuple[float, ...]:
        """Per-input-bit input-branch (array/CABLC/DSWCT) conduction window [ns].

        Sampled bit ``k`` conducts for its sample window ``t_sample__ns[k]``; the
        live bit (K-1) conducts for ``t_other``.
        """
        k_live = self.input_bit_num - 1
        return tuple(self.t_sample__ns[k] if k < k_live else self.t_other__ns for k in range(self.input_bit_num))

    @property
    def window_sc__ns(self) -> tuple[float, ...]:
        """Per-input-bit SINWP-SC leg conduction window [ns].

        A bit-``k`` leg is held from its sample sub-phase to the end, so it
        conducts for the suffix sum of the remaining sample windows plus the tail
        ``t_other``: ``sum(t_sample__ns[k:]) + t_other``. The live bit reduces to
        ``t_other``.
        """
        return tuple(sum(self.t_sample__ns[k:]) + self.t_other__ns for k in range(self.input_bit_num))

    @property
    def conduction_span__ns(self) -> float:
        """Total conduction span [ns] — ``sum(t_sample) + t_other``; must fit in ``t_cycle``."""
        return sum(self.t_sample__ns) + self.t_other__ns

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def validate(self) -> None:
        super().validate()
        self.validate_weight_geometry()
        self.validate_input()
        self.validate_mux()
        self.validate_anchors()
        self.validate_windows()
        self.validate_supply()
        self.validate_reference()
        self.validate_adc_calibration()

    def validate_geometry(self) -> None:
        # Hard bounds only. This scheme runs NO macro-level row-block
        # serialization (the engine owns row-blocking; active_row_num is the
        # kernel row-block size), so the paper geometry (256 rows, 9-row block)
        # need not have row_num divisible by active_row_num.
        if not (self.col_num > 1):
            raise ValueError(f"require: col_num ({self.col_num}) > 1")
        if not (self.row_num > 1):
            raise ValueError(f"require: row_num ({self.row_num}) > 1")
        if not (1 <= self.active_row_num <= self.row_num):
            raise ValueError(f"require: 1 <= active_row_num ({self.active_row_num}) <= row_num ({self.row_num})")

    def validate_weight_geometry(self) -> None:
        # General sign-magnitude weight: >= 1 magnitude digit, radix >= 2 so a
        # digit carries at least the {0, 1} magnitude the P/N pair encodes.
        self._require_pos(self.w_digit_num, "w_digit_num")
        if not (self.w_digit_radix >= 2):
            raise ValueError(f"require: w_digit_radix ({self.w_digit_radix}) >= 2")

    def validate_input(self) -> None:
        self._require_pos(self.input_bit_num, "input_bit_num")

    def validate_mux(self) -> None:
        self._require_pos(self.mux_factor, "mux_factor")
        # The IO regrouping is a reshape, so it needs exact blocking.
        if self.col_num % self.mux_factor != 0:
            raise ValueError(
                f"require: col_num ({self.col_num}) % mux_factor ({self.mux_factor}) == 0 — "
                "the CIM-IO regrouping is an exact reshape"
            )

    def validate_anchors(self) -> None:
        self._require_pos(self.dswct_ratio_msb, "dswct_ratio_msb")
        self._require_pos(self.sc_ratio_msb, "sc_ratio_msb")

    def validate_windows(self) -> None:
        # One sample window per SAMPLED bit; the live bit (K-1) has none.
        if len(self.t_sample__ns) != self.input_bit_num - 1:
            raise ValueError(
                f"require: len(t_sample__ns) ({len(self.t_sample__ns)}) == input_bit_num - 1 "
                f"({self.input_bit_num - 1}) — one window per sampled bit; the live bit uses t_other"
            )
        for k, t in enumerate(self.t_sample__ns):
            self._require_non_neg(t, f"t_sample__ns[{k}]")
        self._require_non_neg(self.t_settle__ns, "t_settle__ns")
        self._require_pos(self.t_cycle__ns, "t_cycle__ns")
        # The static time base must contain the whole conduction span (the read
        # path idles for the remainder of the period).
        if self.t_cycle__ns < self.conduction_span__ns:
            raise ValueError(
                f"require: t_cycle__ns ({self.t_cycle__ns}) >= sum of conduction windows ({self.conduction_span__ns})"
            )
        self._require_non_neg(self.e_control_per_op__fJ, "e_control_per_op__fJ")
        self._require_non_neg(self.e_pn_isub_per_op__fJ, "e_pn_isub_per_op__fJ")

    def validate_supply(self) -> None:
        self._require_non_neg(self.v_dd__V, "v_dd__V")
        self._require_non_neg(self.v_bl_clamp__V, "v_bl_clamp__V")
        # The clamp reference is a BL node between the SL ground and the V_DD
        # supply, so it must not exceed the rail.
        if not (self.v_bl_clamp__V <= self.v_dd__V):
            raise ValueError(f"require: v_bl_clamp__V ({self.v_bl_clamp__V}) <= v_dd__V ({self.v_dd__V})")

    def validate_reference(self) -> None:
        # The TMCSA reads its ladder from the shared CurrentReference, so the tap
        # count must match the binary-search depth exactly.
        want_taps = (1 << self.adc_config.bits) - 1
        if self.reference_config.tap_num != want_taps:
            raise ValueError(
                f"require: reference_config.tap_num ({self.reference_config.tap_num}) == "
                f"2**adc_config.bits - 1 ({want_taps})"
            )

    def validate_adc_calibration(self) -> None:
        if len(self.adc_calibration) == 0:
            raise ValueError("require: adc_calibration must contain at least one entry")
        adc_max_bits = self.adc_config.bits
        mode_num = self.reference_config.mode_num
        seen: set[tuple[int, int]] = set()
        for entry in self.adc_calibration:
            key = (entry.mode, entry.bits)
            if key in seen:
                raise ValueError(f"adc_calibration has duplicate (adc_mode, adc_bits)={key}")
            seen.add(key)
            if not (entry.rescale_factor > 0.0):
                raise ValueError(
                    f"require: rescale_factor ({entry.rescale_factor}) > 0 for "
                    f"(adc_mode={entry.mode}, adc_bits={entry.bits})"
                )
            if entry.bits != adc_max_bits:
                raise ValueError(
                    f"require: adc_calibration entry adc_bits ({entry.bits}) == "
                    f"adc_config.bits ({adc_max_bits}); got (adc_mode={entry.mode}, adc_bits={entry.bits})"
                )
            if not (0 <= entry.mode < mode_num):
                raise ValueError(
                    f"require: adc_calibration entry adc_mode ({entry.mode}) in "
                    f"[0, reference_config.mode_num ({mode_num}))"
                )
        calibrated_modes = {entry.mode for entry in self.adc_calibration}
        missing = sorted(set(range(mode_num)) - calibrated_modes)
        if missing:
            raise ValueError(
                f"require: adc_calibration covers every adc_mode in [0, {mode_num}); missing modes {missing}"
            )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Xue2020JsscCimMacroPolicy(CimMacroPolicy):
    """Composite nonideality policy for :class:`Xue2020JsscCimMacro`.

    One child policy per owned ``ModuleBase`` block. This scheme models no
    mismatch or noise, so the sanctioned ``all_off`` preset (every child at its
    lossless baseline) is the only intended policy.

    Attributes:
        array_policy: 1T1R pure-array policy (the cell policy + the solver chunk knob).
        wl_dac_policy: WL DAC policy.
        cablc_policy: CABLC BL-clamp seat policy.
        sl_driver_policy: SL ideal-driver seat policy.
        adc_policy: TMCSA SAR current-ADC policy.
        reference_policy: Threshold current-reference policy.
        control_policy: Control static-seat policy.
        pn_isub_policy: PN-ISUB static-seat policy.
    """

    array_policy: XbarArray1t1rPolicy
    wl_dac_policy: VoltageDacPolicy
    cablc_policy: VoltageDriverPolicy
    sl_driver_policy: VoltageDriverPolicy
    adc_policy: SarSingleEndedCurrentAdcPolicy
    reference_policy: CurrentReferencePolicy
    control_policy: UnmodeledBlockPolicy
    pn_isub_policy: UnmodeledBlockPolicy


# ---------------------------------------------------------------------------
# Macro
# ---------------------------------------------------------------------------


@CimMacro.register_key(Xue2020JsscCimMacroConfig)
class Xue2020JsscCimMacro(CimMacro[Xue2020JsscCimMacroConfig, Xue2020JsscCimMacroPolicy]):
    """Xue2020 JSSC SINWP 1T1R CIM sub-array with an inline current-mode readout chain.

    Composes the 1T1R pure array (cell grid + wire + solver) with the macro-owned
    WL DAC, the CABLC / SL clamp seats, the TMCSA + shared reference, and the
    control / PN-ISUB static seats. Each logical weight occupies ``w_digit_num *
    2`` grouped cells (P/N per digit); :meth:`program` routes each magnitude digit
    to the polarity matching the sign, and :meth:`vec_mat_mul` drives the array
    solve once (broadcast over the K WL sub-phases), then the DSWCT / SINWP-SC /
    PN-ISUB / TMCSA readout as vectorized tensor operations billed inline at each
    production site, assembling the signed-magnitude codes with primitive trailing
    ``[col_num]``.
    """

    def __init__(
        self,
        *,
        config: Xue2020JsscCimMacroConfig,
        policy: Xue2020JsscCimMacroPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape, dtype=dtype, T__K=T__K)
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW

        gn = config.io_num  # CIM-IO sense-lane count (group_num)

        # --- Programmable weights + wire + solver: the 1T1R pure array ---
        # Grouped layout [group_size(mux slot), group_num(io), P/N, w_digit, row]
        # folds into the array's flat [phys_col, row]: each physical column is an
        # independent BL/SL ladder, so the column-MUX regroup is a pure reshape.
        # The solver returns the per-column BL port current at the
        # position-dependent wire IR drop.
        self.array = XbarArray1t1r(
            config=config.array_config,
            policy=policy.array_policy,
            inst_shape=inst_shape,
            col_num=config.phys_col_num,
            row_num=config.row_num,
            dtype=dtype,
            T__K=T__K,
        )

        # --- WL 1-bit ON/OFF DAC (K single-bit sub-phases; one broadcast solve) ---
        self.wl_dac = VoltageDac.from_config(
            config=config.wl_dac_config,
            policy=policy.wl_dac_policy,
            inst_shape=(*inst_shape, config.row_num),
            dtype=dtype,
            T__K=T__K,
        )

        # --- Clamp seats = the array's boundary drivers (ideal r_out = 0) ---
        # The CABLC is column-MUX time-shared: 4 physical clamps per CIM-IO
        # (P/N x MSB/LSB), so its fabricated inst_shape carries the real device
        # count for PPA. In the value path the solver snapshots it per physical
        # column at the injected v_ref (offset off => the inst-shaped offset
        # buffer is never broadcast against the wider per-column shape). The SL
        # drive is one per-group active clamp forced to 0 V.
        self.cablc = VoltageDriver(
            config=config.cablc_config,
            policy=policy.cablc_policy,
            inst_shape=(*inst_shape, gn, _POLARITY_NUM, config.w_digit_num),
            dtype=dtype,
            T__K=T__K,
        )
        self.sl_driver = VoltageDriver(
            config=config.sl_driver_config,
            policy=policy.sl_driver_policy,
            inst_shape=(*inst_shape, gn, _POLARITY_NUM, config.w_digit_num),
            dtype=dtype,
            T__K=T__K,
        )

        # --- TMCSA SAR current ADC, one per IO; self-holds no ladder ---
        # The macro is the sole latency emitter: its ``t_cycle * serial`` event
        # already spans the whole access period, sensing included. Build the
        # TMCSA with ``record_latency=False`` so its ``convert`` logs no latency
        # event; the ADC keeps its real ``step_latency__ns`` and
        # ``t_conduct_per_step__ns`` (its conduction energy is untouched), and the
        # read-chain window ``t_other`` folds the honest sensing durations back in.
        self.tmcsa = SarSingleEndedCurrentAdc(
            config=config.adc_config,
            policy=policy.adc_policy,
            inst_shape=(*inst_shape, gn),
            dtype=dtype,
            T__K=T__K,
            record_latency=False,
        )
        # One static threshold source per fabricated sub-array copy ([mode, tap]
        # bank), shared across that copy's TMCSAs (no IO axis). It carries the
        # fabrication prefix so its static PPA scales with parallel copies like
        # every other seat; the per-instance ladder is passed straight to the ADC
        # (see :meth:`vec_mat_mul`) as the ADC's per-instance reference contract.
        self.adc_current_reference = CurrentReference(
            config=config.reference_config,
            policy=policy.reference_policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

        # --- Static-PPA seats (dynamic billed by the macro on its channels) ---
        # Both carry the fabrication prefix (one control block / PN-ISUB seat per
        # parallel sub-array copy), so their static PPA scales with the prefix in
        # step with the macro-billed dynamic energy, which already does.
        self.control = UnmodeledBlock(
            config=config.control_config,
            policy=policy.control_policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )
        self.pn_isub = UnmodeledBlock(
            config=config.pn_isub_config,
            policy=policy.pn_isub_policy,
            inst_shape=(*inst_shape, gn),
            dtype=dtype,
            T__K=T__K,
        )

        self._rescale_lut: dict[tuple[int, int], float] = {
            (e.mode, e.bits): e.rescale_factor for e in config.adc_calibration
        }

    # -----------------------------------------------------------------
    # Value-domain semantics
    # -----------------------------------------------------------------

    @property
    def x_range(self) -> tuple[int, int]:
        """Inclusive K-bit activation range — the macro decomposes it into WL sub-phases internally."""
        return (0, (1 << self.config.input_bit_num) - 1)

    @property
    def w_digit_count(self) -> int:
        """Magnitude digits per weight."""
        return self.config.w_digit_num

    @property
    def w_digit_radix(self) -> int:
        """Positional base of the digit combination."""
        return self.config.w_digit_radix

    @property
    def w_digit_range(self) -> tuple[int, int]:
        """Inclusive signed per-digit range — sign-magnitude, ``(-(radix-1), radix-1)``."""
        mag = self.config.w_digit_radix - 1
        return (-mag, mag)

    @property
    def adc_mode_num(self) -> int:
        """Number of ADC operating modes — the shared reference's ladder-row count."""
        return self.config.reference_config.mode_num

    @property
    def adc_max_bits(self) -> int:
        """Maximum ADC magnitude resolution [bits] — the TMCSA's ``bits``."""
        return self.config.adc_config.bits

    def adc_rescale_factor(self, *, adc_mode: int, adc_bits: int) -> float:
        """Rescale factor for ``(adc_mode, adc_bits)``; raises ``KeyError`` if uncalibrated."""
        try:
            return self._rescale_lut[(adc_mode, adc_bits)]
        except KeyError:
            available = sorted(self._rescale_lut)
            raise KeyError(
                f"(adc_mode={adc_mode}, adc_bits={adc_bits}) not in adc_calibration; "
                f"available (adc_mode, adc_bits): {available}"
            ) from None

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def program(self, w: Tensor) -> None:
        """Write the grouped array cells from one sign-magnitude digit tensor (LSB-first).

        Maps each logical weight's magnitude digits to its P/N cells: digit value
        ``+m`` writes the PWG cell to magnitude state ``m`` and the NWG cell to
        HRS (state 0); ``-m`` does the reverse; ``0`` leaves both at HRS. The
        framework transcoder supplies ``w`` LSB-first (digit 0 = LSB), matching the
        DSWCT/regroup digit order. The grouped cells are folded into the array's
        flat ``[phys_col, row]`` layout.

        Args:
            w: Sign-magnitude digit tensor whose shape matches
                ``self._w_layout_shape = (*inst_shape, col_num, w_digit_count,
                row_num)``. Entries must lie in :attr:`w_digit_range`.
        """
        # The magnitude routes to the state index (0 -> HRS, m -> the m-th
        # conductance state); the cell's table lookup is the sole digit-range
        # guard (an out-of-range state index crashes there, radix-irrelevant).
        config = self.config
        mag = w.abs()  # {0, ..., radix-1}
        is_neg = w < 0
        pwg = torch.where(is_neg, torch.zeros_like(mag), mag)
        nwg = torch.where(is_neg, mag, torch.zeros_like(mag))

        # Shape: [*, col, digit, row] -> [*, col, P/N, digit, row]
        w_pol = torch.stack((pwg, nwg), dim=-3)
        # col = io * mux_factor + slot
        # Shape: [*, col, P/N, digit, row] -> [*, io, slot, P/N, digit, row]
        w_grouped = w_pol.unflatten(-4, (config.io_num, config.mux_factor))
        # grouped cell layout [group_size, group_num, P/N, digit, row]
        # Shape: [*, io, slot, P/N, digit, row] -> [*, slot, io, P/N, digit, row]
        w_state_idx = w_grouped.transpose(-5, -4).contiguous()
        # phys_col = slot*io*P/N*digit
        # Shape: [*, slot, io, P/N, digit, row] -> [*, phys_col, row]
        w_flat = w_state_idx.flatten(-5, -2)
        self.array.program(w_flat)

    def vec_mat_mul(self, x: Tensor, *, adc_mode: int, adc_bits: int) -> Tensor:
        """Run the array solve + readout chain over the K WL sub-phases.

        The K activation bits are bit-expanded into K WL planes (LSB first) and
        driven through the WL DAC. All K planes are solved in ONE broadcast array
        solve (the x-bit axis rides the solve leading), each carrying its own
        per-plane conduction window, yielding the per-column BL port current
        ``I_DL``; the DSWCT place-value ratios, the SINWP-SC input-radix combine,
        the PN-ISUB subtraction, and the TMCSA quantization against the per-instance
        reference ladder are pure vectorized tensor operations. Each readout stage
        is billed inline at its production site.

        The column-MUX is a pure reshape (mux slot rides the grouped axis), a valid
        grouped<->flat translation because each physical column is an independent
        BL/SL ladder.

        Args:
            x: Activation tensor with primitive trailing ``[row_num]``; entries in
                :attr:`x_range`. Rows outside the caller's active window (at most
                :attr:`max_active_rows` live rows per sub-phase) must arrive
                zeroed. Every leading axis is anonymous broadcast batch.
            adc_mode: ADC operating-point index selecting the shared reference
                mode row; valid values are ``[0, adc_mode_num)``.
            adc_bits: ADC resolution [bits] the TMCSA quantizes at.

        Returns:
            Signed-magnitude raw-code tensor with the same leading order and
            primitive trailing ``[col_num]``.
        """
        config = self.config
        n_x_bits = config.input_bit_num
        v_dd = config.v_dd__V
        gs = config.mux_factor  # column-MUX slot count per IO (group_size)
        gn = config.io_num  # CIM-IO sense-lane count (group_num)
        wd = config.w_digit_num
        x_long = x.long()  # dtype guard for >> and the bit-expand

        # --- Step 1: Bit-expand x into K WL planes (LSB first) + WL DAC ---

        # The x-bit axis lands at -2 so it becomes the last leading axis of the
        # solve, folding into the array's broadcast leading.
        x_bit_index = torch.arange(n_x_bits, device=x.device).view(n_x_bits, 1)
        # Shape: [*B, row] -> [*B, x_bits, row]
        planes = (x_long.unsqueeze(-2) >> x_bit_index) & 1
        v_wl = self.wl_dac.convert(planes)

        # --- Step 2: Solve the array once (cells + wire IR drop) -> I_DL ---

        # All K WL planes settle in a single DC solve; the per-plane conduction
        # window is a vector applied POST-solve inside the array energy, so the
        # steady currents are window-independent and the x-bit axis rides the
        # solve leading (the array returns both tensors with x_bits already in
        # place, no manual stack).
        bl_v_ref__V = torch.tensor(config.v_bl_clamp__V, dtype=self.dtype, device=x.device)
        sl_v_ref__V = torch.tensor(_V_SL_DRIVE__V, dtype=self.dtype, device=x.device)
        window_array__ns = torch.tensor(config.window_array__ns, dtype=self.dtype, device=x.device)
        steady = self.array.solve_array(
            v_wl,
            bl_driver=self.cablc,
            bl_v_ref__V=bl_v_ref__V,
            sl_driver=self.sl_driver,
            sl_v_ref__V=sl_v_ref__V,
        )
        # Shape: [*B, x_bits, phys_col]
        i_bl_port = steady.i_bl_port__uA

        # Bill cablc INLINE at the I_DL production site: the whole input branch
        # V_DD * I_DL, summed over physical columns per bit and weighted by the bit
        # window. The array bills only its capacitive cycling; the read current
        # conduction energy is entirely the macro's (the macro owns the per-bit
        # conduction window t).
        # Shape: [*B, x_bits, phys_col] -> [*B, x_bits]
        read_power = (v_dd * i_bl_port).sum(dim=-1)
        # Shape: [*B, x_bits] -> [*B]
        e_cablc = (read_power * window_array__ns).sum(dim=-1)
        self._log_dynamic_energy(e_cablc, channel="cablc")

        # Recover the grouped readout layout from the flat BL port;
        # column-separable, so a pure reshape.
        # Shape: [*B, x_bits, phys_col] -> [*B, x_bits, gs, gn, P/N, wd]
        i_dl = i_bl_port.unflatten(-1, (gs, gn, _POLARITY_NUM, wd))

        # --- Step 3: DSWCT place-value weighting -> I_WDL ---

        # LSB-first per-digit mirror ratios.
        digit_ratios = i_dl.new_tensor(config.digit_ratios)
        i_wdl = i_dl * digit_ratios
        # Bill dswct INLINE at the output-leg production site.
        # Shape: [*B, x_bits, gs, gn, P/N, wd] -> [*B, x_bits]
        i_wdl_per_bit = i_wdl.abs().sum(dim=(-4, -3, -2, -1))
        # Shape: [*B, x_bits] -> [*B]
        e_dswct = v_dd * (i_wdl_per_bit * window_array__ns).sum(dim=-1)
        self._log_dynamic_energy(e_dswct, channel="dswct")

        # --- Step 4: SINWP-SC spatial + temporal input-radix combine -> I_DL_PN ---

        # Shape: [*B, x_bits, gs, gn, P/N, wd] -> [*B, x_bits, gs, gn, P/N]
        i_dl_pn_bit = i_wdl.sum(dim=-1)
        # Bill sinwp_sc INLINE at the held/live-leg production site.
        window_sc__ns = i_dl.new_tensor(config.window_sc__ns)
        # Shape: [*B, x_bits, gs, gn, P/N] -> [*B, x_bits]
        i_sc_per_bit = i_dl_pn_bit.sum(dim=(-3, -2, -1))
        # Shape: [*B, x_bits] -> [*B]
        e_sinwp = v_dd * (i_sc_per_bit * window_sc__ns).sum(dim=-1)
        self._log_dynamic_energy(e_sinwp, channel="sinwp_sc")

        # Temporal weighted sum over bits (input radix).
        # Shape: [x_bits, gs=1, gn=1, P/N=1]
        x_bit_ratios = i_dl.new_tensor(config.x_bit_ratios).view(n_x_bits, 1, 1, 1)
        # Shape: [*B, x_bits, gs, gn, P/N] -> [*B, gs, gn, P/N]
        i_dl_pn = (i_dl_pn_bit * x_bit_ratios).sum(dim=-4)

        # --- Step 5: PN-ISUB single-ended magnitude + sign ---

        # Shape: [*B, gs, gn, P/N] -> [*B, gs, gn]
        i_dl_pn_p = i_dl_pn[..., 0]
        i_dl_pn_n = i_dl_pn[..., 1]
        i_sub = (i_dl_pn_p - i_dl_pn_n).abs()
        sign = i_dl_pn_n > i_dl_pn_p
        # Bill pn_isub INLINE (three-branch conduction + per-op comparator).
        e_pnisub = v_dd * config.t_other__ns * (i_dl_pn_p + i_dl_pn_n + i_sub) + config.e_pn_isub_per_op__fJ
        self._log_dynamic_energy(e_pnisub, channel="pn_isub")

        # --- Step 6: TMCSA quantize against the per-instance reference ladder ---

        # The reference carries the fabrication prefix [*inst, mode, tap]; the
        # caller-selected mode row is passed straight to the ADC as the
        # per-instance ladder [*inst, tap] (n_ref = tap), no 1-D collapse.
        ref_snap = self.adc_current_reference.snapshot(
            shape=(
                *self.inst_shape,
                self.adc_current_reference.mode_num,
                self.adc_current_reference.tap_num,
            ),
        )
        adc_i_refs__uA = self.adc_current_reference.i_ref__uA(ref_snap)
        # Shape: [*inst, mode, tap] -> [*inst, tap]
        adc_refs_mode__uA = adc_i_refs__uA[..., adc_mode, :]
        # Shape: [*B, gs, gn]
        code = self.tmcsa.convert(i_sub, adc_refs_mode__uA, bits=adc_bits)
        signed = (1 - 2 * sign.long()) * code

        # --- Step 7: Control energy + the sole latency event ---

        # The control fires once per conversion cycle (shared across the
        # CIM-IOs), billed over the [*B, gs] leading. The latency is one
        # operating period t_cycle scaled by the serial-access count (mux_factor
        # column-MUX steps per fabricated instance); the macro is the sole
        # latency emitter.
        # Shape: [*B, gs]
        e_control = signed.new_full(signed.shape[:-1], config.e_control_per_op__fJ, dtype=torch.float32)
        self._log_dynamic_energy(e_control, channel="control")
        serial_op_count = max(1, signed.numel() // max(self.inst_count * gn, 1))
        latency__ns = torch.tensor(
            config.t_cycle__ns * serial_op_count,
            device=signed.device,
            dtype=torch.float32,
        )
        self._log_latency(latency__ns)

        # col = io * mux_factor + slot = gn * group_size + gs
        # Shape: [*B, gs, gn] -> [*B, col_num]
        return signed.transpose(-2, -1).flatten(-2)
