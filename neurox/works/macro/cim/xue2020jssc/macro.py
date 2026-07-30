"""Xue2020 JSSC SINWP 1T1R CIM sub-array macro composing the current-mode readout chain.

Exposes a signed VMM over ``row_num`` inputs and ``col_num`` outputs. Each
logical weight is a sign-magnitude value (sign + ``w_digit_num`` digits of radix
``w_digit_radix``) carried by ``w_digit_num * 2`` physical cells, a P (PWG) and
an N (NWG) cell per digit; the scheme-local serial-column array, the DSWCT
place-value legs, the SINWP-SC input-radix combine, the PN-ISUB subtraction, and
the TMCSA current SAR ADC recover the signed magnitude.

The macro is the scheme's sole latency emitter, owns the per-call conduction
windows it injects into the self-billing readout modules, and bills the whole
input branch plus the control per-op constant on its own two channels.

See also:
    docs/works/macro/cim/xue2020jssc/model.md
"""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor

from neurox.common.encoding import TrueFormTranscoder
from neurox.primitive.analog import (
    Iref,
    IrefConfig,
    IrefPolicy,
    UnmodeledBlock,
    UnmodeledBlockConfig,
    UnmodeledBlockPolicy,
    VoltageDriver,
    VoltageDriverConfig,
    VoltageDriverPolicy,
    Vref,
    VrefConfig,
    VrefPolicy,
)
from neurox.primitive.analog.current_adc import (
    SarIadc,
    SarIadcConfig,
    SarIadcPolicy,
)
from neurox.primitive.analog.voltage_dac import Vdac, VdacConfig, VdacPolicy
from neurox.primitive.macro.cim import (
    CimMacro,
    CimMacroConfig,
    CimMacroMode,
    CimMacroPolicy,
    IdealCimMacro,
    map_magnitude_input_code,
)
from neurox.primitive.xbar.array import XbarArray1t1rConfig, XbarArray1t1rPolicy

from .array import SerialColumnXbarArray
from .dswct import Dswct, DswctConfig, DswctPolicy
from .pn_isub import PnIsub, PnIsubConfig, PnIsubPolicy
from .sinwp_sc import SinwpSc, SinwpScConfig, SinwpScPolicy
from .tmcsa import Tmcsa, TmcsaConfig, TmcsaPolicy

_POLARITY_NUM = 2  # PWG, NWG per weight digit


class Xue2020JsscCimMacroConfig(CimMacroConfig):
    """Configuration for the xue2020jssc SINWP 1T1R CIM sub-array.

    Every physical / PPA field is required; values live in the TOML, never as a
    code default. The DSWCT / SINWP-SC ratios are derived from the two MSB
    anchor fields. The cell, wire parasitics, and DC solver live inside
    :attr:`array_config`.

    Attributes:
        max_active_num: Input-block size selected per conversion; the engine,
            not the macro, serializes across row blocks.
        w_digit_num: Magnitude digits per weight; ``>= 1``. A single digit is
            one polarity pair with no cross-digit combine.
        w_digit_radix: Positional base of the magnitude digits; ``>= 2``.
            ``> 2`` needs a radix-level conductance table in the cell config.
        input_bit_num: Activation bit width K; ``>= 1``. K serial single-bit WL
            sub-phases, LSB first. ``1`` runs the live bit alone.
        mux_factor: Logical columns per CIM-IO — the column-MUX depth.
            ``col_num`` must divide by it; ``io_num = col_num // mux_factor``.
            Also the serial-access count per conversion (the static-energy
            latency multiplier).
        dswct_ratio_msb: DSWCT MSB-leg mirror ratio anchor. The per-digit ratio
            ``r_d = dswct_ratio_msb * w_digit_radix**(d - (D-1))`` (LSB-first d)
            is derived DOWNWARD from it.
        sc_ratio_msb: SINWP-SC MSB-bit combine ratio anchor. The per-input-bit
            leg ratio ``s_k = sc_ratio_msb * 2**(k - (K-1))`` (LSB-first k) is
            derived DOWNWARD from it; ``input_bit_num = 1`` uses the anchor
            directly with the sample-and-hold leg off.
        t_sample__ns: Sample sub-phase windows, one per SAMPLED input bit
            (length ``input_bit_num - 1``; empty for K=1). Dynamic-energy only.
            The live bit (K-1) conducts in ``t_other`` instead.
        t_settle__ns: Tail non-sensing settle window — the live-bit settle ONLY;
            part of ``t_other``, dynamic-energy only. The SAR sensing durations
            enter ``t_other`` separately, from the ADC step windows.
        t_cycle__ns: Declared operating period, the static-energy time base. The
            macro logs latency ``t_cycle * serial``, so ``leakage_energy =
            leakage_power * latency`` is the static energy over the full period.
            Must be ``>=`` the sum of the conduction windows.
        v_dd__V: Supply-rail voltage every channelled branch is billed across,
            including the whole input branch ``V_DD * I_DL`` on the ``cablc``
            channel.
        e_control_per_op__fJ: Control per-conversion dynamic energy (address
            decode, CMD precharge, timing) billed under the ``control`` channel;
            it includes the CMD precharge, so NO CMD capacitance is modeled.
        control_config: Static-PPA seat for the (functionally unmodeled) control
            block.
        dswct_config: DSWCT place-value mirror-bank module config (cap knob +
            static PPA seat).
        sinwp_sc_config: SINWP-SC sample-and-hold combine module config
            (hold-cap knob + static PPA seat).
        pn_isub_config: PN-ISUB subtractor module config — the comparator
            per-decision energy plus the bias/comparator static PPA seat.
        tmcsa_config: TMCSA phase-resolved conversion-billing module config —
            the PH2/PH3 per-step windows, the per-step fixed constant, and the
            static PPA seat. The window lists span exactly ``adc_config.bits``
            steps and satisfy ``t_ph2[s] + t_ph3[s] <=
            adc_config.step_latency__ns[s]`` (PH1/PH4 occupy the rest).
        array_config: Nested 1T1R pure-array config — the linearized cell tables
            carrying the programmable weights, the BL / SL / WL wire parasitics
            (segment and first-segment ``R > 0``, which the solver requires), and
            the DC solver knobs. Its ``latency_per_op__ns`` should be 0 (the
            macro is the sole latency emitter).
        wl_dac_config: WL 1-bit ON/OFF DAC config; ``latency_per_op__ns`` should
            be 0 (the WL sub-phase folds into ``t_cycle``).
        cablc_config: CABLC BL-clamp seat = the array's ``bl_driver`` (Thevenin
            VoltageDriver, ``r_out__MOhm = 0`` ideal source — the wire IR drop is
            the array's, not the clamp's; ``energy_per_op__fJ = 0``). Its
            reference is injected per call from :attr:`cablc_vref_config`.
        cablc_vref_config: Dedicated CABLC reference source — a
            :class:`~neurox.primitive.analog.Vref` holding the degenerate
            ``[[v]]`` bank (one mode row, one tap) whose sole tap is the BL clamp
            reference, sampled per solve at the array's full call shape and fed
            to the array's ``bl_driver`` as its Thevenin reference; the per-cell
            ``V_BL`` droops below it by the wire IR drop the solver computes. The
            tap must be in ``[0, v_dd__V]``.
        sl_driver_config: SL ideal-clamp seat = the array's ``sl_driver``
            (VoltageDriver, ``r_out__MOhm = 0``). The SL is a direct ground tie,
            so its reference is a plain 0 V tensor, not a reference source.
        adc_config: Kernel SAR current-ADC config (B-form) — the VALUE
            converter. The macro builds it with ``enable_latency_record=False``
            (the macro is the sole latency emitter) and
            ``enable_energy_record=False`` (the conversion energy is billed by
            the :attr:`tmcsa_config` module instead, so the kernel energy knobs
            are inert here). Its ``step_latency__ns`` stays the physical sensing
            duration and feeds the read-chain window :attr:`t_other__ns`.
        reference_config: Static Iref — the ``[mode][tap]`` threshold bank; the
            macro NAMES the mode and the source returns that row as the ladder
            the ADC reads, which handles bit width internally.
            ``tap_num == 2**adc_config.bits - 1``, and every row ascends
            strictly (the macro checks it, since the ladder ordering is the
            TMCSA's knowledge, not the source's).
        modes: One :class:`CimMacroMode` per quantization mode, indexed by
            ``quantization_mode``: the canonical MAC-unit conversion window, the
            ADC input code range the magnitude converter discriminates, and the
            calibrated rescale factor at ``adc_config.bits``. The count must
            match ``reference_config.mode_num`` — one threshold ladder row per
            mode.
    """

    # === Weight / input geometry ===

    w_digit_num: int
    w_digit_radix: int
    input_bit_num: int
    mux_factor: int

    # === Ratio anchors (DSWCT / SINWP-SC ratios derived DOWNWARD from these) ===

    dswct_ratio_msb: float
    sc_ratio_msb: float

    # === Conduction windows (dynamic-energy only) + static time base ===

    t_sample__ns: tuple[float, ...]
    t_settle__ns: float
    t_cycle__ns: float

    # === Supply voltage ===

    v_dd__V: float

    # === Per-op dynamic constants ===

    e_control_per_op__fJ: float

    # === Static-PPA seat (control) ===

    control_config: UnmodeledBlockConfig

    # === Scheme-local readout modules ===

    dswct_config: DswctConfig
    sinwp_sc_config: SinwpScConfig
    pn_isub_config: PnIsubConfig
    tmcsa_config: TmcsaConfig

    # === Device-bearing sub-blocks (full nested configs) ===

    array_config: XbarArray1t1rConfig
    wl_dac_config: VdacConfig
    cablc_config: VoltageDriverConfig
    cablc_vref_config: VrefConfig
    sl_driver_config: VoltageDriverConfig
    adc_config: SarIadcConfig
    reference_config: IrefConfig

    # === Quantization modes ===

    modes: tuple[CimMacroMode, ...]

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
        """Live/tail window — ``t_settle__ns + sum(adc_config.step_latency__ns)``.

        The live input bit (K-1) has no sample sub-phase of its own; its
        conduction, the SINWP-SC combine, the PN-ISUB, and the TMCSA sensing all
        fall in this tail window. The SAR sensing durations are added here, so the
        whole read chain conducts through sensing while ``t_settle__ns`` stays the
        pure non-sensing settle.
        """
        return self.t_settle__ns + sum(self.adc_config.step_latency__ns)

    @property
    def window_array__ns(self) -> tuple[float, ...]:
        """Per-input-bit input-branch (array / CABLC / DSWCT) conduction window.

        Sampled bit ``k`` conducts for its sample window ``t_sample__ns[k]``; the
        live bit (K-1) conducts for ``t_other``.
        """
        k_live = self.input_bit_num - 1
        return tuple(self.t_sample__ns[k] if k < k_live else self.t_other__ns for k in range(self.input_bit_num))

    @property
    def window_sc__ns(self) -> tuple[float, ...]:
        """Per-input-bit SINWP-SC leg conduction window.

        A bit-``k`` leg is held from its sample sub-phase to the end, so it
        conducts for the suffix sum of the remaining sample windows plus the tail
        ``t_other``: ``sum(t_sample__ns[k:]) + t_other``. The live bit reduces to
        ``t_other``.
        """
        return tuple(sum(self.t_sample__ns[k:]) + self.t_other__ns for k in range(self.input_bit_num))

    @property
    def conduction_span__ns(self) -> float:
        """Total conduction span ``sum(t_sample) + t_other``; must fit in ``t_cycle``."""
        return sum(self.t_sample__ns) + self.t_other__ns

    def validate(self) -> None:
        super().validate()

        # --- Data geometry ---

        # General sign-magnitude weight: >= 1 magnitude digit, radix >= 2 so a
        # digit carries at least the {0, 1} magnitude the polarity pair encodes.
        self._require_pos(self.w_digit_num, "w_digit_num")
        if not (self.w_digit_radix >= 2):
            raise ValueError(f"require: w_digit_radix ({self.w_digit_radix}) >= 2")

        self._require_pos(self.input_bit_num, "input_bit_num")
        self._require_pos(self.mux_factor, "mux_factor")

        # --- Analog transfer and timing ---

        self._require_pos(self.dswct_ratio_msb, "dswct_ratio_msb")
        self._require_pos(self.sc_ratio_msb, "sc_ratio_msb")

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

        self._require_non_neg(self.v_dd__V, "v_dd__V")
        # The CABLC consumes exactly one reference tap and holds it across every
        # mode, so its dedicated source is the degenerate single-row single-tap
        # bank. The clamp reference is a BL node between the SL ground and the
        # V_DD supply, so it must not exceed the rail.
        if self.cablc_vref_config.mode_num != 1:
            raise ValueError(
                f"require: cablc_vref_config.mode_num ({self.cablc_vref_config.mode_num}) == 1 "
                "— the CABLC clamp reference does not follow the quantization mode"
            )
        if self.cablc_vref_config.tap_num != 1:
            raise ValueError(
                f"require: cablc_vref_config.tap_num ({self.cablc_vref_config.tap_num}) == 1 "
                "— the CABLC clamp consumes a single reference tap"
            )
        v_bl_clamp__V = self.cablc_vref_config.v_refs__V[0][0]
        if not (v_bl_clamp__V <= self.v_dd__V):
            raise ValueError(
                f"require: cablc_vref_config.v_refs__V[0][0] ({v_bl_clamp__V}) <= v_dd__V ({self.v_dd__V})"
            )

        # --- TMCSA phase windows against the ADC step timing ---

        # The billing module resolves each of the ADC's binary-search steps
        # into PH2/PH3 conduction phases: one window pair per step, and the
        # phases must fit inside that step's latency (PH1/PH4 fill the rest).
        if len(self.tmcsa_config.t_ph2_per_step__ns) != self.adc_config.bits:
            raise ValueError(
                f"require: len(tmcsa_config.t_ph2_per_step__ns) ({len(self.tmcsa_config.t_ph2_per_step__ns)}) == "
                f"adc_config.bits ({self.adc_config.bits}) — one PH2/PH3 window pair per conversion step"
            )
        for s, (t_ph2, t_ph3) in enumerate(
            zip(self.tmcsa_config.t_ph2_per_step__ns, self.tmcsa_config.t_ph3_per_step__ns, strict=True)
        ):
            step = self.adc_config.step_latency__ns[s]
            if t_ph2 + t_ph3 > step:
                raise ValueError(
                    f"require: tmcsa_config t_ph2[{s}] + t_ph3[{s}] ({t_ph2 + t_ph3}) <= "
                    f"adc_config.step_latency__ns[{s}] ({step}) — PH1/PH4 occupy the rest of the step"
                )

        # --- ADC reference and quantization modes ---

        # The TMCSA reads its ladder from the shared Iref, so the tap
        # count must match the binary-search depth exactly. Lower bit widths
        # ride this one max-bits ladder; they need no taps of their own.
        want_taps = (1 << self.adc_config.bits) - 1
        if self.reference_config.tap_num != want_taps:
            raise ValueError(
                f"require: reference_config.tap_num ({self.reference_config.tap_num}) == "
                f"2**adc_config.bits - 1 ({want_taps})"
            )

        # What a reference row MEANS is the consumer's knowledge, so the source
        # does not order its taps: the TMCSA reads each row as a binary-search
        # decision ladder, and only a strictly ascending ladder decodes.
        for m, row in enumerate(self.reference_config.i_refs__uA):
            self._require_increasing(row, f"reference_config.i_refs__uA[{m}]")

        # Each CimMacroMode validates its own canonical window and positive
        # rescale factor on construction; the macro pins the mode count against
        # the reference bank, since a mode IS one ladder row.
        mode_num = self.reference_config.mode_num
        if len(self.modes) != mode_num:
            raise ValueError(
                f"require: len(modes) ({len(self.modes)}) == reference_config.mode_num ({mode_num}) "
                "— one declared quantization mode per threshold ladder row"
            )


class Xue2020JsscCimMacroPolicy(CimMacroPolicy):
    """Composite nonideality policy for :class:`Xue2020JsscCimMacro`.

    One child policy per owned ``ModuleBase`` block. This scheme models no
    mismatch or noise, so the sanctioned ``all_off`` preset (every child at its
    lossless baseline) is the only intended policy.

    Attributes:
        array_policy: 1T1R pure-array policy (the cell policy + the solver chunk knob).
        wl_dac_policy: WL DAC policy.
        cablc_policy: CABLC BL-clamp seat policy.
        cablc_vref_policy: CABLC reference-source policy.
        sl_driver_policy: SL ideal-driver seat policy.
        adc_policy: TMCSA SAR current-ADC policy.
        reference_policy: Threshold current-reference policy.
        control_policy: Control static-seat policy.
        dswct_policy: DSWCT module policy (source-free).
        sinwp_sc_policy: SINWP-SC module policy (source-free).
        pn_isub_policy: PN-ISUB module policy (source-free).
        tmcsa_policy: TMCSA conversion-billing module policy (source-free).
    """

    array_policy: XbarArray1t1rPolicy
    wl_dac_policy: VdacPolicy
    cablc_policy: VoltageDriverPolicy
    cablc_vref_policy: VrefPolicy
    sl_driver_policy: VoltageDriverPolicy
    adc_policy: SarIadcPolicy
    reference_policy: IrefPolicy
    control_policy: UnmodeledBlockPolicy
    dswct_policy: DswctPolicy
    sinwp_sc_policy: SinwpScPolicy
    pn_isub_policy: PnIsubPolicy
    tmcsa_policy: TmcsaPolicy


@CimMacro.register_neurox_module(
    config_type=Xue2020JsscCimMacroConfig,
    policy_type=Xue2020JsscCimMacroPolicy,
)
class Xue2020JsscCimMacro(CimMacro[Xue2020JsscCimMacroConfig, Xue2020JsscCimMacroPolicy]):
    """Xue2020 JSSC SINWP 1T1R CIM sub-array with an inline current-mode readout chain.

    Composes the serial-column 1T1R array (cell grid + wire + solver) with the
    macro-owned WL DAC, the CABLC / SL clamp seats, the DSWCT / SINWP-SC /
    PN-ISUB readout modules, the TMCSA and its shared reference, and the control
    static seat. Each logical weight occupies ``w_digit_num * 2`` grouped cells,
    a polarity pair per digit.

    Args:
        config: Macro configuration.
        policy: Macro nonideality policy.
        input_num: Logical input length, bound to ``row_num`` during construction.
        output_num: Logical output length, bound to ``col_num`` during construction.
        inst_shape: Per-instance multiplicity prefix.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    # === Circuit constant buffers ===

    _sl_v_ref__V: Tensor  # Shape: []
    _window_array__ns: Tensor  # Shape: [x_bits]
    _window_sc__ns: Tensor  # Shape: [x_bits]
    _t_cycle__ns: Tensor  # Shape: []

    def __init__(
        self,
        *,
        config: Xue2020JsscCimMacroConfig,
        policy: Xue2020JsscCimMacroPolicy,
        input_num: int,
        output_num: int,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            input_num=input_num,
            output_num=output_num,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )
        self.row_num = input_num
        self.col_num = output_num
        if self.col_num % config.mux_factor != 0:
            raise ValueError(
                f"require: col_num ({self.col_num}) % mux_factor ({config.mux_factor}) == 0 "
                "(the CIM-IO regrouping is an exact reshape)"
            )
        if config.max_active_num > self.row_num:
            raise ValueError(f"require: max_active_num ({config.max_active_num}) <= row_num ({self.row_num})")
        self._w_transcoder = TrueFormTranscoder(
            radix=config.w_digit_radix,
            digit_count=config.w_digit_num,
        )
        self._x_transcoder = TrueFormTranscoder(radix=2, digit_count=config.input_bit_num)
        self._init_children(dtype=dtype, T__K=T__K)
        self._register_model_buffers(dtype=dtype)

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    def _init_children(self, *, dtype: torch.dtype, T__K: float) -> None:
        """Construct the array, readout chain, and static PPA seats."""
        config = self.config
        policy = self.policy
        gn = self.col_num // config.mux_factor
        phys_col_num = self.col_num * config.w_digit_num * _POLARITY_NUM

        # --- Programmable weights + wire + solver: the serial-column array ---

        # Column-MUX placement: phys_col = ((slot * gn + io) * polarity + pol) *
        # w_digit + digit, the bijection (slot, io, polarity, digit) -> physical
        # column the array seats its cells by.
        slot_map = torch.arange(phys_col_num, dtype=torch.long).reshape(
            config.mux_factor, gn, _POLARITY_NUM, config.w_digit_num
        )
        self.array = SerialColumnXbarArray(
            config=config.array_config,
            policy=policy.array_policy,
            inst_shape=self.inst_shape,
            row_num=self.row_num,
            col_num=phys_col_num,
            slot_map=slot_map,
            dtype=dtype,
            T__K=T__K,
        )

        # --- WL 1-bit ON/OFF DAC (K single-bit sub-phases; one broadcast solve) ---

        self.wl_dac = Vdac.from_config(
            config=config.wl_dac_config,
            policy=policy.wl_dac_policy,
            inst_shape=(*self.inst_shape, self.row_num),
            dtype=dtype,
            T__K=T__K,
        )

        # --- CABLC reference source (dedicated single-tap Vref) ---

        # One reference-generation circuit per fabricated sub-array copy feeds
        # that copy's whole CABLC clamp bank, so the source carries the
        # fabrication prefix and nothing else.
        self.cablc_vref = Vref(
            config=config.cablc_vref_config,
            policy=policy.cablc_vref_policy,
            inst_shape=self.inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

        # --- Clamp seats = the array's boundary drivers (ideal r_out = 0) ---

        # The CABLC is column-MUX time-shared: one physical clamp per (IO, polarity,
        # digit), so the fabricated inst_shape is the real device count for PPA
        # while the solver snapshots it per physical column at the injected
        # v_ref. The SL drive is one per-group active clamp forced to 0 V.
        self.cablc = VoltageDriver(
            config=config.cablc_config,
            policy=policy.cablc_policy,
            inst_shape=(*self.inst_shape, gn, _POLARITY_NUM, config.w_digit_num),
            dtype=dtype,
            T__K=T__K,
        )
        self.sl_driver = VoltageDriver(
            config=config.sl_driver_config,
            policy=policy.sl_driver_policy,
            inst_shape=(*self.inst_shape, gn, _POLARITY_NUM, config.w_digit_num),
            dtype=dtype,
            T__K=T__K,
        )

        # --- Readout modules: DSWCT -> SINWP-SC -> PN-ISUB (self-billing) ---

        # The ratio buffers are config-derived and injected here; the conduction
        # windows are injected per call in vec_mat_mul.
        self.dswct = Dswct(
            config=config.dswct_config,
            policy=policy.dswct_policy,
            inst_shape=(*self.inst_shape, gn, _POLARITY_NUM),
            digit_ratios=torch.tensor(config.digit_ratios, dtype=dtype),
            v_dd__V=config.v_dd__V,
        )
        self.sinwp_sc = SinwpSc(
            config=config.sinwp_sc_config,
            policy=policy.sinwp_sc_policy,
            inst_shape=(*self.inst_shape, gn, _POLARITY_NUM),
            bit_ratios=torch.tensor(config.x_bit_ratios, dtype=dtype),
            v_dd__V=config.v_dd__V,
        )
        self.pn_isub = PnIsub(
            config=config.pn_isub_config,
            policy=policy.pn_isub_policy,
            inst_shape=(*self.inst_shape, gn),
            v_dd__V=config.v_dd__V,
        )

        # --- Kernel SAR current ADC (value only), one per IO; no ladder ---

        # Latency-silent because the macro's t_cycle * serial event already
        # spans the whole access period, sensing included; energy-silent because
        # the tmcsa module below bills the conversion phase-resolved. The ADC
        # keeps its physical step_latency__ns, which feeds t_other.
        self.adc = SarIadc(
            config=config.adc_config,
            policy=policy.adc_policy,
            inst_shape=(*self.inst_shape, gn),
            dtype=dtype,
            T__K=T__K,
            enable_latency_record=False,
            enable_energy_record=False,
        )
        self.tmcsa = Tmcsa(
            config=config.tmcsa_config,
            policy=policy.tmcsa_policy,
            inst_shape=(*self.inst_shape, gn),
            v_dd__V=config.v_dd__V,
            dtype=dtype,
        )
        # One threshold source per fabricated sub-array copy, shared across that
        # copy's TMCSAs: the inst_shape carries the fabrication prefix and no IO
        # axis, so its static PPA scales with parallel copies like every seat.
        self.adc_current_reference = Iref(
            config=config.reference_config,
            policy=policy.reference_policy,
            inst_shape=self.inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

        # --- Control static-PPA seat (dynamic billed on the control channel) ---

        # One control block per parallel sub-array copy, so its static PPA
        # scales with the fabrication prefix in step with the macro-billed
        # dynamic energy.
        self.control = UnmodeledBlock(
            config=config.control_config,
            policy=policy.control_policy,
            inst_shape=self.inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

    def _register_model_buffers(self, *, dtype: torch.dtype) -> None:
        """Register fixed tensors consumed by the readout path."""
        config = self.config
        # SL direct ground tie: a plain all-zeros reference, no Vref module.
        self.register_buffer("_sl_v_ref__V", torch.zeros((), dtype=dtype), persistent=False)
        self.register_buffer("_window_array__ns", torch.tensor(config.window_array__ns, dtype=dtype), persistent=False)
        self.register_buffer("_window_sc__ns", torch.tensor(config.window_sc__ns, dtype=dtype), persistent=False)
        self.register_buffer("_t_cycle__ns", torch.tensor(config.t_cycle__ns, dtype=dtype), persistent=False)

    @property
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive K-bit activation range — the macro decomposes it into WL sub-phases internally."""
        return (0, (1 << self.config.input_bit_num) - 1)

    @property
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive logical sign-magnitude weight range."""
        return self._w_transcoder.value_range

    @property
    def quantization_input_ranges(self) -> tuple[tuple[int, int], ...]:
        """Canonical MAC-unit conversion window per declared quantization mode."""
        return tuple(mode.quantization_input_range for mode in self.config.modes)

    @property
    def adc_max_bits(self) -> int:
        """Maximum ADC magnitude resolution [bits] — the TMCSA's ``bits``."""
        return self.config.adc_config.bits

    def _mode(self, quantization_mode: int) -> CimMacroMode:
        """Return one declared quantization mode.

        Args:
            quantization_mode: Mode index in ``[0, len(config.modes))``.

        Raises:
            ValueError: The index is outside the declared modes.
        """
        modes = self.config.modes
        if not (0 <= quantization_mode < len(modes)):
            raise ValueError(f"require: quantization_mode ({quantization_mode}) in [0, {len(modes)})")
        return modes[quantization_mode]

    def _max_bits_rescale_factor(self, quantization_mode: int) -> float:
        """Return the calibrated rescale factor of one mode at :attr:`adc_max_bits`."""
        return self._mode(quantization_mode).max_bits_rescale_factor

    def map_quantization_input_code(self, code: Tensor, *, quantization_mode: int) -> tuple[Tensor, tuple[int, int]]:
        """Map exact MAC-unit codes onto the sign-magnitude ADC input grid.

        The PN-ISUB hands the TMCSA a single-ended magnitude, so the converter
        discriminates ``|code|``. Its grid is the ladder itself, whose span the
        config declares as the mode's ``adc_input_code_range`` — circuit
        knowledge that does not follow the window width: a window whose top
        magnitude exceeds the ladder simply saturates.

        Args:
            code: Exact integer plane dots.
            quantization_mode: Mode index in ``[0, len(quantization_input_ranges))``.

        Returns:
            The magnitudes and the mode's declared ADC input code range.
        """
        mode = self._mode(quantization_mode)
        magnitude, _ = map_magnitude_input_code(code, code_range=mode.quantization_input_range)
        return magnitude, mode.adc_input_code_range

    def to_ideal(self) -> IdealCimMacro:
        """Return an ideal twin one bit wider than the TMCSA.

        The readout resolves a sign plus ``adc_max_bits`` magnitude bits, a
        signed span the zero-point twin only reaches at ``adc_max_bits + 1``
        bits. The twin is unfaithful by design at the endpoints — it holds one
        phantom bottom level and a single zero where the sign-magnitude
        encoding wastes two — so it is a reference, never a bit-exact model.
        """
        twin = super().to_ideal()
        config = dataclasses.replace(twin.config, adc_max_bits=self.adc_max_bits + 1)
        return IdealCimMacro(
            config=config,
            policy=twin.policy,
            input_num=self.row_num,
            output_num=self.col_num,
            inst_shape=self.inst_shape,
            dtype=self._dtype,
            T__K=self._T__K,
        )

    def _organize_w(self, w: Tensor) -> Tensor:
        """Map logical weights into the array's flat physical-column layout.

        Args:
            w: Logical weight tensor.
                Shape: ``[*inst_shape, row_num, col_num]``.

        Returns:
            State indices in the array's flat physical-column layout.
            Shape: ``[*inst_shape, phys_col_num, row_num]``.
        """
        # Shape: [*inst_shape, row, col] -> [*inst_shape, row, col, w_digit_num]
        w = self._w_transcoder.encode(w, dim=-1)
        # Shape: [*inst_shape, row, col, w_digit_num] -> [*inst_shape, col, w_digit_num, row]
        w = w.movedim(-3, -1)

        # The magnitude routes to the state index (0 -> HRS, m -> the m-th
        # conductance state); the cell's table lookup is the sole digit-range
        # guard (an out-of-range state index crashes there, radix-irrelevant).
        config = self.config
        mag = w.abs()  # {0, ..., radix-1}
        is_neg = w < 0
        pwg = torch.where(is_neg, torch.zeros_like(mag), mag)
        nwg = torch.where(is_neg, mag, torch.zeros_like(mag))

        # Shape: [*inst_shape, col, digit, row] -> [*inst_shape, col, polarity, digit, row]
        w_pol = torch.stack((pwg, nwg), dim=-3)
        # col = io * mux_factor + slot
        # Shape: [*inst_shape, col, polarity, digit, row] -> [*inst_shape, io, slot, polarity, digit, row]
        w_grouped = w_pol.unflatten(-4, (self.col_num // config.mux_factor, config.mux_factor))
        # Shape: [*inst_shape, io, slot, polarity, digit, row] -> [*inst_shape, slot, io, polarity, digit, row]
        w_state_idx = w_grouped.transpose(-5, -4).contiguous()
        # phys_col enumerates (slot, io, polarity, digit) in that order.
        # Shape: [*inst_shape, slot, io, polarity, digit, row] -> [*inst_shape, phys_col, row]
        return w_state_idx.flatten(-5, -2)

    def program(self, w: Tensor) -> None:
        """Encode logical weights and write the grouped array cells.

        The internal true-form transcoder emits magnitude digits LSB-first.
        Each digit maps to its polarity cells: digit value
        ``+m`` writes the PWG cell to magnitude state ``m`` and the NWG cell to
        HRS (state 0); ``-m`` does the reverse; ``0`` leaves both at HRS.
        The grouped cells are folded into the array's flat
        ``[phys_col, physical_row]`` layout.

        Args:
            w: Logical weight tensor; entries must lie in
                :attr:`w_value_range`.
                Shape: ``[*inst_shape, row_num, col_num]``.
        """
        expected_shape = (*self.inst_shape, self.row_num, self.col_num)
        if tuple(w.shape) != expected_shape:
            raise ValueError(f"program() expects w.shape {expected_shape}; got {tuple(w.shape)}")
        self.array.program(self._organize_w(w))

    def vec_mat_mul(self, x: Tensor, *, quantization_mode: int, adc_bits: int | None) -> Tensor:
        """Run the array solve + readout chain over the K WL sub-phases.

        The K activation bits are bit-expanded into K WL planes (LSB first) and
        all of them settle in ONE broadcast array solve, each carrying its own
        conduction window. The readout runs natively per-(slot, IO); only the
        final code assembly maps that layout back to the logical column order.

        Args:
            x: Activation tensor; entries in :attr:`x_value_range`. Positions
                outside the caller-selected set must be zero. Every leading
                axis is anonymous broadcast batch.
                Shape: ``[..., row_num]``.
            quantization_mode: Mode index in
                ``[0, len(quantization_input_ranges))``; names the row the
                threshold source returns as the ADC's ladder.
            adc_bits: ADC resolution [bits] in ``[1, adc_max_bits]``. The full
                ladder is always wired; the TMCSA realizes the width by running
                only the first ``adc_bits`` steps of its max-bits binary search.

        Returns:
            Signed-magnitude raw-code tensor with the same leading order.
            Shape: ``[..., col_num]``.

        Raises:
            ValueError: ``quantization_mode`` is outside the declared modes, or
                ``adc_bits`` is ``None`` (a physical converter has no lossless
                oracle — build :meth:`to_ideal` for that).
        """
        if adc_bits is None:
            raise ValueError("require: adc_bits is an int — the lossless oracle lives on the to_ideal() twin")
        self._mode(quantization_mode)
        config = self.config
        v_dd = config.v_dd__V
        gn = self.col_num // config.mux_factor  # CIM-IO sense-lane count (group_num)
        x_long = x.long()  # dtype guard for >> and the bit-expand

        # --- 1: Bit-expand x into K WL planes (LSB first) + WL DAC ---

        # The x-bit axis lands at -2 so it becomes the last leading axis of the
        # solve, folding into the array's broadcast leading.
        # Shape: [..., row] -> [..., x_bits, row]
        planes = self._x_transcoder.encode(x_long, dim=-2)
        v_wl = self.wl_dac.convert(planes)

        # --- 2: Solve the array once (cells + wire IR drop) -> I_DL ---

        # The boundary references arrive fully shaped, so the macro reproduces
        # the array's own broadcast leading: the seat grid contributes
        # (*inst_shape, serial) and the WL drive its own leading with the serial slot
        # axis open. The clamp trailing is the (gn, polarity, w_digit) lane grid.
        lane_shape = (gn, _POLARITY_NUM, config.w_digit_num)
        leading = torch.broadcast_shapes((*self.inst_shape, config.mux_factor), (*v_wl.shape[:-1], 1))
        ref_shape = (*leading, *lane_shape)

        # One DC solve for all K WL planes: the x-bit and serial slot axes both
        # ride the solve leading, and the steady currents are window-independent
        # (each plane's conduction window applies post-solve).
        steady = self.array.solve_array(
            v_wl,
            bl_driver=self.cablc,
            # The single-tap bank samples at the clamp shape plus its own tap
            # axis, which the clamp then consumes away.
            bl_v_ref__V=self.cablc_vref.snapshot(mode=0, shape=(*ref_shape, 1)).v_refs__V[..., 0],
            sl_driver=self.sl_driver,
            sl_v_ref__V=self._sl_v_ref__V.expand(ref_shape),
        )
        # Shape: [..., x_bits, gs, gn, polarity, wd]
        i_dl = steady.i_bl_port__uA

        # The whole input branch V_DD * I_DL is the macro's to bill, since the
        # macro owns the per-bit conduction window; the array bills only its
        # capacitive cycling.
        record_dynamic_energy = self._is_dynamic_energy_profile_active()
        if record_dynamic_energy:
            # Shape: [..., x_bits, gs, gn, polarity, wd] -> [..., x_bits]
            read_power = (v_dd * i_dl).sum(dim=(-4, -3, -2, -1))
            # Shape: [..., x_bits] -> [...]
            e_cablc = (read_power * self._window_array__ns).sum(dim=-1)
            self._record_dynamic_energy(e_cablc, channel="cablc")

        # --- 3: DSWCT place-value weighting -> I_WDL (self-billing) ---

        # The per-bit DIAGONAL window rides the x-bit leading axis.
        # Shape: [..., x_bits, gs, gn, polarity, wd] -> [..., x_bits, gs, gn, polarity]
        i_wdl = self.dswct(i_dl, window__ns=self._window_array__ns)

        # --- 4: SINWP-SC temporal input-radix combine -> I_DL_PN (self-billing) ---

        # The held-leg suffix-sum window is injected per bit.
        # Shape: [..., x_bits, gs, gn, polarity] -> [..., gs, gn, polarity]
        i_dl_pn = self.sinwp_sc(i_wdl, window_per_bit__ns=self._window_sc__ns)

        # --- 5: PN-ISUB single-ended magnitude + sign (self-billing) ---

        # The three rail branches conduct in the tail window t_other.
        # Shape: [..., gs, gn, polarity] -> [..., gs, gn]
        i_sub, sign = self.pn_isub(i_dl_pn[..., 0], i_dl_pn[..., 1], window__ns=config.t_other__ns)

        # --- 6: TMCSA quantize against the per-instance reference ladder ---

        # The macro only NAMES the mode; the source selects the row and returns
        # the ladder at the full conversion shape. Everything outside the
        # fabricated inst_shape is time-serial on one physical ladder, so the
        # read noise belongs per converted instant, not once per instance.
        # Shape: [..., gs, gn, tap]
        adc_refs_mode__uA = self.adc_current_reference.snapshot(
            mode=quantization_mode,
            shape=(*i_sub.shape, self.adc_current_reference.tap_num),
        ).i_refs__uA
        # Every bit width rides this one max-bits ladder — the ADC truncates
        # its own binary search, the macro never subsets the taps.
        # Shape: [..., gs, gn]
        code = self.adc.convert(i_sub, adc_refs_mode__uA, bits=adc_bits)
        # The kernel ADC is energy-silent; the billing module recovers the
        # per-step reference path from the raw unsigned codes over the full
        # ladder (a lowered bit width replays the leading steps of it).
        self.tmcsa(i_sub, code, adc_refs_mode__uA, bits=adc_bits)
        signed = (1 - 2 * sign.long()) * code

        # --- 7: Control energy + the sole latency event ---

        # The control fires once per conversion cycle, shared across the CIM-IOs,
        # so it bills over the [*B, gs] leading. The latency is one operating
        # period t_cycle per serial access; the macro is the sole latency emitter.
        if record_dynamic_energy:
            # Shape: [*B, gs]
            e_control = signed.new_full(signed.shape[:-1], config.e_control_per_op__fJ, dtype=torch.float32)
            self._record_dynamic_energy(e_control, channel="control")
        parallel_instance_count = self.inst_count * gn
        serial_round_count = (signed.numel() + parallel_instance_count - 1) // parallel_instance_count
        latency__ns = self._t_cycle__ns * serial_round_count
        self._record_latency(latency__ns)

        # col = io * mux_factor + slot
        # Shape: [..., gs, gn] -> [..., col_num]
        result: Tensor = signed.transpose(-2, -1).flatten(-2)
        return result
