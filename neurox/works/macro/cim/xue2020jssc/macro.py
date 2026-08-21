"""Xue2020 JSSC SINWP 1T1R CIM sub-array macro composing the current-mode readout chain.

Exposes a signed VMM over `row_num` inputs and `col_num` outputs. Each logical
weight is a sign-magnitude value (sign + `w_digit_num` digits of radix
`w_digit_radix`) carried by `w_digit_num * polarity` physical cells, a P (PWG) and
an N (NWG) cell per digit; the kernel 1T1R array, the DSWCT place-value legs, the
SINWP-SC input-radix combine, the PN-ISUB subtraction, and the TMCSA current SAR
ADC recover the signed magnitude.

The column MUX is a MACRO axis, not an array one: the array holds every physical
column and settles all of them in one solve per WL plane, while the macro keeps
the `[..., sweep, gn, polarity, w_digit]` seat layout its transcode, readout,
billing, and latency all speak. The two representations meet at exactly two
conversions, `_seat_to_phys` and `_phys_to_seat`, and the slot map that defines
them never leaves the macro.

The macro owns the per-call conduction windows it injects into the self-billing
readout modules, delivers both boundary clamps at the port state the array solve
returns, and bills the whole input branch plus the control per-op constant on its
own two channels.

See Also:
    docs/system_design/cim_execution.md
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

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
from neurox.primitive.xbar.array import (
    XbarArray1t1r,
    XbarArray1t1rConfig,
    XbarArray1t1rPolicy,
    XbarArray1t1rScanMode,
)

from .dswct import Dswct, DswctConfig, DswctPolicy
from .pn_isub import PnIsub, PnIsubConfig, PnIsubPolicy
from .sinwp_sc import SinwpSc, SinwpScConfig, SinwpScPolicy
from .tmcsa import Tmcsa, TmcsaConfig, TmcsaPolicy

_POLARITY_NUM = 2  # PWG, NWG per weight digit


def _move_axis_block(x: Tensor, *, src: int, dst: int, num: int) -> Tensor:
    """Relocate a contiguous block of `num` axes from `src` to `dst`.

    Args:
        x: Tensor to relayout.
        src: Absolute index of the block's first axis in `x`.
        dst: Absolute index of the block's first axis in the RESULT.
        num: Number of axes in the block.

    Returns:
        The relayouted tensor, materialized; `x` itself when the move is empty.
    """
    if num == 0 or src == dst:
        return x
    return x.movedim(list(range(src, src + num)), list(range(dst, dst + num))).contiguous()


def _sample_reference_bank(
    sample: Callable[[tuple[int, ...]], Tensor],
    *,
    shape: tuple[int, ...],
    inst_pos: int,
    inst_num: int,
) -> Tensor:
    """Sample a reference bank at a layout whose instance block is not trailing.

    Both banks expand their per-instance rows RIGHT-ALIGNED onto the requested
    shape, so a request only reaches the intended rows when it ends in
    `(*inst_shape, tap_num)`. This macro seats the MUX slot axis between the
    instance block and the lane grid, so the bank is asked for the same axes with
    the instance block pulled to the trailing position and the block is moved back
    into its seat afterwards.

    Args:
        sample: Reference-bank read — the fabricated buffer indexed by mode and
            broadcast to the requested full shape by view.
        shape: Target shape, ending in the bank's tap axis.
        inst_pos: Absolute index of the instance block's first axis in `shape`.
        inst_num: Number of instance axes.

    Returns:
        The sampled taps at `shape`.
    """
    if inst_num == 0:
        return sample(shape)
    tap = len(shape) - 1
    request = (
        *shape[:inst_pos],
        *shape[inst_pos + inst_num : tap],
        *shape[inst_pos : inst_pos + inst_num],
        shape[tap],
    )
    return _move_axis_block(sample(request), src=tap - inst_num, dst=inst_pos, num=inst_num)


class Xue2020JsscCimMacroConfig(CimMacroConfig):
    # === Weight / input geometry ===

    w_digit_num: int
    """Magnitude digits per weight, >= 1. A single digit is one polarity pair with no
    cross-digit combine."""
    w_digit_radix: int
    """Positional base of the magnitude digits, >= 2. Above 2 needs a radix-level
    conductance table in the cell config."""
    input_bit_num: int
    """Activation bit width K, >= 1 — K serial single-bit WL sub-phases, LSB first."""
    mux_factor: int
    """Logical columns per CIM-IO, the column-MUX depth: `col_num` must divide by it, and
    it is also the serial-access count per conversion."""

    # === Ratio anchors (DSWCT / SINWP-SC ratios derived DOWNWARD from these) ===

    dswct_ratio_msb: float
    """MSB-leg mirror ratio the per-digit DSWCT ratios are derived downward from."""
    sc_ratio_msb: float
    """MSB-bit combine ratio the per-input-bit SINWP-SC ratios are derived downward from;
    at K = 1 the anchor is used directly, with the sample-and-hold leg off."""

    # === Conduction windows (dynamic-energy only) + static time base ===

    t_sample__ns: tuple[float, ...]
    """One sample sub-phase window per SAMPLED input bit, so `input_bit_num - 1` entries;
    the live bit conducts in the tail window instead."""
    t_settle__ns: float
    """Tail non-sensing settle window — the live-bit settle ONLY. The SAR sensing
    durations join it from the ADC step windows."""
    t_cycle__ns: float
    """Declared operating period, the static-energy time base: the leakage integration
    window of one access. Must be >= the total conduction span."""

    # === Supply rails (separate variables even when numerically equal) ===

    v_dd__V: float
    """Rail every channelled branch is billed across, including the whole input branch on
    the `cablc` channel. It is also the BL driver rail handed to the array, so the
    conduction-path capacitance is charged from it."""
    v_dd_wl__V: float
    """Word-line driver rail — the supply behind the WL wire and the per-cell gate
    capacitance, driven rail-to-rail while the read path hangs off the main rail."""

    # === Per-op dynamic constants ===

    e_control_per_op__fJ: float
    """Control per-conversion dynamic energy (address decode, CMD precharge, timing)
    billed on the `control` channel; it covers the CMD precharge, so no CMD capacitance
    is modeled."""

    # === Static-PPA seat (control) ===

    control_config: UnmodeledBlockConfig

    # === Scheme-local readout modules ===

    dswct_config: DswctConfig
    sinwp_sc_config: SinwpScConfig
    pn_isub_config: PnIsubConfig
    tmcsa_config: TmcsaConfig
    """Its window lists span exactly `adc_config.bits` steps and satisfy
    `t_ph2[s] + t_ph3[s] <= adc_config.step_latency__ns[s]`, PH1/PH4 occupying the
    rest."""

    # === Device-bearing sub-blocks (full nested configs) ===

    array_config: XbarArray1t1rConfig
    """The cell tables carrying the programmable weights, the BL / SL / WL wire
    parasitics, and the DC solver knobs; the array's settling sits inside the access
    window this macro times itself."""
    wl_dac_config: VdacConfig
    """1-bit ON/OFF word-line drive, whose sub-phase likewise sits inside the access
    window."""
    cablc_config: VoltageDriverConfig
    """The array's CABLC BL clamp, an ideal source (`r_out__MOhm = 0`) since the wire IR
    drop is the array's, not the clamp's. Its reference is injected per call from
    `cablc_vref_config`."""
    cablc_vref_config: VrefConfig
    """Dedicated CABLC reference source holding the degenerate one-mode single-tap bank
    whose sole tap is the BL clamp reference, read per solve at the array's full call
    shape; the per-cell `V_BL` droops below it by the wire IR drop the solver computes.
    The tap must be in `[0, v_dd__V]`."""
    sl_driver_config: VoltageDriverConfig
    """The array's SL clamp, an ideal source. The SL is a direct ground tie, so its
    reference is a plain 0 V tensor rather than a reference source."""
    adc_config: SarIadcConfig
    """The kernel VALUE converter, built energy-silent because `tmcsa_config` bills the
    conversion instead. Its `step_latency__ns` stays the physical sensing duration: the
    executed prefix is the sensing part of the access time, and the whole tuple feeds
    the read-chain window `t_other__ns`."""
    reference_config: IrefConfig
    """The `[mode][tap]` threshold bank the ADC reads as its ladder — the macro NAMES the
    mode and reads that row. It holds `2**adc_config.bits - 1` taps, and every row must
    ascend strictly, since the ladder ordering is the TMCSA's knowledge rather than the
    source's."""

    # === Quantization modes ===

    modes: tuple[CimMacroMode, ...]
    """One operating point per `quantization_mode` index; the count must match
    `reference_config.mode_num`, one threshold ladder row per mode."""

    @property
    def digit_ratios(self) -> tuple[float, ...]:
        """Per-digit DSWCT mirror ratios (LSB-first): `r_d = dswct_ratio_msb * radix**(d - (D-1))`."""
        d_top = self.w_digit_num - 1
        return tuple(self.dswct_ratio_msb * self.w_digit_radix ** (d - d_top) for d in range(self.w_digit_num))

    @property
    def x_bit_ratios(self) -> tuple[float, ...]:
        """Per-input-bit SINWP-SC combine ratios (LSB-first): `s_k = sc_ratio_msb * 2**(k - (K-1))`."""
        k_top = self.input_bit_num - 1
        return tuple(self.sc_ratio_msb * 2.0 ** (k - k_top) for k in range(self.input_bit_num))

    @property
    def t_other__ns(self) -> float:
        """Live/tail window — `t_settle__ns + sum(adc_config.step_latency__ns)`.

        The live input bit (K-1) has no sample sub-phase of its own; its conduction,
        the SINWP-SC combine, the PN-ISUB, and the TMCSA sensing all fall in this tail
        window. The SAR sensing durations are added here, so the whole read chain
        conducts through sensing while `t_settle__ns` stays the pure non-sensing
        settle.
        """
        return self.t_settle__ns + sum(self.adc_config.step_latency__ns)

    @property
    def window_array__ns(self) -> tuple[float, ...]:
        """Per-input-bit input-branch (array / CABLC / DSWCT) conduction window.

        Sampled bit `k` conducts for its sample window `t_sample__ns[k]`; the live bit
        (K-1) conducts for `t_other`.
        """
        k_live = self.input_bit_num - 1
        return tuple(self.t_sample__ns[k] if k < k_live else self.t_other__ns for k in range(self.input_bit_num))

    @property
    def window_sc__ns(self) -> tuple[float, ...]:
        """Per-input-bit SINWP-SC leg conduction window.

        A bit-`k` leg is held from its sample sub-phase to the end, so it conducts for
        the suffix sum of the remaining sample windows plus the tail:
        `sum(t_sample__ns[k:]) + t_other`. The live bit reduces to `t_other`.
        """
        return tuple(sum(self.t_sample__ns[k:]) + self.t_other__ns for k in range(self.input_bit_num))

    @property
    def conduction_span__ns(self) -> float:
        """Total conduction span `sum(t_sample) + t_other`; must fit in `t_cycle__ns`."""
        return sum(self.t_sample__ns) + self.t_other__ns

    def validate(self) -> None:
        super().validate()

        # --- Data geometry ---

        # General sign-magnitude weight: >= 1 magnitude digit, radix >= 2 so a
        # digit carries at least the {0, 1} magnitude the polarity pair encodes.
        self._require_pos(self.w_digit_num, "w_digit_num")
        self._require_ge(self.w_digit_radix, "w_digit_radix", 2)

        self._require_pos(self.input_bit_num, "input_bit_num")
        self._require_pos(self.mux_factor, "mux_factor")

        # --- Analog transfer and timing ---

        self._require_pos(self.dswct_ratio_msb, "dswct_ratio_msb")
        self._require_pos(self.sc_ratio_msb, "sc_ratio_msb")

        # One sample window per SAMPLED bit; the live bit (K-1) has none.
        self._require_len(self.t_sample__ns, "t_sample__ns", self.input_bit_num - 1)
        for k, t in enumerate(self.t_sample__ns):
            self._require_non_neg(t, f"t_sample__ns[{k}]")
        self._require_non_neg(self.t_settle__ns, "t_settle__ns")
        self._require_pos(self.t_cycle__ns, "t_cycle__ns")
        # The static time base must contain the whole conduction span (the read
        # path idles for the remainder of the period).
        self._require_ge(self.t_cycle__ns, "t_cycle__ns", self.conduction_span__ns)
        self._require_non_neg(self.e_control_per_op__fJ, "e_control_per_op__fJ")

        self._require_non_neg(self.v_dd__V, "v_dd__V")
        self._require_non_neg(self.v_dd_wl__V, "v_dd_wl__V")
        # The CABLC consumes exactly one reference tap and holds it across every
        # mode, so its dedicated source is the degenerate single-row single-tap
        # bank. The clamp reference is a BL node between the SL ground and the
        # V_DD supply, so it must not exceed the rail.
        self._require_len(self.cablc_vref_config.v_refs__V, "cablc_vref_config.v_refs__V", 1)
        self._require_len(self.cablc_vref_config.v_refs__V[0], "cablc_vref_config.v_refs__V[0]", 1)
        v_bl_clamp__V = self.cablc_vref_config.v_refs__V[0][0]
        self._require_le(v_bl_clamp__V, "cablc_vref_config.v_refs__V[0][0]", self.v_dd__V)

        # --- TMCSA phase windows against the ADC step timing ---

        # The billing module resolves each of the ADC's binary-search steps
        # into PH2/PH3 conduction phases: one window pair per step, and the
        # phases must fit inside that step's latency (PH1/PH4 fill the rest).
        self._require_len(
            self.tmcsa_config.t_ph2_per_step__ns,
            "tmcsa_config.t_ph2_per_step__ns",
            self.adc_config.bits,
        )
        for s, (t_ph2, t_ph3) in enumerate(
            zip(self.tmcsa_config.t_ph2_per_step__ns, self.tmcsa_config.t_ph3_per_step__ns, strict=True)
        ):
            step = self.adc_config.step_latency__ns[s]
            self._require_le(
                t_ph2 + t_ph3,
                f"tmcsa_config.t_ph2_per_step__ns[{s}] + t_ph3_per_step__ns[{s}]",
                step,
            )

        # --- ADC reference and quantization modes ---

        # The TMCSA reads its ladder from the shared Iref, so the tap
        # count must match the binary-search depth exactly. Lower bit widths
        # ride this one max-bits ladder; they need no taps of their own.
        want_taps = (1 << self.adc_config.bits) - 1
        self._require_len(self.reference_config.i_refs__uA[0], "reference_config.i_refs__uA[0]", want_taps)

        # What a reference row MEANS is the consumer's knowledge, so the source
        # does not order its taps: the TMCSA reads each row as a binary-search
        # decision ladder, and only a strictly ascending ladder decodes.
        for m, row in enumerate(self.reference_config.i_refs__uA):
            self._require_increasing(row, f"reference_config.i_refs__uA[{m}]")

        # Each CimMacroMode validates its own canonical window and positive
        # rescale factor on construction; the macro pins the mode count against
        # the reference bank, since a mode IS one ladder row.
        self._require_same_len(
            self.modes,
            "modes",
            self.reference_config.i_refs__uA,
            "reference_config.i_refs__uA",
        )


class Xue2020JsscCimMacroPolicy(CimMacroPolicy):
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

    Composes the kernel 1T1R array (cell grid + wire + solver) with the macro-owned
    WL DAC, the CABLC / SL clamp seats, the DSWCT / SINWP-SC / PN-ISUB readout
    modules, the TMCSA and its shared reference, and the control static seat. Each
    logical weight occupies `w_digit_num * polarity` grouped cells, a polarity pair
    per digit.

    Two layouts of the same columns live here. The SEAT layout
    `[..., sweep, gn, polarity, w_digit]` is the macro's own: it names the column-MUX
    slot a column is accessed in and the driver lane it is accessed through, which is
    what the transcode, the readout chain, the conduction windows, and the latency
    arithmetic are all written against. The PHYSICAL layout `[..., phys_col]` is the
    array's, and it is the only thing that crosses into `XbarArray1t1r.program` and
    `XbarArray1t1r.solve_array`. The slot map is the bijection between them and stays
    here; `_seat_to_phys` and `_phys_to_seat` are the only two places either layout
    turns into the other.

    Args:
        input_num: Logical input length, bound to `row_num` during construction.
        output_num: Logical output length, bound to `col_num` during construction.
    """

    # === Functional buffers ===

    _slot_map: Tensor  # Shape: [sweep, gn, polarity, w_digit]
    _seat_of_phys: Tensor  # Shape: [phys_col]

    # === Circuit constant buffers ===

    _sl_v_ref__V: Tensor  # Shape: []
    _window_array__ns: Tensor  # Shape: [x_bits]
    _window_sc__ns: Tensor  # Shape: [x_bits]

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

    def latency__ns(self, *, adc_bits: int | None) -> float:
        """One VMM — the access time of every column-MUX slot it serializes.

        A slot's access is the whole read chain settling once: the `input_bit_num` WL
        sub-phases and the live-bit settle are the macro's own windows, and the
        sensing tail is the TMCSA's, so the resolution enters through the converter
        that owns the search-step axis rather than through a constant here.
        `mux_factor` is the macro's only remaining time axis: the CIM-IO lanes convert
        in parallel, one slot at a time.

        Raises:
            ValueError: `adc_bits` is `None`.
        """
        if adc_bits is None:
            raise ValueError("require: adc_bits is an int — the lossless oracle lives on the to_ideal() twin")
        config = self.config
        chain__ns = sum(config.t_sample__ns) + config.t_settle__ns + self.adc.latency__ns(bits=adc_bits)
        return chain__ns * config.mux_factor

    def _init_children(self, *, dtype: torch.dtype, T__K: float) -> None:
        config = self.config
        policy = self.policy
        gn = self.col_num // config.mux_factor
        phys_col_num = self.col_num * config.w_digit_num * _POLARITY_NUM

        # --- Programmable weights + wire + solver: the whole physical array ---

        # The column MUX is a macro axis: it says WHEN a column is read, not how
        # the cells are wired, so the array holds every physical column and
        # settles all of them in one solve per WL plane. The word line carries
        # the held input and the bit-line boundary is what the MUX scans, hence
        # WL_IN_BL_SCAN; the two rails are declared here, once, and cascade into
        # the array's capacitive billing.
        self.array = XbarArray1t1r(
            config=config.array_config,
            policy=policy.array_policy,
            inst_shape=self.inst_shape,
            row_num=self.row_num,
            col_num=phys_col_num,
            scan_mode=XbarArray1t1rScanMode.WL_IN_BL_SCAN,
            v_dd_wl__V=config.v_dd_wl__V,
            v_dd_bl__V=config.v_dd__V,
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

        # Energy-silent because the tmcsa module below bills the conversion
        # phase-resolved. The ADC keeps its physical step_latency__ns, which
        # answers the macro's access-time query at the executed resolution and
        # sums to the sensing part of t_other.
        self.adc = SarIadc(
            config=config.adc_config,
            policy=policy.adc_policy,
            inst_shape=(*self.inst_shape, gn),
            dtype=dtype,
            T__K=T__K,
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
        config = self.config
        # Column-MUX placement: phys_col = ((slot * gn + io) * polarity + pol) *
        # w_digit + digit, the bijection (slot, io, polarity, digit) -> physical
        # column. It is the macro's own knowledge of WHEN each column is read;
        # the array never sees it.
        gn = self.col_num // config.mux_factor
        phys_col_num = self.col_num * config.w_digit_num * _POLARITY_NUM
        slot_map = torch.arange(phys_col_num, dtype=torch.long).reshape(
            config.mux_factor, gn, _POLARITY_NUM, config.w_digit_num
        )
        # The inverse permutation: the seat a physical column sits in. A gather
        # by this index is what turns a seat-ordered tensor into a physical one,
        # so both directions are a single index_select over a stored bijection.
        seat_of_phys = torch.argsort(slot_map.reshape(-1))
        self.register_buffer("_slot_map", slot_map, persistent=False)
        self.register_buffer("_seat_of_phys", seat_of_phys, persistent=False)
        # SL direct ground tie: a plain all-zeros reference, no Vref module.
        self.register_buffer("_sl_v_ref__V", torch.zeros((), dtype=dtype), persistent=False)
        self.register_buffer("_window_array__ns", torch.tensor(config.window_array__ns, dtype=dtype), persistent=False)
        self.register_buffer("_window_sc__ns", torch.tensor(config.window_sc__ns, dtype=dtype), persistent=False)

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
        """Maximum ADC magnitude resolution [bits] — the TMCSA's binary-search depth."""
        return self.config.adc_config.bits

    def _mode(self, quantization_mode: int) -> CimMacroMode:
        """Return one declared quantization mode.

        Args:
            quantization_mode: Mode index in `[0, len(config.modes))`.

        Raises:
            ValueError: The index is outside the declared modes.
        """
        modes = self.config.modes
        if not (0 <= quantization_mode < len(modes)):
            raise ValueError(f"require: quantization_mode ({quantization_mode}) in [0, {len(modes)})")
        return modes[quantization_mode]

    def _max_bits_rescale_factor(self, quantization_mode: int) -> float:
        """Return the calibrated rescale factor of one mode at `adc_max_bits`."""
        return self._mode(quantization_mode).max_bits_rescale_factor

    def map_quantization_input_code(self, code: Tensor, *, quantization_mode: int) -> tuple[Tensor, tuple[int, int]]:
        """Map exact MAC-unit codes onto the sign-magnitude ADC input grid.

        The PN-ISUB hands the TMCSA a single-ended magnitude, so the converter
        discriminates `|code|`. Its grid is the ladder itself, whose span the config
        declares as the mode's `adc_input_code_range` — circuit knowledge that does
        not follow the window width: a window whose top magnitude exceeds the ladder
        simply saturates.

        Args:
            code: Exact integer plane dots.
            quantization_mode: Mode index in `[0, len(quantization_input_ranges))`.

        Returns:
            The magnitudes and the mode's declared ADC input code range.
        """
        mode = self._mode(quantization_mode)
        magnitude, _ = map_magnitude_input_code(code, code_range=mode.quantization_input_range)
        return magnitude, mode.adc_input_code_range

    def to_ideal(self) -> IdealCimMacro:
        """Return an ideal twin one bit wider than the TMCSA.

        The readout resolves a sign plus `adc_max_bits` magnitude bits, a signed span
        the zero-point twin only reaches at `adc_max_bits + 1` bits. The twin is
        unfaithful by design at the endpoints — it holds one phantom bottom level and
        a single zero where the sign-magnitude encoding wastes two — so it is a
        reference, never a bit-exact model.
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

    def _seat_to_phys(self, x: Tensor, *, dim: int) -> Tensor:
        """Fold the four seat axes starting at `dim` into one physical-column axis.

        The seat block `(sweep, gn, polarity, w_digit)` flattens in its own
        enumeration order and is then gathered by the seat each physical column
        occupies, which is exactly the inverse of the slot map.

        Args:
            x: Tensor whose axes `dim .. dim + 3` are the seat block.
                Shape: `[..., sweep, gn, polarity, w_digit, ...]`.
            dim: Negative index of the seat block's first axis.

        Returns:
            The same tensor on the physical column axis, at `dim + 3`.
            Shape: `[..., phys_col, ...]`.
        """
        return x.flatten(dim, dim + 3).index_select(dim + 3, self._seat_of_phys)

    def _phys_to_seat(self, x: Tensor, *, dim: int) -> Tensor:
        """Split the physical-column axis at `dim` back into the four seat axes.

        Each seat gathers the physical column the slot map assigns it, and the flat
        result unfolds into the seat block.

        Args:
            x: Tensor whose axis `dim` is the physical column axis.
                Shape: `[..., phys_col, ...]`.
            dim: Negative index of that axis.

        Returns:
            The same tensor on the seat block, opening at `dim`.
            Shape: `[..., sweep, gn, polarity, w_digit, ...]`.
        """
        seated: Tensor = x.index_select(dim, self._slot_map.reshape(-1)).unflatten(dim, tuple(self._slot_map.shape))
        return seated

    def _seat_states(self, w: Tensor) -> Tensor:
        """Encode logical weights into per-seat cell state indices.

        Args:
            w: Logical weight tensor.
                Shape: `[*inst_shape, row_num, col_num]`.

        Returns:
            State indices in the macro's own seat layout.
            Shape: `[*inst_shape, sweep, gn, polarity, w_digit, row_num]`.
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
        seated: Tensor = w_grouped.transpose(-5, -4)
        return seated

    def program(self, w: Tensor) -> None:
        """Encode logical weights and write the array cells.

        The internal true-form transcoder emits magnitude digits LSB-first. Each digit
        maps to its polarity cells: digit value `+m` writes the PWG cell to magnitude
        state `m` and the NWG cell to HRS (state 0); `-m` does the reverse; `0` leaves
        both at HRS. Digits land on seats, and the seats are scattered to physical
        columns here — the array's own programming is purely physical and knows
        nothing of slots, polarities, or digit order.

        Args:
            w: Logical weight tensor; entries must lie in `w_value_range`.
                Shape: `[*inst_shape, row_num, col_num]`.
        """
        expected_shape = (*self.inst_shape, self.row_num, self.col_num)
        if tuple(w.shape) != expected_shape:
            raise ValueError(f"program() expects w.shape {expected_shape}; got {tuple(w.shape)}")
        # Shape: [*inst_shape, sweep, gn, 2, wd, row] -> [*inst_shape, phys_col, row]
        self.array.program(self._seat_to_phys(self._seat_states(w), dim=-5))

    def vec_mat_mul(self, x: Tensor, *, quantization_mode: int, adc_bits: int | None) -> Tensor:
        """Run the array solve + readout chain over the K WL sub-phases.

        The K activation bits are bit-expanded into K WL planes (LSB first) and
        all of them settle in ONE broadcast array solve over the whole physical
        column set, each plane carrying its own conduction window. The columns
        come back onto the seat layout immediately, and the readout runs natively
        per-(slot, IO) from there; only the final code assembly maps that layout
        back to the logical column order.

        Args:
            x: Activation tensor; entries in `x_value_range`. Positions outside the
                caller-selected set must be zero. The instance axes are DECLARED
                leading and must be present whenever `inst_shape` is non-empty — every
                fabricated copy holds its own cells, reference banks and comparator
                offsets, so they are never anonymous batch. A size-1 instance axis
                shares one input vector across the whole die ensemble. Any axis ahead
                of them is anonymous broadcast batch.
                Shape: `[..., *inst_shape, row_num]`.
            quantization_mode: Mode index in `[0, len(quantization_input_ranges))`;
                names the row the threshold source returns as the ADC's ladder.
            adc_bits: ADC resolution [bits] in `[1, adc_max_bits]`. The full ladder is
                always wired; the TMCSA realizes the width by running only the first
                `adc_bits` steps of its max-bits binary search.

        Returns:
            Signed-magnitude raw-code tensor with the same leading order.
            Shape: `[..., *inst_shape, col_num]`.

        Raises:
            ValueError: `quantization_mode` is outside the declared modes, `adc_bits`
                is `None` (a physical converter has no lossless oracle — build
                `to_ideal` for that), or `x` has no room for the instance axes.
        """
        if adc_bits is None:
            raise ValueError("require: adc_bits is an int — the lossless oracle lives on the to_ideal() twin")
        self._mode(quantization_mode)
        config = self.config
        v_dd = config.v_dd__V
        gn = self.col_num // config.mux_factor  # CIM-IO sense-lane count (group_num)
        x_long = x.long()  # dtype guard for >> and the bit-expand

        # The instance axes are DECLARED leading, not anonymous batch: the cell
        # grid, both reference banks and the ADC comparators each hold per-copy
        # state, so the input layout has to name a position for them.
        inst_num = len(self.inst_shape)
        if x_long.ndim - 1 < inst_num:
            raise ValueError(
                f"vec_mat_mul() expects x leading (..., *inst_shape) with inst_shape {self.inst_shape}; "
                f"got x.shape {tuple(x.shape)}"
            )
        batch_num = x_long.ndim - 1 - inst_num
        batch = tuple(x_long.shape[:batch_num])
        # A size-1 instance axis shares one input vector across the ensemble.
        inst = tuple(torch.broadcast_shapes(self.inst_shape, x_long.shape[batch_num:-1]))
        x_bits = config.input_bit_num

        # --- 1: Bit-expand x into K WL planes (LSB first) + WL DAC ---

        # Shape: [..., *inst_shape, row] -> [..., *inst_shape, x_bits, row]
        planes = self._x_transcoder.encode(x_long, dim=-2)
        # The array seats its cell grid at (*inst_shape, phys_col, row) and binds
        # it right-aligned, so the solve leading must END with the instance axes.
        # The x-bit axis therefore rides IN FRONT of the instance block rather
        # than between it and the row axis.
        # Shape: -> [..., x_bits, *inst_shape, row]
        planes = _move_axis_block(planes, src=planes.ndim - 2, dst=batch_num, num=1)
        v_wl__V = self.wl_dac.convert(planes)

        # --- 2: Solve the array once (cells + wire IR drop) -> I_DL ---

        # The solve leading is batch, x bit, instance block: the column-MUX slot
        # is NOT here, because every physical column settles in the same solve.
        # It reappears one axis later, as the leading axis of the seat block the
        # boundary snaps are sampled on. Both tuples are assembled by RANK rather
        # than broadcast from the right: right-alignment would seat the instance
        # axes on the x-bit axis, folding two unrelated grids onto one another.
        lane_shape = (gn, _POLARITY_NUM, config.w_digit_num)
        leading = (*batch, x_bits, *inst)
        seat_shape = (*leading, config.mux_factor, *lane_shape)
        # A size-1 input instance axis shares the line drive across the
        # fabricated ensemble. The macro completes that semantic broadcast;
        # the array adds only its physical column axis.
        # Shape: [..., x_bits, *input_inst_shape, row]
        #     -> [*leading, row]
        v_wl__V = v_wl__V.expand(*leading, self.row_num)
        # The single-tap bank broadcasts (by view) at the clamp shape plus its
        # own tap axis, which the clamp then consumes away. The source is
        # fabricate-only (no per-call noise of its own); the CABLC driver
        # draws whatever per-position dynamic noise its policy enables when
        # it snapshots this reference below.
        bl_v_ref__V = _sample_reference_bank(
            lambda shape: self.cablc_vref.v_out__V[..., 0, :].expand(shape),
            shape=(*seat_shape, 1),
            inst_pos=batch_num + 1,
            inst_num=inst_num,
        )[..., 0]

        # The macro owns the event structure, so the macro snapshots: a clamp is
        # re-sampled once per MUX slot, which is what the sweep axis in front of
        # the lane grid says, and each snap folds to physical column order on the
        # very next line — the seat-shaped snap exists only between these two
        # statements, and every array-facing shape is the canonical [..., col].
        # A bank fabricated WITH a fabrication prefix seats the sweep axis
        # between that prefix and the lane grid, so a per-instance offset draw
        # would mis-seat; the sanctioned all-off policy never takes one.
        # Shape: [*leading, sweep, gn, polarity, wd] -> [*leading, phys_col]
        bl_snap = self.cablc.snapshot(v_ref__V=bl_v_ref__V, shape=seat_shape)
        bl_snap = bl_snap.flatten_axes(-4, -1).index_select(-1, self._seat_of_phys)
        # Shape: [*leading, sweep, gn, polarity, wd] -> [*leading, phys_col]
        sl_snap = self.sl_driver.snapshot(v_ref__V=self._sl_v_ref__V.expand(seat_shape), shape=seat_shape)
        sl_snap = sl_snap.flatten_axes(-4, -1).index_select(-1, self._seat_of_phys)

        # One DC solve for all K WL planes over the whole physical array: the
        # x-bit axis rides the solve leading, and the steady currents are
        # window-independent (each plane's conduction window applies post-solve).
        # The array distributes the line-level drive over its own columns.
        # Shape: [*leading, row]
        steady = self.array.solve_array(
            v_wl__V,
            bl_driver=self.cablc,
            bl_driver_snap=bl_snap,
            sl_driver=self.sl_driver,
            sl_driver_snap=sl_snap,
        )
        # Back to the seat layout the whole readout chain speaks: a column's seat
        # is when it is accessed and through which driver lane, which is what the
        # conduction windows and the per-lane readout modules are indexed by.
        # Shape: [*leading, phys_col] -> [*leading, gs, gn, polarity, wd]
        i_bl_seat__uA = self._phys_to_seat(steady.i_bl_port__uA, dim=-1)
        v_bl_seat__V = self._phys_to_seat(steady.v_bl_clamp__V, dim=-1)
        i_sl_seat__uA = self._phys_to_seat(steady.i_sl_port__uA, dim=-1)
        v_sl_seat__V = self._phys_to_seat(steady.v_sl_drive__V, dim=-1)
        # Deliver both boundary clamps at the converged port state, on the seat
        # layout: the lane trailing (gn, polarity, w_digit) is each clamp bank's
        # instance block, and the drive is a layout-blind per-op lump seated by
        # POSITION — so it is billed here, ahead of every axis move below, and a
        # fabrication prefix ahead of the sweep axis does not disturb it.
        # Shape: [..., x_bits, *inst_shape, gs, gn, polarity, wd]
        self.cablc.drive(i_bl_seat__uA, v_bl_seat__V)
        self.sl_driver.drive(i_sl_seat__uA, v_sl_seat__V)

        # Back to the caller's axis order: the readout chain below reads the
        # x-bit axis at a fixed depth from the trailing end.
        # Shape: -> [..., *inst_shape, x_bits, gs, gn, polarity, wd]
        i_dl = _move_axis_block(i_bl_seat__uA, src=batch_num, dst=batch_num + inst_num, num=1)

        # The whole input branch V_DD * I_DL is the macro's to bill, since the
        # macro owns the per-bit conduction window; the array bills only its
        # capacitive cycling. Solving every column at once does NOT mean every
        # column conducts at once: a column conducts during its own slot, for
        # that slot's window, which is why the bill is taken on the seat layout
        # (slot axis explicit) rather than on the flat physical one.
        record_dynamic_energy = self._is_dynamic_energy_profile_active()
        if record_dynamic_energy:
            # Shape: [..., x_bits, gs, gn, polarity, wd] -> [..., x_bits]
            read_power = (v_dd * i_dl).sum(dim=(-4, -3, -2, -1))
            # Every slot shares one per-bit window here, so the slot axis folds
            # into the sum above and the bit axis folds here; what is left is x's
            # leading, whose last axes are this macro's instance axes, which the
            # collector sums past the caller's leading dims.
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

        # The macro only NAMES the mode; the fabricated bank is a single static
        # identity per instance (fabricate-only, no per-call noise of its own),
        # broadcast by view to the full conversion shape — every converted
        # instant outside inst_shape reads the same physical ladder.
        # The converters are fabricated per (instance, CIM-IO) and bind their
        # comparator offsets right-aligned, so the conversion runs on a view
        # whose TRAILING axes are (*inst_shape, gn): the serial slot axis steps
        # ahead of the instance block for the call and back for the result.
        # Shape: [..., *inst_shape, gs, gn] -> [..., gs, *inst_shape, gn]
        i_sub_adc = _move_axis_block(i_sub, src=batch_num, dst=batch_num + 1, num=inst_num)
        # Shape: [..., gs, *inst_shape, gn, tap]
        adc_refs_mode__uA = _sample_reference_bank(
            lambda shape: self.adc_current_reference.i_out__uA[..., quantization_mode, :].expand(shape),
            shape=(*i_sub_adc.shape, self.adc_current_reference.tap_num),
            inst_pos=batch_num + 1,
            inst_num=inst_num,
        )
        # Every bit width rides this one max-bits ladder — the ADC truncates
        # its own binary search, the macro never subsets the taps.
        # Shape: [..., gs, *inst_shape, gn] -> [..., *inst_shape, gs, gn]
        code = _move_axis_block(
            self.adc.convert(i_sub_adc, adc_refs_mode__uA, bits=adc_bits),
            src=batch_num + 1,
            dst=batch_num,
            num=inst_num,
        )
        # The kernel ADC is energy-silent; the billing module recovers the
        # per-step reference path from the raw unsigned codes over the full
        # ladder (a lowered bit width replays the leading steps of it). It reads
        # the three tensors elementwise, so the ladder comes back to the readout
        # chain's own layout with the codes.
        # Shape: [..., *inst_shape, gs, gn, tap]
        self.tmcsa(
            i_sub,
            code,
            _move_axis_block(adc_refs_mode__uA, src=batch_num + 1, dst=batch_num, num=inst_num),
            bits=adc_bits,
        )
        signed = (1 - 2 * sign.long()) * code

        # --- 7: Control energy ---

        # One access is one (leading, mux slot) entry: the gn CIM-IO lanes convert
        # in parallel, so the lane axis is neither billed per lane nor serialized.
        # Shape: [..., gs, gn] -> [..., gs]
        accesses = signed[..., 0]
        # The control fires once per conversion cycle, shared across the CIM-IOs.
        # The expanded constant holds no storage, so no energy tensor is
        # materialized and the energy dtype is the constant's, not the code's.
        # Outside a profiler the call is already a no-op, hence no activity guard.
        # Shape: [] -> [..., *inst_shape, gs]
        e_control__fJ = torch.full((), config.e_control_per_op__fJ, dtype=torch.float32, device=accesses.device)
        self._record_dynamic_energy(e_control__fJ.expand(accesses.shape), channel="control")

        # col = io * mux_factor + slot
        # Shape: [..., gs, gn] -> [..., col_num]
        result: Tensor = signed.transpose(-2, -1).flatten(-2)
        return result
