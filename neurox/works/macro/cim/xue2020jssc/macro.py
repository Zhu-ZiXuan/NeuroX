"""Xue2020 JSSC SINWP 1T1R CIM sub-array.

See Also:
    docs/system_design/cim_execution.md
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from neurox.encoding import Encoding
from neurox.primitive.analog import (
    Reference,
    ReferenceConfig,
    ReferencePolicy,
    UnmodeledBlock,
    UnmodeledBlockConfig,
    UnmodeledBlockPolicy,
    VoltageDriver,
    VoltageDriverConfig,
    VoltageDriverPolicy,
)
from neurox.primitive.macro.cim import (
    CimMacro,
    CimMacroConfig,
    CimMacroPolicy,
    CimMacroQuantizationScheme,
)
from neurox.primitive.physics import e_charge__fJ, q_conduction__fC
from neurox.primitive.xbar.array import (
    XbarArray1t1r,
    XbarArray1t1rConfig,
    XbarArray1t1rPolicy,
)

from .tmcsa import Tmcsa, TmcsaConfig, TmcsaPolicy

_POLARITY_NUM = 2  # PWG, NWG per weight digit


class Xue2020JsscCimMacroConfig(CimMacroConfig):
    # === Digit geometry ===

    w_digit_num: int
    """Magnitude digits per weight."""
    w_digit_radix: int
    """Radix of the magnitude digits."""
    x_bit_num: int
    """Activation bits, processed LSB-first in serial WL phases."""

    # === Readout ratios ===

    dswct_ratio_msb: float
    """MSB-leg mirror ratio the per-digit DSWCT ratios are derived downward from."""
    sc_ratio_msb: float
    """MSB-bit SINWP-SC ratio."""

    # === Biases and supply ===

    v_wl_on__V: float
    """Digital word-line high level."""
    vdd__V: float
    """Core analog supply."""

    # === Timing ===

    t_sample__ns: float
    """Duration of each sampled input-bit phase."""
    t_settle__ns: float
    """Live-bit settle duration before SAR sensing begins."""

    # === Dynamic energy ===

    pn_isub_energy_per_op__fJ: float
    """Data-independent energy of one PN-ISUB sign decision per readout-lane access."""

    # === Submodules ===

    array_config: XbarArray1t1rConfig
    cablc_config: VoltageDriverConfig
    cablc_vref_config: ReferenceConfig
    sl_driver_config: VoltageDriverConfig
    tmcsa_config: TmcsaConfig
    tmcsa_iref_config: ReferenceConfig
    control_config: UnmodeledBlockConfig

    @property
    def supports_signed_weights(self) -> bool:
        return True

    @property
    def w_digit_n(self) -> int:
        return self.w_digit_num

    @property
    def w_digit_r(self) -> int:
        return self.w_digit_radix

    @property
    def w_enc(self) -> Encoding:
        return Encoding.TRUE_FORM

    @property
    def x_digit_n(self) -> int:
        return self.x_bit_num

    @property
    def x_digit_r(self) -> int:
        return 2

    @property
    def x_enc(self) -> Encoding:
        return Encoding.UNSIGNED

    @property
    def quant_scheme(self) -> CimMacroQuantizationScheme:
        return CimMacroQuantizationScheme.SIGN_MAGNITUDE

    def validate(self) -> None:
        super().validate()

        # --- Readout ratios ---

        self._require_pos(self.dswct_ratio_msb, "dswct_ratio_msb")
        self._require_pos(self.sc_ratio_msb, "sc_ratio_msb")

        # --- Biases and supply ---

        self._require_pos(self.v_wl_on__V, "v_wl_on__V")
        self._require_pos(self.vdd__V, "vdd__V")

        # --- Timing ---

        self._require_pos(self.t_sample__ns, "t_sample__ns")
        self._require_non_neg(self.t_settle__ns, "t_settle__ns")

        # --- Dynamic energy ---

        self._require_non_neg(self.pn_isub_energy_per_op__fJ, "pn_isub_energy_per_op__fJ")

        # --- Submodules ---

        if self.cablc_vref_config.shape != ():
            raise ValueError(f"require: cablc_vref_config.shape ({self.cablc_vref_config.shape}) == ()")
        cablc_v_ref__V = self.cablc_vref_config.values
        if isinstance(cablc_v_ref__V, tuple):
            raise TypeError("cablc_vref_config.values must be scalar")
        self._require_in_closed_interval(cablc_v_ref__V, "cablc_vref_config.values", lower=0.0, upper=self.vdd__V)

        want_taps = (1 << self.tmcsa_config.bits) - 1
        expected_shape = (len(self.rescale_factors), want_taps)
        if self.tmcsa_iref_config.shape != expected_shape:
            raise ValueError(f"require: tmcsa_iref_config.shape ({self.tmcsa_iref_config.shape}) == {expected_shape}")
        rows = self.tmcsa_iref_config.values
        if not isinstance(rows, tuple):
            raise TypeError("tmcsa_iref_config.values must be a matrix")
        for mode, row in enumerate(rows):
            if not isinstance(row, tuple):
                raise TypeError("tmcsa_iref_config.values must be a matrix")
            scalars: list[float] = []
            for value in row:
                if isinstance(value, tuple):
                    raise TypeError("tmcsa_iref_config.values must be a matrix")
                scalars.append(value)
            self._require_increasing(scalars, f"tmcsa_iref_config.values[{mode}]")


class Xue2020JsscCimMacroPolicy(CimMacroPolicy):
    array_policy: XbarArray1t1rPolicy
    cablc_policy: VoltageDriverPolicy
    cablc_vref_policy: ReferencePolicy
    sl_driver_policy: VoltageDriverPolicy
    tmcsa_policy: TmcsaPolicy
    tmcsa_iref_policy: ReferencePolicy


_Config = Xue2020JsscCimMacroConfig
_Policy = Xue2020JsscCimMacroPolicy


@CimMacro.register_neurox_impl(config_type=_Config, policy_type=_Policy)
class Xue2020JsscCimMacro(CimMacro):
    """Xue2020 SINWP 1T1R CIM sub-array.

    Use with a matching `Xue2020JsscCimMacroConfig` and policy. Place and
    fabricate the assembled tree, then program logical weights in the macro's
    `(input, output)` orientation, with physical instance axes prepended. The
    macro expands signed weight digits into its physical polarity columns;
    callers supply logical integer weights rather than device-state indices.

    Execute via `vec_mat_mul` with inputs in `x_value_range`. The quantization
    mode selects one configured current-reference ladder and rescale factor.
    Each ladder contains the full-width ascending decision taps even for reduced
    ADC precision. Use `effective_output_num` consistently for execution and
    timing when output ports are padded. Standalone measurement requires
    separately recording the operation duration; this macro's electrical path
    emits circuit energy.

    Args:
        config: Hardware configuration.
        policy: Run policy matching `config`.
        inst_shape: Positive physical instance extents; singletons allow
            broadcasting.
        dtype: Electrical tensor dtype.
    """

    config: _Config
    policy: _Policy

    # === Functional buffers ===

    _sl_v_ref__V: Tensor  # Shape: []
    _dswct_digit_ratios: Tensor  # Shape: [w_digit]
    _sinwp_bit_ratios: Tensor  # Shape: [x_bit]

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
        )
        self.row_num = self.input_num
        self.col_num = self.output_num * config.w_digit_num * _POLARITY_NUM
        self._init_children(dtype=dtype)
        if self.array.w_state_num < self._w_digit_radix:
            raise ValueError(
                f"require: array.w_state_num ({self.array.w_state_num}) >= w_digit_r ({self._w_digit_radix})"
            )
        self._register_functional_buffers(dtype=dtype)

    def _init_children(self, *, dtype: torch.dtype) -> None:
        config = self.config
        policy = self.policy
        lane_num = self.lane_num
        w_digit_num = config.w_digit_num
        self.cablc = VoltageDriver(
            config=config.cablc_config,
            policy=policy.cablc_policy,
            # Shape: [*inst_shape, x_bit=1, row=1, scan=1, lane, polarity, w_digit]
            inst_shape=(*self.inst_shape, 1, 1, 1, lane_num, _POLARITY_NUM, w_digit_num),
            dtype=dtype,
        )
        self.cablc_vref = Reference(
            config=config.cablc_vref_config,
            policy=policy.cablc_vref_policy,
            inst_shape=self.inst_shape,
            dtype=dtype,
        )

        self.sl_driver = VoltageDriver(
            config=config.sl_driver_config,
            policy=policy.sl_driver_policy,
            # Shape: [*inst_shape, x_bit=1, row=1, scan=1, lane, polarity, w_digit]
            inst_shape=(*self.inst_shape, 1, 1, 1, lane_num, _POLARITY_NUM, w_digit_num),
            dtype=dtype,
        )

        # Scan is a macro timing axis; the array contains every physical column.
        self.array = XbarArray1t1r(
            config=config.array_config,
            policy=policy.array_policy,
            # Shape: [*inst_shape, x_bit=1]
            inst_shape=(*self.inst_shape, 1),
            row_num=self.row_num,
            col_num=self.col_num,
            vdd__V=config.vdd__V,
            bl_driver=self.cablc,
            sl_driver=self.sl_driver,
            dtype=dtype,
        )

        self.tmcsa = Tmcsa(
            config=config.tmcsa_config,
            policy=policy.tmcsa_policy,
            # Shape: [*inst_shape, scan=1, lane]
            inst_shape=(*self.inst_shape, 1, lane_num),
            vdd__V=config.vdd__V,
            dtype=dtype,
        )
        self.tmcsa_iref = Reference(
            config=config.tmcsa_iref_config,
            policy=policy.tmcsa_iref_policy,
            inst_shape=self.inst_shape,
            dtype=dtype,
        )

        self.control = UnmodeledBlock(
            config=config.control_config,
            policy=UnmodeledBlockPolicy(),
            # Shape: [*inst_shape, scan=1]
            inst_shape=(*self.inst_shape, 1),
            dtype=dtype,
        )

    def _register_functional_buffers(self, *, dtype: torch.dtype) -> None:
        config = self.config
        w_pv = self._w_transcoder.place_values
        x_pv = self._x_transcoder.place_values
        dswct_digit_ratios = tuple(config.dswct_ratio_msb * pv / w_pv[-1] for pv in w_pv)
        sinwp_bit_ratios = tuple(config.sc_ratio_msb * pv / x_pv[-1] for pv in x_pv)
        self._dtype = dtype
        self._register_nonpersistent_buffer("_sl_v_ref__V", torch.zeros((), dtype=dtype))
        self._register_nonpersistent_buffer("_dswct_digit_ratios", torch.tensor(dswct_digit_ratios, dtype=dtype))
        self._register_nonpersistent_buffer("_sinwp_bit_ratios", torch.tensor(sinwp_bit_ratios, dtype=dtype))

    @property
    def adc_bits(self) -> int:
        """Maximum ADC magnitude resolution [bits]."""
        return self.tmcsa.bits

    def _latency_per_scan__ns(self, *, adc_active_bits: int | None) -> float:
        self._check_adc_active_bits(adc_active_bits)
        adc_active_bits = self._resolve_adc_active_bits(adc_active_bits)
        config = self.config
        return (
            (self._x_digit_num - 1) * config.t_sample__ns
            + self.config.t_settle__ns
            + self.tmcsa.latency__ns(active_bits=adc_active_bits)
        )

    @torch.no_grad()
    def program(self, w: Tensor) -> None:
        expected_shape = (*self.inst_shape, self.input_num, self.output_num)
        if tuple(w.shape) != expected_shape:
            raise ValueError(f"program() expects w.shape {expected_shape}; got {tuple(w.shape)}")

        # Column groups follow logical outputs: every lane in a scan, then the next scan.
        # Shape: [..., input, output] -> [..., row, output, w_digit]
        digits = self._w_transcoder.encode(w, dim=-1)

        # Positive digits occupy PWG; negative digits occupy NWG.
        # Shape: [..., row, output, w_digit] -> [..., row, output, polarity, w_digit]
        state_idx = torch.stack((digits.clamp_min(0), (-digits).clamp_min(0)), dim=-2)

        # Shape: [..., row, output, polarity, w_digit] -> [..., x_bit=1, row, col]
        state_idx = state_idx.flatten(-3, -1).unsqueeze(-3)
        self.array.program(state_idx.contiguous())

    def _record_cablc_dynamic_energy(
        self,
        i_dl__uA: Tensor,
        *,
        sample__ns: float,
        detect__ns: float,
    ) -> None:
        # Shape: [..., x_bit, scan, lane, polarity, w_digit] -> [..., x_bit]
        i_phase__uA = i_dl__uA.sum(dim=(-4, -3, -2, -1))
        # Shape: [..., x_bit] -> [...]
        i_sample__uA = i_phase__uA.narrow(-1, 0, self.config.x_bit_num - 1).sum(dim=-1)
        q__fC = q_conduction__fC(i__uA=i_sample__uA, duration__ns=sample__ns)
        q__fC = q__fC + q_conduction__fC(i__uA=i_phase__uA[..., -1], duration__ns=detect__ns)
        e__fJ = e_charge__fJ(v_supply__V=self.config.vdd__V, delta_q_abs__fC=q__fC)
        self._record_dynamic_energy(e__fJ, channel="cablc")

    def _record_dswct_dynamic_energy(
        self,
        i_wdl__uA: Tensor,
        *,
        sample__ns: float,
        detect__ns: float,
    ) -> None:
        # Shape: [..., x_bit, scan, lane, polarity, w_digit] -> [..., x_bit, scan, lane, polarity]
        i_phase__uA = i_wdl__uA.abs().sum(dim=-1)
        # Shape: [..., x_bit, scan, lane, polarity] -> [..., scan, lane, polarity]
        i_sample__uA = i_phase__uA.narrow(-4, 0, self.config.x_bit_num - 1).sum(dim=-4)
        q__fC = q_conduction__fC(i__uA=i_sample__uA, duration__ns=sample__ns)
        q__fC = q__fC + q_conduction__fC(i__uA=i_phase__uA.select(-4, -1), duration__ns=detect__ns)
        e__fJ = e_charge__fJ(v_supply__V=self.config.vdd__V, delta_q_abs__fC=q__fC)
        self._record_dynamic_energy(e__fJ, channel="dswct")

    def _record_sinwp_sc_dynamic_energy(
        self,
        i_sc_phase__uA: Tensor,
        *,
        sample__ns: float,
        detect__ns: float,
    ) -> None:
        # Shape: [..., x_bit, scan, lane, polarity] -> [..., scan, lane, polarity]
        i_sample__uA = i_sc_phase__uA.narrow(-4, 0, self.config.x_bit_num - 1).sum(dim=-4)
        q__fC = q_conduction__fC(i__uA=i_sample__uA, duration__ns=sample__ns)
        q__fC = q__fC + q_conduction__fC(i__uA=i_sc_phase__uA.select(-4, -1), duration__ns=detect__ns)
        e__fJ = e_charge__fJ(v_supply__V=self.config.vdd__V, delta_q_abs__fC=q__fC)
        self._record_dynamic_energy(e__fJ, channel="sinwp_sc")

    def _record_pn_isub_dynamic_energy(
        self,
        i_p__uA: Tensor,
        i_n__uA: Tensor,
        i_sub__uA: Tensor,
        *,
        detect__ns: float,
        active_mask: Tensor | None,
    ) -> None:
        i_branch__uA = i_p__uA + i_n__uA + i_sub__uA
        q__fC = q_conduction__fC(i__uA=i_branch__uA, duration__ns=detect__ns)
        e__fJ = e_charge__fJ(v_supply__V=self.config.vdd__V, delta_q_abs__fC=q__fC)
        if active_mask is not None:
            e__fJ = e__fJ + active_mask.to(i_branch__uA.dtype) * self.config.pn_isub_energy_per_op__fJ
        else:
            e__fJ = e__fJ + self.config.pn_isub_energy_per_op__fJ
        self._record_dynamic_energy(e__fJ, channel="pn_isub")

    def _vec_mat_mul_impl(
        self,
        x: Tensor,
        *,
        leading_shape: tuple[int, ...],
        quantization_mode: int,
        adc_active_bits: int | None,
        phase_mask: Tensor | None,
    ) -> Tensor:
        adc_active_bits = self._resolve_adc_active_bits(adc_active_bits)
        config = self.config
        sample__ns = config.t_sample__ns
        detect__ns = self.config.t_settle__ns + self.tmcsa.latency__ns(active_bits=adc_active_bits)

        # Shape: [..., mode, tap] -> [..., scan=1, lane=1, tap]
        adc_i_refs__uA = self.tmcsa_iref.values()[..., quantization_mode, :].unsqueeze(-2).unsqueeze(-2)

        # --- 1: Bit-expand x into K binary WL-drive phases (LSB first) ---

        # Shape: [..., row] -> [..., x_bit, row]
        v_wl__V = self._x_transcoder.encode(x, dim=-2).to(dtype=self._dtype) * config.v_wl_on__V
        # Shape: [..., x_bit, row] -> [..., x_bit, row, col=1]
        v_wl__V = v_wl__V.unsqueeze(self.array.col_dim)

        # --- 2: Solve the array once (cells + wire IR drop) -> I_DL ---

        # Shape: [..., x_bit, row=1, scan, lane, polarity, w_digit]
        seat_shape = (
            *leading_shape,
            config.x_bit_num,
            1,
            self.scan_num,
            self.lane_num,
            _POLARITY_NUM,
            config.w_digit_num,
        )
        bl_v_ref__V = self.cablc_vref.values()
        bl_v_ref_shape = (*bl_v_ref__V.shape, 1, 1, 1, 1, 1, 1)
        # Shape: [...] -> [..., x_bit=1, row=1, scan=1, lane=1, polarity=1, w_digit=1]
        bl_v_ref__V = bl_v_ref__V.view(bl_v_ref_shape)
        # Shape: [..., x_bit, row=1, scan, lane, polarity, w_digit]
        bl_driver_snap = self.cablc.snapshot(v_ref__V=bl_v_ref__V, shape=seat_shape)
        sl_driver_snap = self.sl_driver.snapshot(v_ref__V=self._sl_v_ref__V, shape=seat_shape)

        seat_mask = None
        if phase_mask is not None:
            seat_mask_shape = (*phase_mask.shape[:-2], 1, 1, self.scan_num, self.lane_num, 1, 1)
            # Shape: [..., scan, lane] -> [..., x_bit=1, row=1, scan, lane, polarity=1, w_digit=1]
            seat_mask = phase_mask.reshape(seat_mask_shape)
            bl_driver_snap = replace(
                bl_driver_snap, v_open__V=bl_driver_snap.v_open__V.where(seat_mask, self._sl_v_ref__V)
            )
            sl_driver_snap = replace(
                sl_driver_snap, v_open__V=sl_driver_snap.v_open__V.where(seat_mask, self._sl_v_ref__V)
            )
        bl_driver_snap = bl_driver_snap.flatten_axes(-4, -1)
        sl_driver_snap = sl_driver_snap.flatten_axes(-4, -1)

        array_dcop = self.array.solve_dc(
            v_wl__V=v_wl__V,
            leading_shape=(*leading_shape, config.x_bit_num),
            wl_phase_dims=(-3,),
            bl_driver_snap=bl_driver_snap,
            sl_driver_snap=sl_driver_snap,
        )
        seat_axes = (self.scan_num, self.lane_num, _POLARITY_NUM, config.w_digit_num)
        # Shape: [..., x_bit, row=1, col] -> [..., x_bit, row=1, scan, lane, polarity, w_digit]
        i_bl_seat__uA = array_dcop.i_bl_port__uA.unflatten(-1, seat_axes)
        i_sl_seat__uA = array_dcop.i_sl_port__uA.unflatten(-1, seat_axes)
        self.cablc.drive(i_port__uA=i_bl_seat__uA, enable=seat_mask)
        self.sl_driver.drive(i_port__uA=i_sl_seat__uA, enable=seat_mask)

        # Shape: [..., x_bit, scan, lane, polarity, w_digit]
        i_dl = array_dcop.i_bl_port__uA.squeeze(self.array.row_dim).unflatten(-1, seat_axes)

        # CABLC bills the whole VDD·I input branch; the array bills node capacitance.
        record_dynamic_energy = self._is_profiler_active()

        if record_dynamic_energy:
            self._record_cablc_dynamic_energy(i_dl, sample__ns=sample__ns, detect__ns=detect__ns)

        # --- 3: DSWCT place-value weighting -> I_WDL ---

        # Shape: [..., x_bit, scan, lane, polarity, w_digit]
        i_wdl = i_dl * self._dswct_digit_ratios

        if record_dynamic_energy:
            self._record_dswct_dynamic_energy(i_wdl, sample__ns=sample__ns, detect__ns=detect__ns)

        # --- 4: SINWP-SC temporal input-radix combine -> I_DL_PN ---

        # Shape: [x_bit] -> [x_bit, scan=1, lane=1, polarity=1, w_digit=1]
        bit_ratios = self._sinwp_bit_ratios.view(config.x_bit_num, 1, 1, 1, 1)
        # Shape: [..., x_bit, scan, lane, polarity, w_digit] -> [..., x_bit, scan, lane, polarity]
        i_sc_increment = (i_wdl * bit_ratios).sum(dim=-1)
        # Shape: [..., x_bit, scan, lane, polarity]
        i_sc_phase = i_sc_increment.cumsum(dim=-4)
        # Shape: [..., x_bit, scan, lane, polarity] -> [..., scan, lane, polarity]
        i_dl_pn = i_sc_phase.select(-4, -1)

        if record_dynamic_energy:
            self._record_sinwp_sc_dynamic_energy(
                i_sc_phase,
                sample__ns=sample__ns,
                detect__ns=detect__ns,
            )

        # --- 5: PN-ISUB single-ended magnitude + sign ---

        # Shape: [..., scan, lane, polarity] -> [..., scan, lane]
        i_p__uA = i_dl_pn[..., 0]
        i_n__uA = i_dl_pn[..., 1]
        i_sub__uA = (i_p__uA - i_n__uA).abs()
        sign = i_n__uA > i_p__uA

        if record_dynamic_energy:
            self._record_pn_isub_dynamic_energy(
                i_p__uA,
                i_n__uA,
                i_sub__uA,
                detect__ns=detect__ns,
                active_mask=phase_mask,
            )

        # --- 6: TMCSA quantize against the per-instance reference ladder ---

        # Shape: [..., scan, lane]
        code = self.tmcsa.convert(i_sub__uA, i_refs__uA=adc_i_refs__uA, active_bits=adc_active_bits, enable=phase_mask)
        # Shape: [..., scan, lane]
        signed = torch.where(sign, -code, code)

        # --- 7: Control energy ---

        scan_mask = None if phase_mask is None else phase_mask.any(dim=-1)
        self.control.execute((*leading_shape, self.scan_num), enable=scan_mask)

        # Shape: [..., scan, lane] -> [..., output]
        return signed.flatten(-2)
