"""Xue2020 JSSC SINWP 1T1R CIM sub-array.

See Also:
    docs/system_design/cim_execution.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.common.encoding import Encoding
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
from neurox.primitive.physics import e_supply_charge__fJ, q_conduction__fC
from neurox.primitive.xbar.array import (
    XbarArray1t1r,
    XbarArray1t1rConfig,
    XbarArray1t1rPolicy,
)

from .tmcsa import Tmcsa, TmcsaConfig, TmcsaPolicy

_POLARITY_NUM = 2  # PWG, NWG per weight digit


class Xue2020JsscCimMacroConfig(CimMacroConfig):
    # === Weight / input geometry ===

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

    # === Timing ===

    t_sample__ns: float
    """Duration of each sampled input-bit phase."""

    t_settle__ns: float
    """Live-bit settle duration before SAR sensing begins."""

    # === Core analog supply ===

    v_wl_on__V: float
    """Digital word-line high level."""

    vdd__V: float
    """Core analog supply."""

    pn_isub_energy_per_op__fJ: float
    """Data-independent energy of one PN-ISUB sign decision per readout-lane access."""

    # === Submodules ===

    control_config: UnmodeledBlockConfig
    tmcsa_config: TmcsaConfig
    array_config: XbarArray1t1rConfig
    cablc_config: VoltageDriverConfig
    cablc_vref_config: ReferenceConfig
    sl_driver_config: VoltageDriverConfig

    tmcsa_iref_config: ReferenceConfig

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

        # --- Analog transfer and timing ---

        self._require_pos(self.dswct_ratio_msb, "dswct_ratio_msb")
        self._require_pos(self.sc_ratio_msb, "sc_ratio_msb")

        self._require_pos(self.t_sample__ns, "t_sample__ns")
        self._require_non_neg(self.t_settle__ns, "t_settle__ns")
        self._require_pos(self.v_wl_on__V, "v_wl_on__V")
        self._require_pos(self.vdd__V, "vdd__V")
        self._require_non_neg(self.pn_isub_energy_per_op__fJ, "pn_isub_energy_per_op__fJ")
        if self.cablc_vref_config.shape != ():
            raise ValueError(f"require: cablc_vref_config.shape ({self.cablc_vref_config.shape}) == ()")
        cablc_v_ref__V = self.cablc_vref_config.values
        if isinstance(cablc_v_ref__V, tuple):
            raise TypeError("cablc_vref_config.values must be scalar")
        self._require_in_closed_interval(cablc_v_ref__V, "cablc_vref_config.values", 0.0, self.vdd__V)

        # --- ADC reference and quantization modes ---

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


@CimMacro.register_neurox_module(config_type=Xue2020JsscCimMacroConfig, policy_type=Xue2020JsscCimMacroPolicy)
class Xue2020JsscCimMacro(CimMacro[Xue2020JsscCimMacroConfig, Xue2020JsscCimMacroPolicy]):
    """Xue2020 SINWP 1T1R CIM sub-array."""

    # === Functional buffers ===

    _v_wl_on__V: Tensor  # Shape: []
    _sl_v_ref__V: Tensor  # Shape: []
    _dswct_digit_ratios: Tensor  # Shape: [w_digit]
    _sinwp_bit_ratios: Tensor  # Shape: [x_bit]

    def __init__(
        self,
        *,
        config: Xue2020JsscCimMacroConfig,
        policy: Xue2020JsscCimMacroPolicy,
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
        self.row_num = self.input_num
        self.col_num = self.output_num * config.w_digit_num * _POLARITY_NUM
        self._init_children(dtype=dtype, T__K=T__K)
        if self.array.w_state_num < self._w_digit_radix:
            raise ValueError(
                f"require: array.w_state_num ({self.array.w_state_num}) >= w_digit_r ({self._w_digit_radix})"
            )
        self._register_functional_buffers(dtype=dtype)

    def _init_children(self, *, dtype: torch.dtype, T__K: float) -> None:
        config = self.config
        policy = self.policy
        # Scan is a macro timing axis; the array contains every physical column.
        self.array = XbarArray1t1r(
            config=config.array_config,
            policy=policy.array_policy,
            inst_shape=(*self.inst_shape, 1),
            row_num=self.row_num,
            col_num=self.col_num,
            vdd__V=config.vdd__V,
            dtype=dtype,
            T__K=T__K,
        )

        self.cablc = VoltageDriver(
            config=config.cablc_config,
            policy=policy.cablc_policy,
            inst_shape=(*self.inst_shape, 1, self.lane_num, 1, _POLARITY_NUM, config.w_digit_num),
            dtype=dtype,
            T__K=T__K,
        )
        self.cablc_vref = Reference(
            config=config.cablc_vref_config,
            policy=policy.cablc_vref_policy,
            inst_shape=self.inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

        self.sl_driver = VoltageDriver(
            config=config.sl_driver_config,
            policy=policy.sl_driver_policy,
            inst_shape=(*self.inst_shape, 1, self.lane_num, 1, _POLARITY_NUM, config.w_digit_num),
            dtype=dtype,
            T__K=T__K,
        )

        self.tmcsa = Tmcsa(
            config=config.tmcsa_config,
            policy=policy.tmcsa_policy,
            inst_shape=(*self.inst_shape, self.lane_num, 1),
            vdd__V=config.vdd__V,
            dtype=dtype,
            T__K=T__K,
        )
        self.tmcsa_iref = Reference(
            config=config.tmcsa_iref_config,
            policy=policy.tmcsa_iref_policy,
            inst_shape=self.inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

        self.control = UnmodeledBlock(
            config=config.control_config,
            policy=UnmodeledBlockPolicy(),
            inst_shape=(*self.inst_shape, 1),
            dtype=dtype,
            T__K=T__K,
        )

    def _register_functional_buffers(self, *, dtype: torch.dtype) -> None:
        config = self.config
        w_pv = self._w_transcoder.place_values
        x_pv = self._x_transcoder.place_values
        dswct_digit_ratios = tuple(config.dswct_ratio_msb * pv / w_pv[-1] for pv in w_pv)
        sinwp_bit_ratios = tuple(config.sc_ratio_msb * pv / x_pv[-1] for pv in x_pv)
        self._register_nonpersistent_buffer("_v_wl_on__V", torch.tensor(config.v_wl_on__V, dtype=dtype))
        self._register_nonpersistent_buffer("_sl_v_ref__V", torch.zeros((), dtype=dtype))
        self._register_nonpersistent_buffer("_dswct_digit_ratios", torch.tensor(dswct_digit_ratios, dtype=dtype))
        self._register_nonpersistent_buffer("_sinwp_bit_ratios", torch.tensor(sinwp_bit_ratios, dtype=dtype))

    @property
    def adc_bits(self) -> int:
        """Maximum ADC magnitude resolution [bits]."""
        return self.tmcsa.bits

    def latency__ns(self, *, adc_active_bits: int | None) -> float:
        """Circuit latency of all serialized scan positions."""
        self._check_adc_active_bits(adc_active_bits)
        adc_active_bits = self._resolve_adc_active_bits(adc_active_bits)
        config = self.config
        access__ns = (
            (self._x_digit_num - 1) * config.t_sample__ns
            + self.config.t_settle__ns
            + self.tmcsa.latency__ns(active_bits=adc_active_bits)
        )
        return access__ns * self.scan_num

    def program(self, w: Tensor) -> None:
        """Encode logical weights into the physical cell grid.

        Args:
            w: Logical weight tensor; entries must lie in `w_value_range`.
                Shape: `[*inst_shape, input_num, output_num]`.
        """
        expected_shape = (*self.inst_shape, self.input_num, self.output_num)
        if tuple(w.shape) != expected_shape:
            raise ValueError(f"program() expects w.shape {expected_shape}; got {tuple(w.shape)}")

        # Shape: [..., input_num, output_num] -> [..., row, col, w_digit]
        digits = self._w_transcoder.encode(w, dim=-1)

        # Positive digits occupy PWG; negative digits occupy NWG.
        # Shape: [..., row, col, w_digit] -> [..., row, col, polarity, w_digit]
        state_idx = torch.stack((digits.clamp_min(0), (-digits).clamp_min(0)), dim=-2)

        # col = lane * scan_num + scan
        # Shape: [..., row, col, polarity, w_digit] -> [..., lane, scan, polarity, w_digit, row]
        state_idx = state_idx.unflatten(-3, (self.lane_num, self.scan_num)).movedim(-5, -1)
        # Shape: [..., lane, scan, polarity, w_digit, row] -> [..., x_bit=1, phys_col, row]
        state_idx = state_idx.flatten(-5, -2).unsqueeze(-3)
        self.array.program(state_idx)

    def _record_cablc_dynamic_energy(
        self,
        i_dl__uA: Tensor,
        *,
        sample__ns: float,
        detect__ns: float,
    ) -> None:
        # Shape: [..., x_bit, lane, scan, polarity, w_digit] -> [..., x_bit]
        i_phase__uA = i_dl__uA.sum(dim=(-4, -3, -2, -1))
        # Shape: [..., x_bit] -> [...]
        i_sample__uA = i_phase__uA.narrow(-1, 0, self.config.x_bit_num - 1).sum(dim=-1)
        q__fC = q_conduction__fC(i_sample__uA, sample__ns)
        q__fC = q__fC + q_conduction__fC(i_phase__uA.select(-1, -1), detect__ns)
        e__fJ = e_supply_charge__fJ(self.config.vdd__V, q__fC)
        self._record_dynamic_energy(e__fJ, channel="cablc")

    def _record_dswct_dynamic_energy(
        self,
        i_wdl__uA: Tensor,
        *,
        sample__ns: float,
        detect__ns: float,
    ) -> None:
        # Shape: [..., x_bit, lane, scan, polarity, w_digit] -> [..., x_bit, lane, scan, polarity]
        i_phase__uA = i_wdl__uA.abs().sum(dim=-1)
        # Shape: [..., x_bit, lane, scan, polarity] -> [..., lane, scan, polarity]
        i_sample__uA = i_phase__uA.narrow(-4, 0, self.config.x_bit_num - 1).sum(dim=-4)
        q__fC = q_conduction__fC(i_sample__uA, sample__ns)
        q__fC = q__fC + q_conduction__fC(i_phase__uA.select(-4, -1), detect__ns)
        e__fJ = e_supply_charge__fJ(self.config.vdd__V, q__fC)
        self._record_dynamic_energy(e__fJ, channel="dswct")

    def _record_sinwp_sc_dynamic_energy(
        self,
        i_sc_phase__uA: Tensor,
        *,
        sample__ns: float,
        detect__ns: float,
    ) -> None:
        # Shape: [..., x_bit, lane, scan, polarity] -> [..., lane, scan, polarity]
        i_sample__uA = i_sc_phase__uA.narrow(-4, 0, self.config.x_bit_num - 1).sum(dim=-4)
        q__fC = q_conduction__fC(i_sample__uA, sample__ns)
        q__fC = q__fC + q_conduction__fC(i_sc_phase__uA.select(-4, -1), detect__ns)
        e__fJ = e_supply_charge__fJ(self.config.vdd__V, q__fC)
        self._record_dynamic_energy(e__fJ, channel="sinwp_sc")

    def _record_pn_isub_dynamic_energy(
        self,
        i_p__uA: Tensor,
        i_n__uA: Tensor,
        i_sub__uA: Tensor,
        *,
        detect__ns: float,
    ) -> None:
        i_branch__uA = i_p__uA + i_n__uA + i_sub__uA
        q__fC = q_conduction__fC(i_branch__uA, detect__ns)
        e__fJ = e_supply_charge__fJ(self.config.vdd__V, q__fC) + self.config.pn_isub_energy_per_op__fJ
        self._record_dynamic_energy(e__fJ, channel="pn_isub")

    def _vec_mat_mul_impl(
        self,
        x: Tensor,
        *,
        quantization_mode: int,
        adc_active_bits: int | None,
    ) -> Tensor:
        adc_active_bits = self._resolve_adc_active_bits(adc_active_bits)
        config = self.config
        sample__ns = config.t_sample__ns
        detect__ns = self.config.t_settle__ns + self.tmcsa.latency__ns(active_bits=adc_active_bits)

        # Shape: [..., tap] -> [..., lane=1, scan=1, tap]
        adc_i_refs__uA = self.tmcsa_iref.values()[..., quantization_mode, None, None, :]

        # --- 1: Bit-expand x into K binary WL-drive phases (LSB first) ---

        # Shape: [..., row] -> [..., x_bit, row]
        v_wl__V = self._x_transcoder.encode(x.long(), dim=-2) * self._v_wl_on__V

        # --- 2: Solve the array once (cells + wire IR drop) -> I_DL ---

        seat_shape = (*v_wl__V.shape[:-1], self.lane_num, self.scan_num, _POLARITY_NUM, config.w_digit_num)
        # Shape: [..., tap] -> [..., x_bit=1, lane=1, scan=1, polarity=1, w_digit=1]
        bl_v_ref__V = self.cablc_vref.values()[..., None, None, None, None, None]

        # Shape: [..., x_bit, lane, scan, polarity, w_digit] -> [..., x_bit, phys_col]
        bl_snap = self.cablc.snapshot(v_ref__V=bl_v_ref__V, shape=seat_shape).flatten_axes(-4, -1)
        # Shape: [..., x_bit, lane, scan, polarity, w_digit] -> [..., x_bit, phys_col]
        sl_snap = self.sl_driver.snapshot(v_ref__V=self._sl_v_ref__V, shape=seat_shape).flatten_axes(-4, -1)

        # Shape: [..., x_bit, row]
        steady = self.array.solve_array(
            v_wl__V=v_wl__V,
            wl_phase_dims=(-2,),
            bl_driver=self.cablc,
            bl_driver_snap=bl_snap,
            sl_driver=self.sl_driver,
            sl_driver_snap=sl_snap,
        )
        # Shape: [..., x_bit, phys_col] -> [..., x_bit, lane, scan, polarity, w_digit]
        seat_axes = (self.lane_num, self.scan_num, _POLARITY_NUM, config.w_digit_num)
        i_bl_seat__uA = steady.i_bl_port__uA.unflatten(-1, seat_axes)
        i_sl_seat__uA = steady.i_sl_port__uA.unflatten(-1, seat_axes)
        self.cablc.drive(i_port__uA=i_bl_seat__uA)
        self.sl_driver.drive(i_port__uA=i_sl_seat__uA)

        # Shape: [..., x_bit, lane, scan, polarity, w_digit]
        i_dl = i_bl_seat__uA

        # CABLC bills the whole VDD·I input branch; the array bills node capacitance.
        record_dynamic_energy = self._is_dynamic_energy_profile_active()

        if record_dynamic_energy:
            self._record_cablc_dynamic_energy(i_dl, sample__ns=sample__ns, detect__ns=detect__ns)

        # --- 3: DSWCT place-value weighting -> I_WDL ---

        # Shape: [..., x_bit, lane, scan, polarity, w_digit]
        i_wdl = i_dl * self._dswct_digit_ratios

        if record_dynamic_energy:
            self._record_dswct_dynamic_energy(i_wdl, sample__ns=sample__ns, detect__ns=detect__ns)

        # --- 4: SINWP-SC temporal input-radix combine -> I_DL_PN ---

        # Shape: [..., x_bit, lane, scan, polarity, w_digit] -> [..., x_bit, lane, scan, polarity]
        i_sc_increment = (i_wdl * self._sinwp_bit_ratios.view(config.x_bit_num, 1, 1, 1, 1)).sum(dim=-1)
        # Shape: [..., x_bit, lane, scan, polarity]
        i_sc_phase = i_sc_increment.cumsum(dim=-4)
        # Shape: [..., x_bit, lane, scan, polarity] -> [..., lane, scan, polarity]
        i_dl_pn = i_sc_phase.select(-4, -1)

        if record_dynamic_energy:
            self._record_sinwp_sc_dynamic_energy(i_sc_phase, sample__ns=sample__ns, detect__ns=detect__ns)

        # --- 5: PN-ISUB single-ended magnitude + sign ---

        # Shape: [..., lane, scan, polarity] -> [..., lane, scan]
        i_p__uA = i_dl_pn[..., 0]
        i_n__uA = i_dl_pn[..., 1]
        i_sub__uA = (i_p__uA - i_n__uA).abs()
        sign = i_n__uA > i_p__uA

        if record_dynamic_energy:
            self._record_pn_isub_dynamic_energy(i_p__uA, i_n__uA, i_sub__uA, detect__ns=detect__ns)

        # --- 6: TMCSA quantize against the per-instance reference ladder ---

        # Shape: [..., lane, scan]
        code = self.tmcsa.convert(i_sub__uA, adc_i_refs__uA, active_bits=adc_active_bits)
        # Shape: [..., lane, scan]
        signed = torch.where(sign, -code, code)

        # --- 7: Control energy ---

        self.control.execute((*signed.shape[:-2], signed.shape[-1]))

        return signed
