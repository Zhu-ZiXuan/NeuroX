"""Xue2020 JSSC SINWP 1T1R CIM sub-array.

See Also:
    docs/system_design/cim_execution.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.common.encoding import TrueFormTranscoder
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
from neurox.primitive.analog.voltage_dac import Vdac, VdacConfig, VdacPolicy
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
    XbarArray1t1rScanMode,
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

    scan_num: int
    """Scan positions time-multiplexed onto each parallel readout lane."""

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

    t_cycle__ns: float
    """Operating period and leakage integration window."""

    # === Core analog supply ===

    vdd__V: float
    """Core analog supply."""

    pn_isub_energy_per_op__fJ: float
    """Data-independent energy of one PN-ISUB sign decision per readout-lane access."""

    # === Submodules ===

    control_config: UnmodeledBlockConfig
    tmcsa_config: TmcsaConfig
    array_config: XbarArray1t1rConfig
    wl_dac_config: VdacConfig
    cablc_config: VoltageDriverConfig
    cablc_vref_config: ReferenceConfig
    sl_driver_config: VoltageDriverConfig

    tmcsa_iref_config: ReferenceConfig

    @property
    def digit_ratios(self) -> tuple[float, ...]:
        """Per-digit DSWCT mirror ratios (LSB-first): `r_d = dswct_ratio_msb * radix**(d - (D-1))`."""
        d_top = self.w_digit_num - 1
        return tuple(self.dswct_ratio_msb * self.w_digit_radix ** (d - d_top) for d in range(self.w_digit_num))

    @property
    def x_bit_ratios(self) -> tuple[float, ...]:
        """Per-input-bit SINWP-SC combine ratios (LSB-first): `s_k = sc_ratio_msb * 2**(k - (K-1))`."""
        k_top = self.x_bit_num - 1
        return tuple(self.sc_ratio_msb * 2.0 ** (k - k_top) for k in range(self.x_bit_num))

    def detect__ns(self, adc_active_bits: int) -> float:
        """Settle and sensing duration of the final input phase."""
        if not (1 <= adc_active_bits <= self.tmcsa_config.bits):
            raise ValueError(
                f"require: adc_active_bits ({adc_active_bits}) in [1, adc_bits ({self.tmcsa_config.bits})]"
            )
        return self.t_settle__ns + adc_active_bits * self.tmcsa_config.latency_per_bit__ns

    def access_latency__ns(self, adc_active_bits: int) -> float:
        """Circuit latency of one column-MUX access."""
        return (self.x_bit_num - 1) * self.t_sample__ns + self.detect__ns(adc_active_bits)

    def validate(self) -> None:
        super().validate()

        # --- Data geometry ---

        self._require_pos(self.w_digit_num, "w_digit_num")
        self._require_ge(self.w_digit_radix, "w_digit_radix", 2)

        self._require_pos(self.x_bit_num, "x_bit_num")
        self._require_pos(self.scan_num, "scan_num")

        # --- Analog transfer and timing ---

        self._require_pos(self.dswct_ratio_msb, "dswct_ratio_msb")
        self._require_pos(self.sc_ratio_msb, "sc_ratio_msb")

        self._require_pos(self.t_sample__ns, "t_sample__ns")
        self._require_non_neg(self.t_settle__ns, "t_settle__ns")
        self._require_pos(self.t_cycle__ns, "t_cycle__ns")
        self._require_ge(self.t_cycle__ns, "t_cycle__ns", self.access_latency__ns(self.tmcsa_config.bits))
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
    wl_dac_policy: VdacPolicy
    cablc_policy: VoltageDriverPolicy
    cablc_vref_policy: ReferencePolicy
    sl_driver_policy: VoltageDriverPolicy
    tmcsa_policy: TmcsaPolicy
    tmcsa_iref_policy: ReferencePolicy


@CimMacro.register_neurox_module(config_type=Xue2020JsscCimMacroConfig, policy_type=Xue2020JsscCimMacroPolicy)
class Xue2020JsscCimMacro(CimMacro[Xue2020JsscCimMacroConfig, Xue2020JsscCimMacroPolicy]):
    """Xue2020 SINWP 1T1R CIM sub-array."""

    # === Functional buffers ===

    _sl_v_ref__V: Tensor  # Shape: []
    _dswct_digit_ratios: Tensor  # Shape: [w_digit]
    _sinwp_bit_ratios: Tensor  # Shape: [x_bit]

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
        if self.col_num % config.scan_num != 0:
            raise ValueError(
                f"require: col_num ({self.col_num}) % scan_num ({config.scan_num}) == 0 "
                "(the readout-lane regrouping is an exact reshape)"
            )
        self.lane_num = self.col_num // config.scan_num
        if config.max_active_num > self.row_num:
            raise ValueError(f"require: max_active_num ({config.max_active_num}) <= row_num ({self.row_num})")
        self._w_transcoder = TrueFormTranscoder(
            radix=config.w_digit_radix,
            digit_count=config.w_digit_num,
        )
        self._x_transcoder = TrueFormTranscoder(radix=2, digit_count=config.x_bit_num)
        self._init_children(dtype=dtype, T__K=T__K)
        self._register_functional_buffers(dtype=dtype)

    def latency__ns(self, *, adc_active_bits: int) -> float:
        """Circuit latency of all serialized scan positions."""
        config = self.config
        return config.access_latency__ns(adc_active_bits) * config.scan_num

    def initiation_interval__ns(self, *, adc_active_bits: int) -> float:
        """Scheduled duration of all serialized scan positions."""
        self.tmcsa.latency__ns(active_bits=adc_active_bits)
        return self.config.t_cycle__ns * self.config.scan_num

    def _init_children(self, *, dtype: torch.dtype, T__K: float) -> None:
        config = self.config
        policy = self.policy
        phys_col_num = self.col_num * config.w_digit_num * _POLARITY_NUM

        # Scan is a macro timing axis; the array contains every physical column.
        self.array = XbarArray1t1r(
            config=config.array_config,
            policy=policy.array_policy,
            inst_shape=(*self.inst_shape, 1),
            row_num=self.row_num,
            col_num=phys_col_num,
            scan_mode=XbarArray1t1rScanMode.WL_IN_BL_SCAN,
            vdd__V=config.vdd__V,
            dtype=dtype,
            T__K=T__K,
        )

        self.wl_dac = Vdac.from_config(
            config=config.wl_dac_config,
            policy=policy.wl_dac_policy,
            inst_shape=(*self.inst_shape, 1, self.row_num),
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
        self._register_nonpersistent_buffer("_sl_v_ref__V", torch.zeros((), dtype=dtype))
        self._register_nonpersistent_buffer("_dswct_digit_ratios", torch.tensor(config.digit_ratios, dtype=dtype))
        self._register_nonpersistent_buffer("_sinwp_bit_ratios", torch.tensor(config.x_bit_ratios, dtype=dtype))

    @property
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive activation range."""
        return (0, (1 << self.config.x_bit_num) - 1)

    @property
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive weight range."""
        return self._w_transcoder.value_range

    @property
    def adc_bits(self) -> int:
        """Maximum ADC magnitude resolution [bits]."""
        return self.config.tmcsa_config.bits

    @property
    def _quantization_scheme(self) -> CimMacroQuantizationScheme:
        return CimMacroQuantizationScheme.SIGN_MAGNITUDE

    def restore_adc_layout(self, value: Tensor) -> Tensor:
        """Arrange the TMCSA call layout onto logical output columns."""
        return value.flatten(-2)

    def program(self, w: Tensor) -> None:
        """Encode logical weights into the physical cell grid.

        Args:
            w: Logical weight tensor; entries must lie in `w_value_range`.
                Shape: `[*inst_shape, input_num, output_num]`.
        """
        expected_shape = (*self.inst_shape, *self._logical_shape)
        if tuple(w.shape) != expected_shape:
            raise ValueError(f"program() expects w.shape {expected_shape}; got {tuple(w.shape)}")

        # Shape: [..., input_num, output_num] -> [..., row, col, w_digit]
        digits = self._w_transcoder.encode(w, dim=-1)

        # Positive digits occupy PWG; negative digits occupy NWG.
        # Shape: [..., row, col, w_digit] -> [..., row, col, polarity, w_digit]
        states = torch.stack((digits.clamp_min(0), (-digits).clamp_min(0)), dim=-2)

        # col = lane * scan_num + scan
        # Shape: [..., row, col, polarity, w_digit] -> [..., lane, scan, polarity, w_digit, row]
        states = states.unflatten(-3, (self.lane_num, self.config.scan_num)).movedim(-5, -1)
        # Shape: [..., lane, scan, polarity, w_digit, row] -> [..., instance_slot=1, phys_col, row]
        states = states.flatten(-5, -2).unsqueeze(-3)
        self.array.program(states)

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

    def vec_mat_mul(self, x: Tensor, *, quantization_mode: int, adc_active_bits: int) -> Tensor:
        """Run the array solve and current-mode readout.

        Args:
            x: Activation tensor; entries in `x_value_range`. Positions outside the
                caller-selected set must be zero.
                Shape: `[..., row_num]`.
            quantization_mode: Index naming the threshold-ladder row and its
                calibrated output scale.
            adc_active_bits: Active ADC resolution in `[1, adc_bits]`.

        Returns:
            Signed-magnitude raw-code tensor retaining the inherited leading axes.
            Shape: `[..., col_num]`.

        Raises:
            ValueError: `quantization_mode` or `adc_active_bits` is outside its declared range.
        """
        self._check_quantization_mode(quantization_mode)
        self._check_adc_active_bits(adc_active_bits)

        config = self.config
        sample__ns = config.t_sample__ns
        detect__ns = config.detect__ns(adc_active_bits)

        # Shape: [..., tap] -> [..., lane=1, scan=1, tap]
        adc_i_refs__uA = self.tmcsa_iref.values()[..., quantization_mode, None, None, :]

        # --- 1: Bit-expand x into K WL planes (LSB first) + WL DAC ---

        # Shape: [..., row] -> [..., x_bit, row]
        planes = self._x_transcoder.encode(x.long(), dim=-2)
        v_wl__V = self.wl_dac.convert(planes)

        # --- 2: Solve the array once (cells + wire IR drop) -> I_DL ---

        seat_shape = (*planes.shape[:-1], self.lane_num, config.scan_num, _POLARITY_NUM, config.w_digit_num)
        # Shape: [..., tap] -> [..., x_bit=1, lane=1, scan=1, polarity=1, w_digit=1]
        bl_v_ref__V = self.cablc_vref.values()[..., None, None, None, None, None]

        # Shape: [..., x_bit, lane, scan, polarity, w_digit] -> [..., x_bit, phys_col]
        bl_snap = self.cablc.snapshot(v_ref__V=bl_v_ref__V, shape=seat_shape).flatten_axes(-4, -1)
        # Shape: [..., x_bit, lane, scan, polarity, w_digit] -> [..., x_bit, phys_col]
        sl_snap = self.sl_driver.snapshot(v_ref__V=self._sl_v_ref__V, shape=seat_shape).flatten_axes(-4, -1)

        # Shape: [..., x_bit, row]
        steady = self.array.solve_array(
            v_wl__V,
            bl_driver=self.cablc,
            bl_driver_snap=bl_snap,
            sl_driver=self.sl_driver,
            sl_driver_snap=sl_snap,
        )
        # Shape: [..., x_bit, phys_col] -> [..., x_bit, lane, scan, polarity, w_digit]
        seat_axes = (self.lane_num, config.scan_num, _POLARITY_NUM, config.w_digit_num)
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
        signed = (1 - 2 * sign.long()) * code

        # --- 7: Control energy ---

        self.control.execute((*signed.shape[:-2], signed.shape[-1]))

        return self.restore_adc_layout(signed)
