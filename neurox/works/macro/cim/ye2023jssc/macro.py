"""Ye 2023 JSSC WH-2T1R RRAM CIM macro.

See Also:
    docs/reference/primitive/macro/cim/family.md
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

from .array import (
    Ye2023Jssc2t1rArray,
    Ye2023Jssc2t1rArrayConfig,
    Ye2023Jssc2t1rArrayPolicy,
)
from .rscsa import RsCsaIadc, RsCsaIadcConfig, RsCsaIadcPolicy


class Ye2023JsscCimMacroConfig(CimMacroConfig):
    # === Digit geometry ===

    w_digit_num: int
    """Digits per weight."""
    w_digit_radix: int
    """Weight-digit radix."""

    # === Biases and supply ===

    v_bl__V: float
    v_wl_on__V: float
    v_tbl__V: float
    """TBL clamp voltage shared by the array and readout path."""
    v_sl__V: float
    vdd__V: float
    """Core supply behind array-node charging."""

    # === Timing ===

    t_settle__ns: float
    """Array settling duration before each RS-CSA conversion."""

    # === Submodules ===

    array_config: Ye2023Jssc2t1rArrayConfig
    bl_driver_config: VoltageDriverConfig
    sl_driver_config: VoltageDriverConfig
    adc_config: RsCsaIadcConfig
    reference_config: ReferenceConfig
    mux_driver_config: UnmodeledBlockConfig
    timing_ctrl_config: UnmodeledBlockConfig

    @property
    def supports_signed_weights(self) -> bool:
        return False

    @property
    def w_digit_n(self) -> int:
        return self.w_digit_num

    @property
    def w_digit_r(self) -> int:
        return self.w_digit_radix

    @property
    def w_enc(self) -> Encoding:
        return Encoding.UNSIGNED

    @property
    def x_digit_n(self) -> int:
        return 1

    @property
    def x_digit_r(self) -> int:
        return 2

    @property
    def x_enc(self) -> Encoding:
        return Encoding.UNSIGNED

    @property
    def quant_scheme(self) -> CimMacroQuantizationScheme:
        return CimMacroQuantizationScheme.ZERO_POINT

    def validate(self) -> None:
        super().validate()

        # --- Biases and supply ---

        self._require_non_neg(self.v_bl__V, "v_bl__V")
        self._require_pos(self.v_wl_on__V, "v_wl_on__V")
        self._require_non_neg(self.v_tbl__V, "v_tbl__V")
        self._require_non_neg(self.v_sl__V, "v_sl__V")
        self._require_non_neg(self.vdd__V, "vdd__V")

        # --- Timing ---

        self._require_non_neg(self.t_settle__ns, "t_settle__ns")

        # --- Submodules ---

        expected_shape = (len(self.rescale_factors),)
        if self.reference_config.shape != expected_shape:
            raise ValueError(f"require: reference_config.shape ({self.reference_config.shape}) == {expected_shape}")
        if not isinstance(self.reference_config.values, tuple):
            raise TypeError("require: reference_config.values a one-dimensional tuple")
        for mode, value in enumerate(self.reference_config.values):
            if isinstance(value, tuple):
                raise TypeError("require: reference_config.values a one-dimensional tuple")
            self._require_pos(value, f"reference_config.values[{mode}]")


class Ye2023JsscCimMacroPolicy(CimMacroPolicy):
    array_policy: Ye2023Jssc2t1rArrayPolicy
    adc_policy: RsCsaIadcPolicy
    reference_policy: ReferencePolicy
    bl_driver_policy: VoltageDriverPolicy
    sl_driver_policy: VoltageDriverPolicy


_Config = Ye2023JsscCimMacroConfig
_Policy = Ye2023JsscCimMacroPolicy


@CimMacro.register_neurox_impl(config_type=_Config, policy_type=_Policy)
class Ye2023JsscCimMacro(CimMacro):
    """WH-2T1R array with RSM disabled and a time-shared RS-CSA readout."""

    config: _Config
    policy: _Policy

    # === Functional buffers ===

    _v_wl_scan__V: Tensor  # Shape: [scan, row, col=1]
    _scan_indices: Tensor  # Shape: [scan, lane=1]
    _tbl_row_indices: Tensor  # Shape: [scan, lane]
    _v_bl__V: Tensor  # Shape: []
    _v_sl__V: Tensor  # Shape: []

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
        )
        self.row_num = self.output_num
        self.col_num = self.input_num * (self._w_digit_num + 1)
        if config.max_active_num != self.input_num:
            raise ValueError(
                f"require: max_active_num ({config.max_active_num}) == input_num ({self.input_num}) "
                "(input-parallel design)"
            )
        self._init_children(dtype=dtype)
        if self.array.w_state_num < self._w_digit_radix:
            raise ValueError(
                f"require: array.w_state_num ({self.array.w_state_num}) >= w_digit_r ({self._w_digit_radix})"
            )
        self._register_functional_buffers(dtype=dtype)

    def _latency_per_scan__ns(self, *, adc_active_bits: int | None) -> float:
        self._check_adc_active_bits(adc_active_bits)
        adc_active_bits = self._resolve_adc_active_bits(adc_active_bits)
        return self.config.t_settle__ns + self.rscsa.latency__ns(active_bits=adc_active_bits)

    def _init_children(self, *, dtype: torch.dtype) -> None:
        config = self.config
        policy = self.policy
        w_pv = self._w_transcoder.place_values
        # [m0, m0, m0, ..., m1, m1, m1, ..., m2, m2, m2, ...], each multiplier is repeated input_num times
        col_t2_multipliers = tuple(m for m in (*w_pv, w_pv[-1]) for _ in range(self.input_num))

        # --- BL and SL boundary clamps ---

        self.bl_driver = VoltageDriver(
            config=config.bl_driver_config,
            policy=policy.bl_driver_policy,
            # Shape: [*inst_shape, scan=1, row=1, col]
            inst_shape=(*self.inst_shape, 1, 1, self.col_num),
            dtype=dtype,
        )
        self.sl_driver = VoltageDriver(
            config=config.sl_driver_config,
            policy=policy.sl_driver_policy,
            # Shape: [*inst_shape, scan=1, row=1, col]
            inst_shape=(*self.inst_shape, 1, 1, self.col_num),
            dtype=dtype,
        )

        # --- Transposed array: physical rows are outputs ---

        self.array = Ye2023Jssc2t1rArray(
            config=config.array_config,
            policy=policy.array_policy,
            # Shape: [*inst_shape, scan=1]
            inst_shape=(*self.inst_shape, 1),
            row_num=self.row_num,
            col_num=self.col_num,
            t2_multipliers=col_t2_multipliers,
            v_tbl__V=config.v_tbl__V,
            vdd__V=config.vdd__V,
            bl_driver=self.bl_driver,
            sl_driver=self.sl_driver,
            dtype=dtype,
        )

        # --- Time-shared readout and its independent reference source ---

        self.rscsa = RsCsaIadc(
            config=config.adc_config,
            policy=policy.adc_policy,
            # Shape: [*inst_shape, scan=1, lane]
            inst_shape=(*self.inst_shape, 1, self.lane_num),
            dtype=dtype,
        )

        self.rscsa_reference = Reference(
            config=config.reference_config,
            policy=policy.reference_policy,
            inst_shape=self.inst_shape,
            dtype=dtype,
        )

        # --- Unmodeled peripheral blocks ---

        self.mux_driver = UnmodeledBlock(
            config=config.mux_driver_config,
            policy=UnmodeledBlockPolicy(),
            # Shape: [*inst_shape, scan=1]
            inst_shape=(*self.inst_shape, 1),
            dtype=dtype,
        )
        self.timing_ctrl = UnmodeledBlock(
            config=config.timing_ctrl_config,
            policy=UnmodeledBlockPolicy(),
            # Shape: [*inst_shape, scan=1]
            inst_shape=(*self.inst_shape, 1),
            dtype=dtype,
        )

    def _register_functional_buffers(self, *, dtype: torch.dtype) -> None:
        # Shape: [row, row] -> [scan, lane, row]
        scan_wl_on = torch.eye(self.row_num, dtype=dtype).unflatten(0, (self.scan_num, self.lane_num))
        # Shape: [scan, lane, row] -> [scan, row, col=1]
        scan_wl_on = scan_wl_on.sum(dim=1).unsqueeze(self.array.col_dim)
        # Shape: [scan] -> [scan, lane=1]
        scan_indices = torch.arange(self.scan_num).unsqueeze(-1)
        # Shape: [scan, lane]
        tbl_row_indices = scan_indices * self.lane_num + torch.arange(self.lane_num).unsqueeze(0)

        self._register_nonpersistent_buffer("_v_wl_scan__V", scan_wl_on * self.config.v_wl_on__V)
        self._register_nonpersistent_buffer("_scan_indices", scan_indices)
        self._register_nonpersistent_buffer("_tbl_row_indices", tbl_row_indices)
        self._register_nonpersistent_buffer("_v_bl__V", torch.tensor(self.config.v_bl__V, dtype=dtype))
        self._register_nonpersistent_buffer("_v_sl__V", torch.tensor(self.config.v_sl__V, dtype=dtype))

    @property
    def adc_bits(self) -> int:
        return self.rscsa.bits

    def _append_disabled_rsm(self, value: Tensor, *, dim: int) -> Tensor:
        rsm = torch.zeros_like(value.narrow(dim, 0, 1))
        return torch.cat((value, rsm), dim=dim)

    @torch.no_grad()
    def program(self, w: Tensor) -> None:
        expected_shape = (*self.inst_shape, self.input_num, self.output_num)
        if tuple(w.shape) != expected_shape:
            raise ValueError(f"program() expects w.shape {expected_shape}; got {tuple(w.shape)}")

        # Physical rows follow logical outputs: every lane in a scan, then the next scan.
        # Shape: [..., input, output] -> [..., w_digit, input, row]
        digits = self._w_transcoder.encode(w, dim=-3)
        # Shape: [..., w_digit, input, row] -> [..., w_digit=_w_digit_num+1, input, row]
        digits = self._append_disabled_rsm(digits, dim=-3)
        # Shape: [..., w_digit, input, row] -> [..., row, col]
        state_idx = digits.movedim(-1, -3).flatten(-2, -1)
        # Shape: [..., row, col] -> [..., scan=1, row, col]
        state_idx = state_idx.unsqueeze(-3)
        self.array.program(state_idx.contiguous())

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
        access__ns = config.t_settle__ns + self.rscsa.latency__ns(active_bits=adc_active_bits)

        # --- 1: drive each logical input across the physical columns ---

        # Shape: [..., input] -> [..., w_digit, input]
        x_code = x.unsqueeze(-2).expand(*x.shape[:-1], self._w_digit_num, self.input_num)
        # Shape: [..., w_digit, input] -> [..., col]
        bl_code = self._append_disabled_rsm(x_code, dim=-2).flatten(-2)
        # Shape: [..., col]
        v_bl__V = bl_code * self._v_bl__V

        # --- 2: append the serialized row-scan axis ---

        # Shape: [scan, row, col=1]
        v_wl__V = self._v_wl_scan__V
        if phase_mask is not None:
            # Shape: [..., scan, lane] -> [..., scan=1, row, col=1]
            row_mask = phase_mask.flatten(-2).unsqueeze(-2).unsqueeze(-1)
            v_wl__V = v_wl__V.where(row_mask, 0)

        # --- 3: solve every scan phase ---

        port_shape = (*leading_shape, self.scan_num, 1, self.col_num)
        # Shape: [..., col] -> [..., scan=1, row=1, col]
        bl_v_ref__V = v_bl__V.unsqueeze(-2).unsqueeze(self.array.row_dim)
        sl_v_ref__V = self._v_sl__V.expand(port_shape)
        bl_driver_snap = self.bl_driver.snapshot(v_ref__V=bl_v_ref__V, shape=port_shape)
        sl_driver_snap = self.sl_driver.snapshot(v_ref__V=sl_v_ref__V, shape=port_shape)
        # Enable BL/SL driving when at least one lane uses the scan.
        # Shape: [..., scan, lane] -> [..., scan, row=1, col=1]
        scan_enable = None if phase_mask is None else phase_mask.any(dim=-1).unsqueeze(-1).unsqueeze(-1)
        if scan_enable is not None:
            bl_driver_snap = replace(bl_driver_snap, v_open__V=bl_driver_snap.v_open__V.where(scan_enable, 0))
            sl_driver_snap = replace(sl_driver_snap, v_open__V=sl_driver_snap.v_open__V.where(scan_enable, 0))
        array_dcop = self.array.solve_dc(
            v_wl__V=v_wl__V,
            leading_shape=(*leading_shape, self.scan_num),
            wl_phase_dims=(-3,),
            bl_driver_snap=bl_driver_snap,
            sl_driver_snap=sl_driver_snap,
        )

        # Complete both boundary accesses at the converged port-current layout.
        self.bl_driver.drive(i_port__uA=array_dcop.i_bl_port__uA, enable=scan_enable)
        self.sl_driver.drive(i_port__uA=array_dcop.i_sl_port__uA, enable=scan_enable)

        # Shape: [..., scan, row] -> [..., scan, lane]
        i_tbl__uA = array_dcop.i_tbl_by_row__uA[..., self._scan_indices, self._tbl_row_indices]

        # --- 4: array conduction ---

        if self._is_profiler_active():
            self._record_dynamic_energy(
                array_dcop.v_bl_port__V * array_dcop.i_bl_port__uA * access__ns,
                channel="bl_conduction",
            )
            self._record_dynamic_energy(
                config.v_tbl__V * i_tbl__uA * access__ns,
                channel="tbl_conduction",
            )

        # --- 5: RS-CSA quantize against its single reference current ---

        # Shape: [..., mode] -> [...]
        i_refs__uA = self.rscsa_reference.values()[..., quantization_mode]
        i_refs_shape = (*i_refs__uA.shape, 1, 1, 1)
        # Shape: [...] -> [..., scan=1, lane=1, tap=1]
        i_refs__uA = i_refs__uA.view(i_refs_shape)
        # Shape: [..., scan, lane]
        i_signal__uA = i_tbl__uA - self.array.i_tbl_leak__uA
        # Shape: [..., scan, lane]
        code = self.rscsa.convert(i_signal__uA, i_refs__uA, active_bits=adc_active_bits, enable=phase_mask)

        # --- 6: shared peripheral energy per active scan ---

        control_mask = None if phase_mask is None else phase_mask.any(dim=-1)
        self.mux_driver.execute((*leading_shape, self.scan_num), enable=control_mask)
        self.timing_ctrl.execute((*leading_shape, self.scan_num), enable=control_mask)

        # Shape: [..., scan, lane] -> [..., output]
        return code.flatten(-2)
