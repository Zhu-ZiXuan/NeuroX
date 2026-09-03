"""Ye 2023 JSSC WH-2T1R RRAM CIM macro.

See Also:
    docs/reference/primitive/macro/cim/family.md
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

from .array import Ye2023Jssc2t1rArray, Ye2023Jssc2t1rArrayConfig, Ye2023Jssc2t1rArrayPolicy
from .rscsa import RsCsaIadc, RsCsaIadcConfig, RsCsaIadcPolicy


class Ye2023JsscCimMacroConfig(CimMacroConfig):
    w_digit_num: int
    """Digits per weight."""

    w_digit_radix: int
    """Weight-digit radix."""

    # === Device-bearing sub-blocks ===

    array_config: Ye2023Jssc2t1rArrayConfig
    adc_config: RsCsaIadcConfig
    reference_config: ReferenceConfig
    bl_driver_config: VoltageDriverConfig
    sl_driver_config: VoltageDriverConfig

    # === Unmodeled peripheral blocks ===

    mux_driver_config: UnmodeledBlockConfig
    timing_ctrl_config: UnmodeledBlockConfig

    # === Timing ===

    t_settle__ns: float
    """Array settling duration before each RS-CSA conversion."""

    # === Biases and supply ===

    v_bl__V: float
    v_wl_on__V: float
    v_tbl__V: float
    """TBL clamp voltage shared by the array and readout path."""
    v_sl__V: float
    vdd__V: float
    """Core supply behind array-node charging."""

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

        self._require_non_neg(self.t_settle__ns, "t_settle__ns")
        self._require_non_neg(self.v_bl__V, "v_bl__V")
        self._require_pos(self.v_wl_on__V, "v_wl_on__V")
        self._require_non_neg(self.v_tbl__V, "v_tbl__V")
        self._require_non_neg(self.v_sl__V, "v_sl__V")
        self._require_non_neg(self.vdd__V, "vdd__V")

        # --- Readout reference ---

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


@CimMacro.register_neurox_module(
    config_type=Ye2023JsscCimMacroConfig,
    policy_type=Ye2023JsscCimMacroPolicy,
)
class Ye2023JsscCimMacro(CimMacro[Ye2023JsscCimMacroConfig, Ye2023JsscCimMacroPolicy]):
    """WH-2T1R array with RSM disabled and a time-shared RS-CSA readout."""

    # === Functional buffers ===

    _v_wl_scan__V: Tensor  # Shape: [scan, row]
    _scan_indices: Tensor  # Shape: [scan, lane=1]
    _tbl_row_indices: Tensor  # Shape: [scan, lane]
    _v_bl__V: Tensor  # Shape: []
    _v_sl__V: Tensor  # Shape: []

    def __init__(
        self,
        *,
        config: Ye2023JsscCimMacroConfig,
        policy: Ye2023JsscCimMacroPolicy,
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
        self.row_num = self.output_num
        self.col_num = self.input_num * (self._w_digit_num + 1)
        if config.max_active_num != self.input_num:
            raise ValueError(
                f"require: max_active_num ({config.max_active_num}) == input_num ({self.input_num}) "
                "(input-parallel design)"
            )
        self._init_children(dtype=dtype, T__K=T__K)
        if self.array.w_state_num < self._w_digit_radix:
            raise ValueError(
                f"require: array.w_state_num ({self.array.w_state_num}) >= w_digit_r ({self._w_digit_radix})"
            )
        self._register_functional_buffers(dtype=dtype)

    def latency__ns(self, *, adc_active_bits: int | None) -> float:
        """Circuit latency of all serialized output-row scans."""
        self._check_adc_active_bits(adc_active_bits)
        adc_active_bits = self._resolve_adc_active_bits(adc_active_bits)
        access__ns = self.config.t_settle__ns + self.rscsa.latency__ns(active_bits=adc_active_bits)
        return self.scan_num * access__ns

    def _init_children(self, *, dtype: torch.dtype, T__K: float) -> None:
        config = self.config
        policy = self.policy
        w_pv = self._w_transcoder.place_values
        # [m0, m0, m0, ..., m1, m1, m1, ..., m2, m2, m2, ...], each multiplier is repeated input_num times
        col_t2_multipliers = tuple(m for m in (*w_pv, w_pv[-1]) for _ in range(self.input_num))

        # --- Transposed array: physical rows are outputs ---

        self.array = Ye2023Jssc2t1rArray(
            config=config.array_config,
            policy=policy.array_policy,
            inst_shape=(*self.inst_shape, 1),
            row_num=self.row_num,
            col_num=self.col_num,
            t2_multipliers=col_t2_multipliers,
            v_tbl__V=config.v_tbl__V,
            vdd__V=config.vdd__V,
            dtype=dtype,
            T__K=T__K,
        )

        # --- BL and SL boundary clamps ---

        self.bl_driver = VoltageDriver(
            config=config.bl_driver_config,
            policy=policy.bl_driver_policy,
            inst_shape=(*self.inst_shape, 1, self.col_num),
            dtype=dtype,
            T__K=T__K,
        )
        self.sl_driver = VoltageDriver(
            config=config.sl_driver_config,
            policy=policy.sl_driver_policy,
            inst_shape=(*self.inst_shape, 1, self.col_num),
            dtype=dtype,
            T__K=T__K,
        )

        # --- Time-shared readout and its independent reference source ---

        self.rscsa = RsCsaIadc(
            config=config.adc_config,
            policy=policy.adc_policy,
            inst_shape=(*self.inst_shape, self.lane_num, 1),
            dtype=dtype,
            T__K=T__K,
        )

        self.rscsa_reference = Reference(
            config=config.reference_config,
            policy=policy.reference_policy,
            inst_shape=self.inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

        # --- Unmodeled peripheral blocks ---

        self.mux_driver = UnmodeledBlock(
            config=config.mux_driver_config,
            policy=UnmodeledBlockPolicy(),
            inst_shape=(*self.inst_shape, 1),
            dtype=dtype,
            T__K=T__K,
        )
        self.timing_ctrl = UnmodeledBlock(
            config=config.timing_ctrl_config,
            policy=UnmodeledBlockPolicy(),
            inst_shape=(*self.inst_shape, 1),
            dtype=dtype,
            T__K=T__K,
        )

    def _register_functional_buffers(self, *, dtype: torch.dtype) -> None:
        # Shape: [row, row] -> [lane, scan, row]
        scan_wl_on = torch.eye(self.row_num, dtype=dtype).unflatten(0, (self.lane_num, self.scan_num))
        # Shape: [lane, scan, row] -> [scan, row]
        scan_wl_on = scan_wl_on.sum(dim=0)
        # Shape: [scan] -> [scan, lane=1]
        scan_indices = torch.arange(self.scan_num).unsqueeze(-1)
        # Shape: [scan, lane]
        tbl_row_indices = scan_indices + torch.arange(self.lane_num).unsqueeze(0) * self.scan_num

        self._register_nonpersistent_buffer("_v_wl_scan__V", scan_wl_on * self.config.v_wl_on__V)
        self._register_nonpersistent_buffer("_scan_indices", scan_indices)
        self._register_nonpersistent_buffer("_tbl_row_indices", tbl_row_indices)
        self._register_nonpersistent_buffer("_v_bl__V", torch.tensor(self.config.v_bl__V, dtype=dtype))
        self._register_nonpersistent_buffer("_v_sl__V", torch.tensor(self.config.v_sl__V, dtype=dtype))

    @property
    def adc_bits(self) -> int:
        """Maximum ADC resolution [bits] — the RS-CSA's physical resolution."""
        return self.rscsa.bits

    def _append_disabled_rsm(self, value: Tensor, *, dim: int) -> Tensor:
        rsm = torch.zeros_like(value.narrow(dim, 0, 1))
        return torch.cat((value, rsm), dim=dim)

    def program(self, w: Tensor) -> None:
        """Encode logical weights onto the physical cell grid.

        Args:
            w: Logical unsigned weight tensor.
                Shape: `[*inst_shape, input, output]`.
        """
        expected_shape = (*self.inst_shape, self.input_num, self.output_num)
        if tuple(w.shape) != expected_shape:
            raise ValueError(f"program() expects w.shape {expected_shape}; got {tuple(w.shape)}")

        # Shape: [..., input, output] -> [..., w_digit, input, output]
        digits = self._w_transcoder.encode(w.long(), dim=-3)
        # Shape: [..., w_digit, input, output] -> [..., col, row]
        state_idx = self._append_disabled_rsm(digits, dim=-3).flatten(-3, -2)
        # Shape: [..., col, row] -> [..., scan=1, col, row]
        state_idx = state_idx.unsqueeze(-3)
        self.array.program(state_idx)

    def _vec_mat_mul_impl(
        self,
        x: Tensor,
        *,
        quantization_mode: int,
        adc_active_bits: int | None,
    ) -> Tensor:
        adc_active_bits = self._resolve_adc_active_bits(adc_active_bits)
        config = self.config
        leading_shape = x.shape[:-1]
        access__ns = config.t_settle__ns + self.rscsa.latency__ns(active_bits=adc_active_bits)

        # --- 1: drive each logical input across the physical columns ---

        # Shape: [..., input] -> [..., w_digit, input]
        x_code = x.unsqueeze(-2).expand(*leading_shape, self._w_digit_num, self.input_num)
        # Shape: [..., w_digit, input] -> [..., col]
        bl_code = self._append_disabled_rsm(x_code, dim=-2).flatten(-2)
        # Shape: [..., col]
        v_bl__V = bl_code * self._v_bl__V

        # --- 2: append the serialized row-scan axis ---

        # Shape: [scan, row] -> [..., scan, row]
        v_wl__V = self._v_wl_scan__V.expand(*leading_shape, self.scan_num, self.row_num)

        # --- 3: solve every scan phase ---

        port_shape = (*leading_shape, self.scan_num, self.col_num)
        bl_snap = self.bl_driver.snapshot(v_ref__V=v_bl__V.unsqueeze(-2), shape=port_shape)
        sl_snap = self.sl_driver.snapshot(v_ref__V=self._v_sl__V, shape=port_shape)
        steady_state = self.array.solve_array(
            v_wl__V=v_wl__V,
            wl_phase_dims=(-2,),
            bl_driver=self.bl_driver,
            bl_driver_snap=bl_snap,
            sl_driver=self.sl_driver,
            sl_driver_snap=sl_snap,
        )

        # Complete both boundary accesses at the converged port-current layout.
        self.bl_driver.drive(i_port__uA=steady_state.i_bl_port__uA)
        self.sl_driver.drive(i_port__uA=steady_state.i_sl_port__uA)

        # Shape: [..., scan, row] -> [..., scan, lane]
        i_tbl__uA = steady_state.i_tbl_by_row__uA[..., self._scan_indices, self._tbl_row_indices]
        # Shape: [..., scan, lane] -> [..., lane, scan]
        i_tbl__uA = i_tbl__uA.movedim(-1, -2)

        # --- 4: array conduction ---

        if self._is_dynamic_energy_profile_active():
            self._record_dynamic_energy(
                steady_state.v_bl_clamp__V * steady_state.i_bl_port__uA * access__ns,
                channel="bl_conduction",
            )
            self._record_dynamic_energy(
                config.v_tbl__V * i_tbl__uA * access__ns,
                channel="tbl_conduction",
            )

        # --- 5: RS-CSA quantize against its single reference current ---

        # Shape: [..., mode] -> [...]
        i_refs__uA = self.rscsa_reference.values()[..., quantization_mode]
        # Shape: [...] -> [..., lane=1, scan=1, tap=1]
        i_refs_shape = (*i_refs__uA.shape, 1, 1, 1)
        i_refs__uA = i_refs__uA.view(i_refs_shape)
        # Shape: [..., lane, scan]
        i_signal__uA = i_tbl__uA - self.array.i_tbl_leak__uA
        # Shape: [..., lane, scan]
        code = self.rscsa.convert(i_signal__uA, i_refs__uA, active_bits=adc_active_bits)

        # --- 6: unmodeled peripheral energy (per output access) ---

        self.mux_driver.execute(code.shape)
        self.timing_ctrl.execute(code.shape)

        return code
