"""Ye 2023 JSSC WH-2T1R RRAM CIM macro composing the array, readout, and seats.

Exposes an unsigned logical VMM over ``row_num`` inputs and ``col_num`` outputs
on a transposed, digit-folded physical array read by one time-shared RS-CSA.
The macro is the scheme's sole latency emitter and bills every energy branch its
children do not.

See also:
    docs/works/macro/cim/ye2023jssc/model.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.primitive.analog import (
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
    CimMacroMode,
    CimMacroPolicy,
    decimate_references,
    map_zero_point_input_code,
)

from .array import Ye2023Jssc2t1rArray, Ye2023Jssc2t1rArrayConfig, Ye2023Jssc2t1rArrayPolicy
from .cell import Ye2023Jssc2t1rCellConfig
from .rscsa import RsCsaIadc, RsCsaIadcConfig, RsCsaIadcPolicy

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class Ye2023JsscCimMacroConfig(CimMacroConfig):
    """Configuration for the Ye2023 JSSC WH-2T1R CIM macro.

    Attributes:
        max_active_num: Simultaneously selected inputs; must equal ``row_num``.
        array_config: Nested WH-2T1R array config.
        adc_config: RS-CSA current-ADC config; its ``bits`` is the macro's
            ``adc_max_bits`` and its phase durations set
            :attr:`~Ye2023JsscCimMacro.t_ac__ns`.
        bl_driver_config: Per-column BL input clamp (Thevenin VoltageDriver).
        sl_driver_config: SL grounded clamp (VoltageDriver).
        mux_driver_config: Static-PPA seat for the Mux & Driver block.
        timing_ctrl_config: Static-PPA seat for the Timing & Mode Ctrl block.
        v_wl_sel__V: Selected word-line drive [V]; > 0.
        v_bl_in1__V: BL voltage for input bit 1 [V]; > 0 and equal to
            ``array_config.v_bl_in1__V``.
        v_tbl__V: Transpose-bitline clamp voltage [V]; an operating-point datum.
        v_sl__V: Source-line drive [V].
        v_dd_core__V: Core supply rail [V]; must equal ``adc_config.v_rail__V``.
        e_mux_driver_per_op__fJ: Mux & Driver energy [fJ] per output access.
        e_timing_ctrl_per_op__fJ: Timing & Ctrl energy [fJ] per output access.
        modes: Quantization operating points, one per ``quantization_mode``
            index; at least one. Each carries the canonical MAC-unit window the
            mode converts, the ADC input code range its converter
            discriminates, and the rescale factor of a code at
            ``adc_config.bits``.
    """

    # --- Device-bearing sub-blocks ---
    array_config: Ye2023Jssc2t1rArrayConfig
    adc_config: RsCsaIadcConfig
    bl_driver_config: VoltageDriverConfig
    sl_driver_config: VoltageDriverConfig

    # --- Flat peripheral seats (static PPA only) ---
    mux_driver_config: UnmodeledBlockConfig
    timing_ctrl_config: UnmodeledBlockConfig

    # --- Biases ---
    v_wl_sel__V: float
    v_bl_in1__V: float
    v_tbl__V: float
    v_sl__V: float
    v_dd_core__V: float

    # --- Flat peripheral per-op energies ---
    e_mux_driver_per_op__fJ: float
    e_timing_ctrl_per_op__fJ: float

    # --- Quantization operating points ---
    modes: tuple[CimMacroMode, ...]

    @property
    def w_digit_num(self) -> int:
        """Weight bit-plane count — ``len(array_config.weight_radix)``."""
        return len(self.array_config.weight_radix)

    @property
    def cell_config(self) -> Ye2023Jssc2t1rCellConfig:
        """The nested WH-2T1R cell config; the array config declares the family base."""
        cell_config = self.array_config.cell_config
        if not isinstance(cell_config, Ye2023Jssc2t1rCellConfig):
            raise TypeError(
                f"require: array_config.cell_config a Ye2023Jssc2t1rCellConfig; got {type(cell_config).__name__}"
            )
        return cell_config

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def validate(self) -> None:
        super().validate()

        # --- Cross-block bias consistency ---

        if not (self.v_bl_in1__V == self.array_config.v_bl_in1__V):
            raise ValueError(
                f"require: v_bl_in1__V ({self.v_bl_in1__V}) == array_config.v_bl_in1__V "
                f"({self.array_config.v_bl_in1__V})"
            )
        if not (self.adc_config.v_rail__V == self.v_dd_core__V):
            raise ValueError(
                f"require: adc_config.v_rail__V ({self.adc_config.v_rail__V}) == v_dd_core__V ({self.v_dd_core__V})"
            )
        # The derived compensation current reads ONE input-0 floor entry, so
        # that floor must not depend on the programmed state.
        i_floor_row__uA = self.cell_config.i_t2_table__uA[0]
        if len(set(i_floor_row__uA)) != 1:
            raise ValueError(f"require: i_t2_table__uA[0] (input-0 floor) state-independent; got {i_floor_row__uA}")
        self._require_pos(self.v_wl_sel__V, "v_wl_sel__V")
        self._require_pos(self.v_bl_in1__V, "v_bl_in1__V")
        self._require_non_neg(self.v_tbl__V, "v_tbl__V")
        self._require_non_neg(self.v_sl__V, "v_sl__V")
        self._require_non_neg(self.v_dd_core__V, "v_dd_core__V")
        self._require_non_neg(self.e_mux_driver_per_op__fJ, "e_mux_driver_per_op__fJ")
        self._require_non_neg(self.e_timing_ctrl_per_op__fJ, "e_timing_ctrl_per_op__fJ")

        # --- Quantization modes ---

        if len(self.modes) == 0:
            raise ValueError("require: modes must declare at least one quantization operating point")


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class Ye2023JsscCimMacroPolicy(CimMacroPolicy):
    """Composite nonideality policy for :class:`Ye2023JsscCimMacro`.

    Attributes:
        array_policy: WH-2T1R array policy (cell policy + solver chunk knob).
        adc_policy: RS-CSA current-ADC policy.
        bl_driver_policy: BL clamp policy.
        sl_driver_policy: SL clamp policy.
        mux_driver_policy: Mux & Driver static-seat policy.
        timing_ctrl_policy: Timing & Ctrl static-seat policy.
    """

    array_policy: Ye2023Jssc2t1rArrayPolicy
    adc_policy: RsCsaIadcPolicy
    bl_driver_policy: VoltageDriverPolicy
    sl_driver_policy: VoltageDriverPolicy
    mux_driver_policy: UnmodeledBlockPolicy
    timing_ctrl_policy: UnmodeledBlockPolicy


# ---------------------------------------------------------------------------
# Macro
# ---------------------------------------------------------------------------


@CimMacro.register_neurox_module(
    config_type=Ye2023JsscCimMacroConfig,
    policy_type=Ye2023JsscCimMacroPolicy,
)
class Ye2023JsscCimMacro(CimMacro[Ye2023JsscCimMacroConfig, Ye2023JsscCimMacroPolicy]):
    """Ye2023 JSSC WH-2T1R CIM macro: transposed 2T1R array + time-shared RS-CSA.

    Owns the dedicated WH-2T1R array, the per-column BL clamp and the SL clamp,
    one RS-CSA, and two static-PPA peripheral seats. It derives the readout's
    static compensation current from the array's own tables at construction.

    Args:
        config: Macro configuration.
        policy: Macro nonideality policy.
        input_num: Logical input length, bound to ``row_num`` during construction.
        output_num: Logical output length, bound to ``col_num`` during construction.
        inst_shape: Per-instance multiplicity prefix.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    # --- Immutable model buffers ---

    _v_wl_onehot__V: Tensor
    _sl_v_ref__V: Tensor
    _i_ref_ladder__uA: Tensor
    _w_encode_lut: Tensor

    def __init__(
        self,
        *,
        config: Ye2023JsscCimMacroConfig,
        policy: Ye2023JsscCimMacroPolicy,
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
        if config.max_active_num != self.row_num:
            raise ValueError(
                f"require: max_active_num ({config.max_active_num}) == row_num ({self.row_num}) (input-parallel design)"
            )
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW
        self._init_children(dtype=dtype, T__K=T__K)
        self._register_model_buffers(dtype=dtype)

        # Total capacitance one BL column drives: its wire segments plus, at every
        # physical row, the cell BL node and the cell X node.
        array_config = config.array_config
        cell_config = array_config.cell_config
        self._c_bl_column__fF = (
            array_config.bl_first_c__fF
            + (self.col_num - 1) * array_config.bl_segment_c__fF
            + self.col_num * (cell_config.c_bl__fF + cell_config.c_x__fF)
        )

    def _init_children(self, *, dtype: torch.dtype, T__K: float) -> None:
        """Construct the array, readout, clamps, and flat peripheral seats."""
        config = self.config
        policy = self.policy
        plane_num = config.w_digit_num + len(config.array_config.redundant_radix)
        phys_col_num = self.row_num * plane_num

        # --- Dedicated WH-2T1R array (TRANSPOSED: rows = outputs, cols = in x plane) ---

        self.array = Ye2023Jssc2t1rArray(
            config=config.array_config,
            policy=policy.array_policy,
            inst_shape=self.inst_shape,
            row_num=self.col_num,  # physical rows = logical outputs
            col_num=phys_col_num,  # physical cols = row_num * (weight planes + redundant planes)
            dtype=dtype,
            T__K=T__K,
            enable_latency_record=False,  # macro owns latency
        )

        # --- Boundary clamps (ideal r_out = 0 sources; wire IR drop is the array's) ---

        self.bl_driver = VoltageDriver(
            config=config.bl_driver_config,
            policy=policy.bl_driver_policy,
            inst_shape=(*self.inst_shape, phys_col_num),
            dtype=dtype,
            T__K=T__K,
        )
        self.sl_driver = VoltageDriver(
            config=config.sl_driver_config,
            policy=policy.sl_driver_policy,
            inst_shape=self.inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

        # --- Single time-shared RS-CSA (the outputs ride the convert leading) ---

        # The compensation current is the array's all-off row current: the
        # off-cell floor over every place value one row sums.
        all_radix = (*config.array_config.weight_radix, *config.array_config.redundant_radix)
        i_ph0_comp__uA = config.cell_config.i_t2_table__uA[0][0] * self.row_num * sum(all_radix)
        self.rscsa = RsCsaIadc(
            config=config.adc_config,
            policy=policy.adc_policy,
            inst_shape=self.inst_shape,
            dtype=dtype,
            T__K=T__K,
            i_ph0_comp__uA=i_ph0_comp__uA,
            enable_latency_record=False,  # macro owns latency
        )

        # --- Flat peripheral seats (static PPA; dynamic billed by the macro) ---

        self.mux_driver = UnmodeledBlock(
            config=config.mux_driver_config,
            policy=policy.mux_driver_policy,
            inst_shape=self.inst_shape,
            dtype=dtype,
            T__K=T__K,
        )
        self.timing_ctrl = UnmodeledBlock(
            config=config.timing_ctrl_config,
            policy=policy.timing_ctrl_policy,
            inst_shape=self.inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

    def _register_model_buffers(self, *, dtype: torch.dtype) -> None:
        """Register the one-hot WL grid, the SL drive, and the RS-CSA ladder."""
        config = self.config
        # Shape: [col_num, col_num]
        wl_onehot = config.v_wl_sel__V * torch.eye(self.col_num, dtype=dtype)
        self.register_buffer("_v_wl_onehot__V", wl_onehot, persistent=False)
        self.register_buffer("_sl_v_ref__V", torch.tensor(config.v_sl__V, dtype=dtype), persistent=False)

        # Per-mode reference bank at the RS-CSA's max-bits resolution: row
        # ``quantization_mode`` is the ascending decision ladder c * i_lsb for
        # c = 1 .. 2**bits - 1. The readout has ONE physical current step, so
        # every mode shares it. Shape: [mode_num, 2**bits - 1]
        adc_config = config.adc_config
        n_tap = (1 << adc_config.bits) - 1
        ladder = torch.arange(1, n_tap + 1, dtype=dtype) * adc_config.i_lsb__uA
        self.register_buffer(
            "_i_ref_ladder__uA",
            ladder.expand(len(config.modes), n_tap).contiguous(),
            persistent=False,
        )

        digit_num = config.w_digit_num
        patterns = torch.arange(1 << digit_num, dtype=torch.long)
        digits = (patterns.unsqueeze(-1) >> torch.arange(digit_num)) & 1
        values = digits @ torch.tensor(config.array_config.weight_radix, dtype=torch.long)
        encode_lut = torch.zeros((sum(config.array_config.weight_radix) + 1, digit_num), dtype=torch.long)
        encode_lut[values] = digits
        self.register_buffer("_w_encode_lut", encode_lut, persistent=False)

    # -----------------------------------------------------------------
    # Value-domain semantics
    # -----------------------------------------------------------------

    @property
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive 1-bit activation range."""
        return (0, 1)

    @property
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive envelope of logical unsigned weights."""
        return (0, sum(self.config.array_config.weight_radix))

    @property
    def quantization_input_ranges(self) -> tuple[tuple[int, int], ...]:
        """Canonical conversion window per mode, in MAC units."""
        return tuple(mode.quantization_input_range for mode in self.config.modes)

    @property
    def adc_max_bits(self) -> int:
        """Maximum ADC resolution [bits] — the RS-CSA's ``bits``."""
        return self.config.adc_config.bits

    def _max_bits_rescale_factor(self, quantization_mode: int) -> float:
        """Return the configured rescale factor of one mode at :attr:`adc_max_bits`.

        Args:
            quantization_mode: Mode index in ``[0, len(config.modes))``.
        """
        return self.config.modes[self._check_mode(quantization_mode)].max_bits_rescale_factor

    def map_quantization_input_code(self, code: Tensor, *, quantization_mode: int) -> tuple[Tensor, tuple[int, int]]:
        """Map exact MAC-unit codes onto the readout's input grid.

        The row current the RS-CSA discriminates rises with the unsigned MAC
        from the mode's window bottom, so the grid is the zero-point one; it is
        the identity for an unsigned window.

        Args:
            code: Exact integer plane dots.
            quantization_mode: Mode index in ``[0, len(config.modes))``.

        Returns:
            The offset codes and the mode's declared ADC input code range.
        """
        mode = self.config.modes[self._check_mode(quantization_mode)]
        mapped, _ = map_zero_point_input_code(code, code_range=mode.quantization_input_range)
        return mapped, mode.adc_input_code_range

    def _check_mode(self, quantization_mode: int) -> int:
        """Return ``quantization_mode`` after bounding it against the declared modes."""
        mode_num = len(self.config.modes)
        if not (0 <= quantization_mode < mode_num):
            raise ValueError(f"require: quantization_mode ({quantization_mode}) in [0, {mode_num})")
        return quantization_mode

    # -----------------------------------------------------------------
    # Timing
    # -----------------------------------------------------------------

    @property
    def t_ac__ns(self) -> Tensor:
        """Access window T_AC [ns] (0-d) — the RS-CSA conversion window.

        This one window times every conduction branch and the per-access latency
        event.
        """
        return self.rscsa.t_conversion__ns

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def program(self, w: Tensor) -> None:
        """Encode logical weights and fold the planes into the physical grid.

        Args:
            w: Logical weight tensor with shape
                ``(*inst_shape, row_num, col_num)``; every entry must be
                representable by a binary selection over
                ``array_config.weight_radix``.
        """
        expected_shape = (*self.inst_shape, self.row_num, self.col_num)
        if tuple(w.shape) != expected_shape:
            raise ValueError(f"program() expects w.shape {expected_shape}; got {tuple(w.shape)}")

        # Shape: [*, row, col] -> [*, row, col, w_digit_num]
        digits = self._w_encode_lut[w.long()]
        # Shape: [*, row, col, w_digit_num] -> [*, w_digit_num, row, col]
        digits = digits.movedim(-1, -3)
        # Shape: [*, w_digit_num, row, col] -> [*, w_digit_num * row, col]
        w_state_idx = digits.flatten(-3, -2)
        # Redundant (non-weight) planes are programmed all-HRS (state 0).
        n_redundant_col = len(self.config.array_config.redundant_radix) * self.row_num
        redundant = w_state_idx.new_zeros((*w_state_idx.shape[:-2], n_redundant_col, w_state_idx.shape[-1]))
        # Shape: [*, w_digit_num * row + redundant * row, col] = [*, phys_col_num, col]
        w_state_idx = torch.cat((w_state_idx, redundant), dim=-2).contiguous()
        self.array.program(w_state_idx)

    def vec_mat_mul(self, x: Tensor, *, quantization_mode: int, adc_bits: int | None) -> Tensor:
        """Run one broadcast array solve over the output-serial word lines + RS-CSA.

        Args:
            x: 1-bit activation tensor with primitive trailing ``[row_num]`` and
                leading ``(*batch, *inst_shape)``; entries in
                :attr:`x_value_range`. The instance axes must be present when
                ``inst_shape`` is non-empty, and a size-1 instance axis shares one
                input vector across the whole die ensemble.
            quantization_mode: Mode index in ``[0, len(config.modes))``; selects
                the reference bank row.
            adc_bits: RS-CSA resolution [bits] in ``[1, adc_max_bits]``. Below
                the maximum the mode's ladder is decimated, which drops the
                code's low bits; the readout has no lossless oracle, so ``None``
                is rejected.

        Returns:
            Unsigned RS-CSA code tensor with leading ``(*batch, *inst_shape)`` and
            primitive trailing ``[col_num]``.
        """
        self._check_mode(quantization_mode)
        if adc_bits is None:
            raise ValueError("require: adc_bits is an int — the physical readout has no lossless oracle")
        if not (1 <= adc_bits <= self.adc_max_bits):
            raise ValueError(f"require: adc_bits ({adc_bits}) in [1, adc_max_bits ({self.adc_max_bits})]")
        config = self.config
        n_weight_plane = config.w_digit_num
        n_redundant_plane = len(config.array_config.redundant_radix)
        row_num = self.row_num
        v_dd_core = config.v_dd_core__V
        record = self._is_dynamic_energy_profile_active()
        x_long = x.long()

        n_inst = len(self.inst_shape)
        if x_long.ndim - 1 < n_inst:
            raise ValueError(
                f"vec_mat_mul() expects x leading (*batch, *inst_shape) with inst_shape {self.inst_shape}; "
                f"got x.shape {tuple(x.shape)}"
            )

        # --- Step 1: per-column BL input voltages, tiled plane-major ---

        # [*B, row] -> [*B, weight_plane, row] -> [*B, weight_plane * row]; the
        # first row_num carry plane 0, matching the array's plane-major radix
        # layout. The redundant planes are forced input-0.
        leading = x_long.shape[:-1]
        x_weight = (
            x_long.unsqueeze(-2).expand(*leading, n_weight_plane, row_num).reshape(*leading, n_weight_plane * row_num)
        )
        x_redundant = x_weight.new_zeros((*leading, n_redundant_plane * row_num))
        x_tiled = torch.cat((x_weight, x_redundant), dim=-1)
        v_bl = torch.where(
            x_tiled > 0,
            self._v_wl_onehot__V.new_tensor(config.v_bl_in1__V),
            self._v_wl_onehot__V.new_tensor(0.0),
        )

        # --- Step 2: stack the output-serial one-hot word lines on the leading ---

        # The array carries its instance prefix on the LAST leading axes, so the
        # output-serial axis is inserted BEFORE the instance axes x's leading ends
        # with: the solve leading is (*batch, out, *inst).
        batch = leading[: len(leading) - n_inst]
        inst_slots = leading[len(leading) - n_inst :]
        solve_leading = (*batch, self.col_num, *inst_slots)
        # [*batch, out, *inst, array_row]: output o activates array row o at V_WL_sel.
        v_wl = self._v_wl_onehot__V.view(self.col_num, *(1,) * n_inst, self.col_num).expand(
            *solve_leading, self.col_num
        )
        # [*batch, out, *inst, phys_col]: the same per-column inputs for every output.
        bl_v_ref = v_bl.unsqueeze(-(n_inst + 2)).expand(*solve_leading, x_tiled.shape[-1])

        # --- Step 3: one broadcast solve; leading becomes (*batch, out, *inst) ---

        steady = self.array.solve(
            v_wl,
            bl_driver=self.bl_driver,
            bl_v_ref__V=bl_v_ref,
            sl_driver=self.sl_driver,
            sl_v_ref__V=self._sl_v_ref__V,
        )
        i_bl_port = steady.i_bl_port__uA  # [*batch, out, *inst, phys_col]
        i_tbl = steady.i_tbl__uA  # [*batch, out, *inst], RAW (includes the IN=0 floor)

        # --- Step 4: conduction branches + per-vector BL charge (macro-billed) ---

        if record:
            t_ac__ns = self.t_ac__ns
            # BL input branch, PER ACCESS.
            # [*batch, out, *inst, phys_col] -> [*batch]
            e_bl_cond = (config.v_bl_in1__V * i_bl_port).sum(dim=tuple(range(-(n_inst + 2), 0))) * t_ac__ns
            self._record_dynamic_energy(e_bl_cond, channel="bl_cond")
            # DL branch, PER ACCESS: the RAW row current, before the readout's
            # compensation subtraction, on the core rail.
            # [*batch, out, *inst] -> [*batch]
            e_dl_cond = (v_dd_core * i_tbl).sum(dim=tuple(range(-(n_inst + 1), 0))) * t_ac__ns
            self._record_dynamic_energy(e_dl_cond, channel="dl_cond")
            # BL column charge, PER VECTOR (full-cycle convention): one charge
            # event per input-high column per call, not per access. Every die
            # charges its own columns, so a shared (size-1) input vector still
            # bills once per instance.
            # [*batch, *inst, phys_col] -> [*batch, *inst]
            high_col_count = (x_tiled > 0).sum(dim=-1).to(i_tbl.dtype)
            e_bl_cap = (config.v_bl_in1__V**2 * self._c_bl_column__fF) * high_col_count
            self._record_dynamic_energy(
                e_bl_cap.expand(torch.broadcast_shapes(e_bl_cap.shape, self.inst_shape)),
                channel="bl_cap",
            )

        # --- Step 5: RS-CSA quantize against the mode's uniform i_lsb ladder ---

        i_refs__uA = decimate_references(
            self._i_ref_ladder__uA[quantization_mode],
            adc_max_bits=self.adc_max_bits,
            adc_bits=adc_bits,
        )
        code = self.rscsa.convert(i_tbl, i_refs__uA, bits=adc_bits)  # [*batch, out, *inst]

        # --- Step 6: flat peripheral energy (per output access) + latency ---

        if record:
            e_mux = i_tbl.new_full(i_tbl.shape, config.e_mux_driver_per_op__fJ)
            self._record_dynamic_energy(e_mux, channel="mux_driver")
            e_ctrl = i_tbl.new_full(i_tbl.shape, config.e_timing_ctrl_per_op__fJ)
            self._record_dynamic_energy(e_ctrl, channel="timing_ctrl")

        # The single time-shared RS-CSA serializes every output over the leading;
        # each round takes one access window. The instances are parallel dies, so
        # their accesses share rounds instead of adding them.
        parallel_instance_count = self.inst_count
        serial_round_count = (code.numel() + parallel_instance_count - 1) // parallel_instance_count
        self._record_latency(self.t_ac__ns * serial_round_count)

        # [*batch, out, *inst] -> [*batch, *inst, out]
        return code.movedim(-(n_inst + 1), -1)
