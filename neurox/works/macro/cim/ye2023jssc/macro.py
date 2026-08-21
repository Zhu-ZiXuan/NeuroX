"""Ye 2023 JSSC WH-2T1R RRAM CIM macro composing the array, readout, and seats.

Exposes an unsigned logical VMM over `row_num` inputs and `col_num` outputs on a
transposed, digit-folded physical array read by one time-shared RS-CSA.
The macro bills every energy branch its children do not.

See Also:
    docs/reference/primitive/macro/cim/family.md
"""

from __future__ import annotations

import torch
from torch import Tensor

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
)
from neurox.primitive.analog.voltage_dac import Vdac, VdacConfig, VdacPolicy
from neurox.primitive.macro.cim import (
    CimMacro,
    CimMacroConfig,
    CimMacroMode,
    CimMacroPolicy,
    map_zero_point_input_code,
)

from .array import Ye2023Jssc2t1rArray, Ye2023Jssc2t1rArrayConfig, Ye2023Jssc2t1rArrayPolicy
from .cell import Ye2023Jssc2t1rCellConfig
from .rscsa import RsCsaIadc, RsCsaIadcConfig, RsCsaIadcPolicy


class Ye2023JsscCimMacroConfig(CimMacroConfig):
    # === Device-bearing sub-blocks ===

    array_config: Ye2023Jssc2t1rArrayConfig
    adc_config: RsCsaIadcConfig
    """Its `bits` is the macro's `adc_max_bits`, and its phase durations set the access
    window every conduction branch rides."""
    reference_config: IrefConfig
    """Dedicated RS-CSA reference source. The readout scales one current by its own
    compare-phase weights, so the bank is single-tap and carries one row per declared
    mode: the macro NAMES the mode and reads that row from the fabricated bank."""
    wl_dac_config: VdacConfig
    """One 1-bit ON/OFF converter seat per word line; its `code_to_signal` states the
    selected and deselected WL levels and its per-code energy is the drive event a
    scanned row costs."""
    bl_dac_config: VdacConfig
    """One 1-bit converter seat per physical column; its `code_to_signal` states the
    IN = 1 and IN = 0 input levels and its per-code energy is the drive event one held
    input vector costs."""
    bl_driver_config: VoltageDriverConfig
    """Per-column BL input clamp, a Thevenin source."""
    sl_driver_config: VoltageDriverConfig
    """Per-column SL grounded clamp."""

    # === Flat peripheral seats (static PPA only) ===

    mux_driver_config: UnmodeledBlockConfig
    timing_ctrl_config: UnmodeledBlockConfig

    # === Readout calibration ===

    i_ph0_comp__uA: float
    """PH0 compensation current the readout subtracts once per conversion — a calibration
    product, measured as the array's all-off row leakage; non-negative.

    The compensation is static by design: the circuit carries no replica, dummy, or
    tracking branch, so the activity-dependent part of the row leakage it over-subtracts
    is a prediction of this model rather than a defect in it.
    """

    # === Biases and supply ===

    v_tbl__V: float
    """Transpose-bitline clamp voltage and per-access node excursion."""
    v_sl__V: float
    vdd__V: float
    """Core analog supply behind array-node charging, BL/TBL conduction, and
    the readout; it must equal `adc_config.v_rail__V`."""

    # === Flat peripheral per-op energies ===

    e_mux_driver_per_op__fJ: float
    """Billed once per output access."""
    e_timing_ctrl_per_op__fJ: float
    """Billed once per output access."""

    # === Quantization operating points ===

    modes: tuple[CimMacroMode, ...]
    """Quantization operating points indexed by `quantization_mode`; at least one, and
    one per reference row."""

    @property
    def w_digit_num(self) -> int:
        """Weight bit-plane count — one per weight-radix place value."""
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

    def validate(self) -> None:
        super().validate()

        # --- Cross-block bias consistency ---

        if not (self.adc_config.v_rail__V == self.vdd__V):
            raise ValueError(f"require: adc_config.v_rail__V ({self.adc_config.v_rail__V}) == vdd__V ({self.vdd__V})")
        self._require_non_neg(self.i_ph0_comp__uA, "i_ph0_comp__uA")
        self._require_non_neg(self.v_tbl__V, "v_tbl__V")
        self._require_non_neg(self.v_sl__V, "v_sl__V")
        self._require_non_neg(self.vdd__V, "vdd__V")
        self._require_non_neg(self.e_mux_driver_per_op__fJ, "e_mux_driver_per_op__fJ")
        self._require_non_neg(self.e_timing_ctrl_per_op__fJ, "e_timing_ctrl_per_op__fJ")

        # --- Readout reference ---

        # The RS-CSA takes ONE reference current and derives its whole ladder
        # from it, so the source is single-tap; the row set is the mode set,
        # because the macro names a mode and the source returns that row.
        self._require_len(self.reference_config.i_refs__uA[0], "reference_config.i_refs__uA[0]", 1)

        # --- Quantization modes ---

        self._require_non_empty(self.modes, "modes")
        self._require_same_len(
            self.modes,
            "modes",
            self.reference_config.i_refs__uA,
            "reference_config.i_refs__uA",
        )


class Ye2023JsscCimMacroPolicy(CimMacroPolicy):
    array_policy: Ye2023Jssc2t1rArrayPolicy
    adc_policy: RsCsaIadcPolicy
    reference_policy: IrefPolicy
    wl_dac_policy: VdacPolicy
    bl_dac_policy: VdacPolicy
    bl_driver_policy: VoltageDriverPolicy
    sl_driver_policy: VoltageDriverPolicy
    mux_driver_policy: UnmodeledBlockPolicy
    timing_ctrl_policy: UnmodeledBlockPolicy


@CimMacro.register_neurox_module(
    config_type=Ye2023JsscCimMacroConfig,
    policy_type=Ye2023JsscCimMacroPolicy,
)
class Ye2023JsscCimMacro(CimMacro[Ye2023JsscCimMacroConfig, Ye2023JsscCimMacroPolicy]):
    """Ye2023 JSSC WH-2T1R CIM macro: transposed 2T1R array + time-shared RS-CSA.

    Owns the dedicated WH-2T1R array, the word-line and BL input converter
    banks, the per-column BL and SL clamps, one RS-CSA with its reference
    source, and two static-PPA peripheral seats. It hands the readout its
    configured static compensation current at construction.

    Args:
        input_num: Logical input length, bound to `row_num` during construction.
        output_num: Logical output length, bound to `col_num` during construction.
    """

    # === Functional buffers ===

    _w_encode_lut: Tensor  # Shape: [w_value_num, w_digit_num]

    # === Circuit constant buffers ===

    _wl_onehot_code: Tensor  # Shape: [col_num, col_num]
    _sl_v_ref__V: Tensor  # Shape: []

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
        self._init_children(dtype=dtype, T__K=T__K)
        self._register_model_buffers(dtype=dtype)

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    def latency__ns(self, *, adc_bits: int | None) -> float:
        """One VMM — one RS-CSA access window per logical output.

        The single readout is time-shared, so `col_num` outputs run sequentially on
        it; the activation is 1-bit, so no input-bit axis multiplies them. One access
        is the readout's executed conversion window, which already spans the array
        solve the phases run over, so the macro multiplies its converter instead of
        summing its children.

        Raises:
            ValueError: `adc_bits` is `None`.
        """
        if adc_bits is None:
            raise ValueError("require: adc_bits is an int — the physical readout has no lossless oracle")
        return self.col_num * self.rscsa.latency__ns(bits=adc_bits)

    def _init_children(self, *, dtype: torch.dtype, T__K: float) -> None:
        config = self.config
        policy = self.policy
        plane_num = config.w_digit_num + len(config.array_config.redundant_radix)
        phys_col_num = self.row_num * plane_num

        # --- Dedicated WH-2T1R array (TRANSPOSED: rows = outputs, cols = in x plane) ---

        # The BL boundary holds the input pattern while the word lines are
        # scanned, which the array fixes for itself. The core supply is declared
        # once at this macro's top level and cascades into the array's capacitive
        # billing from here.
        self.array = Ye2023Jssc2t1rArray(
            config=config.array_config,
            policy=policy.array_policy,
            inst_shape=self.inst_shape,
            row_num=self.col_num,  # physical rows = logical outputs
            col_num=phys_col_num,  # physical cols = row_num * (weight planes + redundant planes)
            v_tbl__V=config.v_tbl__V,
            vdd__V=config.vdd__V,
            dtype=dtype,
            T__K=T__K,
        )

        # --- Input converters (the two 1-bit drives the array is scanned with) ---

        # One WL converter per word line: a line spans its whole row, so the seat
        # count is the row count and a scan drives every seat once per access.
        self.wl_dac = Vdac.from_config(
            config=config.wl_dac_config,
            policy=policy.wl_dac_policy,
            inst_shape=(*self.inst_shape, self.col_num),
            dtype=dtype,
            T__K=T__K,
        )
        # One BL converter per physical column, on the same column grid the
        # boundary clamps sit on.
        self.bl_dac = Vdac.from_config(
            config=config.bl_dac_config,
            policy=policy.bl_dac_policy,
            inst_shape=(*self.inst_shape, phys_col_num),
            dtype=dtype,
            T__K=T__K,
        )

        # --- Boundary clamps (ideal r_out = 0 sources; wire IR drop is the array's) ---

        # BOTH rails hold one clamp per physical column, each biased from its own
        # per-column reference, so the instance grid of either clamp is the
        # column grid. The macro hands them to the solve and drives them itself
        # at the converged port state.
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
            inst_shape=(*self.inst_shape, phys_col_num),
            dtype=dtype,
            T__K=T__K,
        )

        # --- Single time-shared RS-CSA (the outputs ride the convert leading) ---

        # The compensation current is a calibrated seat: the array's measured
        # all-off row current, which this macro only hands over.
        self.rscsa = RsCsaIadc(
            config=config.adc_config,
            policy=policy.adc_policy,
            inst_shape=self.inst_shape,
            dtype=dtype,
            T__K=T__K,
            i_ph0_comp__uA=config.i_ph0_comp__uA,
        )

        # --- The readout's single reference current source ---

        # One reference-generation circuit per fabricated die, feeding that die's
        # RS-CSA alone; the readout scales it into its own decision ladder.
        self.rscsa_reference = Iref(
            config=config.reference_config,
            policy=policy.reference_policy,
            inst_shape=self.inst_shape,
            dtype=dtype,
            T__K=T__K,
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
        config = self.config
        # The scan pattern is DIGITAL: output o raises word line o and holds every
        # other line at its off code, and the WL converter turns that into levels.
        self.register_buffer("_wl_onehot_code", torch.eye(self.col_num, dtype=torch.long), persistent=False)
        self.register_buffer("_sl_v_ref__V", torch.tensor(config.v_sl__V, dtype=dtype), persistent=False)

        # An unrepresentable logical value keeps this table's all-zero row: an
        # arbitrary radix list may leave holes inside `w_value_range`, and config
        # validation deliberately imposes no dense positional-radix pattern.
        digit_num = config.w_digit_num
        patterns = torch.arange(1 << digit_num, dtype=torch.long)
        digits = (patterns.unsqueeze(-1) >> torch.arange(digit_num)) & 1
        values = digits @ torch.tensor(config.array_config.weight_radix, dtype=torch.long)
        encode_lut = torch.zeros((sum(config.array_config.weight_radix) + 1, digit_num), dtype=torch.long)
        encode_lut[values] = digits
        self.register_buffer("_w_encode_lut", encode_lut, persistent=False)

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
        """Maximum ADC resolution [bits] — the RS-CSA's physical resolution."""
        return self.config.adc_config.bits

    def _max_bits_rescale_factor(self, quantization_mode: int) -> float:
        """Return the configured rescale factor of one mode at `adc_max_bits`.

        Args:
            quantization_mode: Mode index in `[0, len(config.modes))`.
        """
        return self.config.modes[self._check_mode(quantization_mode)].max_bits_rescale_factor

    def map_quantization_input_code(self, code: Tensor, *, quantization_mode: int) -> tuple[Tensor, tuple[int, int]]:
        """Map exact MAC-unit codes onto the readout's input grid.

        The row current the RS-CSA discriminates rises with the unsigned MAC
        from the mode's window bottom, so the grid is the zero-point one; it is
        the identity for an unsigned window.

        Args:
            code: Exact integer plane dots.
            quantization_mode: Mode index in `[0, len(config.modes))`.

        Returns:
            The offset codes and the mode's declared ADC input code range.
        """
        mode = self.config.modes[self._check_mode(quantization_mode)]
        mapped, _ = map_zero_point_input_code(code, code_range=mode.quantization_input_range)
        return mapped, mode.adc_input_code_range

    def _check_mode(self, quantization_mode: int) -> int:
        """Return `quantization_mode` after bounding it against the declared modes."""
        mode_num = len(self.config.modes)
        if not (0 <= quantization_mode < mode_num):
            raise ValueError(f"require: quantization_mode ({quantization_mode}) in [0, {mode_num})")
        return quantization_mode

    @property
    def t_ac__ns(self) -> Tensor:
        """Nominal access window T_AC [ns] at `adc_max_bits`.

        The RS-CSA runs the compensation phase plus one compare phase per requested
        bit, so the window an access actually holds is the executed one of its
        `adc_bits`; this property reports the operating point where the readout runs
        its whole phase set.
        """
        return self.rscsa.t_conversion__ns(self.adc_max_bits)

    def _organize_w(self, w: Tensor) -> Tensor:
        """Map logical weights into the array's physical-column plane layout.

        Args:
            w: Logical weight tensor.
                Shape: `[*inst_shape, row_num, col_num]`.

        Returns:
            State indices.
            Shape: `[*inst_shape, phys_col_num, col_num]`.
        """
        # Shape: [..., row, col] -> [..., row, col, w_digit_num]
        digits = self._w_encode_lut[w.long()]
        # Shape: [..., row, col, w_digit_num] -> [..., w_digit_num, row, col]
        digits = digits.movedim(-1, -3)
        # Shape: [..., w_digit_num, row, col] -> [..., w_digit_num * row, col]
        w_state_idx = digits.flatten(-3, -2)
        # Redundant (non-weight) planes are programmed all-HRS (state 0).
        n_redundant_col = len(self.config.array_config.redundant_radix) * self.row_num
        redundant = w_state_idx.new_zeros((*w_state_idx.shape[:-2], n_redundant_col, w_state_idx.shape[-1]))
        # phys_col_num = (w_digit_num + redundant_plane_num) * row.
        # Shape: [..., w_digit_num * row, col] -> [..., phys_col_num, col]
        return torch.cat((w_state_idx, redundant), dim=-2).contiguous()

    def program(self, w: Tensor) -> None:
        """Encode logical weights and fold the planes into the physical grid.

        Args:
            w: Logical weight tensor; every entry must be representable by a binary
                selection over the array's weight radix.
                Shape: `[*inst_shape, row_num, col_num]`.
        """
        expected_shape = (*self.inst_shape, self.row_num, self.col_num)
        if tuple(w.shape) != expected_shape:
            raise ValueError(f"program() expects w.shape {expected_shape}; got {tuple(w.shape)}")
        self.array.program(self._organize_w(w))

    def vec_mat_mul(self, x: Tensor, *, quantization_mode: int, adc_bits: int | None) -> Tensor:
        """Run one broadcast array solve over the output-serial word lines + RS-CSA.

        Args:
            x: 1-bit activation tensor; entries in `x_value_range`. The instance axes
                must be present when `inst_shape` is non-empty, and a size-1 instance
                axis shares one input vector across the whole die ensemble.
                Shape: `[..., *inst_shape, row_num]`.
            quantization_mode: Mode index in `[0, len(config.modes))`; names the
                reference row the source selects.
            adc_bits: RS-CSA resolution [bits] in `[1, adc_max_bits]`. The whole
                ladder is always wired; below the maximum the readout drops the code's
                low bits internally and runs fewer compare phases, so the access
                window shortens with it. The readout has no lossless oracle, so `None`
                is rejected.

        Returns:
            Unsigned RS-CSA code tensor.
            Shape: `[..., *inst_shape, col_num]`.
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
        vdd__V = config.vdd__V
        record = self._is_dynamic_energy_profile_active()
        x_long = x.long()
        # The EXECUTED access window: no sample-and-hold stands between the array
        # and the readout, so the DC biases are held for the phases this resolution
        # runs and every conduction branch and the per-access latency ride it.
        t_ac__ns = self.rscsa.t_conversion__ns(adc_bits)

        n_inst = len(self.inst_shape)
        if x_long.ndim - 1 < n_inst:
            raise ValueError(
                f"vec_mat_mul() expects x leading (..., *inst_shape) with inst_shape {self.inst_shape}; "
                f"got x.shape {tuple(x.shape)}"
            )

        # --- 1: per-column BL input voltages, tiled plane-major ---

        # The first row_num carry plane 0, matching the array's plane-major radix
        # layout. The redundant planes are forced input-0.
        leading = x_long.shape[:-1]
        # The array carries its instance prefix on the LAST leading axes, so the
        # ensemble width is resolved HERE: a size-1 instance slot shares one input
        # vector across the die ensemble, while every die still drives its own
        # columns and the solve still runs one instance per fabricated die.
        batch = leading[: len(leading) - n_inst]
        inst = tuple(torch.broadcast_shapes(self.inst_shape, leading[len(leading) - n_inst :]))
        # Shape: [..., row] -> [..., weight_plane * row]
        x_weight = (
            x_long.unsqueeze(-2).expand(*leading, n_weight_plane, row_num).reshape(*leading, n_weight_plane * row_num)
        )
        x_redundant = x_weight.new_zeros((*leading, n_redundant_plane * row_num))
        x_tiled = torch.cat((x_weight, x_redundant), dim=-1)
        phys_col_num = x_tiled.shape[-1]
        # The BL converter states the two input levels and bills its own drive:
        # ONE event per vector per column, since the level is held across the
        # whole row scan, and an input-0 column bills that code's own entry. The
        # codes arrive at the bank's own (*inst_shape, phys_col) seat layout, so a
        # shared input vector still pays once per die.
        # Shape: [..., *inst_shape, phys_col]
        v_bl__V = self.bl_dac.convert((x_tiled > 0).long().expand(*batch, *inst, phys_col_num))

        # --- 2: stack the output-serial one-hot word lines on the leading ---

        # The output-serial axis is inserted BEFORE the instance axes x's leading
        # ends with: the solve leading is (..., out, *inst_shape). The array
        # itself normalizes no shape.
        solve_leading = (*batch, self.col_num, *inst)
        # Output o raises word line o. The converter bank holds one seat per word
        # line, so its own instance block is the LINE axis and the scan sits ahead
        # of it, exactly where a right-aligned bank expects a time axis: the codes
        # arrive at the seat layout and the bank bills one drive event per line
        # per access, on every die and for every input vector.
        # Shape: [out, array_row] -> [..., out, *inst_shape, array_row]
        wl_code = self._wl_onehot_code.view(self.col_num, *(1,) * n_inst, self.col_num).expand(
            *solve_leading, self.col_num
        )
        # Shape: [..., out, *inst_shape, array_row]
        v_wl__V = self.wl_dac.convert(wl_code)
        # The same per-column inputs for every output.
        # Shape: [..., out, *inst_shape, phys_col]
        bl_v_ref__V = v_bl__V.unsqueeze(-(n_inst + 2)).expand(*solve_leading, phys_col_num)

        # --- 3: one broadcast solve; leading becomes (..., out, *inst_shape) ---

        # The macro owns the event structure, so the macro snapshots: one draw
        # per (output access, instance, column), the output axis sitting ahead
        # of each clamp bank's own (*inst_shape, phys_col) block, which is where
        # a right-aligned bank expects a time axis. Both snaps arrive at the
        # canonical [..., col] shape; the array distributes the line-level WL
        # drive over its own columns.
        # Shape: [..., out, *inst_shape, phys_col]
        event_shape = (*solve_leading, phys_col_num)
        bl_snap = self.bl_driver.snapshot(v_ref__V=bl_v_ref__V, shape=event_shape)
        sl_snap = self.sl_driver.snapshot(v_ref__V=self._sl_v_ref__V.expand(event_shape), shape=event_shape)
        steady = self.array.solve_array(
            v_wl__V,
            bl_driver=self.bl_driver,
            bl_driver_snap=bl_snap,
            sl_driver=self.sl_driver,
            sl_driver_snap=sl_snap,
        )
        # Shape: [..., out, *inst_shape, phys_col]
        i_bl_port = steady.i_bl_port__uA
        # The RAW row current, including the IN=0 floor.
        # Shape: [..., out, *inst_shape]
        i_tbl = steady.i_tbl__uA

        # Deliver both boundary clamps at the converged port state: each is one
        # instance per physical column, so the column axis IS its instance axis,
        # and the array's layout already seats the clamp instance block last.
        # Both conduct, so both are driven. The drive is billed per driven
        # position, which is why it happens here, on the layout the solve
        # returned and BEFORE the output axis moves past the instance axes.
        # Shape: [..., out, *inst_shape, phys_col]
        self.bl_driver.drive(i_bl_port, steady.v_bl_clamp__V)
        self.sl_driver.drive(steady.i_sl_port__uA, steady.v_sl_drive__V)

        # --- 4: conduction branches (macro-billed) ---

        # Every energy tensor below keeps the caller's leading dims and lets the
        # collector sum the macro's own output, instance and column axes past
        # them. No channel pre-reduces the caller block. The array's node ledger
        # is the sole account of the capacitance inside it, this column's own
        # included, so no capacitive channel is billed here.
        if record:
            # BL input branch, PER ACCESS, on the core supply: a branch
            # bill states what rail the charge leaves, never the level the node
            # it feeds sits at. An input-0 column carries no port current and
            # self-zeroes.
            # Shape: [..., out, *inst_shape, phys_col]
            e_bl_cond = (vdd__V * i_bl_port) * t_ac__ns
            self._record_dynamic_energy(e_bl_cond, channel="bl_cond")
            # DL branch, PER ACCESS: the RAW row current, before the readout's
            # compensation subtraction, on the same core supply.
            # Shape: [..., out, *inst_shape]
            e_dl_cond = (vdd__V * i_tbl) * t_ac__ns
            self._record_dynamic_energy(e_dl_cond, channel="dl_cond")

        # --- 5: RS-CSA quantize against its single reference current ---

        # The DL summing node ends the per-column layout: every readout bill
        # below is per access with no column axis — the collector sums past the
        # caller block, counting a column-shaped bill once per column.

        # The macro only NAMES the mode; the fabricated bank is a single static
        # identity per instance (fabricate-only, no per-call noise of its own),
        # broadcast by view to the full conversion shape — every conversion
        # serialized on this one physical source reads the same row.
        # Shape: [..., out, *inst_shape, 1]
        i_refs__uA = self.rscsa_reference.i_out__uA[..., quantization_mode, :].expand(
            *i_tbl.shape, self.rscsa_reference.tap_num
        )
        # The readout scales that one reference into its own ladder and handles
        # the requested bit width internally.
        # Shape: [..., out, *inst_shape]
        code = self.rscsa.convert(i_tbl, i_refs__uA, bits=adc_bits)

        # --- 6: flat peripheral energy (per output access) ---

        # A flat per-op lump over the accesses `code` carries — a constant, so the
        # expanded view holds no storage and no energy tensor is materialized, and the
        # energy dtype is the constant's rather than the integer code's; outside
        # a profiler the call is already a no-op, hence no `record` guard.
        # Shape: [] -> [..., out, *inst_shape]
        e_mux__fJ = torch.full((), config.e_mux_driver_per_op__fJ, dtype=torch.float32, device=code.device)
        e_timing__fJ = torch.full((), config.e_timing_ctrl_per_op__fJ, dtype=torch.float32, device=code.device)
        self._record_dynamic_energy(e_mux__fJ.expand(code.shape), channel="mux_driver")
        self._record_dynamic_energy(e_timing__fJ.expand(code.shape), channel="timing_ctrl")

        # Shape: [..., out, *inst_shape] -> [..., *inst_shape, out]
        return code.movedim(-(n_inst + 1), -1)
