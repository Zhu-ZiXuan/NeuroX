"""Ye 2023 JSSC WH-2T1R RRAM CIM macro composing the array, readout, and seats.

Exposes an unsigned logical VMM over ``row_num`` inputs and ``col_num`` outputs
on a transposed, digit-folded physical array read by one time-shared RS-CSA.
The macro bills every energy branch its children do not.

See also:
    docs/works/macro/cim/ye2023jssc/model.md
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
    """Configuration for the Ye2023 JSSC WH-2T1R CIM macro.

    Attributes:
        max_active_num: Simultaneously selected inputs; must equal ``row_num``.
        array_config: Nested WH-2T1R array config.
        adc_config: RS-CSA current-ADC config; its ``bits`` is the macro's
            ``adc_max_bits`` and its phase durations set the access window every
            conduction branch rides.
        reference_config: Dedicated RS-CSA reference source — an
            :class:`~neurox.primitive.analog.Iref` holding the ``[mode][tap]``
            bank the readout's single reference input reads. The RS-CSA scales
            that one current by its own compare-phase weights, so the bank is
            single-tap (``tap_num == 1``) and carries one row per declared mode:
            the macro NAMES the mode and reads that row from the source's
            fabricated bank.
        wl_dac_config: Word-line 1-bit ON/OFF DAC config; one converter seat per
            word line, whose ``code_to_signal`` states the selected and
            deselected WL levels and whose per-code energy is the drive event a
            scanned row costs.
        bl_dac_config: BL input 1-bit DAC config; one converter seat per physical
            column, whose ``code_to_signal`` states the IN = 1 and IN = 0 input
            levels and whose per-code energy is the drive event one held input
            vector costs.
        bl_driver_config: Per-column BL input clamp (Thevenin VoltageDriver).
        sl_driver_config: Per-column SL grounded clamp (VoltageDriver).
        mux_driver_config: Static-PPA seat for the Mux & Driver block.
        timing_ctrl_config: Static-PPA seat for the Timing & Mode Ctrl block.
        i_ph0_comp__uA: PH0 compensation current [uA] the readout subtracts once
            per conversion — a calibration product, measured as the array's
            all-off row leakage; non-negative.
        v_tbl__V: Transpose-bitline clamp voltage [V]; an operating-point datum.
        v_sl__V: Source-line drive [V].
        v_dd_core__V: Core supply rail [V]; must equal ``adc_config.v_rail__V``.
            The readout's own rail: the DL conduction branch is billed across
            it. It is NOT a driver rail — the two below are.
        v_dd_bl__V: Bit-line driver rail [V], handed to the array as the supply
            behind every conduction-path node it bills (BL, X, SL) and carrying
            the macro's own BL input conduction branch. A separate variable from
            :attr:`v_dd_core__V` even when numerically equal: what a charge is
            drawn FROM is the driver's supply, not the block the current ends
            up in.
        v_dd_wl__V: Word-line driver rail [V] handed to the array — the supply
            behind every WL node it bills. Separate
            from :attr:`v_dd_bl__V` by the same rule: the word line is its own
            supply domain, driven rail-to-rail while the read path hangs off
            the bit-line side.
        e_mux_driver_per_op__fJ: Mux & Driver energy [fJ] per output access.
        e_timing_ctrl_per_op__fJ: Timing & Ctrl energy [fJ] per output access.
        modes: Quantization operating points, one per ``quantization_mode``
            index; at least one. Each carries the canonical MAC-unit window the
            mode converts, the ADC input code range its converter
            discriminates, and the rescale factor of a code at
            ``adc_config.bits``.
    """

    # === Device-bearing sub-blocks ===

    array_config: Ye2023Jssc2t1rArrayConfig
    adc_config: RsCsaIadcConfig
    reference_config: IrefConfig
    wl_dac_config: VdacConfig
    bl_dac_config: VdacConfig
    bl_driver_config: VoltageDriverConfig
    sl_driver_config: VoltageDriverConfig

    # === Flat peripheral seats (static PPA only) ===

    mux_driver_config: UnmodeledBlockConfig
    timing_ctrl_config: UnmodeledBlockConfig

    # === Readout calibration ===

    i_ph0_comp__uA: float

    # === Biases ===

    v_tbl__V: float
    v_sl__V: float

    # === Supply rails (separate variables even when numerically equal) ===

    v_dd_core__V: float
    v_dd_bl__V: float
    v_dd_wl__V: float

    # === Flat peripheral per-op energies ===

    e_mux_driver_per_op__fJ: float
    e_timing_ctrl_per_op__fJ: float

    # === Quantization operating points ===

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

    def validate(self) -> None:
        super().validate()

        # --- Cross-block bias consistency ---

        if not (self.adc_config.v_rail__V == self.v_dd_core__V):
            raise ValueError(
                f"require: adc_config.v_rail__V ({self.adc_config.v_rail__V}) == v_dd_core__V ({self.v_dd_core__V})"
            )
        self._require_non_neg(self.i_ph0_comp__uA, "i_ph0_comp__uA")
        self._require_non_neg(self.v_tbl__V, "v_tbl__V")
        self._require_non_neg(self.v_sl__V, "v_sl__V")
        self._require_non_neg(self.v_dd_core__V, "v_dd_core__V")
        self._require_non_neg(self.v_dd_bl__V, "v_dd_bl__V")
        self._require_non_neg(self.v_dd_wl__V, "v_dd_wl__V")
        self._require_non_neg(self.e_mux_driver_per_op__fJ, "e_mux_driver_per_op__fJ")
        self._require_non_neg(self.e_timing_ctrl_per_op__fJ, "e_timing_ctrl_per_op__fJ")

        # --- Readout reference ---

        # The RS-CSA takes ONE reference current and derives its whole ladder
        # from it, so the source is single-tap; the row set is the mode set,
        # because the macro names a mode and the source returns that row.
        if self.reference_config.tap_num != 1:
            raise ValueError(
                f"require: reference_config.tap_num ({self.reference_config.tap_num}) == 1 "
                "— the RS-CSA takes a single reference current"
            )

        # --- Quantization modes ---

        if len(self.modes) == 0:
            raise ValueError("require: modes must declare at least one quantization operating point")
        if len(self.modes) != self.reference_config.mode_num:
            raise ValueError(
                f"require: len(modes) ({len(self.modes)}) == reference_config.mode_num "
                f"({self.reference_config.mode_num}) — one reference row per quantization mode"
            )


class Ye2023JsscCimMacroPolicy(CimMacroPolicy):
    """Composite nonideality policy for :class:`Ye2023JsscCimMacro`.

    Attributes:
        array_policy: WH-2T1R array policy (cell policy + solver chunk knob).
        adc_policy: RS-CSA current-ADC policy.
        reference_policy: RS-CSA reference-source policy.
        wl_dac_policy: Word-line DAC policy.
        bl_dac_policy: BL input DAC policy.
        bl_driver_policy: BL clamp policy.
        sl_driver_policy: SL clamp policy.
        mux_driver_policy: Mux & Driver static-seat policy.
        timing_ctrl_policy: Timing & Ctrl static-seat policy.
    """

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
        config: Macro configuration.
        policy: Macro nonideality policy.
        input_num: Logical input length, bound to ``row_num`` during construction.
        output_num: Logical output length, bound to ``col_num`` during construction.
        inst_shape: Per-instance multiplicity prefix.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
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

        The single readout is time-shared, so ``col_num`` outputs run
        sequentially on it; the activation is 1-bit, so no input-bit axis
        multiplies them. One access is the readout's executed conversion
        window, which already spans the array solve the phases run over, so
        the macro multiplies its converter instead of summing its children.

        Raises:
            ValueError: ``adc_bits`` is ``None``.
        """
        if adc_bits is None:
            raise ValueError("require: adc_bits is an int — the physical readout has no lossless oracle")
        return self.col_num * self.rscsa.latency__ns(bits=adc_bits)

    def _init_children(self, *, dtype: torch.dtype, T__K: float) -> None:
        """Construct the array, readout, clamps, and flat peripheral seats."""
        config = self.config
        policy = self.policy
        plane_num = config.w_digit_num + len(config.array_config.redundant_radix)
        phys_col_num = self.row_num * plane_num

        # --- Dedicated WH-2T1R array (TRANSPOSED: rows = outputs, cols = in x plane) ---

        # The BL boundary holds the input pattern while the word lines are
        # scanned, which the array fixes for itself. The two rails are declared
        # once at this macro's top level and cascade into the array's capacitive
        # billing from here.
        self.array = Ye2023Jssc2t1rArray(
            config=config.array_config,
            policy=policy.array_policy,
            inst_shape=self.inst_shape,
            row_num=self.col_num,  # physical rows = logical outputs
            col_num=phys_col_num,  # physical cols = row_num * (weight planes + redundant planes)
            v_dd_wl__V=config.v_dd_wl__V,
            v_dd_bl__V=config.v_dd_bl__V,
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
        """Register the one-hot WL code grid, the SL drive, and the weight-encode LUT."""
        config = self.config
        # The scan pattern is DIGITAL: output o raises word line o and holds every
        # other line at its off code, and the WL converter turns that into levels.
        self.register_buffer("_wl_onehot_code", torch.eye(self.col_num, dtype=torch.long), persistent=False)
        self.register_buffer("_sl_v_ref__V", torch.tensor(config.v_sl__V, dtype=dtype), persistent=False)

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

    @property
    def t_ac__ns(self) -> Tensor:
        """Nominal access window T_AC [ns] at :attr:`adc_max_bits`.

        The RS-CSA runs the compensation phase plus one compare phase per
        requested bit, so the window an access actually holds is the executed one
        of its ``adc_bits``; this property reports the operating point where the
        readout runs its whole phase set.
        """
        return self.rscsa.t_conversion__ns(self.adc_max_bits)

    def _organize_w(self, w: Tensor) -> Tensor:
        """Map logical weights into the array's physical-column plane layout.

        Args:
            w: Logical weight tensor.
                Shape: ``[*inst_shape, row_num, col_num]``.

        Returns:
            State indices.
            Shape: ``[*inst_shape, phys_col_num, col_num]``.
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
            w: Logical weight tensor; every entry must be representable by a
                binary selection over ``array_config.weight_radix``.
                Shape: ``[*inst_shape, row_num, col_num]``.
        """
        expected_shape = (*self.inst_shape, self.row_num, self.col_num)
        if tuple(w.shape) != expected_shape:
            raise ValueError(f"program() expects w.shape {expected_shape}; got {tuple(w.shape)}")
        self.array.program(self._organize_w(w))

    def vec_mat_mul(self, x: Tensor, *, quantization_mode: int, adc_bits: int | None) -> Tensor:
        """Run one broadcast array solve over the output-serial word lines + RS-CSA.

        Args:
            x: 1-bit activation tensor; entries in :attr:`x_value_range`. The
                instance axes must be present when ``inst_shape`` is non-empty,
                and a size-1 instance axis shares one input vector across the
                whole die ensemble.
                Shape: ``[..., *inst_shape, row_num]``.
            quantization_mode: Mode index in ``[0, len(config.modes))``; names
                the reference row the source selects.
            adc_bits: RS-CSA resolution [bits] in ``[1, adc_max_bits]``. The
                whole ladder is always wired; below the maximum the readout drops
                the code's low bits internally and runs fewer compare phases, so
                the access window shortens with it. The readout has no lossless
                oracle, so ``None`` is rejected.

        Returns:
            Unsigned RS-CSA code tensor.
            Shape: ``[..., *inst_shape, col_num]``.
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
        # The EXECUTED access window: the readout holds its DC biases for the
        # phases this resolution runs, so every conduction branch and the
        # per-access latency ride it.
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
        v_bl = self.bl_dac.convert((x_tiled > 0).long().expand(*batch, *inst, phys_col_num))

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
        v_wl_lines = self.wl_dac.convert(wl_code)
        # The array reads one value per cell GATE, and one word line spans every
        # column of its row, which the stride-0 expand over the column axis
        # states — it is also this macro's declaration that a scanned row carries
        # no per-column structure.
        # Shape: [..., out, *inst_shape, phys_col, array_row]
        v_wl = v_wl_lines.unsqueeze(-2).expand(*solve_leading, phys_col_num, self.col_num)
        # The same per-column inputs for every output.
        # Shape: [..., out, *inst_shape, phys_col]
        bl_v_ref = v_bl.unsqueeze(-(n_inst + 2)).expand(*solve_leading, phys_col_num)

        # --- 3: one broadcast solve; leading becomes (..., out, *inst_shape) ---

        # The macro owns the event structure, so the macro snapshots: one draw
        # per (output access, instance, column), the output axis sitting ahead
        # of each clamp bank's own (*inst_shape, phys_col) block, which is where
        # a right-aligned bank expects a time axis. Both snaps arrive at the
        # canonical [..., col] shape, so the array normalizes nothing.
        # Shape: [..., out, *inst_shape, phys_col]
        event_shape = (*solve_leading, phys_col_num)
        bl_snap = self.bl_driver.snapshot(v_ref__V=bl_v_ref, shape=event_shape)
        sl_snap = self.sl_driver.snapshot(v_ref__V=self._sl_v_ref__V.expand(event_shape), shape=event_shape)
        steady = self.array.solve_array(
            v_wl,
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
            # BL input branch, PER ACCESS, on the BL driver's SUPPLY: a branch
            # bill states what rail the charge leaves, never the level the node
            # it feeds sits at. An input-0 column carries no port current and
            # self-zeroes.
            # Shape: [..., out, *inst_shape, phys_col]
            e_bl_cond = (config.v_dd_bl__V * i_bl_port) * t_ac__ns
            self._record_dynamic_energy(e_bl_cond, channel="bl_cond")
            # DL branch, PER ACCESS: the RAW row current, before the readout's
            # compensation subtraction, on the core rail.
            # Shape: [..., out, *inst_shape]
            e_dl_cond = (v_dd_core * i_tbl) * t_ac__ns
            self._record_dynamic_energy(e_dl_cond, channel="dl_cond")

        # --- 5: RS-CSA quantize against its single reference current ---

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
