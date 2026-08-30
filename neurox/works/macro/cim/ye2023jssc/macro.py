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

from .array import Ye2023Jssc2t1rArray, Ye2023Jssc2t1rArrayConfig, Ye2023Jssc2t1rArrayPolicy
from .cell import Ye2023Jssc2t1rCellConfig
from .rscsa import RsCsaIadc, RsCsaIadcConfig, RsCsaIadcPolicy


class Ye2023JsscCimMacroConfig(CimMacroConfig):
    # === Device-bearing sub-blocks ===

    array_config: Ye2023Jssc2t1rArrayConfig
    adc_config: RsCsaIadcConfig
    """Its `bits` is the macro's `adc_bits`, and its phase durations set the access
    window every conduction branch rides."""
    reference_config: ReferenceConfig
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

    # === Unmodeled peripheral blocks ===

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

        # --- Readout reference ---

        expected_shape = (len(self.rescale_factors),)
        if self.reference_config.shape != expected_shape:
            raise ValueError(f"require: reference_config.shape ({self.reference_config.shape}) == {expected_shape}")


class Ye2023JsscCimMacroPolicy(CimMacroPolicy):
    array_policy: Ye2023Jssc2t1rArrayPolicy
    adc_policy: RsCsaIadcPolicy
    reference_policy: ReferencePolicy
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
    source, and two lumped-PPA peripheral blocks. It hands the readout its
    configured static compensation current at construction.

    Args:
        input_num: Logical input length, bound to `row_num` during construction.
        output_num: Logical output length, bound to `col_num` during construction.
    """

    # === Functional buffers ===

    _wl_onehot_code: Tensor  # Shape: [col_num, col_num]
    _sl_v_ref__V: Tensor  # Shape: []
    _w_encode_lut: Tensor  # Shape: [w_value_num, w_digit_num]

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
        self._register_functional_buffers(dtype=dtype)

    def latency__ns(self, *, adc_active_bits: int) -> float:
        """One VMM — one RS-CSA access window per logical output.

        The single readout is time-shared, so `col_num` outputs run sequentially on
        it; the activation is 1-bit, so no input-bit axis multiplies them. One access
        is the readout's executed conversion window, which already spans the array
        solve the phases run over, so the macro multiplies its converter instead of
        summing its children.

        """
        return self.col_num * self.rscsa.latency__ns(active_bits=adc_active_bits)

    def initiation_interval__ns(self, *, adc_active_bits: int) -> float:
        """Scheduled duration of one VMM under back-to-back conversions."""
        return self.latency__ns(adc_active_bits=adc_active_bits)

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
            inst_shape=(*self.inst_shape, 1),
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
            inst_shape=(*self.inst_shape, 1, self.col_num),
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
            inst_shape=(*self.inst_shape, 1, phys_col_num),
            dtype=dtype,
            T__K=T__K,
        )
        self.sl_driver = VoltageDriver(
            config=config.sl_driver_config,
            policy=policy.sl_driver_policy,
            inst_shape=(*self.inst_shape, 1, phys_col_num),
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
            policy=policy.mux_driver_policy,
            inst_shape=(*self.inst_shape, 1),
            dtype=dtype,
            T__K=T__K,
        )
        self.timing_ctrl = UnmodeledBlock(
            config=config.timing_ctrl_config,
            policy=policy.timing_ctrl_policy,
            inst_shape=(*self.inst_shape, 1),
            dtype=dtype,
            T__K=T__K,
        )

    def _register_functional_buffers(self, *, dtype: torch.dtype) -> None:
        config = self.config
        # The scan pattern is DIGITAL: output o raises word line o and holds every
        # other line at its off code, and the WL converter turns that into levels.
        self._register_nonpersistent_buffer("_wl_onehot_code", torch.eye(self.col_num, dtype=torch.long))
        self._register_nonpersistent_buffer("_sl_v_ref__V", torch.tensor(config.v_sl__V, dtype=dtype))

        # An unrepresentable logical value keeps this table's all-zero row: an
        # arbitrary radix list may leave holes inside `w_value_range`, and config
        # validation deliberately imposes no dense positional-radix pattern.
        digit_num = config.w_digit_num
        patterns = torch.arange(1 << digit_num, dtype=torch.long)
        digits = (patterns.unsqueeze(-1) >> torch.arange(digit_num)) & 1
        values = digits @ torch.tensor(config.array_config.weight_radix, dtype=torch.long)
        encode_lut = torch.zeros((sum(config.array_config.weight_radix) + 1, digit_num), dtype=torch.long)
        encode_lut[values] = digits
        self._register_nonpersistent_buffer("_w_encode_lut", encode_lut)

    @property
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive 1-bit activation range."""
        return (0, 1)

    @property
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive envelope of logical unsigned weights."""
        return (0, sum(self.config.array_config.weight_radix))

    @property
    def adc_bits(self) -> int:
        """Maximum ADC resolution [bits] — the RS-CSA's physical resolution."""
        return self.config.adc_config.bits

    @property
    def _quantization_scheme(self) -> CimMacroQuantizationScheme:
        return CimMacroQuantizationScheme.ZERO_POINT

    def restore_adc_layout(self, value: Tensor) -> Tensor:
        """Return the already-logical output-serial layout."""
        return value

    @property
    def t_ac__ns(self) -> float:
        """Nominal access window T_AC [ns] at `adc_bits`.

        The RS-CSA runs the compensation phase plus one compare phase per requested
        bit, so the window an access actually holds is the executed one of its
        `adc_active_bits`; this property reports the operating point where the readout runs
        its whole phase set.
        """
        return self.rscsa.t_conversion__ns(self.adc_bits)

    def _organize_w(self, w: Tensor) -> Tensor:
        """Map logical weights into the array's physical-column plane layout.

        Args:
            w: Logical weight tensor.
                Shape: `[..., row_num, col_num]`.

        Returns:
            State indices.
            Shape: `[..., phys_col_num, col_num]`.
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
        # The array reserves its runtime output axis as a singleton instance
        # slot; programmed state keeps that same layout.
        # Shape: [..., phys_col_num, col_num] -> [..., out=1, phys_col_num, col_num]
        self.array.program(self._organize_w(w).unsqueeze(-3))

    def vec_mat_mul(self, x: Tensor, *, quantization_mode: int, adc_active_bits: int) -> Tensor:
        """Run one broadcast array solve over the output-serial word lines + RS-CSA.

        Args:
            x: 1-bit activation tensor; entries in `x_value_range`.
                Shape: `[..., row_num]`.
            quantization_mode: Index naming the reference row and its calibrated
                output scale.
            adc_active_bits: Active RS-CSA resolution in `[1, adc_bits]`. The whole
                ladder is always wired; below the maximum the readout drops the code's
                low bits internally and runs fewer compare phases, so the access
                window shortens with it.

        Returns:
            Unsigned RS-CSA code tensor retaining the inherited leading axes.
            Shape: `[..., col_num]`.
        """
        self._check_quantization_mode(quantization_mode)
        self._check_adc_active_bits(adc_active_bits)

        config = self.config
        n_weight_plane = config.w_digit_num
        n_redundant_plane = len(config.array_config.redundant_radix)
        row_num = self.row_num
        vdd__V = config.vdd__V
        record = self._is_dynamic_energy_profile_active()
        leading_shape = x.shape[:-1]
        x_long = x.long()
        # The EXECUTED access window: no sample-and-hold stands between the array
        # and the readout, so the DC biases are held for the phases this resolution
        # runs and every conduction branch and the per-access latency ride it.
        t_ac__ns = self.rscsa.t_conversion__ns(adc_active_bits)

        # --- 1: per-column BL input voltages, tiled plane-major ---

        # The first row_num carry plane 0, matching the array's plane-major radix
        # layout. The redundant planes are forced input-0.
        # Shape: [..., row] -> [..., weight_plane * row]
        x_weight = (
            x_long.unsqueeze(-2)
            .expand(*leading_shape, n_weight_plane, row_num)
            .reshape(*leading_shape, n_weight_plane * row_num)
        )
        x_redundant = x_weight.new_zeros((*leading_shape, n_redundant_plane * row_num))
        x_tiled = torch.cat((x_weight, x_redundant), dim=-1)
        phys_col_num = x_tiled.shape[-1]
        # The BL converter states the two input levels and bills its own drive:
        # ONE event per vector per column, since the level is held across the
        # whole row scan, and an input-0 column bills that code's own entry.
        # Shape: [..., phys_col]
        v_bl__V = self.bl_dac.convert((x_tiled > 0).long())

        # --- 2: append the output-serial one-hot word-line axis ---

        solve_leading = (*leading_shape, self.col_num)
        # Output o raises word line o. The converter bank holds one seat per word
        # line. The parent-aligned prefix stays opaque while the scan axis is
        # appended immediately before the line axis.
        # Shape: [out, array_row] -> [..., out, array_row]
        wl_code = self._wl_onehot_code.expand(*solve_leading, self.col_num)
        # Shape: [..., out, array_row]
        v_wl__V = self.wl_dac.convert(wl_code)
        # The same per-column inputs for every output.
        # Shape: [..., phys_col] -> [..., out=1, phys_col]
        bl_v_ref__V = v_bl__V.unsqueeze(-2)

        # --- 3: one solve over the inserted output-serial axis ---

        # The macro owns the event structure, so the macro snapshots: one draw
        # per output access and column. Both snaps arrive at the canonical
        # [..., out, phys_col] shape; the array distributes the line-level WL
        # drive over its own columns.
        # Shape: [..., out, phys_col]
        event_shape = (*solve_leading, phys_col_num)
        bl_snap = self.bl_driver.snapshot(v_ref__V=bl_v_ref__V, shape=event_shape)
        sl_snap = self.sl_driver.snapshot(v_ref__V=self._sl_v_ref__V, shape=event_shape)
        steady = self.array.solve_array(
            v_wl__V,
            bl_driver=self.bl_driver,
            bl_driver_snap=bl_snap,
            sl_driver=self.sl_driver,
            sl_driver_snap=sl_snap,
        )
        # Shape: [..., out, phys_col]
        i_bl_port = steady.i_bl_port__uA
        # The RAW row current, including the IN=0 floor.
        # Shape: [..., out]
        i_tbl = steady.i_tbl__uA

        # Complete both boundary accesses at the converged port-current layout.
        # Shape: [..., out, phys_col]
        self.bl_driver.drive(i_port__uA=i_bl_port)
        self.sl_driver.drive(i_port__uA=steady.i_sl_port__uA)

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
            # Shape: [..., out, phys_col]
            e_bl_cond = (vdd__V * i_bl_port) * t_ac__ns
            self._record_dynamic_energy(e_bl_cond, channel="bl_cond")
            # DL branch, PER ACCESS: the RAW row current, before the readout's
            # compensation subtraction, on the same core supply.
            # Shape: [..., out]
            e_dl_cond = (vdd__V * i_tbl) * t_ac__ns
            self._record_dynamic_energy(e_dl_cond, channel="dl_cond")

        # --- 5: RS-CSA quantize against its single reference current ---

        # The DL summing node ends the per-column layout: every readout bill
        # below is per access with no column axis — the collector sums past the
        # caller block, counting a column-shaped bill once per column.

        # Shape: [..., mode] -> [..., out=1, tap=1]
        i_refs__uA = self.rscsa_reference.values()[..., quantization_mode, None, None]
        # The readout scales that one reference into its own ladder and handles
        # the requested bit width internally.
        # Shape: [..., out]
        code = self.rscsa.convert(i_tbl, i_refs__uA, active_bits=adc_active_bits)

        # --- 6: unmodeled peripheral energy (per output access) ---

        self.mux_driver.execute(code.shape)
        self.timing_ctrl.execute(code.shape)

        # Shape: [..., out] -> [..., col_num]
        return self.restore_adc_layout(code)
