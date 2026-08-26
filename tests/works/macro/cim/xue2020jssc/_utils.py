"""Shared hand-built witness configs + in-code calibration for the xue2020jssc scheme tests.

Every solve-bearing test builds through `build_macro` from the
hand-constructed `build_config` witness — every config dataclass is built
directly in Python with small explicit values (no disk TOML). The witness ships
a NEAR-IDEAL analog chain so the integer MAC is analytic: a linearized 1T1R cell
with an exact-zero HRS branch (`g_cell_on_table__uS = (0.0, g_lrs)` and WL-off
= 0) programmed through the composed `XbarArray1t1r`, a fixed `V_BLC`
clamp reference, and a small positive BL/SL wire resistance (a required array
field — never assumed zero in code; the DC solver needs `R > 0`). The word line
is gate-only: it carries no DC current, so it has capacitance fields but no
resistance. The wire R is
tiny relative to the cell branch, so the array's IR drop is a fraction of a
percent and the per-cell current is `I = g_chord * V_BLC`. Every array node and
read branch draws from the shared core supply `vdd__V`. Under this chain the ADC
input current `I_SUB` is monotone in the signed integer MAC,
so a mid-point ladder probed from the macro's own transfer decodes any MAC
bit-exactly.

The geometry mirrors the paper design in miniature: `output_num = 4`
(`mux_factor = 2` -> `io_num = 2`), `input_num = max_active_num = 4`,
`input_bit_num = 2` (K serial WL sub-phases, LSB first), a 3-bit ADC magnitude.
The ADC uses one fixed SAR decision period; the static-energy time base is
`t_cycle` alone.
`build_config` is parameterised by geometry and window knobs so other test
files reuse it.

The analog `I_SUB(M)` grid depends on the whole electrical config, so the
witness ships a placeholder ladder and decode-bearing tests calibrate in-code
through `build_calibrated_macro`: probe the macro's own `I_SUB(M)` grid
(`probe_i_sub_grid`, captured through the ADC's own record prober) on
an all-`+1` column, install the mid-point thresholds (`midpoint_refs` +
`with_ref_levels`), and rebuild — a law-level calibration derived from the
config under test, not from shipped numbers.
"""

from __future__ import annotations

import dataclasses
from typing import TypedDict, Unpack

import torch
from torch import Tensor

from neurox import stamp_names
from neurox.primitive.analog import (
    AdcProber,
    IrefConfig,
    IrefPolicy,
    UnmodeledBlockConfig,
    VoltageDriverConfig,
    VoltageDriverPolicy,
    VrefConfig,
    VrefPolicy,
)
from neurox.primitive.analog.current_adc import (
    SarIadcConfig,
    SarIadcPolicy,
)
from neurox.primitive.analog.voltage_dac import GeneralVdacConfig, GeneralVdacPolicy
from neurox.primitive.macro.cim import CimMacro, CimMacroMode
from neurox.primitive.xbar.array import XbarArray1t1rConfig, XbarArray1t1rPolicy
from neurox.primitive.xbar.cell import XbarCell1t1rLinearConfig, XbarCell1t1rLinearPolicy
from neurox.primitive.xbar.solver import ColBlColSlSolverConfig
from neurox.works.macro.cim.xue2020jssc import (
    Xue2020JsscCimMacro,
    Xue2020JsscCimMacroConfig,
    Xue2020JsscCimMacroPolicy,
)
from neurox.works.macro.cim.xue2020jssc.pn_isub import PnIsubConfig
from neurox.works.macro.cim.xue2020jssc.tmcsa import TmcsaConfig

# --- Tiny witness geometry ---
TINY_OUTPUT_NUM = 4
TINY_INPUT_NUM = 4
TINY_MAX_ACTIVE_SIZE = 4
TINY_MUX_FACTOR = 2  # io_num = output_num // mux_factor = 2
TINY_K = 2  # input_bit_num: two serial WL sub-phases, LSB first
TINY_ADC_BITS = 3
MAG_MAX = (1 << TINY_ADC_BITS) - 1  # 7 — the 3-bit magnitude saturation
QUANTIZATION_MODE = 0  # witness operating mode (single-mode reference)

_DTYPE = torch.float64  # analytic near-ideal chain: double precision keeps the ladder crisp
_G_LRS__uS = 100.0
_V_BLC__V = 0.3
# Small positive wire R (an [uncertain] physical estimate; the solver needs R > 0
# and never assumes it in code). Tiny relative to the ~0.01 MOhm cell branch, so
# the array IR drop is a fraction of a percent — the chain stays near-ideal.
_WIRE_SEGMENT_R__MOhm = 5.0e-6  # 5 Ohm


class _BuildConfigKwargs(TypedDict, total=False):
    max_active_num: int
    mux_factor: int
    w_digit_num: int
    w_digit_radix: int
    input_bit_num: int
    adc_bits: int
    t_sample__ns: float
    t_settle__ns: float
    t_cycle__ns: float
    t_conduct_per_step__ns: float
    latency_per_step__ns: float
    ref_levels__uA: tuple[float, ...] | None


def _default_ref_levels(adc_bits: int) -> tuple[float, ...]:
    """Placeholder strictly-increasing single-mode ladder (`2**adc_bits - 1` taps)."""
    return tuple(float(k) for k in range(1, 1 << adc_bits))


def _default_mode(adc_bits: int) -> CimMacroMode:
    """The witness's single quantization mode at `adc_bits` magnitude resolution.

    The sign-magnitude readout attains `+-(2**adc_bits - 1)`, so the canonical
    mid-zero window holding it is `[-2**adc_bits, 2**adc_bits - 1]` (its bottom
    level is the phantom the encoding never emits) and the converter's own input
    code grid is the magnitude range `[0, 2**adc_bits - 1]`. The mid-point
    ladder makes the code the MAC magnitude itself, so the rescale factor is 1.
    """
    return CimMacroMode(
        quantization_input_range=(-(1 << adc_bits), (1 << adc_bits) - 1),
        adc_input_code_range=(0, (1 << adc_bits) - 1),
        max_bits_rescale_factor=1.0,
    )


def _linear_cell_config() -> XbarCell1t1rLinearConfig:
    """Near-ideal linearized 1T1R cell: exact-zero HRS branch, LRS chord, WL-off cut off."""
    return XbarCell1t1rLinearConfig(
        g_cell_on_table__uS=(0.0, _G_LRS__uS),  # state 0 = HRS -> exact 0, state 1 = LRS
        g_cell_off_table__uS=(0.0, 0.0),  # WL off -> no conduction
        vx_ratio_on_table=(0.5, 0.5),
        vx_ratio_off_table=(0.5, 0.5),
        v_wl_on_threshold__V=0.5,
    )


def _array_config() -> XbarArray1t1rConfig:
    """1T1R pure array: linear cell + small positive wire parasitics + nested DC solver.

    The wire R is a required positive field (never assumed zero in code); the near-
    zero value keeps the array's IR drop negligible so the MAC stays near-analytic.
    The four per-node caps are distinct positive placeholders (a test law: only
    distinctness and positivity are asserted, never a specific value).
    """
    return XbarArray1t1rConfig(
        row_cell_space__um=1.0,
        col_cell_space__um=1.0,
        bl_segment_r__MOhm=_WIRE_SEGMENT_R__MOhm,
        sl_segment_r__MOhm=_WIRE_SEGMENT_R__MOhm,
        bl_node_c__fF=0.11,
        x_node_c__fF=0.13,
        sl_node_c__fF=0.17,
        wl_node_c__fF=0.19,
        cell_config=_linear_cell_config(),
        solver_config=ColBlColSlSolverConfig(n_outer=3, n_inner=3),
    )


def build_config(
    *,
    max_active_num: int = TINY_INPUT_NUM,
    mux_factor: int = TINY_MUX_FACTOR,
    w_digit_num: int = 2,
    w_digit_radix: int = 2,
    input_bit_num: int = TINY_K,
    adc_bits: int = TINY_ADC_BITS,
    t_sample__ns: float = 1.0,
    t_settle__ns: float = 2.0,
    t_cycle__ns: float = 50.0,
    t_conduct_per_step__ns: float = 0.1,
    latency_per_step__ns: float = 1.0,
    ref_levels__uA: tuple[float, ...] | None = None,
) -> Xue2020JsscCimMacroConfig:
    """Hand-built near-ideal witness config, parameterised by geometry / windows.

    All physical / PPA fields are explicit small round values; only the linear
    cell / driver / array sub-configs are built in Python (no disk TOML). The
    near-ideal chain keeps the analog MAC analytic. The threshold ladder defaults
    to the placeholder `_default_ref_levels`; decode-bearing tests calibrate
    it in-code via `build_calibrated_macro`.

    Args:
        max_active_num: Per-conversion selection limit.
        mux_factor: Column-MUX depth; `io_num = output_num // mux_factor`.
        w_digit_num: Magnitude digits per weight (>= 1).
        w_digit_radix: Positional base of the magnitude digits (>= 2).
        input_bit_num: Activation bit width K (K serial WL sub-phases, LSB first).
        adc_bits: TMCSA magnitude resolution; the reference carries
            `2**adc_bits - 1` taps.
        t_sample__ns: Duration of each sampled input-bit phase.
        t_settle__ns: Tail settle window.
        t_cycle__ns: Declared operating period (the static-energy time base).
        t_conduct_per_step__ns: Kernel ADC conduction window shared by every
            decision step. Energy-path only.
        latency_per_step__ns: Duration of one TMCSA decision step.
        ref_levels__uA: Single-mode threshold ladder (defaults to the placeholder).
    """
    if ref_levels__uA is None:
        ref_levels__uA = _default_ref_levels(adc_bits)

    return Xue2020JsscCimMacroConfig(
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=8.0,  # macro-owned lump (un-attributed remainder)
        max_active_num=max_active_num,
        w_digit_num=w_digit_num,
        w_digit_radix=w_digit_radix,
        input_bit_num=input_bit_num,
        mux_factor=mux_factor,
        dswct_ratio_msb=0.5,
        sc_ratio_msb=0.5,
        t_sample__ns=t_sample__ns,
        t_settle__ns=t_settle__ns,
        t_cycle__ns=t_cycle__ns,
        vdd__V=1.0,
        control_config=UnmodeledBlockConfig(
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=6.0,
            energy_per_op__fJ=5.0,
        ),
        pn_isub_config=PnIsubConfig(e_per_op__fJ=1.0),
        tmcsa_config=TmcsaConfig(
            t_ph2__ns=0.2,
            t_ph3__ns=0.3,
            e_per_step__fJ=0.75,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=2.5,
        ),
        array_config=_array_config(),
        wl_dac_config=GeneralVdacConfig(
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
            code_to_signal=(0.0, 0.9),  # 1-bit ON/OFF WL drive
            drive_thermal__V=0.0,
            code_to_per_op_energy__fJ=(0.0, 0.0),
        ),
        cablc_config=VoltageDriverConfig(
            r_out__MOhm=0.0,  # ideal flat clamp (V_BL = V_BLC at the port); static leakage seat only
            offset_sigma__V=0.0,
            thermal_sigma__V=0.0,
            energy_per_op__fJ=0.0,  # CMD precharge belongs to the control block
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=2.0,
        ),
        cablc_vref_config=VrefConfig(  # dedicated V_BLC clamp source: one mode row, one tap
            v_refs__V=((_V_BLC__V,),),
            tolerance_sigma_relative=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        sl_driver_config=VoltageDriverConfig(
            r_out__MOhm=0.0,  # ideal flat clamp; the SL reference is a plain 0 V tensor (ground tie)
            offset_sigma__V=0.0,
            thermal_sigma__V=0.0,
            energy_per_op__fJ=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=1.0,
        ),
        adc_config=SarIadcConfig(
            bits=adc_bits,
            margin_gain=3.0,
            e_fixed_per_op__fJ=2.0,
            v_rail__V=1.0,
            t_conduct_per_step__ns=t_conduct_per_step__ns,
            latency_per_step__ns=latency_per_step__ns,
            comparator_offset_sigma__uA=0.0,
            coupling_mismatch_sigma__uA=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=4.0,
        ),
        reference_config=IrefConfig(
            i_refs__uA=(tuple(ref_levels__uA),),  # outer tuple = mode axis (single mode)
            tolerance_sigma_relative=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=5.0,
        ),
        modes=(_default_mode(adc_bits),),
    )


def build_all_off_policy() -> Xue2020JsscCimMacroPolicy:
    """All-off (lossless baseline) composite policy — the scheme's only intended policy."""
    return Xue2020JsscCimMacroPolicy(
        array_policy=XbarArray1t1rPolicy(cell_policy=XbarCell1t1rLinearPolicy(), solve_chunk_size=0),
        wl_dac_policy=GeneralVdacPolicy(drive_thermal=False),
        cablc_policy=VoltageDriverPolicy(offset=False, thermal=False),
        cablc_vref_policy=VrefPolicy(tolerance=False),
        sl_driver_policy=VoltageDriverPolicy(offset=False, thermal=False),
        adc_policy=SarIadcPolicy(
            comparator_offset=False,
            coupling_mismatch=False,
        ),
        reference_policy=IrefPolicy(tolerance=False),
    )


def build_macro(
    config: Xue2020JsscCimMacroConfig,
    *,
    input_num: int = TINY_INPUT_NUM,
    output_num: int = TINY_OUTPUT_NUM,
    device: torch.device | None = None,
    inst_shape: tuple[int, ...] = (),
) -> Xue2020JsscCimMacro:
    """Build + fabricate one macro on `device` under the all-off policy.

    The macro is name-stamped once assembled, so every profiled run built here
    emits records a `neurox.common.reporter.Reporter` can name.
    """
    macro = CimMacro.from_config(
        config=config,
        policy=build_all_off_policy(),
        input_num=input_num,
        output_num=output_num,
        inst_shape=inst_shape,
        dtype=_DTYPE,
        T__K=300.0,
    )
    assert isinstance(macro, Xue2020JsscCimMacro)
    if device is not None:
        macro.to(device)
    macro.eval()
    macro.fabricate()
    stamp_names(macro)
    return macro


def macro_device(macro: Xue2020JsscCimMacro) -> torch.device:
    """Device the macro lives on (first buffer of the module tree)."""
    return next(macro.buffers()).device


def with_ref_levels(config: Xue2020JsscCimMacroConfig, ref_levels__uA: tuple[float, ...]) -> Xue2020JsscCimMacroConfig:
    """Install one single-mode ladder on the Iref (the single ladder source)."""
    return dataclasses.replace(
        config,
        reference_config=dataclasses.replace(config.reference_config, i_refs__uA=(tuple(ref_levels__uA),)),
    )


def probe_i_sub_grid(macro: Xue2020JsscCimMacro, *, m_max: int) -> list[float]:
    """Probe the analog `I_SUB(M)` grid [uA] for MAC `M = 0..m_max` on an all-`+1` column.

    Programs logical column 0 all `+1` (others 0) and drives inputs whose row
    sum equals `M` (greedy fill, per-row value in `x_value_range`); column 0 lives
    at mux slot 0 of IO 0, so the grid rides `i_sub[m, 0, 0]`. The pre-ADC
    magnitude `I_SUB` is captured through the ADC's own
    `AdcProber` (`i_in__uA` per convert). All-off makes
    the probe deterministic. NOTE: reprograms the macro.
    """
    device = macro_device(macro)
    cfg = macro.config
    row_num = macro.row_num
    x_max = (1 << cfg.input_bit_num) - 1

    w_signed = torch.zeros((*macro.inst_shape, macro.row_num, macro.col_num), dtype=torch.long, device=device)
    w_signed[:, 0] = 1
    macro.program(w_signed)

    x = torch.zeros((m_max + 1, row_num), dtype=torch.long, device=device)
    for m in range(m_max + 1):
        remaining = m
        for r in range(row_num):
            v = min(x_max, remaining)
            x[m, r] = v
            remaining -= v
        assert remaining == 0, f"cannot reach MAC {m} with {row_num} rows of max {x_max}"

    with AdcProber() as probe, torch.no_grad():
        macro.vec_mat_mul(x, quantization_mode=QUANTIZATION_MODE, adc_bits=TINY_ADC_BITS)
    # One convert per vec_mat_mul; i_in__uA is the pre-ADC magnitude I_SUB.
    # Shape: [m_max + 1, group_size, group_num]
    i_sub = probe.records[-1].input_value()
    return [float(v) for v in i_sub[:, 0, 0].cpu()]


def midpoint_refs(grid: list[float], *, adc_bits: int = TINY_ADC_BITS) -> tuple[float, ...]:
    """The `2**adc_bits - 1` mid-point thresholds `ref[k] = 0.5 * (I(k) + I(k+1))`."""
    level_num = (1 << adc_bits) - 1
    assert len(grid) >= level_num + 1, f"grid too short: {len(grid)} < {level_num + 1}"
    return tuple(0.5 * (grid[k] + grid[k + 1]) for k in range(level_num))


def build_calibrated_macro(
    *,
    device: torch.device | None = None,
    inst_shape: tuple[int, ...] = (),
    **config_kwargs: Unpack[_BuildConfigKwargs],
) -> Xue2020JsscCimMacro:
    """Macro with an in-code calibrated ladder: probe the `I_SUB(M)` grid, install mid-points, rebuild.

    A first (placeholder-ladder) build probes the analog `I_SUB(M)` grid on an
    all-`+1` column — the ladder is irrelevant before the ADC — and the rebuild
    installs the grid's mid-points as the calibrated thresholds. Under all-off
    the analog chain is deterministic, so the same ladder serves every instance.
    Extra keyword arguments pass through to `build_config`.
    """
    config = build_config(**config_kwargs)
    adc_bits = config.adc_config.bits
    mag_max = (1 << adc_bits) - 1
    probe = build_macro(config, device=device)
    grid = probe_i_sub_grid(probe, m_max=mag_max)
    calibrated = with_ref_levels(config, midpoint_refs(grid, adc_bits=adc_bits))
    return build_macro(calibrated, device=device, inst_shape=inst_shape)


def ideal_mac(w_signed: Tensor, x: Tensor, *, mag_max: int = MAG_MAX) -> Tensor:
    """CPU int64 reference for the logical VMM.

    Args:
        w_signed: Weights in the logical value domain, not digit planes.
            Shape: `[input_num, output_num]`.
        x: Integer activations.
            Shape: `[..., input_num]`.
        mag_max: Signed-magnitude clip bound.

    Returns:
        Expected signed codes on CPU.
        Shape: `[..., output_num]`.
    """
    w2 = w_signed.cpu().long()
    x2 = x.cpu().long()
    mac = x2 @ w2
    return mac.clamp(-mag_max, mag_max)


def decode(
    macro: Xue2020JsscCimMacro,
    w_signed: Tensor,
    x: Tensor,
    *,
    quantization_mode: int = QUANTIZATION_MODE,
    adc_bits: int = TINY_ADC_BITS,
) -> Tensor:
    """Program logical weights and run one VMM.

    Returns the signed-magnitude codes on CPU.
    """
    device = macro_device(macro)
    macro.program(w_signed.to(device))
    with torch.no_grad():
        out = macro.vec_mat_mul(x.to(device), quantization_mode=quantization_mode, adc_bits=adc_bits)
    return out.cpu()
