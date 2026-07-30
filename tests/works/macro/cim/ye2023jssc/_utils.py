"""Shared hand-built witness configs + oracles for the ye2023jssc scheme tests.

Every config dataclass is built directly in Python with small explicit values (no
disk TOML). The witness ships an ANALYTIC readout chain so the integer MAC is
exact:

  * the T2 lookup returns the same ``FLOOR__uA`` for every input-0 cell AND for
    an input-1 HRS cell, and ``I_UNIT__uA`` for (IN = 1, LRS);
  * the macro DERIVES the PH0 compensation as ``FLOOR * row_num * sum(radix)``
    over the weight AND redundant planes, so it cancels every non-LRS
    contribution exactly;
  * ``i_lsb = I_UNIT - FLOOR`` is the resulting per-MAC-unit current step, so the
    RS-CSA code equals the UNSIGNED MAC bit-exactly (rescale factor 1.0).

Every current is a dyadic fraction, so the whole chain is exact in float64 and a
MAC landing on a decision boundary is unambiguous.

Miniature geometry: ``output_num = 4`` logical outputs, ``input_num = 2`` logical
1-bit inputs, ``weight_radix = (1, 2, 4)`` (three binary weight planes, LSB-first)
plus ``redundant_radix = (4,)`` (the non-weight SUBA4 plane, programmed all-HRS
and driven input-0), 4-bit RS-CSA. The array is TRANSPOSED (physical rows = 4
outputs, physical columns = ``input_num * 4 planes = 8``).

The witness phase set is deliberately NOT the paper's, so any test reading the
access window pins the DERIVATION ``T_AC(b) = sum(t_phase[:b]) + t4_intrinsic``
over the EXECUTED phases rather than a shipped number.
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.primitive.analog import (
    UnmodeledBlockConfig,
    UnmodeledBlockPolicy,
    VoltageDriverConfig,
    VoltageDriverPolicy,
)
from neurox.primitive.macro.cim import CimMacro, CimMacroMode
from neurox.primitive.xbar.solver import NestedParallelRailSolverConfig
from neurox.works.macro.cim.ye2023jssc import (
    Ye2023JsscCimMacro,
    Ye2023JsscCimMacroConfig,
    Ye2023JsscCimMacroPolicy,
)
from neurox.works.macro.cim.ye2023jssc.array import Ye2023Jssc2t1rArrayConfig, Ye2023Jssc2t1rArrayPolicy
from neurox.works.macro.cim.ye2023jssc.cell import Ye2023Jssc2t1rCellConfig, Ye2023Jssc2t1rCellPolicy
from neurox.works.macro.cim.ye2023jssc.rscsa import RsCsaIadcConfig, RsCsaIadcPolicy

# --- Tiny witness geometry ---
TINY_OUTPUT_NUM = 4
TINY_INPUT_NUM = 2
TINY_WEIGHT_RADIX = (1, 2, 4)  # three binary WEIGHT planes, LSB-first (digit 0 = m = 1)
TINY_REDUNDANT_RADIX = (4,)  # the SUBA4 non-weight plane
TINY_PLANE_NUM = len(TINY_WEIGHT_RADIX) + len(TINY_REDUNDANT_RADIX)
TINY_ADC_BITS = 4
QUANTIZATION_MODE = 0
W_MAX = sum(TINY_WEIGHT_RADIX)  # 7 — the three-plane unsigned weight envelope
MAG_MAX = (1 << TINY_ADC_BITS) - 1  # 15 — the 4-bit code saturation

# --- Analytic current chain (dyadic: exact in float64) ---
FLOOR__uA = 0.125  # IN = 0 off-cell floor AND the (IN = 1, HRS) weight leakage
I_UNIT__uA = 0.5  # (IN = 1, LRS) T2 compute current
I_LSB__uA = I_UNIT__uA - FLOOR__uA  # 0.375 — one MAC unit after the PH0 subtraction

# --- Biases ---
V_WL_SEL__V = 0.6
V_WL_ON_THRESHOLD__V = 0.3
V_BL_IN1__V = 0.3
V_BL_IN_THRESHOLD__V = 0.15
V_SL__V = 0.0
V_DD_CORE__V = 0.8
V_TBL__V = 0.1

# --- RS-CSA timing / energy (witness values, NOT the paper's) ---
T_PHASE__ns = (1.0, 2.0, 4.0, 8.0, 16.0)  # PH0 + one compare phase per bit, MSB-first
T4_INTRINSIC__ns = 0.5
T_AC__ns = sum(T_PHASE__ns[:-1]) + T4_INTRINSIC__ns  # the derived window at the FULL phase set
REF_RADIX = (8, 4, 2, 1)
MIRROR_SCALE = 0.25
E_FIXED__fJ = 2.0

# --- Capacitances [fF] ---
C_BL__fF = 0.2
C_X__fF = 0.3
C_SL__fF = 0.1
C_WL__fF = 0.2
BL_FIRST_C__fF = 0.4
BL_SEGMENT_C__fF = 0.1
SL_FIRST_C__fF = 0.4
SL_SEGMENT_C__fF = 0.1
WL_FIRST_C__fF = 0.4
WL_SEGMENT_C__fF = 0.1

# --- Flat peripheral per-op energies ---
E_MUX_DRIVER__fJ = 3.0
E_TIMING_CTRL__fJ = 7.0

DTYPE = torch.float64  # crisp analytic ladder

_G_HRS__uS = 1.0  # every cell conducts on step1 -> BL conduction is state-independent
_G_LRS__uS = 100.0
_VX_RATIO_ON = 0.5
# Small positive wire R (solver needs R > 0); tiny vs the cell branch.
_WIRE_FIRST_R__MOhm = 2.0e-5
_WIRE_SEGMENT_R__MOhm = 5.0e-6


def cell_config() -> Ye2023Jssc2t1rCellConfig:
    """WH-2T1R cell witness: linear divider chords + the analytic I_T2 table."""
    return Ye2023Jssc2t1rCellConfig(
        c_bl__fF=C_BL__fF,
        c_x__fF=C_X__fF,
        c_sl__fF=C_SL__fF,
        c_wl__fF=C_WL__fF,
        g_cell_on_table__uS=(_G_HRS__uS, _G_LRS__uS),  # state 0 = HRS, state 1 = LRS
        g_cell_off_table__uS=(0.0, 0.0),  # WL off -> no branch conduction
        vx_ratio_on_table=(_VX_RATIO_ON, _VX_RATIO_ON),
        vx_ratio_off_table=(0.0, 0.0),
        v_wl_on_threshold__V=V_WL_ON_THRESHOLD__V,
        # i_t2_table__uA[input_bit][state]: IN = 0 floor, IN = 1 -> HRS floor / LRS I_unit.
        i_t2_table__uA=((FLOOR__uA, FLOOR__uA), (FLOOR__uA, I_UNIT__uA)),
    )


def array_config() -> Ye2023Jssc2t1rArrayConfig:
    """WH-2T1R array witness: cell + small positive wire parasitics + nested solver."""
    return Ye2023Jssc2t1rArrayConfig(
        row_first_space__um=1.0,
        row_cell_space__um=1.0,
        col_first_space__um=1.0,
        col_cell_space__um=1.0,
        bl_first_r__MOhm=_WIRE_FIRST_R__MOhm,
        bl_first_c__fF=BL_FIRST_C__fF,
        bl_segment_r__MOhm=_WIRE_SEGMENT_R__MOhm,
        bl_segment_c__fF=BL_SEGMENT_C__fF,
        sl_first_r__MOhm=_WIRE_FIRST_R__MOhm,
        sl_first_c__fF=SL_FIRST_C__fF,
        sl_segment_r__MOhm=_WIRE_SEGMENT_R__MOhm,
        sl_segment_c__fF=SL_SEGMENT_C__fF,
        wl_first_r__MOhm=_WIRE_FIRST_R__MOhm,
        wl_first_c__fF=WL_FIRST_C__fF,
        wl_segment_r__MOhm=_WIRE_SEGMENT_R__MOhm,
        wl_segment_c__fF=WL_SEGMENT_C__fF,
        cell_config=cell_config(),
        solver_config=NestedParallelRailSolverConfig(n_outer=3, n_inner=3),
        latency_per_op__ns=0.0,  # macro is the sole latency emitter
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=7.0,
        weight_radix=TINY_WEIGHT_RADIX,
        redundant_radix=TINY_REDUNDANT_RADIX,
        v_bl_in1__V=V_BL_IN1__V,
        v_bl_in_threshold__V=V_BL_IN_THRESHOLD__V,
    )


def adc_config(*, adc_bits: int = TINY_ADC_BITS) -> RsCsaIadcConfig:
    """RS-CSA witness: uniform ``i_lsb`` ladder + physical phase set."""
    return RsCsaIadcConfig(
        bits=adc_bits,
        i_lsb__uA=I_LSB__uA,  # LSB = one MAC unit -> code == MAC
        ref_radix=REF_RADIX,
        v_rail__V=V_DD_CORE__V,
        t_phase__ns=T_PHASE__ns,
        t4_intrinsic__ns=T4_INTRINSIC__ns,
        mirror_scale=MIRROR_SCALE,
        e_fixed_per_op__fJ=E_FIXED__fJ,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=4.0,
    )


def build_config(
    *,
    max_active_num: int = TINY_INPUT_NUM,
    adc_bits: int = TINY_ADC_BITS,
) -> Ye2023JsscCimMacroConfig:
    """Hand-built analytic witness config; ``code == UNSIGNED MAC`` (rescale 1.0)."""
    return Ye2023JsscCimMacroConfig(
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=8.0,
        max_active_num=max_active_num,
        array_config=array_config(),
        adc_config=adc_config(adc_bits=adc_bits),
        bl_driver_config=VoltageDriverConfig(
            r_out__MOhm=0.0,
            offset_sigma__V=0.0,
            thermal_sigma__V=0.0,
            energy_per_op__fJ=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=2.0,
        ),
        sl_driver_config=VoltageDriverConfig(
            r_out__MOhm=0.0,
            offset_sigma__V=0.0,
            thermal_sigma__V=0.0,
            energy_per_op__fJ=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=1.0,
        ),
        mux_driver_config=UnmodeledBlockConfig(area_per_inst__um2=0.0, leakage_per_inst__uW=5.0),
        timing_ctrl_config=UnmodeledBlockConfig(area_per_inst__um2=0.0, leakage_per_inst__uW=14.0),
        v_wl_sel__V=V_WL_SEL__V,
        v_bl_in1__V=V_BL_IN1__V,
        v_tbl__V=V_TBL__V,
        v_sl__V=V_SL__V,
        v_dd_core__V=V_DD_CORE__V,
        e_mux_driver_per_op__fJ=E_MUX_DRIVER__fJ,
        e_timing_ctrl_per_op__fJ=E_TIMING_CTRL__fJ,
        # One code carries one MAC unit, so the window holds the 2**adc_bits
        # codes the readout resolves and the rescale factor is the identity.
        modes=(
            CimMacroMode(
                quantization_input_range=(0, (1 << adc_bits) - 1),
                adc_input_code_range=(0, (1 << adc_bits) - 1),
                max_bits_rescale_factor=1.0,
            ),
        ),
    )


def build_all_off_policy() -> Ye2023JsscCimMacroPolicy:
    """All-off (lossless baseline) composite policy — the scheme's only intended policy."""
    return Ye2023JsscCimMacroPolicy(
        array_policy=Ye2023Jssc2t1rArrayPolicy(
            cell_policy=Ye2023Jssc2t1rCellPolicy(),
            solve_chunk_size=0,
        ),
        adc_policy=RsCsaIadcPolicy(),
        bl_driver_policy=VoltageDriverPolicy(offset=False, thermal=False),
        sl_driver_policy=VoltageDriverPolicy(offset=False, thermal=False),
        mux_driver_policy=UnmodeledBlockPolicy(),
        timing_ctrl_policy=UnmodeledBlockPolicy(),
    )


def build_macro(
    config: Ye2023JsscCimMacroConfig,
    *,
    input_num: int = TINY_INPUT_NUM,
    output_num: int = TINY_OUTPUT_NUM,
    device: torch.device | None = None,
    inst_shape: tuple[int, ...] = (),
) -> Ye2023JsscCimMacro:
    """Build + fabricate one macro on ``device`` under the all-off policy."""
    macro = CimMacro.from_config(
        config=config,
        policy=build_all_off_policy(),
        input_num=input_num,
        output_num=output_num,
        inst_shape=inst_shape,
        dtype=DTYPE,
        T__K=300.0,
    )
    assert isinstance(macro, Ye2023JsscCimMacro)
    if device is not None:
        macro.to(device)
    macro.eval()
    macro.fabricate()
    return macro


def macro_device(macro: Ye2023JsscCimMacro) -> torch.device:
    """Device the macro lives on (first buffer of the module tree)."""
    return next(macro.buffers()).device


def expected_ph0__uA(*, input_num: int = TINY_INPUT_NUM) -> float:
    """All-off row floor sum: ``FLOOR * input_num * sum(weight + redundant radix)``."""
    return FLOOR__uA * input_num * sum((*TINY_WEIGHT_RADIX, *TINY_REDUNDANT_RADIX))


def ideal_mac(w_val: Tensor, x: Tensor, *, clamp: bool = True, mag_max: int = MAG_MAX) -> Tensor:
    """CPU int64 unsigned VMM reference.

    Args:
        w_val: Unsigned logical weights.
            Shape: ``[input_num, output_num]``.
        x: 1-bit activations.
            Shape: ``[..., input_num]``.
        clamp: Clamp to ``[0, mag_max]`` (the RS-CSA code saturation).
        mag_max: Upper code bound.

    Returns:
        Expected MAC / code on CPU (int64).
        Shape: ``[..., out]``.
    """
    w2 = w_val.cpu().long()
    x2 = x.cpu().long()
    mac = x2 @ w2
    return mac.clamp(0, mag_max) if clamp else mac


def decode(
    macro: Ye2023JsscCimMacro,
    w_val: Tensor,
    x: Tensor,
    *,
    quantization_mode: int = QUANTIZATION_MODE,
    adc_bits: int = TINY_ADC_BITS,
) -> Tensor:
    """Program logical unsigned weights and run one VMM.

    Returns the unsigned codes on CPU.
    """
    device = macro_device(macro)
    macro.program(w_val.to(device))
    with torch.no_grad():
        out = macro.vec_mat_mul(x.to(device), quantization_mode=quantization_mode, adc_bits=adc_bits)
    return out.cpu()
