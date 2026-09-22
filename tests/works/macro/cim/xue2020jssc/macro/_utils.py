"""Shared Xue2020 test fixtures and in-code ADC calibration."""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor

from neurox import stamp_names
from neurox.primitive.analog import (
    AdcProber,
    ReferenceConfig,
    ReferencePolicy,
    UnmodeledBlockConfig,
    VoltageDriverConfig,
    VoltageDriverPolicy,
)
from neurox.primitive.macro.cim import CimMacro
from neurox.primitive.xbar.array import XbarArray1t1rConfig, XbarArray1t1rPolicy
from neurox.primitive.xbar.cell import XbarCell1t1rLinearConfig, XbarCell1t1rLinearPolicy
from neurox.works.macro.cim.xue2020jssc import (
    Xue2020JsscCimMacro,
    Xue2020JsscCimMacroConfig,
    Xue2020JsscCimMacroPolicy,
)
from neurox.works.macro.cim.xue2020jssc.tmcsa import TmcsaConfig, TmcsaPolicy

TINY_OUTPUT_NUM = 4
TINY_INPUT_NUM = 4
TINY_LANE_NUM = 2
TINY_SCAN_NUM = 2
TINY_K = 2  # x_bit_num: two serial WL sub-phases, LSB first
TINY_ADC_BITS = 3
MAG_MAX = (1 << TINY_ADC_BITS) - 1
QUANTIZATION_MODE = 0

_DTYPE = torch.float64
_G_LRS__uS = 100.0
_V_BLC__V = 0.3
_WIRE_SEGMENT_R__MOhm = 5.0e-6  # Positive and negligible beside the cell branch.


def _default_ref_levels(adc_bits: int) -> tuple[float, ...]:
    return tuple(float(k) for k in range(1, 1 << adc_bits))


def _linear_cell_config() -> XbarCell1t1rLinearConfig:
    """Linear cell with exact-zero HRS and WL-off currents."""
    return XbarCell1t1rLinearConfig(
        g_cell_on_table__uS=(0.0, _G_LRS__uS),
        g_cell_off_table__uS=(0.0, 0.0),
        vx_ratio_on_table=(0.5, 0.5),
        vx_ratio_off_table=(0.5, 0.5),
        v_wl_on_threshold__V=0.5,
    )


def _array_config() -> XbarArray1t1rConfig:
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
    )


def build_config(
    *,
    input_num: int = TINY_INPUT_NUM,
    max_active_num: int = TINY_INPUT_NUM,
    lane_num: int = TINY_LANE_NUM,
    scan_num: int = TINY_SCAN_NUM,
    w_digit_num: int = 2,
    w_digit_radix: int = 2,
    x_bit_num: int = TINY_K,
    adc_bits: int = TINY_ADC_BITS,
    t_sample__ns: float = 1.0,
    t_settle__ns: float = 2.0,
    latency_per_bit__ns: float = 1.0,
    ref_levels__uA: tuple[float, ...] | None = None,
) -> Xue2020JsscCimMacroConfig:
    """Build a parameterized near-ideal test configuration."""
    if ref_levels__uA is None:
        ref_levels__uA = _default_ref_levels(adc_bits)

    return Xue2020JsscCimMacroConfig(
        input_num=input_num,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=8.0,
        max_active_num=max_active_num,
        lane_num=lane_num,
        scan_num=scan_num,
        w_digit_num=w_digit_num,
        w_digit_radix=w_digit_radix,
        x_bit_num=x_bit_num,
        dswct_ratio_msb=0.5,
        sc_ratio_msb=0.5,
        t_sample__ns=t_sample__ns,
        t_settle__ns=t_settle__ns,
        v_wl_on__V=0.9,
        vdd__V=1.0,
        pn_isub_energy_per_op__fJ=1.0,
        control_config=UnmodeledBlockConfig(
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=6.0,
            energy_per_op__fJ=5.0,
        ),
        tmcsa_config=TmcsaConfig(
            bits=adc_bits,
            t_ph2__ns=0.2,
            t_ph3__ns=0.3,
            energy_per_bit__fJ=0.75,
            latency_per_bit__ns=latency_per_bit__ns,
            comparator_offset_sigma__uA=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=2.5,
        ),
        array_config=_array_config(),
        cablc_config=VoltageDriverConfig(
            r_out__MOhm=0.0,
            offset_sigma__V=0.0,
            thermal_sigma__V=0.0,
            energy_per_op__fJ=0.0,  # CMD precharge belongs to the control block
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=2.0,
        ),
        cablc_vref_config=ReferenceConfig(
            values=_V_BLC__V,
            tolerance_sigma_relative=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        sl_driver_config=VoltageDriverConfig(
            r_out__MOhm=0.0,
            offset_sigma__V=0.0,
            thermal_sigma__V=0.0,
            energy_per_op__fJ=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=1.0,
        ),
        tmcsa_iref_config=ReferenceConfig(
            values=(tuple(ref_levels__uA),),
            tolerance_sigma_relative=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=5.0,
        ),
        rescale_factors=(1.0,),
    )


def build_all_off_policy() -> Xue2020JsscCimMacroPolicy:
    return Xue2020JsscCimMacroPolicy(
        array_policy=XbarArray1t1rPolicy(cell_policy=XbarCell1t1rLinearPolicy(), solve_chunk_size=0),
        cablc_policy=VoltageDriverPolicy(offset=False, thermal=False),
        cablc_vref_policy=ReferencePolicy(tolerance=False),
        sl_driver_policy=VoltageDriverPolicy(offset=False, thermal=False),
        tmcsa_policy=TmcsaPolicy(
            comparator_offset=False,
        ),
        tmcsa_iref_policy=ReferencePolicy(tolerance=False),
    )


def build_macro(
    config: Xue2020JsscCimMacroConfig,
    *,
    device: torch.device | None = None,
    inst_shape: tuple[int, ...] = (),
) -> Xue2020JsscCimMacro:
    macro = CimMacro.from_config(
        config=config,
        policy=build_all_off_policy(),
        inst_shape=inst_shape,
        dtype=_DTYPE,
    )
    assert isinstance(macro, Xue2020JsscCimMacro)
    if device is not None:
        macro.to(device)
    macro.eval()
    macro.fabricate()
    stamp_names(macro)
    return macro


def macro_device(macro: Xue2020JsscCimMacro) -> torch.device:
    return next(macro.buffers()).device


def with_ref_levels(config: Xue2020JsscCimMacroConfig, ref_levels__uA: tuple[float, ...]) -> Xue2020JsscCimMacroConfig:
    return dataclasses.replace(
        config,
        tmcsa_iref_config=dataclasses.replace(
            config.tmcsa_iref_config,
            values=(tuple(ref_levels__uA),),
        ),
    )


def probe_i_sub_grid(macro: Xue2020JsscCimMacro, *, m_max: int) -> list[float]:
    """Probe `I_SUB(M)` for `M = 0..m_max` on an all-`+1` column."""
    device = macro_device(macro)
    cfg = macro.config
    input_num = macro.input_num
    x_max = (1 << cfg.x_bit_num) - 1

    w_signed = torch.zeros((*macro.inst_shape, macro.input_num, macro.output_num), dtype=torch.int32, device=device)
    w_signed[:, 0] = 1
    macro.program(w_signed)

    x = torch.zeros((m_max + 1, input_num), dtype=torch.int32, device=device)
    for m in range(m_max + 1):
        remaining = m
        for r in range(input_num):
            v = min(x_max, remaining)
            x[m, r] = v
            remaining -= v
        assert remaining == 0, f"cannot reach MAC {m} with {input_num} inputs of max {x_max}"

    with AdcProber() as probe, torch.no_grad():
        macro.vec_mat_mul(x, quantization_mode=QUANTIZATION_MODE, adc_active_bits=TINY_ADC_BITS)
    # Shape: [magnitude, scan, lane]
    i_sub = probe.result[-1].input_value()
    return [float(v) for v in i_sub[:, 0, 0].cpu()]


def midpoint_refs(grid: list[float], *, adc_bits: int = TINY_ADC_BITS) -> tuple[float, ...]:
    """Place thresholds between adjacent measured integer-MAC currents."""
    level_num = (1 << adc_bits) - 1
    assert len(grid) >= level_num + 1, f"grid too short: {len(grid)} < {level_num + 1}"
    return tuple(0.5 * (grid[k] + grid[k + 1]) for k in range(level_num))


def build_calibrated_macro(
    *,
    device: torch.device | None = None,
    inst_shape: tuple[int, ...] = (),
    w_digit_num: int = 2,
) -> Xue2020JsscCimMacro:
    """Build a macro with midpoint references derived from its transfer."""
    config = build_config(w_digit_num=w_digit_num)
    adc_bits = config.tmcsa_config.bits
    mag_max = (1 << adc_bits) - 1
    probe = build_macro(config, device=device)
    grid = probe_i_sub_grid(probe, m_max=mag_max)
    calibrated = with_ref_levels(config, midpoint_refs(grid, adc_bits=adc_bits))
    return build_macro(calibrated, device=device, inst_shape=inst_shape)


def ideal_mac(w_signed: Tensor, x: Tensor, *, mag_max: int = MAG_MAX) -> Tensor:
    """Return the clamped CPU integer VMM reference."""
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
    adc_active_bits: int = TINY_ADC_BITS,
) -> Tensor:
    """Program weights and return one VMM result on CPU."""
    device = macro_device(macro)
    macro.program(w_signed.to(device))
    with torch.no_grad():
        out = macro.vec_mat_mul(
            x.to(device),
            quantization_mode=quantization_mode,
            adc_active_bits=adc_active_bits,
        )
    return out.cpu()
