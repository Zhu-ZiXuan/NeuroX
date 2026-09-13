"""Small deterministic Ye2023 witness."""

from __future__ import annotations

import torch

from neurox.primitive.analog import (
    ReferenceConfig,
    ReferencePolicy,
    UnmodeledBlockConfig,
    VoltageDriverConfig,
    VoltageDriverPolicy,
)
from neurox.primitive.macro.cim import CimMacro
from neurox.works.macro.cim.ye2023jssc import (
    Ye2023JsscCimMacro,
    Ye2023JsscCimMacroConfig,
    Ye2023JsscCimMacroPolicy,
)
from neurox.works.macro.cim.ye2023jssc.array import Ye2023Jssc2t1rArrayConfig, Ye2023Jssc2t1rArrayPolicy
from neurox.works.macro.cim.ye2023jssc.cell import Ye2023Jssc2t1rCellConfig, Ye2023Jssc2t1rCellPolicy
from neurox.works.macro.cim.ye2023jssc.rscsa import RsCsaIadcConfig, RsCsaIadcPolicy

INPUT_NUM = 2
OUTPUT_NUM = 4
ADC_BITS = 4
W_DIGIT_NUM = 3
W_DIGIT_RADIX = 2
T2_LEAK__uA = 0.125
T2_SIGNAL__uA = 0.375
V_TBL__V = 0.1
T_SETTLE__ns = 1.0
ADC_LATENCY_PER_BIT__ns = 2.0


def cell_config() -> Ye2023Jssc2t1rCellConfig:
    return Ye2023Jssc2t1rCellConfig(
        g_cell_off_table__uS=(0.001, 0.001),
        g_cell_on_table__uS=(1.0, 10.0),
        vx_ratio_off_table=(0.0, 0.0),
        vx_ratio_on_table=(0.75, 0.25),
        v_wl_on_threshold__V=0.3,
        v_x_on_threshold__V=0.15,
        i_t2_leak__uA=T2_LEAK__uA,
        i_t2_unit_signal__uA=T2_SIGNAL__uA,
    )


def adc_config() -> RsCsaIadcConfig:
    return RsCsaIadcConfig(
        bits=ADC_BITS,
        latency_per_bit__ns=ADC_LATENCY_PER_BIT__ns,
        comparator_offset_sigma__uA=0.0,
        init__ns=0.0,
        energy_per_bit__fJ=0.5,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
    )


def build_config(
    *,
    input_num: int = INPUT_NUM,
    lane_num: int = 1,
    scan_num: int = OUTPUT_NUM,
) -> Ye2023JsscCimMacroConfig:
    array = Ye2023Jssc2t1rArrayConfig(
        row_cell_space__um=1.0,
        col_cell_space__um=1.0,
        bl_segment_r__MOhm=5.0e-6,
        sl_segment_r__MOhm=5.0e-6,
        bl_node_c__fF=0.3,
        x_node_c__fF=0.2,
        sl_node_c__fF=0.1,
        wl_node_c__fF=0.4,
        t2_gate_unit_c__fF=0.05,
        tbl_node_unit_c__fF=0.5,
        cell_config=cell_config(),
    )
    zero_block = UnmodeledBlockConfig(area_per_inst__um2=0.0, leakage_per_inst__uW=0.0, energy_per_op__fJ=0.0)
    zero_driver = VoltageDriverConfig(
        r_out__MOhm=0.0,
        offset_sigma__V=0.0,
        thermal_sigma__V=0.0,
        energy_per_op__fJ=0.0,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
    )
    return Ye2023JsscCimMacroConfig(
        input_num=input_num,
        max_active_num=input_num,
        lane_num=lane_num,
        scan_num=scan_num,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        rescale_factors=(1.0,),
        w_digit_num=W_DIGIT_NUM,
        w_digit_radix=W_DIGIT_RADIX,
        t_settle__ns=T_SETTLE__ns,
        array_config=array,
        adc_config=adc_config(),
        reference_config=ReferenceConfig(
            values=(T2_SIGNAL__uA,),
            tolerance_sigma_relative=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        bl_driver_config=zero_driver,
        sl_driver_config=zero_driver,
        mux_driver_config=zero_block,
        timing_ctrl_config=zero_block,
        v_bl__V=0.3,
        v_wl_on__V=0.6,
        v_tbl__V=V_TBL__V,
        v_sl__V=0.0,
        vdd__V=0.8,
    )


def build_policy(*, solve_chunk_size: int = 0) -> Ye2023JsscCimMacroPolicy:
    return Ye2023JsscCimMacroPolicy(
        array_policy=Ye2023Jssc2t1rArrayPolicy(
            cell_policy=Ye2023Jssc2t1rCellPolicy(),
            solve_chunk_size=solve_chunk_size,
        ),
        adc_policy=RsCsaIadcPolicy(comparator_offset=False),
        reference_policy=ReferencePolicy(tolerance=False),
        bl_driver_policy=VoltageDriverPolicy(offset=False, thermal=False),
        sl_driver_policy=VoltageDriverPolicy(offset=False, thermal=False),
    )


def build_macro(
    *,
    device: torch.device,
    input_num: int = INPUT_NUM,
    lane_num: int = 1,
    scan_num: int = OUTPUT_NUM,
    inst_shape: tuple[int, ...] = (),
    solve_chunk_size: int = 0,
) -> Ye2023JsscCimMacro:
    macro = CimMacro.from_config(
        config=build_config(input_num=input_num, lane_num=lane_num, scan_num=scan_num),
        policy=build_policy(solve_chunk_size=solve_chunk_size),
        inst_shape=inst_shape,
        dtype=torch.float64,
    )
    assert isinstance(macro, Ye2023JsscCimMacro)
    macro.to(device)
    macro.eval()
    macro.fabricate()
    return macro
