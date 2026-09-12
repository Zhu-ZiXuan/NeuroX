"""Xue2020 construction, geometry, timing, and validation tests."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from neurox.common.encoding import Encoding
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig
from neurox.primitive.xbar.cell import XbarCell1t1rLinearConfig
from neurox.works.macro.cim.xue2020jssc import Xue2020JsscCimMacro, Xue2020JsscCimMacroConfig

from ._utils import (
    TINY_ADC_BITS,
    build_all_off_policy,
    build_config,
    build_macro,
)


def test_registry_dispatch() -> None:
    config = build_config()
    macro = CimMacro.from_config(
        config=config,
        policy=build_all_off_policy(),
        inst_shape=(),
        dtype=torch.float64,
        T__K=300.0,
    )
    assert isinstance(macro, Xue2020JsscCimMacro)


def test_dict_reflection_round_trip() -> None:
    config = build_config()
    restored = CimMacroConfig.from_dict(config.to_dict())
    assert isinstance(restored, Xue2020JsscCimMacroConfig)
    assert restored == config


def test_derived_geometry_laws() -> None:
    macro = build_macro(build_config(lane_num=2, scan_num=4))
    assert macro.lane_num == 2
    assert macro.scan_num == 4
    assert macro.array.cell.inst_shape[-2:] == (macro.row_num, macro.col_num)

    macro_d1 = build_macro(build_config(w_digit_num=1))
    macro_d3 = build_macro(build_config(w_digit_num=3))
    assert macro_d1.array.cell.inst_shape[-2:] == (macro_d1.row_num, macro_d1.col_num)
    assert macro_d3.array.cell.inst_shape[-2:] == (macro_d3.row_num, macro_d3.col_num)


def test_derived_ratio_anchors() -> None:
    macro = build_macro(build_config())
    digit_ratios = tuple(macro._dswct_digit_ratios.tolist())
    bit_ratios = tuple(macro._sinwp_bit_ratios.tolist())
    assert digit_ratios == (0.25, 0.5)
    assert bit_ratios == (0.25, 0.5)
    composite = sorted(r * s for r in digit_ratios for s in bit_ratios)
    assert composite == pytest.approx([1 / 16, 1 / 8, 1 / 8, 1 / 4])


def test_derived_ratios_degenerate_to_anchor_at_size_one() -> None:
    d1 = build_macro(build_config(w_digit_num=1))
    assert tuple(d1._dswct_digit_ratios.tolist()) == (0.5,)
    k1 = build_macro(build_config(x_bit_num=1))
    assert tuple(k1._sinwp_bit_ratios.tolist()) == (0.5,)


def test_window_laws_default() -> None:
    config = build_config(t_sample__ns=1.0, t_settle__ns=2.0, latency_per_bit__ns=3.0)
    macro = build_macro(config)
    for bits in (1, 3):
        detect__ns = 2.0 + bits * 3.0
        access__ns = 1.0 + detect__ns
        assert macro.latency__ns(adc_active_bits=bits) == access__ns * macro.scan_num


def test_window_laws_three_bit() -> None:
    config = build_config(
        x_bit_num=3,
        t_sample__ns=2.0,
        t_settle__ns=1.0,
        latency_per_bit__ns=3.0,
    )
    macro = build_macro(config)
    detect__ns = 1.0 + 3 * 3.0
    access__ns = 2 * 2.0 + detect__ns
    assert macro.latency__ns(adc_active_bits=3) == access__ns * macro.scan_num


def test_value_domain_contract() -> None:
    config = build_config()
    macro = build_macro(config)
    assert config.w_digit_n == config.w_digit_num
    assert config.w_digit_r == config.w_digit_radix
    assert config.w_enc is Encoding.TRUE_FORM
    assert config.x_digit_n == config.x_bit_num
    assert config.x_digit_r == 2
    assert config.x_enc is Encoding.UNSIGNED
    assert macro.x_value_range == (0, (1 << config.x_bit_num) - 1) == (0, 3)
    assert macro.w_value_range == (-3, 3)
    assert macro.adc_bits == config.tmcsa_config.bits == TINY_ADC_BITS
    assert config.tmcsa_iref_config.shape == (len(macro.config.rescale_factors), 7)


def test_quantization_mode_out_of_range_rejected() -> None:
    macro = build_macro(build_config())
    beyond = len(macro.config.rescale_factors)
    with pytest.raises(ValueError, match="quantization_mode"):
        macro.rescale_factor(quantization_mode=beyond, adc_active_bits=macro.adc_bits)


def test_rescale_factor_bit_width_law() -> None:
    macro = build_macro(build_config())
    max_bits = macro.adc_bits
    r_max = macro.rescale_factor(quantization_mode=0, adc_active_bits=max_bits)
    assert r_max > 0.0
    for bits in range(1, max_bits):
        assert macro.rescale_factor(quantization_mode=0, adc_active_bits=bits) == pytest.approx(
            r_max * 2.0 ** (max_bits - bits)
        )
    with pytest.raises(ValueError, match="adc_active_bits"):
        macro.rescale_factor(quantization_mode=0, adc_active_bits=max_bits + 1)


def test_ideal_twin_preserves_the_quantization_surface() -> None:
    config = build_config()
    macro = build_macro(config)
    twin = macro.to_ideal()
    assert twin.adc_bits == macro.adc_bits
    assert twin.config.rescale_factors == macro.config.rescale_factors
    assert twin.x_value_range == macro.x_value_range
    assert twin.w_value_range == macro.w_value_range
    assert twin.max_active_num == macro.max_active_num
    assert twin.rescale_factor(quantization_mode=0, adc_active_bits=twin.adc_bits) == 1.0


def test_validate_rejects_mode_count_mismatch() -> None:
    config = build_config()
    with pytest.raises(ValueError, match=r"tmcsa_iref_config\.shape"):
        dataclasses.replace(config, rescale_factors=(*config.rescale_factors, *config.rescale_factors))


def test_output_num_is_fixed_by_readout_geometry() -> None:
    macro = build_macro(build_config(lane_num=3, scan_num=2))
    assert macro.output_num == 6


def test_validate_rejects_sub_binary_radix() -> None:
    config = build_config()
    with pytest.raises(ValueError, match=r"w_digit_r \(1\) >= 2"):
        dataclasses.replace(config, w_digit_radix=1)


def test_validate_rejects_negative_input_phase_duration() -> None:
    config = build_config()
    with pytest.raises(ValueError, match="t_sample__ns"):
        dataclasses.replace(config, t_sample__ns=-1.0)


def test_generalized_w_digit_num_accepted() -> None:
    d1 = build_config(w_digit_num=1)
    assert d1.w_digit_num == 1
    macro1 = build_macro(d1)
    assert macro1.w_value_range == (-1, 1)
    assert macro1.array.cell.inst_shape[-2:] == (macro1.row_num, macro1.col_num)

    d3 = build_config(w_digit_num=3)
    assert d3.w_digit_num == 3
    macro3 = build_macro(d3)
    assert macro3.array.cell.inst_shape[-2:] == (macro3.row_num, macro3.col_num)


def test_weight_radix_cannot_exceed_the_cell_state_count() -> None:
    config = build_config(w_digit_radix=3)
    with pytest.raises(ValueError, match=r"array\.w_state_num \(2\) >= w_digit_r \(3\)"):
        build_macro(config)


def test_weight_radix_uses_a_matching_multistate_cell() -> None:
    config = build_config(w_digit_radix=3)
    cell = config.array_config.cell_config
    assert isinstance(cell, XbarCell1t1rLinearConfig)
    cell = dataclasses.replace(
        cell,
        g_cell_on_table__uS=(*cell.g_cell_on_table__uS, 200.0),
        g_cell_off_table__uS=(*cell.g_cell_off_table__uS, 0.0),
        vx_ratio_on_table=(*cell.vx_ratio_on_table, 0.5),
        vx_ratio_off_table=(*cell.vx_ratio_off_table, 0.5),
    )
    config = dataclasses.replace(
        config,
        array_config=dataclasses.replace(config.array_config, cell_config=cell),
    )
    macro = build_macro(config)
    assert macro.w_value_range == (-8, 8)


def test_input_bit_num_one_accepted() -> None:
    config = build_config(x_bit_num=1)
    assert config.x_bit_num == 1
    macro = build_macro(config)
    detect__ns = config.t_settle__ns + macro.tmcsa.latency__ns(active_bits=macro.adc_bits)
    assert macro.latency__ns(adc_active_bits=macro.adc_bits) == detect__ns * macro.scan_num
    assert macro.x_value_range == (0, 1)  # single input bit


def test_non_divisible_max_active_num_accepted() -> None:
    assert 256 % 9 != 0  # max_active_num need not divide input_num
    config = dataclasses.replace(build_config(), input_num=256, max_active_num=9)
    macro = build_macro(config)
    assert config.max_active_num == 9
    assert macro.row_num == 256

    macro = build_macro(build_config(input_num=5, max_active_num=3))
    assert macro.max_active_num == 3
    assert isinstance(macro, Xue2020JsscCimMacro)
