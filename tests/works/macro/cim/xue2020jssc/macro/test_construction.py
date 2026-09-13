"""Xue2020 construction, geometry, and timing."""

from __future__ import annotations

from tests.works.macro.cim.xue2020jssc._utils import (
    build_config,
    build_macro,
)


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


def test_input_bit_num_one_accepted() -> None:
    config = build_config(x_bit_num=1)
    macro = build_macro(config)
    detect__ns = config.t_settle__ns + macro.tmcsa.latency__ns(active_bits=macro.adc_bits)
    assert macro.latency__ns(adc_active_bits=macro.adc_bits) == detect__ns * macro.scan_num
