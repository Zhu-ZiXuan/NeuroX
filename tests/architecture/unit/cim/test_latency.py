"""Local CIM timing composes serial accesses and parallel macro scan pipelines."""

from __future__ import annotations

from dataclasses import replace

import pytest

from neurox.encoding import Encoding
from neurox.primitive.digital import AdderConfig

from ._utils import build_unit, unit_config


def test_serial_accesses_wait_for_the_slowest_active_macro(monkeypatch: pytest.MonkeyPatch) -> None:
    unit = build_unit(unit_config(), matrix_shape=(5, 4))
    monkeypatch.setattr(unit.cim_macro, "_latency_per_scan__ns", lambda **kwargs: 1.3)
    # Ten slice outputs occupy [4, 4] and [2, 0] ports. The two access
    # steps finish at 6.2 and 9.8 ns, including their digital tails.
    # Three input slices each need two phases: local completion is 58.8 ns.
    assert unit._vmm_local_latency__ns(adc_active_bits=None) == pytest.approx(58.8)


@pytest.mark.parametrize(
    ("encoding", "w_value_range", "difference_circuit", "lane_num", "scan_num", "analog_scans", "macro_num"),
    [
        (Encoding.UNSIGNED, (0, 3), False, 1, 4, 8, 3),
        (Encoding.UNSIGNED, (0, 3), True, 1, 4, 8, 3),
        (Encoding.UNSIGNED, (0, 3), False, 1, 5, 8, 3),
        (Encoding.UNSIGNED, (0, 3), True, 1, 5, 8, 3),
        (Encoding.UNSIGNED, (0, 3), False, 2, 2, 4, 3),
        (Encoding.UNSIGNED, (0, 3), True, 2, 2, 4, 3),
        (Encoding.UNSIGNED, (0, 3), False, 3, 2, 4, 2),
        (Encoding.UNSIGNED, (0, 3), True, 3, 2, 4, 2),
        (Encoding.UNSIGNED, (0, 3), False, 4, 1, 2, 3),
        (Encoding.UNSIGNED, (0, 3), True, 4, 1, 2, 3),
        (Encoding.TRUE_FORM, (0, 1), True, 1, 4, 8, 3),
    ],
)
def test_adjacent_polarity_ports_share_local_update_timing_with_or_without_adder(
    encoding: Encoding,
    w_value_range: tuple[int, int],
    difference_circuit: bool,
    lane_num: int,
    scan_num: int,
    analog_scans: int,
    macro_num: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = unit_config()
    config = replace(
        config,
        cim_macro_config=replace(
            config.cim_macro_config,
            w_encoding=encoding,
            w_signed=False,
            w_value_range=w_value_range,
            lane_num=lane_num,
            scan_num=scan_num,
        ),
        w_polarity_adder_config=AdderConfig(
            bit_width=32, energy_per_op__fJ=0.0, area_per_inst__um2=0.0, leakage_per_inst__uW=0.0
        )
        if difference_circuit
        else None,
    )
    unit = build_unit(config, matrix_shape=(5, 4))
    monkeypatch.setattr(unit.cim_macro, "_latency_per_scan__ns", lambda **kwargs: 1.3)
    # Six phase/slice accesses each read two reuse slots with one final update.
    # Only complete polarity pairs occupy scans and local update circuits.
    assert unit.cim_macro.inst_count == macro_num
    expected = 6 * (analog_scans * 1.3 + 2 * config.clock_period__ns)
    assert unit._vmm_local_latency__ns(adc_active_bits=None) == pytest.approx(expected)
