"""CPU-only eager foundation smoke test for the isub_iadc_1t1r macro.

Covers the build / dispatch / validation foundation of
``neurox/works/macro/cim/isub_iadc_1t1r`` on the hand-built tiny witness
config (``_utils.build_tiny_config``):

  * ``CimMacro.from_config`` registry dispatch reaches
    :class:`IsubIadc1t1rCimMacro`, and the ``to_dict`` / ``from_dict``
    reflection round trip re-selects :class:`IsubIadc1t1rCimMacroConfig`
    through the family discriminator,
  * the geometry derivation laws: ``n_lane = col_num // mux_factor``,
    ``n_io = col_num // io_col_num``, ``phys_col_num = 2 * col_num``, and the
    WL sub-phase count ``row_num // active_row_num`` — all computed from the
    witness config's own values,
  * the value-domain properties expose the fixed ternary / binary
    signed-magnitude contract at the config's ADC width,
  * config validation rejects every strict-divisibility violation
    (``row_num % active_row_num``, ``col_num % mux_factor``,
    ``col_num % io_col_num``, ``io_col_num % mux_factor``), an out-of-range
    ``active_row_num``, a subtractor rail diverging from ``v_dd__V``,
    inconsistent ADC-vs-Reference threshold copies, and a calibration record
    at the wrong bit width,
  * :meth:`IsubIadc1t1rCimMacro.program` rejects an out-of-range ternary
    digit and a wrong weight-tensor shape with clear messages.

Construction / validation only — no DC solve. Runs eagerly (dynamo disabled)
so the ``@torch.compile`` solver leaf is not unrolled.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest
import torch
import torch._dynamo

from neurox.primitive.analog.adc_common import AdcCalibrationRecord, AdcOperationPoint
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig
from neurox.works.macro.cim.isub_iadc_1t1r.macro import IsubIadc1t1rCimMacro, IsubIadc1t1rCimMacroConfig
from tests.works.macro.cim.isub_iadc_1t1r._utils import (
    TINY_ADC_BITS,
    build_all_off_policy,
    build_tile,
    build_tiny_config,
)


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is ``@torch.compile``; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


def test_registry_dispatch_and_geometry_laws() -> None:
    """``CimMacro.from_config`` dispatch reaches the scheme class; geometry derives from config."""
    config = build_tiny_config()
    xbar = CimMacro.from_config(
        config=config,
        policy=build_all_off_policy(),
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    assert isinstance(xbar, IsubIadc1t1rCimMacro)
    # Geometry derivation laws over the witness config's own values.
    assert xbar.physical_col_num == 2 * config.col_num
    assert (xbar.n_lane, xbar.n_io) == (config.col_num // config.mux_factor, config.col_num // config.io_col_num)
    assert xbar.max_active_rows == config.active_row_num
    assert config.row_num % config.active_row_num == 0


def test_dict_reflection_round_trip() -> None:
    """``to_dict`` / ``from_dict`` re-selects the scheme config through the family tag."""
    config = build_tiny_config()
    restored = CimMacroConfig.from_dict(config.to_dict())
    assert isinstance(restored, IsubIadc1t1rCimMacroConfig)
    assert restored == config


def test_build_value_domain_contract() -> None:
    """Direct config + policy construction exposes the fixed value-domain contract."""
    config = build_tiny_config()
    xbar = build_tile(config)
    assert isinstance(xbar, IsubIadc1t1rCimMacro)
    assert xbar.col_num == config.col_num
    assert xbar.physical_col_num == 2 * config.col_num

    # --- Value-domain contract: ternary weight, binary input, SM out ---
    assert xbar.x_range == (0, 1)
    assert xbar.w_digit_count == 1
    assert xbar.w_digit_radix == 2
    assert xbar.w_digit_range == (-1, 1)
    assert xbar.adc_max_bits == config.adc_config.n_bits
    assert xbar.adc_mode_num == config.adc_config.mode_num
    for mode in range(config.adc_config.mode_num):
        op = AdcOperationPoint(adc_mode=mode, adc_bits=config.adc_config.n_bits)
        assert xbar.adc_rescale_factor(op) > 0.0
    with pytest.raises(KeyError, match="adc_calibration"):
        xbar.adc_rescale_factor(
            AdcOperationPoint(adc_mode=config.adc_config.mode_num, adc_bits=config.adc_config.n_bits)
        )


def test_validate_rejects_active_row_num_out_of_range() -> None:
    """``active_row_num`` must lie in ``[1, row_num]``."""
    config = build_tiny_config()
    with pytest.raises(ValueError, match="1 <= active_row_num"):
        dataclasses.replace(config, active_row_num=0)
    with pytest.raises(ValueError, match="1 <= active_row_num"):
        dataclasses.replace(config, active_row_num=2 * config.row_num)


def test_validate_rejects_active_row_num_non_divisor() -> None:
    """``row_num`` must divide exactly into uniform active phases."""
    config = build_tiny_config()
    # 16 % 3 != 0 -> non-uniform phases.
    with pytest.raises(ValueError, match="% active_row_num"):
        dataclasses.replace(config, active_row_num=3)


def test_validate_rejects_bad_lane_blocking() -> None:
    """``col_num % mux_factor != 0`` breaks the exact lane reshape (no clamping)."""
    config = build_tiny_config()
    # 2 % 4 != 0 -> non-exact front-end lane blocking.
    with pytest.raises(ValueError, match=r"col_num \(2\) % mux_factor"):
        dataclasses.replace(config, mux_factor=4)


def test_validate_rejects_bad_io_blocking() -> None:
    """``col_num % io_col_num != 0`` breaks the exact CIM-IO reshape (no clamping)."""
    config = build_tiny_config()
    with pytest.raises(ValueError, match="% io_col_num"):
        dataclasses.replace(config, io_col_num=3)


def test_validate_rejects_io_not_whole_lanes() -> None:
    """``io_col_num % mux_factor != 0`` breaks the exact lane-to-IO regroup."""
    config = build_tiny_config()
    # col_num = 2 divides by both, but io_col_num = 1 does not hold whole
    # mux_factor = 2 lanes.
    with pytest.raises(ValueError, match=r"io_col_num \(1\) % mux_factor"):
        dataclasses.replace(config, mux_factor=2, io_col_num=1)


def test_validate_rejects_subtractor_rail_divergence() -> None:
    """``subtractor_config.v_rail__V`` must equal the tile's ``v_dd__V``."""
    config = build_tiny_config()
    bad_sub = dataclasses.replace(config.subtractor_config, v_rail__V=config.v_dd__V + 0.1)
    with pytest.raises(ValueError, match=r"v_rail__V .* == v_dd__V"):
        dataclasses.replace(config, subtractor_config=bad_sub)


def test_validate_rejects_ref_level_mismatch() -> None:
    """ADC threshold copy must equal the CurrentReference taps (single source of truth)."""
    config = build_tiny_config()
    shifted = tuple(tuple(v + 0.05 for v in row) for row in config.adc_config.ref_levels__uA)
    bad_adc = dataclasses.replace(config.adc_config, ref_levels__uA=shifted)
    with pytest.raises(ValueError, match="single source of truth"):
        dataclasses.replace(config, adc_config=bad_adc)


def test_validate_rejects_calibration_bits_mismatch() -> None:
    """A calibration record at a bit width the quantizer cannot produce is rejected."""
    config = build_tiny_config()
    bad_record = AdcCalibrationRecord(adc_mode=0, adc_bits=TINY_ADC_BITS + 1, rescale_factor=1.0)
    with pytest.raises(ValueError, match=r"adc_config\.n_bits"):
        dataclasses.replace(config, adc_calibration=(bad_record,))


def test_program_rejects_out_of_range_digit() -> None:
    """A non-ternary digit fails fast with a message naming ``w_digit_range``."""
    config = build_tiny_config()
    xbar = build_tile(config)
    w = torch.zeros((config.col_num, 1, config.row_num), dtype=torch.long)
    w[0, 0, 0] = 2  # outside {-1, 0, +1}
    with pytest.raises(ValueError, match="w_digit_range"):
        xbar.program(w)


def test_program_rejects_bad_shape() -> None:
    """A weight tensor without the size-1 digit axis is rejected up front."""
    config = build_tiny_config()
    xbar = build_tile(config)
    w = torch.zeros((config.col_num, config.row_num), dtype=torch.long)  # missing digit axis
    with pytest.raises(ValueError, match=r"program\(\) expects"):
        xbar.program(w)
