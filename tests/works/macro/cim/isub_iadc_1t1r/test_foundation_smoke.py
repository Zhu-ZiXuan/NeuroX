"""CPU-only eager foundation smoke test for the isub_iadc_1t1r macro.

Covers the build / dispatch / validation foundation of
``neurox/works/macro/cim/isub_iadc_1t1r``:

  * ``CimMacroConfig.from_file`` on ``params/default.toml`` reflects into
    ``IsubIadc1t1rCimMacroConfig`` and ``CimMacro.from_config`` registry
    dispatch reaches :class:`IsubIadc1t1rCimMacro`,
  * the geometry derivation at the canonical tile: ``n_lane`` / ``n_io`` /
    the WL sub-phase count from the strict-divisibility declarations,
  * the value-domain properties expose the fixed ternary / binary / 4-bit
    signed-magnitude contract,
  * config validation rejects every strict-divisibility violation
    (``row_num % active_row_num``, ``col_num % mux_factor``,
    ``col_num % io_col_num``, ``io_col_num % mux_factor``), an out-of-range
    ``active_row_num``, a subtractor rail diverging from ``v_dd__V``,
    inconsistent ADC-vs-Reference threshold copies, and a calibration record
    at the wrong bit width,
  * :meth:`IsubIadc1t1rCimMacro.program` rejects an out-of-range ternary
    digit and a wrong weight-tensor shape with clear messages.

Construction / validation only — no DC solve — so the canonical 256x256
geometry builds in well under a second on CPU. Runs eagerly (dynamo disabled)
so the ``@torch.compile`` solver leaf is not unrolled.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest
import torch
import torch._dynamo

from neurox.primitive.analog.adc_common import AdcCalibrationRecord, AdcOperationPoint
from neurox.primitive.macro.cim import CimMacro, CimMacroPolicy
from neurox.works.macro.cim.isub_iadc_1t1r.macro import IsubIadc1t1rCimMacro
from tests.works.macro.cim.isub_iadc_1t1r._utils import (
    CONFIG_PATH,
    DEF_ACTIVE_ROW_NUM,
    DEF_ADC_BITS,
    DEF_ADC_MODE_NUM,
    DEF_COL_NUM,
    DEF_PHASE_NUM,
    DEF_ROW_NUM,
    POLICY_PATH,
    build_tile,
    load_config,
)

_PHYS_COL_NUM = 2 * DEF_COL_NUM


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is ``@torch.compile``; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


def test_from_file_registry_dispatch() -> None:
    """TOML reflection + ``CimMacro.from_config`` dispatch reach the scheme class."""
    config = load_config(CONFIG_PATH)
    policy = CimMacroPolicy.from_file(POLICY_PATH, section="policy")
    xbar = CimMacro.from_config(
        config=config,
        policy=policy,
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    assert isinstance(xbar, IsubIadc1t1rCimMacro)
    assert xbar.physical_col_num == _PHYS_COL_NUM
    # Geometry derivation from the strict-divisibility declarations.
    assert (xbar.n_lane, xbar.n_io) == (DEF_COL_NUM // config.mux_factor, DEF_COL_NUM // config.io_col_num)
    assert (xbar.n_lane, xbar.n_io) == (16, 4)
    assert config.row_num // config.active_row_num == DEF_ROW_NUM // DEF_ACTIVE_ROW_NUM == DEF_PHASE_NUM


def test_default_build_contract() -> None:
    """Direct config + policy construction exposes the fixed value-domain contract."""
    xbar = build_tile(load_config(CONFIG_PATH))
    assert isinstance(xbar, IsubIadc1t1rCimMacro)
    assert xbar.col_num == DEF_COL_NUM
    assert xbar.physical_col_num == _PHYS_COL_NUM
    assert xbar.max_active_rows == DEF_ACTIVE_ROW_NUM

    # --- Value-domain contract: ternary weight, binary input, 6-bit SM out ---
    assert xbar.x_range == (0, 1)
    assert xbar.w_digit_count == 1
    assert xbar.w_digit_radix == 2
    assert xbar.w_digit_range == (-1, 1)
    assert xbar.adc_max_bits == DEF_ADC_BITS
    assert xbar.adc_mode_num == DEF_ADC_MODE_NUM
    for mode in range(DEF_ADC_MODE_NUM):
        op = AdcOperationPoint(adc_mode=mode, adc_bits=DEF_ADC_BITS)
        assert xbar.adc_rescale_factor(op) > 0.0
    with pytest.raises(KeyError, match="adc_calibration"):
        xbar.adc_rescale_factor(AdcOperationPoint(adc_mode=DEF_ADC_MODE_NUM, adc_bits=DEF_ADC_BITS))


def test_validate_rejects_active_row_num_out_of_range() -> None:
    """``active_row_num`` must lie in ``[1, row_num]``."""
    config = load_config(CONFIG_PATH)
    with pytest.raises(ValueError, match="1 <= active_row_num"):
        dataclasses.replace(config, active_row_num=0)
    with pytest.raises(ValueError, match="1 <= active_row_num"):
        dataclasses.replace(config, active_row_num=2 * DEF_ROW_NUM)


def test_validate_rejects_active_row_num_non_divisor() -> None:
    """``row_num`` must divide exactly into uniform active phases."""
    config = load_config(CONFIG_PATH)
    # 256 % 48 != 0 -> non-uniform phases.
    with pytest.raises(ValueError, match="% active_row_num"):
        dataclasses.replace(config, active_row_num=48)


def test_validate_rejects_bad_lane_blocking() -> None:
    """``col_num % mux_factor != 0`` breaks the exact lane reshape (no clamping)."""
    config = load_config(CONFIG_PATH)
    # 256 % 48 != 0 -> non-exact front-end lane blocking.
    with pytest.raises(ValueError, match="% mux_factor"):
        dataclasses.replace(config, mux_factor=48)


def test_validate_rejects_bad_io_blocking() -> None:
    """``col_num % io_col_num != 0`` breaks the exact CIM-IO reshape (no clamping)."""
    config = load_config(CONFIG_PATH)
    with pytest.raises(ValueError, match="% io_col_num"):
        dataclasses.replace(config, io_col_num=48)


def test_validate_rejects_io_not_whole_lanes() -> None:
    """``io_col_num % mux_factor != 0`` breaks the exact lane-to-IO regroup."""
    config = load_config(CONFIG_PATH)
    # col_num = 256 divides by 8, but io_col_num = 8 does not hold whole
    # mux_factor = 16 lanes.
    with pytest.raises(ValueError, match=r"io_col_num \(8\) % mux_factor"):
        dataclasses.replace(config, io_col_num=8)


def test_validate_rejects_subtractor_rail_divergence() -> None:
    """``subtractor_config.v_rail__V`` must equal the tile's ``v_dd__V``."""
    config = load_config(CONFIG_PATH)
    bad_sub = dataclasses.replace(config.subtractor_config, v_rail__V=config.v_dd__V + 0.1)
    with pytest.raises(ValueError, match=r"v_rail__V .* == v_dd__V"):
        dataclasses.replace(config, subtractor_config=bad_sub)


def test_validate_rejects_ref_level_mismatch() -> None:
    """ADC threshold copy must equal the CurrentReference taps (single source of truth)."""
    config = load_config(CONFIG_PATH)
    shifted = tuple(tuple(v + 0.05 for v in row) for row in config.adc_config.ref_levels__uA)
    bad_adc = dataclasses.replace(config.adc_config, ref_levels__uA=shifted)
    with pytest.raises(ValueError, match="single source of truth"):
        dataclasses.replace(config, adc_config=bad_adc)


def test_validate_rejects_calibration_bits_mismatch() -> None:
    """A calibration record at a bit width the quantizer cannot produce is rejected."""
    config = load_config(CONFIG_PATH)
    bad_record = AdcCalibrationRecord(adc_mode=0, adc_bits=4, rescale_factor=1.0)
    with pytest.raises(ValueError, match=r"adc_config\.n_bits"):
        dataclasses.replace(config, adc_calibration=(bad_record,))


def test_program_rejects_out_of_range_digit() -> None:
    """A non-ternary digit fails fast with a message naming ``w_digit_range``."""
    xbar = build_tile(load_config(CONFIG_PATH))
    w = torch.zeros((DEF_COL_NUM, 1, DEF_ROW_NUM), dtype=torch.long)
    w[0, 0, 0] = 2  # outside {-1, 0, +1}
    with pytest.raises(ValueError, match="w_digit_range"):
        xbar.program(w)


def test_program_rejects_bad_shape() -> None:
    """A weight tensor without the size-1 digit axis is rejected up front."""
    xbar = build_tile(load_config(CONFIG_PATH))
    w = torch.zeros((DEF_COL_NUM, DEF_ROW_NUM), dtype=torch.long)  # missing digit axis
    with pytest.raises(ValueError, match=r"program\(\) expects"):
        xbar.program(w)
