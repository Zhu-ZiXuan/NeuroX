"""Static-leakage instance-multiplicity + reconciliation test for the simplified tile.

The readout circuits are column-MUX time-multiplexed, so every reporter's
static leakage must equal its GENUINE per-circuit ``leakage_per_inst__uW``
times the REAL shared instance count derived from the geometry (never the raw
column count), and the profiler's static total must equal the sum of the
per-reporter products. The kernel mirrors are non-reporters: their silicon is
the xbar's lumped ``leakage_per_inst__uW``.

Asserts, on the hand-built tiny witness config at two SMALL geometries
(``col_num = 2`` and ``4`` via frozen-dataclass replace — exact divisors of
``mux_factor`` / ``io_col_num``, strict divisibility, no clamping):

  (a) each reporter's collected leakage == the config's own per-inst seed x
      the geometry-derived count (core (pure array) x1, bl_clamp x
      ``2 * (col_num // mux_factor)``, sl_driver x ``2 * col_num``,
      subtractor x n_io, adc x n_io, reference x1, macro lump x1);
  (b) the non-reporters (p/n mirrors, cell) are ABSENT from the static walk;
  (c) the profiler total == the sum derived from the config seeds;
  (d) across the two geometries the per-circuit config seeds are identical
      (geometry-independent) while the counts scale with the real lane / IO
      derivation, and the shared reference count stays 1.

CPU-only; the static walk needs no forward pass, so this is millisecond-fast
apart from the builds.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest
import torch._dynamo

from neurox.common.profiler import NeuroxProfiler
from neurox.works.macro.cim.isub_iadc_1t1r.macro import IsubIadc1t1rCimMacro
from tests.works.macro.cim.isub_iadc_1t1r._utils import build_tile, build_tiny_config

_SMALL_COL_NUM = 2
_LARGE_COL_NUM = 4


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — module construction only, but keep the suite uniform."""
    with torch._dynamo.config.patch(disable=True):
        yield


def _build(col_num: int) -> IsubIadc1t1rCimMacro:
    """Build the witness tile at ``col_num`` (only the geometry differs).

    Overrides only ``col_num`` via the frozen-dataclass ``replace``, so every
    per-circuit leakage seed is identical across geometries — exactly what
    check (d) relies on. ``col_num`` must divide exactly by the witness
    ``mux_factor`` / ``io_col_num`` (strict divisibility).
    """
    config = build_tiny_config()
    if col_num != config.col_num:
        config = dataclasses.replace(config, col_num=col_num)
    return build_tile(config)


def _static_by_name(xbar: IsubIadc1t1rCimMacro) -> dict[str, float]:
    return {r.qualified_name: r.leakage_power__uW for r in NeuroxProfiler.collect_static(xbar)}


def _assert_reporter_products(xbar: IsubIadc1t1rCimMacro) -> float:
    """Check (a)-(c) at one geometry; return the collected static total [uW]."""
    cfg = xbar.config
    leak = _static_by_name(xbar)

    # --- (a) per-reporter leakage = config per-inst seed x geometry-derived count ---
    n_cablc = 2 * (cfg.col_num // cfg.mux_factor)
    n_io = cfg.col_num // cfg.io_col_num
    assert leak[""] == pytest.approx(cfg.leakage_per_inst__uW)  # root: the xbar's own lump, x1
    assert leak["core"] == pytest.approx(cfg.array_config.leakage_per_inst__uW * 1)
    assert xbar.bl_clamp.inst_count == n_cablc
    assert leak["bl_clamp"] == pytest.approx(cfg.bl_clamp_config.leakage_per_inst__uW * n_cablc)
    assert xbar.sl_driver.inst_count == cfg.phys_col_num == 2 * cfg.col_num
    assert leak["sl_driver"] == pytest.approx(cfg.sl_driver_config.leakage_per_inst__uW * cfg.phys_col_num)
    assert leak["clamp_ref"] == pytest.approx(cfg.clamp_ref_config.leakage_per_inst__uW)
    assert leak["wl_dac"] == pytest.approx(cfg.wl_dac_config.leakage_per_inst__uW * cfg.row_num)
    assert xbar.subtractor.inst_count == n_io == xbar.n_io
    assert leak["subtractor"] == pytest.approx(cfg.subtractor_config.leakage_per_inst__uW * n_io)
    assert xbar.bl_adc.inst_count == n_io
    assert leak["bl_adc"] == pytest.approx(cfg.adc_config.leakage_per_inst__uW * n_io)
    assert xbar.reference.inst_count == 1
    assert leak["reference"] == pytest.approx(cfg.reference_config.leakage_per_inst__uW * 1)

    # --- (b) non-reporters are absent from the static walk (rolled up) ---
    for non_reporter in ("p_mirror", "n_mirror", "core.cell"):
        assert non_reporter not in leak, f"non-reporter {non_reporter} leaked into the static walk"

    # --- (c) the profiler total equals the config-derived breakdown sum ---
    expected_total__uW = (
        cfg.leakage_per_inst__uW
        + cfg.array_config.leakage_per_inst__uW
        + cfg.bl_clamp_config.leakage_per_inst__uW * n_cablc
        + cfg.sl_driver_config.leakage_per_inst__uW * cfg.phys_col_num
        + cfg.clamp_ref_config.leakage_per_inst__uW
        + cfg.wl_dac_config.leakage_per_inst__uW * cfg.row_num
        + cfg.subtractor_config.leakage_per_inst__uW * n_io
        + cfg.adc_config.leakage_per_inst__uW * n_io
        + cfg.reference_config.leakage_per_inst__uW
    )
    total__uW = NeuroxProfiler.analyze_static(xbar).leakage_power__uW
    assert total__uW == pytest.approx(sum(leak.values()))
    assert total__uW == pytest.approx(expected_total__uW)
    return total__uW


def test_static_totals_match_config_derivation() -> None:
    """Witness tile: per-reporter leakage == per-inst x real count, total == derived sum."""
    xbar = _build(_SMALL_COL_NUM)
    total__uW = _assert_reporter_products(xbar)
    assert total__uW > 0.0


def test_leakage_scales_with_real_shared_instance_count() -> None:
    """Counts scale with the lane / IO derivation, seeds stay geometry-independent."""
    small = _build(_SMALL_COL_NUM)  # n_lane = 2, n_io = 1
    large = _build(_LARGE_COL_NUM)  # n_lane = 4, n_io = 2
    assert (small.n_lane, small.n_io) == (2, 1)
    assert (large.n_lane, large.n_io) == (4, 2)
    ratio = _LARGE_COL_NUM // _SMALL_COL_NUM

    # (d) the per-circuit seeds are identical across geometries.
    for field in ("bl_clamp_config", "subtractor_config", "adc_config", "reference_config"):
        seed_small = getattr(small.config, field).leakage_per_inst__uW
        seed_large = getattr(large.config, field).leakage_per_inst__uW
        assert seed_small == seed_large

    # The full reporter-product law holds at both geometries.
    _assert_reporter_products(small)
    _assert_reporter_products(large)

    # The counts scale with the real lane / IO derivation.
    leak_small = _static_by_name(small)
    leak_large = _static_by_name(large)
    assert leak_large["bl_clamp"] == pytest.approx(ratio * leak_small["bl_clamp"])
    assert leak_large["subtractor"] == pytest.approx(ratio * leak_small["subtractor"])
    assert leak_large["bl_adc"] == pytest.approx(ratio * leak_small["bl_adc"])
    # The shared reference is one per tile at EVERY geometry.
    assert large.reference.inst_count == small.reference.inst_count == 1
    assert leak_large["reference"] == pytest.approx(leak_small["reference"])
