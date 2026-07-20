"""Static-leakage instance-multiplicity + reconciliation test for the simplified tile.

The readout circuits are column-MUX time-multiplexed, so every reporter's
static leakage must equal its GENUINE per-circuit ``leakage_per_inst__uW``
times the REAL shared instance count derived from the geometry (never the raw
column count), and the profiler's static total must equal the sum of the
per-reporter products. The kernel mirrors / subtractor are non-reporters:
their silicon is the xbar's lumped ``leakage_per_inst__uW``, whose breakdown
is pinned against the TOML derivation comment:

    lump = Control 187.25 + p-stage 32 x 4.609375 + n-stage 8 x 12.825
         + subtractor 4 x 5.450625 = 459.1525 uW   (at the canonical tile)

Asserts, at the canonical geometry (col_num = 256 -> n_lane = 16, n_io = 4):

  (a) each reporter's collected leakage == per-inst x inst_count with the
      geometry-derived counts (core (pure array) x1, bl_clamp x 2*n_lane,
      sl_driver x phys_col, adc x n_io, reference x1, macro lump x1);
  (b) the non-reporters (p/n mirrors, subtractor, cell) are ABSENT from the
      static walk;
  (c) the profiler total == the sum of the TOML-seeded breakdown;

and, across geometries (col_num = 64 / 128 / 256 by frozen-dataclass replace,
all exact divisors of mux_factor / io_col_num — strict divisibility, no
clamping):

  (d) the per-circuit config seeds are byte-identical (geometry-independent)
      while the counts scale with the real lane / IO derivation — bl_clamp
      with ``2 * (col_num // mux_factor)``, the ADC with ``n_io`` — and the
      shared reference count stays 1.

CPU-only; the static walk needs no forward pass, so this is millisecond-fast
apart from the builds.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest
import torch
import torch._dynamo

from neurox.common.profiler import NeuroxProfiler
from neurox.primitive.macro.cim import CimMacro
from neurox.works.macro.cim.isub_iadc_1t1r.macro import IsubIadc1t1rCimMacro
from tests.works.macro.cim.isub_iadc_1t1r._utils import (
    CONFIG_PATH,
    DEF_COL_NUM,
    load_all_off_policy,
    load_config,
)

# --- The TOML derivation comments, pinned (params/default.toml) ---
# Xbar lump breakdown at the canonical tile (n_lane = 16, n_io = 4).
_CONTROL__uW = 187.25
_P_STAGE_PER_DEVICE__uW = 4.609375
_N_STAGE_PER_DEVICE__uW = 12.825
_SUB_PER_DEVICE__uW = 5.450625
_N_LANE = 16
_N_IO = 4
_LUMP__uW = (
    _CONTROL__uW
    + 2 * _N_LANE * _P_STAGE_PER_DEVICE__uW
    + 2 * _N_IO * _N_STAGE_PER_DEVICE__uW
    + _N_IO * _SUB_PER_DEVICE__uW
)
# Genuine per-circuit reporter seeds. The Reference seed carries the 31-tap
# (5-bit) replica-bank scaling of the paper's 7-tap share:
# 151.97625 x 31 / 7 uW.
_CORE__uW = 1.0
_CABLC_PER_INST__uW = 5.96875
_ADC_PER_INST__uW = 14.90625
_REFERENCE_PER_INST__uW = 151.97625 * 31 / 7


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — module construction only, but keep the suite uniform."""
    with torch._dynamo.config.patch(disable=True):
        yield


def _build(col_num: int) -> IsubIadc1t1rCimMacro:
    """Build the tile at ``col_num`` from the canonical TOMLs.

    Loads ``params/default.toml`` once and overrides only ``col_num`` via the
    frozen-dataclass ``replace``, so every per-circuit leakage seed is
    byte-identical across geometries — exactly what check (d) relies on.
    ``col_num`` must divide exactly by the canonical ``mux_factor`` /
    ``io_col_num`` (strict divisibility).
    """
    config = load_config(CONFIG_PATH)
    if col_num != config.col_num:
        config = dataclasses.replace(config, col_num=col_num)
    xbar = CimMacro.from_config(
        config=config,
        policy=load_all_off_policy(),
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    assert isinstance(xbar, IsubIadc1t1rCimMacro)
    return xbar


def _static_by_name(xbar: IsubIadc1t1rCimMacro) -> dict[str, float]:
    return {r.qualified_name: r.leakage_power__uW for r in NeuroxProfiler.collect_static(xbar)}


def test_static_totals_match_toml_breakdown() -> None:
    """Canonical tile: per-reporter leakage == per-inst x real count, total == breakdown."""
    xbar = _build(DEF_COL_NUM)
    leak = _static_by_name(xbar)

    # --- The xbar lump matches its TOML derivation comment ---
    assert xbar.config.leakage_per_inst__uW == pytest.approx(_LUMP__uW)
    assert leak[""] == pytest.approx(_LUMP__uW)  # root: the xbar's own lump, inst_count = 1

    # --- (a) per-reporter leakage = genuine per-inst x geometry-derived count ---
    phys_col = xbar.config.phys_col_num
    n_cablc = 2 * (xbar.config.col_num // xbar.config.mux_factor)
    assert xbar.bl_clamp.inst_count == n_cablc == 2 * _N_LANE
    assert leak["core"] == pytest.approx(_CORE__uW * 1)
    assert leak["bl_clamp"] == pytest.approx(_CABLC_PER_INST__uW * n_cablc)
    assert leak["sl_driver"] == pytest.approx(0.0)
    assert xbar.sl_driver.inst_count == phys_col
    assert leak["clamp_ref"] == pytest.approx(0.0)
    assert leak["wl_dac"] == pytest.approx(0.0)
    assert xbar.bl_adc.inst_count == xbar.n_io == _N_IO
    assert leak["bl_adc"] == pytest.approx(_ADC_PER_INST__uW * xbar.n_io)
    assert xbar.reference.inst_count == 1
    assert leak["reference"] == pytest.approx(_REFERENCE_PER_INST__uW * 1)

    # --- (b) non-reporters are absent from the static walk (rolled up) ---
    for non_reporter in ("p_mirror", "n_mirror", "subtractor", "core.cell"):
        assert non_reporter not in leak, f"non-reporter {non_reporter} leaked into the static walk"

    # --- (c) the profiler total equals the breakdown sum ---
    expected_total__uW = (
        _LUMP__uW + _CORE__uW + _CABLC_PER_INST__uW * n_cablc + _ADC_PER_INST__uW * xbar.n_io + _REFERENCE_PER_INST__uW
    )
    total__uW = NeuroxProfiler.analyze_static(xbar).leakage_power__uW
    assert total__uW == pytest.approx(sum(leak.values()))
    assert total__uW == pytest.approx(expected_total__uW)

    print(f"\n[leakage] canonical-tile static total = {total__uW:.6f} uW")
    for name in sorted(leak, key=lambda n: -leak[n]):
        print(f"  {name or '<xbar lump>':<16s} {leak[name]:>12.6f}")


def test_leakage_scales_with_real_shared_instance_count() -> None:
    """Counts scale with the lane / IO derivation, seeds stay geometry-independent."""
    small = _build(64)  # n_lane = 4, n_io = 1
    mid = _build(128)  # n_lane = 8, n_io = 2
    full = _build(256)  # n_lane = 16, n_io = 4

    assert (small.n_lane, small.n_io) == (4, 1)
    assert (mid.n_lane, mid.n_io) == (8, 2)
    assert (full.n_lane, full.n_io) == (16, 4)

    for xbar in (small, mid, full):
        cfg = xbar.config
        # (d) the per-circuit seeds are byte-identical across geometries.
        assert cfg.bl_clamp_config.leakage_per_inst__uW == _CABLC_PER_INST__uW
        assert cfg.adc_config.leakage_per_inst__uW == _ADC_PER_INST__uW
        assert cfg.reference_config.leakage_per_inst__uW == _REFERENCE_PER_INST__uW

        leak = _static_by_name(xbar)
        n_cablc = 2 * (cfg.col_num // cfg.mux_factor)
        assert xbar.bl_clamp.inst_count == n_cablc
        assert leak["bl_clamp"] == pytest.approx(_CABLC_PER_INST__uW * n_cablc)
        assert xbar.bl_adc.inst_count == xbar.n_io
        assert leak["bl_adc"] == pytest.approx(_ADC_PER_INST__uW * xbar.n_io)
        # The shared reference is one per tile at EVERY geometry.
        assert xbar.reference.inst_count == 1
        assert leak["reference"] == pytest.approx(_REFERENCE_PER_INST__uW)

    # The scaling is lane / IO derived, never the raw 4x column growth.
    assert _static_by_name(mid)["bl_clamp"] == pytest.approx(2 * _static_by_name(small)["bl_clamp"])
    assert _static_by_name(full)["bl_clamp"] == pytest.approx(4 * _static_by_name(small)["bl_clamp"])
    assert _static_by_name(full)["bl_adc"] == pytest.approx(4 * _static_by_name(small)["bl_adc"])
