"""Eager end-to-end smoke test for the simplified crossbar tile (device-threaded).

Builds the full folded :class:`IsubIadc1t1rCimMacro` (array + the inline
kernel readout chain) from the hand-built tiny witness config
(``_utils.build_tiny_config``: ``col_num = 2`` -> 4 physical columns,
``row_num = 16``, ``active_row_num = 8`` -> P = 2 serial WL sub-phases,
``n_lane = 2`` front-end lanes per polarity) with the in-code all-off policy,
the ADC ladder calibrated in-code from the tile's own analog transfer
(``_utils.build_calibrated_tile``).

Programs a small mixed-sign ternary weight, expands a binary activation
batch into zero-masked WL planes (engine mask formula), runs
``vec_mat_mul`` on the planes, and asserts:

  * the output is an integer per-plane code tensor with the leading order
    preserved and primitive trailing ``[col_num]`` (the sub-phase axis rides
    the leading as ``[batch, P, col_num]``), every value a 4-bit
    signed-magnitude code in ``[-7, 7]``,
  * the per-plane codes equal the unit-role reference
    ``clamp(x_p . w_p, -7, 7)`` bit-exactly (both signs exercised, not stuck
    at a rail),
  * the VMM is deterministic under ``all_off``: a second call is bit-exact,
    and an independently rebuilt + refabricated tile reproduces the output,
  * a :class:`NeuroxProfiler` report shows per-module dynamic energy for the
    forward reporters (``array``, ``bl_clamp``, ``subtractor``, ``bl_adc``)
    plus the xbar-owned clamp-drop + seam-branch events, zero-valued
    ``sl_driver`` (physical zero — direct ground tie) and ``wl_dac``
    (driver-circuit seed zero; the WL LOAD caps are array-billed) events, a
    separately-derived nonzero static leakage that reconciles as
    ``leakage_power__uW x total_latency__ns``, and NO dynamic event from the
    static-only shared CurrentReference.

Runs eagerly (dynamo disabled) so the ``@torch.compile`` solver leaf is not
unrolled — the whole test finishes in a few seconds.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
import torch._dynamo
from torch import Tensor

from neurox.common.profiler import NeuroxProfiler
from neurox.works.macro.cim.isub_iadc_1t1r.macro import IsubIadc1t1rCimMacro
from tests.works.macro.cim.isub_iadc_1t1r._utils import (
    ADC_OP,
    MAG_MAX,
    TINY_ACTIVE_ROW_NUM,
    TINY_COL_NUM,
    TINY_PHASE_NUM,
    TINY_ROW_NUM,
    build_calibrated_tile,
    masked_planes,
    per_phase_clamp_reference,
)

_PHYS_COL_NUM = 2 * TINY_COL_NUM

# Forward reporters, named as the xbar root's own traversal names them (the
# SAR quantizer leaf is named ``bl_adc`` — the inner-ADC role). The kernel
# mirrors are pure transports that log nothing of their own; the three seam
# branches and the clamp-side array-branch split are logged by the xbar
# itself (asserted separately). The shared CurrentReference is NOT here —
# it is a static-only source asserted on the static-leakage side instead.
_FORWARD_REPORTER_NAMES = ("array", "bl_clamp", "subtractor", "bl_adc")
_FORWARD_REPORTER_TYPES = ("XbarArray1t1r", "VoltageDriver", "CurrentSubtractor", "SarCurrentAdc")
_REFERENCE_BLOCK_NAME = "reference"
_REFERENCE_BLOCK_TYPE = "CurrentReference"


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is ``@torch.compile``; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


def _build(device: torch.device) -> IsubIadc1t1rCimMacro:
    """Build the fabricated calibrated tiny tile from the hand-built witness config."""
    xbar = build_calibrated_tile(device)
    assert xbar.physical_col_num == _PHYS_COL_NUM
    assert (xbar.n_lane, xbar.n_io) == (2, 1)
    assert xbar.config.row_num // xbar.max_active_rows == TINY_PHASE_NUM
    return xbar


def _mixed_sign_weight() -> Tensor:
    """A small ternary weight spanning both signs, mid-scale on both columns.

    Column 0 carries ``+1`` on 2 phase-0 rows, column 1 ``-1`` on 3 phase-0
    rows. Shape ``[col_num, w_digit_count=1, row_num]``; entries in
    ``{-1, 0, +1}``.
    """
    w = torch.zeros((TINY_COL_NUM, 1, TINY_ROW_NUM), dtype=torch.long)
    w[0, 0, :2] = 1
    w[1, 0, :3] = -1
    return w


def _activation_batch() -> Tensor:
    """A 3-row binary batch: full drive, phase-0-only drive, single row."""
    x = torch.zeros((3, TINY_ROW_NUM), dtype=torch.long)
    x[0, :] = 1  # all rows on               -> phase 0: (+2, -3), phase 1: (0, 0)
    x[1, :TINY_ACTIVE_ROW_NUM] = 1  # phase-0 rows only -> same codes
    x[2, 0] = 1  # row 0 only               -> phase 0: (+1, -1), phase 1: (0, 0)
    return x


def test_xbar_end_to_end_smoke(device: torch.device) -> None:
    """Full tile VMM: per-plane shape, signed-magnitude decode, per-block PPA."""
    xbar = _build(device)
    w = _mixed_sign_weight()
    xbar.program(w.to(device))

    x = _activation_batch()
    assert xbar.x_range == (0, 1)

    # Shape: [batch, row_num] -> [batch, P, row_num]   zero-masked WL planes
    planes = masked_planes(x.to(device), row_num=TINY_ROW_NUM, max_active_rows=TINY_ACTIVE_ROW_NUM)
    with NeuroxProfiler() as profiler, torch.no_grad():
        out = xbar.vec_mat_mul(planes, adc_operation_point=ADC_OP)
    report = profiler.report(xbar)
    out = out.cpu()

    # --- 1. Integer per-plane codes; leading [batch, P] preserved, trailing [col_num] ---
    assert out.dtype in (torch.int64, torch.long)
    assert tuple(out.shape) == (x.shape[0], TINY_PHASE_NUM, TINY_COL_NUM)

    # --- 2. Values are 4-bit signed-magnitude codes in [-7, 7] ---
    assert int(out.min()) >= -MAG_MAX
    assert int(out.max()) <= MAG_MAX

    # --- 3. Bit-exact per-plane decode; both signs, no rail ---
    expected = per_phase_clamp_reference(w, x, active_row_num=TINY_ACTIVE_ROW_NUM)
    assert torch.equal(out, expected), f"per-plane decode mismatch:\n{out}\nvs\n{expected}"
    assert int(out.min()) < 0 and int(out.max()) > 0
    assert int(out.abs().max()) < MAG_MAX

    # --- 4a. Per-module dynamic energy: forward reporters present + positive ---
    by_name = report.energy_by_name
    by_type = profiler.energy_by_type
    for block in _FORWARD_REPORTER_NAMES:
        assert block in by_name, f"missing dynamic-energy event for {block}; have {sorted(by_name)}"
        assert by_name[block] > 0.0, f"non-positive dynamic energy for {block}: {by_name[block]}"
    for block_type in _FORWARD_REPORTER_TYPES:
        assert block_type in by_type, f"missing dynamic-energy type {block_type}; have {sorted(by_type)}"
        assert by_type[block_type] > 0.0

    # The clamp-side array-branch split and the three whole seam branches are
    # logged by the xbar itself (the kernel mirrors log none): strictly
    # positive under conduction.
    xbar_owned__fJ = sum(e.dynamic_energy__fJ for e in report.energy_events if e.module is xbar)
    assert xbar_owned__fJ > 0.0, f"non-positive xbar-owned readout energy: {xbar_owned__fJ}"

    # The SL driver's per-op interface energy is a physical zero (direct
    # ground tie): its events exist but carry no energy.
    assert by_name.get("sl_driver", 0.0) == 0.0

    # The WL DAC bills only its own driver-circuit energy (seeded zero);
    # the WL LOAD caps (wire + gates) are array-billed, so the ``array``
    # energy above carries them and the DAC events are zero.
    assert by_name.get("wl_dac", 0.0) == 0.0

    # The shared CurrentReference is static-only: no dynamic event at all.
    assert _REFERENCE_BLOCK_NAME not in by_name
    assert _REFERENCE_BLOCK_TYPE not in by_type

    # --- 4b. Leakage reconciliation: static power x total latency ---
    assert profiler.total_dynamic_energy__fJ > 0.0
    assert profiler.total_latency__ns > 0.0
    assert report.static.leakage_power__uW > 0.0
    assert report.leakage_energy__fJ == pytest.approx(report.static.leakage_power__uW * profiler.total_latency__ns)

    # The CurrentReference carries its cost ENTIRELY as static leakage.
    static_leak_by_name = {r.qualified_name: r.leakage_power__uW for r in NeuroxProfiler.collect_static(xbar)}
    assert _REFERENCE_BLOCK_NAME in static_leak_by_name
    assert static_leak_by_name[_REFERENCE_BLOCK_NAME] > 0.0

    # --- 5. Determinism under all_off: bit-exact repeat + rebuild ---
    with torch.no_grad():
        again = xbar.vec_mat_mul(planes, adc_operation_point=ADC_OP)
    assert torch.equal(out, again.cpu()), "all_off VMM must be bit-exact repeatable"

    rebuilt = _build(device)
    rebuilt.program(w.to(device))
    with torch.no_grad():
        out_rebuilt = rebuilt.vec_mat_mul(planes, adc_operation_point=ADC_OP)
    assert torch.equal(out, out_rebuilt.cpu()), "all_off VMM must survive rebuild + refabricate"

    # --- Surface the per-module PPA breakdown for the runner log ---
    print("\n[xbar smoke] per-plane codes:", out.tolist())
    print(f"[xbar smoke] leakage_power_uW = {report.static.leakage_power__uW:.4f}")
    print(f"[xbar smoke] total_latency_ns = {profiler.total_latency__ns:.4f}")
    print("[xbar smoke] dynamic_energy_by_name_fJ:")
    for k in sorted(by_name, key=lambda n: -by_name[n]):
        print(f"  {k:<28s} {by_name[k]:.4f}")
