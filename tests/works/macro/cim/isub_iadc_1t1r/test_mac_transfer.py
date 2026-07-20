"""Eager integer-MAC transfer test for the simplified crossbar tile (device-threaded).

Verifies the calibrated read chain decodes the ternary integer MAC exactly in
the row-active-phase regime: the per-phase signed-magnitude codes equal
``clamp(x_p . w_p, -mag_max, mag_max)`` bit-exactly per phase (the intended
lossy per-phase magnitude clip — ``mag_max = 7`` on the 3-bit tiny fixture,
``31`` on the canonical 5-bit tile), and the summed codes equal the
sum-of-per-phase-clamp unit-role reference — including the mixed-sign and
saturating-phase cases where ``Q(sum) != sum(Q)``. The per-row HRS leakage
cancels inside each logical column's P/N pair (the subtractor takes the
polarity difference), so each active ``+1`` / ``-1`` row contributes exactly
one signed per-phase MAC unit.

Three geometries are exercised:

  * the TINY overlay tile (``tests/tiny_xbar.toml``: ``col_num = 2``,
    ``row_num = 16``, ``active_row_num = 8`` -> P = 2, ``n_lane = 2``) whose
    shipped ``ref_levels__uA`` are the calibrated values at that geometry —
    exhaustive per-phase sweeps, saturating phases, mixed signs across
    phases, and a random-batch bit-exactness gate;
  * an INLINE-OVERLAY tile written to a throwaway temp TOML (the
    parent-scheme mechanism, including the ``active_row_num`` line) at the
    same geometry class; the analog grid is geometry-dependent, so the tile
    re-probes its own mid-point thresholds in-test and rebuilds with them
    (``dataclasses.replace`` on the frozen config);
  * the CANONICAL default geometry (``col_num = 256``, ``row_num = 256``,
    ``active_row_num = 64`` -> P = 4, 5-bit magnitude + sign, two operating
    modes) with the shipped battery-calibrated thresholds —
    battery-consistent random ternary patterns plus above-range clamping
    columns, run per mode on GPU (a default-geometry solve is a GPU-scale
    workload).

Determinism comes from the ``all_off`` policy (every nonideality sigma
disabled): the SAR quantizer is a deterministic hard-threshold comparator, so
``.eval()`` is nn.Module hygiene, not the source of determinism. Runs eagerly
(dynamo disabled).
"""

from __future__ import annotations

import itertools
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import torch._dynamo
from torch import Tensor

from neurox.primitive.analog.adc_common import AdcOperationPoint
from neurox.works.macro.cim.isub_iadc_1t1r.macro import IsubIadc1t1rCimMacro
from tests.works.macro.cim.isub_iadc_1t1r._utils import (
    CONFIG_PATH,
    DEF_ACTIVE_ROW_NUM,
    DEF_ADC_BITS,
    DEF_ADC_MODE_NUM,
    DEF_COL_NUM,
    DEF_MAG_MAX,
    DEF_PHASE_NUM,
    DEF_ROW_NUM,
    MAG_MAX,
    TINY_ACTIVE_ROW_NUM,
    TINY_COL_NUM,
    TINY_OVERLAY_PATH,
    TINY_PHASE_NUM,
    TINY_ROW_NUM,
    build_tile,
    decode,
    load_config,
    masked_planes,
    per_phase_clamp_reference,
    probe_i_sub,
    tile_device,
    with_ref_levels,
)


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is ``@torch.compile``; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


# ---------------------------------------------------------------------------
# Build + probe helpers
# ---------------------------------------------------------------------------


def _build_tiny(device: torch.device) -> IsubIadc1t1rCimMacro:
    """Fabricated tiny tile with its shipped calibrated thresholds."""
    return build_tile(load_config(TINY_OVERLAY_PATH, CONFIG_PATH), device=device)


def _build_inline_overlay_reprobe(device: torch.device) -> IsubIadc1t1rCimMacro:
    """Build an inline-overlay tile with in-test re-probed mid-point thresholds.

    The geometry overlay — including the mandatory ``active_row_num`` and the
    strict-divisibility ``mux_factor`` / ``io_col_num`` declarations — is
    written to a throwaway temp file and merged first-wins on top of the
    canonical config (the parent-scheme mechanism). A first build probes the
    analog ``I_SUB(M)`` grid (thresholds do not matter before the ADC); the
    rebuild installs the grid's mid-points as this geometry's calibrated
    thresholds.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write(
            '[cim_macro]\n_neurox_class = "IsubIadc1t1rCimMacroConfig"\n'
            f"col_num = {TINY_COL_NUM}\nrow_num = {TINY_ROW_NUM}\n"
            f"active_row_num = {TINY_ACTIVE_ROW_NUM}\n"
            "mux_factor = 1\nio_col_num = 2\n"
            # 3-bit single-mode fixture pinning (the canonical config is a
            # 5-bit multi-mode bank; the overlay replaces the quantizer
            # width, sensing steps, and calibration list wholesale).
            "[[cim_macro.adc_calibration]]\n"
            "adc_mode = 0\nadc_bits = 3\nrescale_factor = 1.0\n"
            "[cim_macro.adc_config]\nn_bits = 3\n"
            "step_latency__ns = [3.16, 3.07, 3.11]\n"
            # Seed 7-tap ladder for the probe build; the rebuild installs
            # the re-probed mid-points via with_ref_levels.
            "ref_levels__uA = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]\n"
            "[cim_macro.reference_config]\n"
            "i_refs__uA = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]\n"
        )
        overlay = Path(f.name)
    try:
        probe_xbar = build_tile(load_config(overlay, CONFIG_PATH), device=device)
        grid = _probe_i_sub_grid(probe_xbar, m_max=TINY_ACTIVE_ROW_NUM)
        refs = _midpoint_refs(grid)
        return build_tile(with_ref_levels(load_config(overlay, CONFIG_PATH), refs), device=device)
    finally:
        overlay.unlink(missing_ok=True)


def _probe_i_sub_grid(xbar: IsubIadc1t1rCimMacro, *, m_max: int) -> list[float]:
    """Probe the analog ``I_SUB(M)`` grid [uA] at the ADC input, M = 0..m_max.

    Programs logical column 0 all ``+1`` (others 0) and drives exactly ``M``
    sub-phase-0 rows per batch entry (``m_max <= max_active_rows``, so the
    whole grid lives inside one conversion — the per-conversion range). The
    zero-masked planes go through the modeled chain up to the ADC input
    via :func:`probe_i_sub`; plane 0 of IO 0 carries the grid. ``all_off``
    makes the probe deterministic. NOTE: reprograms the tile.
    """
    assert m_max <= xbar.max_active_rows
    device = tile_device(xbar)
    row_num = xbar.config.row_num

    w = torch.zeros((xbar.col_num, 1, row_num), dtype=torch.long, device=device)
    w[0, 0, :] = 1
    xbar.program(w)

    x = torch.zeros((m_max + 1, row_num), dtype=torch.long, device=device)
    for m in range(m_max + 1):
        x[m, :m] = 1

    # Shape: [m_max + 1, row_num] -> [m_max + 1, P, row_num]
    x_planes = masked_planes(x, row_num=row_num, max_active_rows=xbar.max_active_rows)
    i_sub, _sign = probe_i_sub(xbar, x_planes)  # [m_max + 1, P, n_io, col/n_io]
    return [float(v) for v in i_sub[:, 0, 0, 0].cpu()]


def _midpoint_refs(grid: list[float]) -> tuple[float, ...]:
    """The 7 mid-point thresholds ``ref[k] = 0.5 * (I(k) + I(k+1))``, k = 0..6."""
    assert len(grid) >= 8
    return tuple(0.5 * (grid[k] + grid[k + 1]) for k in range(7))


def _assert_decode_matches_reference(xbar: IsubIadc1t1rCimMacro, w: Tensor, x: Tensor) -> Tensor:
    """Decode and assert per-plane codes == per-plane clamp reference; return codes."""
    out = decode(xbar, w, x)
    expected = per_phase_clamp_reference(w, x, active_row_num=xbar.max_active_rows)
    assert torch.equal(out, expected), f"per-plane decode mismatch:\n{out}\nvs reference\n{expected}"
    return out


# ---------------------------------------------------------------------------
# Tiny tile (shipped thresholds): sweeps, saturating phases, mixed signs
# ---------------------------------------------------------------------------


def test_tiny_grid_monotone_and_thresholds_consistent(device: torch.device) -> None:
    """The re-probed phase-0 ``I_SUB(M)`` grid is monotone and zero-anchored."""
    xbar = _build_tiny(device)
    grid = _probe_i_sub_grid(xbar, m_max=TINY_ACTIVE_ROW_NUM)
    assert grid[0] == pytest.approx(0.0, abs=1e-3)  # M = 0: HRS leakage cancels
    assert all(b > a for a, b in itertools.pairwise(grid)), f"non-monotone I_SUB grid: {grid}"
    # The shipped thresholds separate the grid points: I(M) decodes to code M.
    assert xbar.config.reference_config.i_refs__uA == xbar.config.adc_config.ref_levels__uA  # single source of truth
    shipped = xbar.config.adc_config.ref_levels__uA[0]  # single-mode fixture: mode row 0
    for m in range(TINY_ACTIVE_ROW_NUM + 1):
        expected_code = min(m, MAG_MAX)
        below = sum(1 for r in shipped if grid[m] > r)
        assert below == expected_code, f"M={m}: I_SUB {grid[m]:.6f} uA decodes {below}, expect {expected_code}"


def test_tiny_row_sweep_across_phases(device: torch.device) -> None:
    """A +1 / -1 column pair swept over M = 0..16 active rows, crossing phases.

    Row ``m`` of the batch drives the first ``m`` word lines, so the sweep
    walks phase 0 through 0..8 (saturating at 7 for the 8-row full phase) and
    then fills phase 1. Per-phase codes and the phase-summed codes both match
    the clamp reference.
    """
    xbar = _build_tiny(device)
    w = torch.zeros((TINY_COL_NUM, 1, TINY_ROW_NUM), dtype=torch.long)
    w[0, 0, :] = 1
    w[1, 0, :] = -1

    x = torch.zeros((TINY_ROW_NUM + 1, TINY_ROW_NUM), dtype=torch.long)
    for m in range(TINY_ROW_NUM + 1):
        x[m, :m] = 1

    out = _assert_decode_matches_reference(xbar, w, x)
    assert tuple(out.shape) == (TINY_ROW_NUM + 1, TINY_PHASE_NUM, TINY_COL_NUM)

    # Phase-summed codes reproduce the sum-of-per-phase-clamp reference; the
    # full-drive row saturates BOTH phases (Q(sum) != sum(Q): true M = 16).
    summed = out.sum(dim=-2)
    a = TINY_ACTIVE_ROW_NUM
    for m in range(TINY_ROW_NUM + 1):
        expected = min(m, a, MAG_MAX) + min(max(m - a, 0), MAG_MAX)
        assert int(summed[m, 0]) == expected
        assert int(summed[m, 1]) == -expected
    assert int(summed[-1, 0]) == 2 * MAG_MAX  # 14, not the true 16


def test_tiny_saturating_and_mixed_sign_phases(device: torch.device) -> None:
    """Per-phase saturation and opposite-sign phases: expected = per-phase clamps.

    Column patterns (full word-line drive):
      * ``+1`` on every row -> partials (8, 8) -> codes (7, 7), sum 14;
      * ``+1`` phase-0 rows, ``-1`` phase-1 rows -> partials (8, -8) -> codes
        (7, -7), sum 0 (a saturating phase of EACH sign);
      * ``+1`` phase-0 rows, ``-1`` on 3 phase-1 rows -> partials (8, -3) ->
        codes (7, -3), sum 4 (one saturating, one linear, mixed signs).
    """
    xbar = _build_tiny(device)
    a = TINY_ACTIVE_ROW_NUM
    x = torch.ones((TINY_ROW_NUM,), dtype=torch.long)

    cases: list[tuple[Tensor, list[list[int]]]] = []

    w_all = torch.zeros((TINY_COL_NUM, 1, TINY_ROW_NUM), dtype=torch.long)
    w_all[0, 0, :] = 1
    cases.append((w_all, [[7, 0], [7, 0]]))

    w_anti = torch.zeros((TINY_COL_NUM, 1, TINY_ROW_NUM), dtype=torch.long)
    w_anti[0, 0, :a] = 1
    w_anti[0, 0, a:] = -1
    cases.append((w_anti, [[7, 0], [-7, 0]]))

    w_mixed = torch.zeros((TINY_COL_NUM, 1, TINY_ROW_NUM), dtype=torch.long)
    w_mixed[0, 0, :a] = 1
    w_mixed[0, 0, a : a + 3] = -1
    cases.append((w_mixed, [[7, 0], [-3, 0]]))

    for w, expected_codes in cases:
        out = _assert_decode_matches_reference(xbar, w, x)
        assert out.tolist() == expected_codes, f"expected {expected_codes}; got {out.tolist()}"


def test_tiny_random_batch_bit_exact(device: torch.device) -> None:
    """Random ternary weights x random binary batch == per-phase clamp reference."""
    xbar = _build_tiny(device)
    torch.manual_seed(7)
    w = torch.randint(-1, 2, (TINY_COL_NUM, 1, TINY_ROW_NUM), dtype=torch.long)
    x = torch.randint(0, 2, (16, TINY_ROW_NUM), dtype=torch.long)
    _assert_decode_matches_reference(xbar, w, x)


# ---------------------------------------------------------------------------
# Inline-overlay tile (in-test re-probed thresholds): transfer sweeps
# ---------------------------------------------------------------------------


def _single_column_sweep(xbar: IsubIadc1t1rCimMacro, sign: int) -> list[tuple[int, list[int]]]:
    """(true_M, per-phase codes of column 0) for M = 0..row_num rows of ``sign``."""
    row_num = xbar.config.row_num
    x = torch.ones((row_num,), dtype=torch.long)
    sweep: list[tuple[int, list[int]]] = []
    for m in range(row_num + 1):
        w = torch.zeros((TINY_COL_NUM, 1, row_num), dtype=torch.long)
        w[0, 0, :m] = sign
        out = _assert_decode_matches_reference(xbar, w, x)
        assert tuple(out.shape) == (TINY_PHASE_NUM, TINY_COL_NUM)
        sweep.append((sign * m, out[:, 0].tolist()))
    return sweep


def test_overlay_positive_transfer_monotone_and_saturating(device: torch.device) -> None:
    """+1 column: phase-summed decode is monotone and saturates at 2 x 7."""
    xbar = _build_inline_overlay_reprobe(device)
    sweep = _single_column_sweep(xbar, sign=+1)

    summed = [sum(codes) for _m, codes in sweep]
    assert summed == sorted(summed), f"non-monotone decode: {summed}"
    assert max(m for m, _ in sweep) > MAG_MAX  # the sweep crosses saturation
    assert summed[-1] == 2 * MAG_MAX


def test_overlay_negative_transfer_monotone_and_saturating(device: torch.device) -> None:
    """-1 column: phase-summed decode is monotone decreasing, saturating at -2 x 7."""
    xbar = _build_inline_overlay_reprobe(device)
    sweep = _single_column_sweep(xbar, sign=-1)

    summed = [sum(codes) for _m, codes in sweep]
    assert summed == sorted(summed, reverse=True), f"non-monotone decode: {summed}"
    assert min(m for m, _ in sweep) < -MAG_MAX
    assert summed[-1] == -2 * MAG_MAX


def test_overlay_mixed_sign_rows_single_column(device: torch.device) -> None:
    """Disjoint +1 / -1 row spans in ONE column: per-phase clamp reference holds."""
    xbar = _build_inline_overlay_reprobe(device)
    row_num = TINY_ROW_NUM
    x = torch.ones((row_num,), dtype=torch.long)

    # (a positive rows, b negative rows) on disjoint spans; spans cross the
    # phase boundary at a = 8 so mixed-sign PHASES occur too.
    for a, b in [(1, 0), (3, 2), (2, 5), (0, 4), (2, 2), (5, 5), (9, 0), (0, 9), (10, 2), (8, 8)]:
        assert a + b <= row_num
        w = torch.zeros((TINY_COL_NUM, 1, row_num), dtype=torch.long)
        w[0, 0, :a] = +1
        w[0, 0, a : a + b] = -1
        _assert_decode_matches_reference(xbar, w, x)


def test_overlay_zero_decodes_zero(device: torch.device) -> None:
    """M = 0 decodes to all-zero per-phase codes: zero weights and zero input."""
    xbar = _build_inline_overlay_reprobe(device)
    row_num = TINY_ROW_NUM

    # No active weight, full input: both polarity columns carry only the
    # common HRS leakage (I_P ~= I_N), so the sign comparator sits at the
    # balance point and the magnitude stays below the first threshold.
    w_zero = torch.zeros((TINY_COL_NUM, 1, row_num), dtype=torch.long)
    out = decode(xbar, w_zero, torch.ones((row_num,), dtype=torch.long))
    assert out.abs().sum() == 0, f"zero weight must decode 0; got {out.tolist()}"

    # Active weight, zero input: no word line on, no MAC in any phase.
    w = torch.zeros((TINY_COL_NUM, 1, row_num), dtype=torch.long)
    w[0, 0, :] = 1
    w[1, 0, :] = -1
    out = decode(xbar, w, torch.zeros((row_num,), dtype=torch.long))
    assert out.abs().sum() == 0, f"zero input must decode 0; got {out.tolist()}"


# ---------------------------------------------------------------------------
# Canonical default geometry (GPU): battery-consistent 6-bit per-mode decode
# ---------------------------------------------------------------------------


def _capped_battery_w(gen_seed: int, *, per_block: int) -> torch.Tensor:
    """Random single-sign count-capped ternary battery weights ``(col, 1, row)``."""
    torch.manual_seed(gen_seed)
    n_phase = DEF_PHASE_NUM
    a = DEF_ACTIVE_ROW_NUM
    counts = torch.randint(0, per_block + 1, (DEF_COL_NUM, n_phase, 1))
    pick = torch.argsort(torch.rand(DEF_COL_NUM, n_phase, a), dim=-1) < counts
    signs = (torch.randint(0, 2, (DEF_COL_NUM, n_phase, 1), dtype=torch.long) * 2 - 1).expand(-1, -1, a)
    w_blocks = torch.where(pick, signs, torch.zeros_like(signs))  # (col, P, A)
    return w_blocks.flatten(-2, -1).unsqueeze(-2).contiguous()  # (col, 1, row)


def _battery_x(gen_seed: int) -> torch.Tensor:
    """Drive battery: zero, full, two random d0.5, two random d0.25 planes."""
    torch.manual_seed(gen_seed)
    x = torch.zeros((6, DEF_ROW_NUM), dtype=torch.long)
    x[1, :] = 1
    x[2:4] = torch.randint(0, 2, (2, DEF_ROW_NUM), dtype=torch.long)
    x[4:] = (torch.rand(2, DEF_ROW_NUM) < 0.25).to(torch.long)
    return x


# The shipped 6-bit ladders (see the calibration note on
# adc_config.ref_levels__uA) hold every band margin positive in both
# modes, so decode is bit-exact over the full code span at default
# geometry — this test asserts exact equality everywhere so any
# recalibration or config regression that reintroduces band overlap
# fails loudly.
_SAT_COL_NUM = 3  # trailing above-range (saturating) columns in battery 1


@pytest.mark.parametrize("adc_mode", range(DEF_ADC_MODE_NUM))
def test_default_geometry_battery_decode(device: torch.device, adc_mode: int) -> None:
    """256x256 tile, P = 4: 6-bit per-phase decode vs clamp reference, per mode.

    Two batteries mirroring the calibration battery documented on
    ``adc_config.ref_levels__uA``, driven by zero / full / random <= 50%
    density word-line planes. The shipped ladders hold every band margin
    positive in both modes, so every code must equal the
    sum-of-per-phase-clamp unit-role reference bit-exactly:

      * IN-RANGE + SATURATING: random ternary columns with at most 5
        cells per (column, phase) block, plus above-range columns (48
        cells per block, past the +-31 clamp but inside the
        solver-convergent loading envelope) saturating both signs and a
        phase-antisymmetric pair. Per-phase codes AND phase-summed codes
        must be bit-exact; the above-range columns clamp at +-31 exactly
        under full word-line drive and land mid-range under the
        partial-drive planes — bit-exact there too.
      * FULL-SPAN: counts up to 31 exercise the entire band grid;
        bit-exact against the clamp reference.
    """
    if device.type == "cpu":
        pytest.skip("default-geometry solve is a GPU-scale workload (run with --device cuda)")

    xbar = build_tile(load_config(CONFIG_PATH), device=device)
    assert (xbar.col_num, xbar.config.row_num) == (DEF_COL_NUM, DEF_ROW_NUM)
    assert xbar.config.row_num // xbar.max_active_rows == DEF_PHASE_NUM
    op = AdcOperationPoint(adc_mode=adc_mode, adc_bits=DEF_ADC_BITS)
    n_phase = DEF_PHASE_NUM
    a = DEF_ACTIVE_ROW_NUM

    # --- Battery 1: in-range (<= 5 cells / block) + above-range columns ---
    w = _capped_battery_w(20200709 + adc_mode, per_block=5)
    sat = 48  # > 31 clamp, < active_row_num, inside the convergent envelope
    w[-3:, 0, :] = 0
    for p in range(n_phase):
        base = p * a
        w[-3, 0, base : base + sat] = 1
        w[-2, 0, base : base + sat] = -1
        w[-1, 0, base : base + sat] = 1 if p % 2 == 0 else -1
    x = _battery_x(377 + adc_mode)

    out = decode(xbar, w, x, adc_operation_point=op)
    assert tuple(out.shape) == (6, DEF_PHASE_NUM, DEF_COL_NUM)
    assert out.dtype in (torch.int64, torch.long)

    expected = per_phase_clamp_reference(w, x, active_row_num=a, mag_max=DEF_MAG_MAX)
    mismatch = (out != expected).nonzero()
    assert torch.equal(out, expected), (
        f"mode {adc_mode}: {mismatch.shape[0]} per-phase code mismatches; first few: "
        f"{[(tuple(i.tolist()), int(out[tuple(i.tolist())]), int(expected[tuple(i.tolist())])) for i in mismatch[:8]]}"
    )
    # Phase-summed codes reproduce the sum-of-per-phase-clamp reference.
    assert torch.equal(out.sum(dim=-2), expected.sum(dim=-2))
    # The above-range columns decode 0 with no drive and clamp at +-31 in
    # every phase under full drive.
    assert (out[0, :, -_SAT_COL_NUM:] == 0).all()
    assert (out[1, :, -3] == DEF_MAG_MAX).all()
    assert (out[1, :, -2] == -DEF_MAG_MAX).all()
    sat_signs = torch.tensor([1 if p % 2 == 0 else -1 for p in range(DEF_PHASE_NUM)])
    assert torch.equal(out[1, :, -1], sat_signs * DEF_MAG_MAX)

    # --- Battery 2: full code span (<= 31 cells / block), bit-exact ---
    w_full = _capped_battery_w(4096 + adc_mode, per_block=31)
    x_full = _battery_x(733 + adc_mode)
    out_full = decode(xbar, w_full, x_full, adc_operation_point=op)
    expected_full = per_phase_clamp_reference(w_full, x_full, active_row_num=a, mag_max=DEF_MAG_MAX)
    mismatch_full = (out_full != expected_full).nonzero()
    assert torch.equal(out_full, expected_full), (
        f"mode {adc_mode}: {mismatch_full.shape[0]} full-span code mismatches; first few: "
        f"{[(tuple(i.tolist()), int(out_full[tuple(i.tolist())]), int(expected_full[tuple(i.tolist())])) for i in mismatch_full[:8]]}"
    )
