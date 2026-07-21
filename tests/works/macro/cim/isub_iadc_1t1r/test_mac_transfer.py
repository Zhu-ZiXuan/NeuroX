"""Eager integer-MAC transfer test for the simplified crossbar tile (device-threaded).

Verifies the calibrated read chain decodes the ternary integer MAC exactly in
the row-active-phase regime on the hand-built tiny witness tile
(``_utils.build_tiny_config``: ``col_num = 2``, ``row_num = 16``,
``active_row_num = 8`` -> P = 2, ``n_lane = 2``): the per-phase
signed-magnitude codes equal ``clamp(x_p . w_p, -MAG_MAX, MAG_MAX)``
bit-exactly per phase (the intended lossy per-phase magnitude clip —
``MAG_MAX = 7`` at the 3-bit witness width), and the summed codes equal the
sum-of-per-phase-clamp unit-role reference — including the mixed-sign and
saturating-phase cases where ``Q(sum) != sum(Q)``. The per-row HRS leakage
cancels inside each logical column's P/N pair (the subtractor takes the
polarity difference), so each active ``+1`` / ``-1`` row contributes exactly
one signed per-phase MAC unit.

The analog ``I_SUB(M)`` grid is config-dependent, so the ladder is calibrated
in-code from the tile's own transfer (``_utils.build_calibrated_tile``: probe
the grid, install the mid-point thresholds, rebuild) — a law-level
calibration derived from the config under test. Coverage: grid monotonicity
+ threshold consistency, exhaustive row sweeps crossing phases, saturating
and mixed-sign phases, single-column monotone transfer of both signs, zero
decode, a random-batch bit-exactness gate, and a CUDA-gated tiny random
battery (skips without CUDA).

Determinism comes from the ``all_off`` policy (every nonideality sigma
disabled): the SAR quantizer is a deterministic hard-threshold comparator, so
``.eval()`` is nn.Module hygiene, not the source of determinism. Runs eagerly
(dynamo disabled).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import pytest
import torch
import torch._dynamo
from torch import Tensor

from neurox.works.macro.cim.isub_iadc_1t1r.macro import IsubIadc1t1rCimMacro
from tests.works.macro.cim.isub_iadc_1t1r._utils import (
    MAG_MAX,
    TINY_ACTIVE_ROW_NUM,
    TINY_COL_NUM,
    TINY_PHASE_NUM,
    TINY_ROW_NUM,
    build_calibrated_tile,
    decode,
    per_phase_clamp_reference,
    probe_i_sub_grid,
)


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is ``@torch.compile``; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


def _assert_decode_matches_reference(xbar: IsubIadc1t1rCimMacro, w: Tensor, x: Tensor) -> Tensor:
    """Decode and assert per-plane codes == per-plane clamp reference; return codes."""
    out = decode(xbar, w, x)
    expected = per_phase_clamp_reference(w, x, active_row_num=xbar.max_active_rows)
    assert torch.equal(out, expected), f"per-plane decode mismatch:\n{out}\nvs reference\n{expected}"
    return out


# ---------------------------------------------------------------------------
# Grid + threshold consistency
# ---------------------------------------------------------------------------


def test_grid_monotone_and_thresholds_consistent(device: torch.device) -> None:
    """The probed phase-0 ``I_SUB(M)`` grid is monotone; the ladder decodes it."""
    xbar = build_calibrated_tile(device)
    grid = probe_i_sub_grid(xbar, m_max=TINY_ACTIVE_ROW_NUM)
    assert grid[0] == pytest.approx(0.0, abs=1e-3)  # M = 0: HRS leakage cancels
    assert all(b > a for a, b in itertools.pairwise(grid)), f"non-monotone I_SUB grid: {grid}"
    # Both threshold copies stay equal (single source of truth) and the
    # installed mid-points separate the grid points: I(M) decodes to code M.
    assert xbar.config.reference_config.i_refs__uA == xbar.config.adc_config.ref_levels__uA
    ladder = xbar.config.adc_config.ref_levels__uA[0]  # single-mode witness: mode row 0
    for m in range(TINY_ACTIVE_ROW_NUM + 1):
        expected_code = min(m, MAG_MAX)
        below = sum(1 for r in ladder if grid[m] > r)
        assert below == expected_code, f"M={m}: I_SUB {grid[m]:.6f} uA decodes {below}, expect {expected_code}"


# ---------------------------------------------------------------------------
# Row sweeps, saturating phases, mixed signs
# ---------------------------------------------------------------------------


def test_row_sweep_across_phases(device: torch.device) -> None:
    """A +1 / -1 column pair swept over M = 0..16 active rows, crossing phases.

    Row ``m`` of the batch drives the first ``m`` word lines, so the sweep
    walks phase 0 through 0..8 (saturating at 7 for the 8-row full phase) and
    then fills phase 1. Per-phase codes and the phase-summed codes both match
    the clamp reference.
    """
    xbar = build_calibrated_tile(device)
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


def test_saturating_and_mixed_sign_phases(device: torch.device) -> None:
    """Per-phase saturation and opposite-sign phases: expected = per-phase clamps.

    Column patterns (full word-line drive):
      * ``+1`` on every row -> partials (8, 8) -> codes (7, 7), sum 14;
      * ``+1`` phase-0 rows, ``-1`` phase-1 rows -> partials (8, -8) -> codes
        (7, -7), sum 0 (a saturating phase of EACH sign);
      * ``+1`` phase-0 rows, ``-1`` on 3 phase-1 rows -> partials (8, -3) ->
        codes (7, -3), sum 4 (one saturating, one linear, mixed signs).
    """
    xbar = build_calibrated_tile(device)
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


def test_random_batch_bit_exact(device: torch.device) -> None:
    """Random ternary weights x random binary batch == per-phase clamp reference."""
    xbar = build_calibrated_tile(device)
    torch.manual_seed(7)
    w = torch.randint(-1, 2, (TINY_COL_NUM, 1, TINY_ROW_NUM), dtype=torch.long)
    x = torch.randint(0, 2, (8, TINY_ROW_NUM), dtype=torch.long)
    _assert_decode_matches_reference(xbar, w, x)


# ---------------------------------------------------------------------------
# Single-column transfer sweeps
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


def test_positive_transfer_monotone_and_saturating(device: torch.device) -> None:
    """+1 column: phase-summed decode is monotone and saturates at 2 x 7."""
    xbar = build_calibrated_tile(device)
    sweep = _single_column_sweep(xbar, sign=+1)

    summed = [sum(codes) for _m, codes in sweep]
    assert summed == sorted(summed), f"non-monotone decode: {summed}"
    assert max(m for m, _ in sweep) > MAG_MAX  # the sweep crosses saturation
    assert summed[-1] == 2 * MAG_MAX


def test_negative_transfer_monotone_and_saturating(device: torch.device) -> None:
    """-1 column: phase-summed decode is monotone decreasing, saturating at -2 x 7."""
    xbar = build_calibrated_tile(device)
    sweep = _single_column_sweep(xbar, sign=-1)

    summed = [sum(codes) for _m, codes in sweep]
    assert summed == sorted(summed, reverse=True), f"non-monotone decode: {summed}"
    assert min(m for m, _ in sweep) < -MAG_MAX
    assert summed[-1] == -2 * MAG_MAX


def test_mixed_sign_rows_single_column(device: torch.device) -> None:
    """Disjoint +1 / -1 row spans in ONE column: per-phase clamp reference holds."""
    xbar = build_calibrated_tile(device)
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


def test_zero_decodes_zero(device: torch.device) -> None:
    """M = 0 decodes to all-zero per-phase codes: zero weights and zero input."""
    xbar = build_calibrated_tile(device)
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
# CUDA-gated tiny random battery
# ---------------------------------------------------------------------------


def test_tiny_battery_decode_cuda() -> None:
    """Tiny-geometry random battery on CUDA: bit-exact per-phase decode.

    The same witness tile and clamp-reference law as the CPU tests, solved on
    the CUDA backend — a device-parity gate, not a canonical-geometry
    workload. Skips when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cuda = torch.device("cuda")
    xbar = build_calibrated_tile(cuda)
    torch.manual_seed(11)
    w = torch.randint(-1, 2, (TINY_COL_NUM, 1, TINY_ROW_NUM), dtype=torch.long)
    x = torch.randint(0, 2, (4, TINY_ROW_NUM), dtype=torch.long)
    _assert_decode_matches_reference(xbar, w, x)
