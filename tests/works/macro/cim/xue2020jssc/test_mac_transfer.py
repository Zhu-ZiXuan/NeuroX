"""Eager integer-MAC transfer test for the xue2020jssc SINWP 1T1R CIM sub-array.

Verifies the calibrated read chain decodes the signed integer MAC exactly on the
hand-built near-ideal witness macro (``_utils.build_config``: ``col_num = 4`` ->
``io_num = 2`` at ``mux_factor = 2``, ``row_num = active_row_num = 4``,
``input_bit_num = 2``, 3-bit ADC). The macro consumes K-bit integer activations
directly (it runs the K serial WL sub-phases internally, LSB first) and returns
signed-magnitude codes; the reference is the CPU int64 unit-role MAC
``clamp(sum_row w * x, -MAG_MAX, MAG_MAX)`` (``MAG_MAX = 7`` — the intended lossy
3-bit magnitude clip). Coverage: probed ``I_SUB(M)`` grid monotonicity + ladder
consistency, a strictly monotone single-row input sweep over ``0..2**K - 1``,
saturation clipping at ``+-(2**adc_bits - 1)``, zero-weight / zero-input decode,
mixed-sign columns, a random-batch bit-exactness gate, a generalized
``w_digit_num = 1`` (ternary weight) transfer check, and — the LSB-first guard
per S5 — a TrueFormTranscoder-driven asymmetric-weight regression plus a focused
single-row place-value check that a reversed-but-consistent digit convention
would fail.

The analog ``I_SUB(M)`` grid is config-dependent, so the ladder is calibrated
in-code from the macro's own transfer (``_utils.build_calibrated_macro``: probe
the grid on an all-``+1`` column, install the mid-point thresholds, rebuild) — a
law-level calibration derived from the config under test. Determinism comes from
the ``all_off`` policy (every nonideality sigma disabled); the SAR quantizer is a
deterministic hard-threshold comparator. Runs eagerly (dynamo disabled).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import pytest
import torch
import torch._dynamo
from torch import Tensor

from neurox.works.macro.cim.xue2020jssc import Xue2020JsscCimMacro
from tests.works.macro.cim.xue2020jssc._utils import (
    MAG_MAX,
    TINY_ADC_BITS,
    TINY_COL_NUM,
    TINY_K,
    TINY_ROW_NUM,
    TRANSCODER,
    build_calibrated_macro,
    decode,
    ideal_mac,
    probe_i_sub_grid,
)


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is ``@torch.compile``; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


def _assert_decode_matches_ideal(macro: Xue2020JsscCimMacro, w: Tensor, x: Tensor) -> Tensor:
    """Decode ``(w, x)`` and assert codes == clamped ideal integer MAC; return codes."""
    out = decode(macro, w, x)
    expected = ideal_mac(w, x)
    assert torch.equal(out, expected), f"MAC decode mismatch:\n{out.tolist()}\nvs ideal\n{expected.tolist()}"
    return out


# ---------------------------------------------------------------------------
# Grid + threshold consistency
# ---------------------------------------------------------------------------


def test_grid_monotone_and_thresholds_consistent(device: torch.device) -> None:
    """The probed ``I_SUB(M)`` grid is monotone from 0; the installed ladder decodes it."""
    macro = build_calibrated_macro(device=device)
    grid = probe_i_sub_grid(macro, m_max=MAG_MAX)
    assert grid[0] == pytest.approx(0.0, abs=1e-6)  # M = 0: exact-zero HRS branch, no leakage
    assert all(b > a for a, b in itertools.pairwise(grid)), f"non-monotone I_SUB grid: {grid}"
    # The installed mid-points (the single CurrentReference ladder source)
    # separate the grid points: I_SUB(M) decodes to code M.
    ladder = macro.config.reference_config.i_refs__uA[0]  # single-mode witness: mode row 0
    for m in range(MAG_MAX + 1):
        below = sum(1 for r in ladder if grid[m] > r)
        assert below == m, f"M={m}: I_SUB {grid[m]:.9f} uA decodes {below}, expect {m}"


# ---------------------------------------------------------------------------
# Monotone input sweep + saturation
# ---------------------------------------------------------------------------


def test_single_row_input_sweep_monotone(device: torch.device) -> None:
    """A single ``+1`` row swept over inputs ``0..2**K - 1`` decodes strictly monotone == x."""
    macro = build_calibrated_macro(device=device)
    w = torch.zeros((TINY_COL_NUM, TINY_ROW_NUM), dtype=torch.long)
    w[0, 0] = 1  # one active weight on column 0, row 0

    codes: list[int] = []
    for v in range(1 << TINY_K):
        x = torch.zeros((TINY_ROW_NUM,), dtype=torch.long)
        x[0] = v
        out = _assert_decode_matches_ideal(macro, w, x)
        codes.append(int(out[0]))
    assert codes == list(range(1 << TINY_K)), f"non-identity input sweep: {codes}"
    assert codes == sorted(codes)  # strictly monotone in x


def test_saturation_clips_at_magnitude_max(device: torch.device) -> None:
    """Beyond ``+-(2**adc_bits - 1)`` the signed-magnitude code saturates, both signs."""
    macro = build_calibrated_macro(device=device)
    x_full = torch.full((TINY_ROW_NUM,), (1 << TINY_K) - 1, dtype=torch.long)  # x = 3 everywhere

    # All +3 weights, x = 3 -> true MAC 4*3*3 = 36, clips at +7.
    w_pos = torch.full((TINY_COL_NUM, TINY_ROW_NUM), 3, dtype=torch.long)
    out_pos = _assert_decode_matches_ideal(macro, w_pos, x_full)
    assert bool((out_pos == MAG_MAX).all()), out_pos.tolist()

    # All -3 weights -> clips at -7.
    w_neg = torch.full((TINY_COL_NUM, TINY_ROW_NUM), -3, dtype=torch.long)
    out_neg = _assert_decode_matches_ideal(macro, w_neg, x_full)
    assert bool((out_neg == -MAG_MAX).all()), out_neg.tolist()


def test_full_column_input_sweep_saturates_monotone(device: torch.device) -> None:
    """A full ``+1`` column swept over uniform inputs decodes non-decreasing into saturation."""
    macro = build_calibrated_macro(device=device)
    w = torch.zeros((TINY_COL_NUM, TINY_ROW_NUM), dtype=torch.long)
    w[0, :] = 1  # every row of column 0 active

    summed: list[int] = []
    for v in range(1 << TINY_K):
        x = torch.full((TINY_ROW_NUM,), v, dtype=torch.long)  # uniform input -> MAC = row_num * v
        out = _assert_decode_matches_ideal(macro, w, x)
        summed.append(int(out[0]))
    assert summed == sorted(summed), f"non-monotone sweep: {summed}"
    assert summed[-1] == MAG_MAX  # row_num * (2**K - 1) = 12 clips at 7


# ---------------------------------------------------------------------------
# Zero, mixed-sign, random
# ---------------------------------------------------------------------------


def test_zero_weight_and_zero_input_decode_zero(device: torch.device) -> None:
    """Zero weight (any input) and zero input (any weight) both decode all-zero."""
    macro = build_calibrated_macro(device=device)

    # No active weight, non-zero input: both polarity legs carry only the
    # exact-zero HRS branch, so I_SUB sits below the first threshold.
    w_zero = torch.zeros((TINY_COL_NUM, TINY_ROW_NUM), dtype=torch.long)
    x_some = torch.tensor([1, 2, 3, 1], dtype=torch.long)
    out = _assert_decode_matches_ideal(macro, w_zero, x_some)
    assert int(out.abs().sum()) == 0

    # Active weight, zero input: no word line on, no MAC in any column.
    w_some = torch.tensor([[1, 1, -1, 0], [-2, 1, 0, 0], [3, -1, 0, 0], [0, 0, 0, 0]], dtype=torch.long)
    out = _assert_decode_matches_ideal(macro, w_some, torch.zeros((TINY_ROW_NUM,), dtype=torch.long))
    assert int(out.abs().sum()) == 0


def test_mixed_sign_columns(device: torch.device) -> None:
    """Columns with per-row sign mixes decode the signed integer MAC exactly."""
    macro = build_calibrated_macro(device=device)
    w = torch.tensor(
        [
            [1, 1, -1, 0],  # MAC = 1 + 2 - 1 = 2
            [-2, 1, 0, 0],  # MAC = -2 + 2 = 0
            [3, -1, 0, 0],  # MAC = 3 - 2 = 1
            [0, 0, 0, 0],  # MAC = 0
        ],
        dtype=torch.long,
    )
    x = torch.tensor([1, 2, 1, 0], dtype=torch.long)
    out = _assert_decode_matches_ideal(macro, w, x)
    assert out.tolist() == [2, 0, 1, 0]


def test_random_batch_bit_exact(device: torch.device) -> None:
    """Random signed weights x a random input batch decode the clamped ideal MAC bit-exactly."""
    macro = build_calibrated_macro(device=device)
    torch.manual_seed(7)
    w = torch.randint(-3, 4, (TINY_COL_NUM, TINY_ROW_NUM), dtype=torch.long)
    x = torch.randint(0, 1 << TINY_K, (6, TINY_ROW_NUM), dtype=torch.long)  # batch (6,)
    out = _assert_decode_matches_ideal(macro, w, x)
    assert tuple(out.shape) == (6, TINY_COL_NUM)


# ---------------------------------------------------------------------------
# Generalized geometry: single-digit (ternary) weight transfer
# ---------------------------------------------------------------------------


def test_w_digit_num_1_ternary_transfer(device: torch.device) -> None:
    """A single-digit (``w_digit_num = 1``, radix 2) macro decodes the ternary-weight MAC exactly.

    The old ``w_digit_num == 2`` guard is gone: with one magnitude digit the
    DSWCT digit sum degenerates to identity, so weights live in ``{-1, 0, 1}`` and
    the read chain still resolves the signed integer MAC. The calibration helpers
    build the matching single-digit transcoder from the macro's own geometry.
    """
    macro = build_calibrated_macro(device=device, w_digit_num=1)
    assert macro.w_digit_count == 1
    assert macro.w_digit_range == (-1, 1)

    torch.manual_seed(11)
    w = torch.randint(-1, 2, (TINY_COL_NUM, TINY_ROW_NUM), dtype=torch.long)  # ternary
    x = torch.randint(0, 1 << TINY_K, (5, TINY_ROW_NUM), dtype=torch.long)  # batch (5,)
    out = _assert_decode_matches_ideal(macro, w, x)
    assert tuple(out.shape) == (5, TINY_COL_NUM)
    # Both signs are exercised somewhere in the random draw.
    assert int(out.min()) < 0 and int(out.max()) > 0


# ---------------------------------------------------------------------------
# LSB-first digit guard (per S5)
# ---------------------------------------------------------------------------


def test_asymmetric_weight_regression_lsb_first(device: torch.device) -> None:
    """TrueFormTranscoder-driven asymmetric weights decode exactly (a reversed digit order would not).

    The columns carry distinct magnitudes 0..3 of both signs, so the MSB / LSB
    place values (DSWCT ratios 0.5 / 0.25) are exercised asymmetrically. A
    reversed-but-consistent digit convention swaps the place values and mistakes
    e.g. a weight-2 (digits [0, 1] LSB-first) for a weight-1, so this bit-exact
    match is the LSB-first guard.
    """
    macro = build_calibrated_macro(device=device)
    # Digit layout must be LSB-first (digit 0 = LSB) so MSB carries 2x place value.
    d1 = TRANSCODER.encode(torch.tensor(1, dtype=torch.long), dim=-1).tolist()  # value 1 -> [1, 0]
    d2 = TRANSCODER.encode(torch.tensor(2, dtype=torch.long), dim=-1).tolist()  # value 2 -> [0, 1]
    assert d1 == [1, 0] and d2 == [0, 1], (d1, d2)

    x = torch.tensor([1, 2, 2, 2], dtype=torch.long)  # per-row inputs
    w = torch.tensor(
        [
            [3, -2, 1, 0],  # MAC = 3*1 - 2*2 + 1*2 + 0*2 = 1
            [-3, 2, -1, 0],  # MAC = -3*1 + 2*2 - 1*2 + 0*2 = -1
            [2, 0, 3, -1],  # MAC = 2*1 + 0*2 + 3*2 - 1*2 = 6
            [1, -1, 2, 3],  # MAC = 1*1 - 1*2 + 2*2 + 3*2 = 9 -> clips at 7
        ],
        dtype=torch.long,
    )
    out = _assert_decode_matches_ideal(macro, w, x)
    # Cross-check the hand-computed ideal (with the clip) explicitly.
    assert out.tolist() == [1, -1, 6, 7]


def test_lsb_first_place_value_single_row(device: torch.device) -> None:
    """A single-row weight of each magnitude decodes to that magnitude — a crisp place-value guard.

    The ladder is calibrated on a weight-1 unit; a single active row of weight m
    with input 1 must decode to m (positive and negative). Under a reversed digit
    convention the weight-2 (MSB set) reads as the weight-1 LSB place and decodes
    to 1, so this exact match pins the LSB-first mapping.
    """
    macro = build_calibrated_macro(device=device)
    x = torch.tensor([1, 0, 0, 0], dtype=torch.long)  # drive row 0 only, input 1
    for m in range(1, 1 << TINY_ADC_BITS):
        if m > (1 << macro.config.w_digit_num) - 1:
            break  # weight magnitude bounded by the 2-digit radix-2 envelope (max 3)
        for sign in (+1, -1):
            w = torch.zeros((TINY_COL_NUM, TINY_ROW_NUM), dtype=torch.long)
            w[0, 0] = sign * m
            out = _assert_decode_matches_ideal(macro, w, x)
            assert int(out[0]) == sign * m, f"weight {sign * m} decoded {int(out[0])}"
