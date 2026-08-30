"""Xue2020 integer-MAC transfer tests."""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import pytest
import torch
import torch._dynamo
from torch import Tensor

from neurox.works.macro.cim.xue2020jssc import Xue2020JsscCimMacro

from ._utils import (
    MAG_MAX,
    TINY_ADC_BITS,
    TINY_INPUT_NUM,
    TINY_K,
    TINY_OUTPUT_NUM,
    build_calibrated_macro,
    decode,
    ideal_mac,
    probe_i_sub_grid,
)


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is `@torch.compile`; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


def _assert_decode_matches_ideal(macro: Xue2020JsscCimMacro, w: Tensor, x: Tensor) -> Tensor:
    """Decode `(w, x)` and assert codes == clamped ideal integer MAC; return codes."""
    out = decode(macro, w, x)
    expected = ideal_mac(w, x)
    assert torch.equal(out, expected), f"MAC decode mismatch:\n{out.tolist()}\nvs ideal\n{expected.tolist()}"
    return out


def test_grid_monotone_and_thresholds_consistent(device: torch.device) -> None:
    macro = build_calibrated_macro(device=device)
    grid = probe_i_sub_grid(macro, m_max=MAG_MAX)
    assert grid[0] == pytest.approx(0.0, abs=1e-6)  # M = 0: exact-zero HRS branch, no leakage
    assert all(b > a for a, b in itertools.pairwise(grid)), f"non-monotone I_SUB grid: {grid}"
    values = macro.config.tmcsa_iref_config.values
    assert isinstance(values, tuple)
    ladder = values[0]
    assert isinstance(ladder, tuple)
    for m in range(MAG_MAX + 1):
        below = sum(1 for r in ladder if not isinstance(r, tuple) and grid[m] > r)
        assert below == m, f"M={m}: I_SUB {grid[m]:.9f} uA decodes {below}, expect {m}"


def test_single_row_input_sweep_monotone(device: torch.device) -> None:
    macro = build_calibrated_macro(device=device)
    w = torch.zeros((TINY_INPUT_NUM, TINY_OUTPUT_NUM), dtype=torch.long)
    w[0, 0] = 1

    codes: list[int] = []
    for v in range(1 << TINY_K):
        x = torch.zeros((TINY_INPUT_NUM,), dtype=torch.long)
        x[0] = v
        out = _assert_decode_matches_ideal(macro, w, x)
        codes.append(int(out[0]))
    assert codes == list(range(1 << TINY_K)), f"non-identity input sweep: {codes}"
    assert codes == sorted(codes)  # strictly monotone in x


def test_saturation_clips_at_magnitude_max(device: torch.device) -> None:
    macro = build_calibrated_macro(device=device)
    x_full = torch.full((TINY_INPUT_NUM,), (1 << TINY_K) - 1, dtype=torch.long)

    w_pos = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), 3, dtype=torch.long)
    out_pos = _assert_decode_matches_ideal(macro, w_pos, x_full)
    assert bool((out_pos == MAG_MAX).all()), out_pos.tolist()

    w_neg = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), -3, dtype=torch.long)
    out_neg = _assert_decode_matches_ideal(macro, w_neg, x_full)
    assert bool((out_neg == -MAG_MAX).all()), out_neg.tolist()


def test_full_column_input_sweep_saturates_monotone(device: torch.device) -> None:
    macro = build_calibrated_macro(device=device)
    w = torch.zeros((TINY_INPUT_NUM, TINY_OUTPUT_NUM), dtype=torch.long)
    w[:, 0] = 1

    summed: list[int] = []
    for v in range(1 << TINY_K):
        x = torch.full((TINY_INPUT_NUM,), v, dtype=torch.long)
        out = _assert_decode_matches_ideal(macro, w, x)
        summed.append(int(out[0]))
    assert summed == sorted(summed), f"non-monotone sweep: {summed}"
    assert summed[-1] == MAG_MAX


def test_zero_weight_and_zero_input_decode_zero(device: torch.device) -> None:
    macro = build_calibrated_macro(device=device)

    w_zero = torch.zeros((TINY_INPUT_NUM, TINY_OUTPUT_NUM), dtype=torch.long)
    x_some = torch.tensor([1, 2, 3, 1], dtype=torch.long)
    out = _assert_decode_matches_ideal(macro, w_zero, x_some)
    assert int(out.abs().sum()) == 0

    w_some = torch.tensor(
        [[1, 1, -1, 0], [-2, 1, 0, 0], [3, -1, 0, 0], [0, 0, 0, 0]],
        dtype=torch.long,
    ).transpose(-1, -2)
    out = _assert_decode_matches_ideal(macro, w_some, torch.zeros((TINY_INPUT_NUM,), dtype=torch.long))
    assert int(out.abs().sum()) == 0


def test_mixed_sign_columns(device: torch.device) -> None:
    macro = build_calibrated_macro(device=device)
    w = torch.tensor(
        [
            [1, 1, -1, 0],  # MAC = 1 + 2 - 1 = 2
            [-2, 1, 0, 0],  # MAC = -2 + 2 = 0
            [3, -1, 0, 0],  # MAC = 3 - 2 = 1
            [0, 0, 0, 0],  # MAC = 0
        ],
        dtype=torch.long,
    ).transpose(-1, -2)
    x = torch.tensor([1, 2, 1, 0], dtype=torch.long)
    out = _assert_decode_matches_ideal(macro, w, x)
    assert out.tolist() == [2, 0, 1, 0]


def test_random_batch_bit_exact(device: torch.device) -> None:
    macro = build_calibrated_macro(device=device)
    torch.manual_seed(7)
    w = torch.randint(-3, 4, (TINY_INPUT_NUM, TINY_OUTPUT_NUM), dtype=torch.long)
    x = torch.randint(0, 1 << TINY_K, (6, TINY_INPUT_NUM), dtype=torch.long)
    out = _assert_decode_matches_ideal(macro, w, x)
    assert tuple(out.shape) == (6, TINY_OUTPUT_NUM)


def test_w_digit_num_1_ternary_transfer(device: torch.device) -> None:
    macro = build_calibrated_macro(device=device, w_digit_num=1)
    assert macro.w_value_range == (-1, 1)

    torch.manual_seed(11)
    w = torch.randint(-1, 2, (TINY_INPUT_NUM, TINY_OUTPUT_NUM), dtype=torch.long)
    x = torch.randint(0, 1 << TINY_K, (5, TINY_INPUT_NUM), dtype=torch.long)
    out = _assert_decode_matches_ideal(macro, w, x)
    assert tuple(out.shape) == (5, TINY_OUTPUT_NUM)
    assert int(out.min()) < 0 < int(out.max())


def test_asymmetric_weight_regression_lsb_first(device: torch.device) -> None:
    macro = build_calibrated_macro(device=device)
    x = torch.tensor([1, 2, 2, 2], dtype=torch.long)  # per-row inputs
    w = torch.tensor(
        [
            [3, -2, 1, 0],  # MAC = 3*1 - 2*2 + 1*2 + 0*2 = 1
            [-3, 2, -1, 0],  # MAC = -3*1 + 2*2 - 1*2 + 0*2 = -1
            [2, 0, 3, -1],  # MAC = 2*1 + 0*2 + 3*2 - 1*2 = 6
            [1, -1, 2, 3],  # MAC = 1*1 - 1*2 + 2*2 + 3*2 = 9 -> clips at 7
        ],
        dtype=torch.long,
    ).transpose(-1, -2)
    out = _assert_decode_matches_ideal(macro, w, x)
    assert out.tolist() == [1, -1, 6, 7]


def test_lsb_first_place_value_single_row(device: torch.device) -> None:
    macro = build_calibrated_macro(device=device)
    x = torch.tensor([1, 0, 0, 0], dtype=torch.long)  # drive row 0 only, input 1
    for m in range(1, 1 << TINY_ADC_BITS):
        if m > (1 << macro.config.w_digit_num) - 1:
            break  # weight magnitude bounded by the 2-digit radix-2 envelope (max 3)
        for sign in (+1, -1):
            w = torch.zeros((TINY_INPUT_NUM, TINY_OUTPUT_NUM), dtype=torch.long)
            w[0, 0] = sign * m
            out = _assert_decode_matches_ideal(macro, w, x)
            assert int(out[0]) == sign * m, f"weight {sign * m} decoded {int(out[0])}"


def test_shared_ladder_raw_code_law(device: torch.device) -> None:
    macro = build_calibrated_macro(device=device)
    max_bits = macro.adc_bits
    gen = torch.Generator().manual_seed(0)
    w = torch.randint(-3, 4, (TINY_INPUT_NUM, TINY_OUTPUT_NUM), generator=gen, dtype=torch.long)
    x = torch.randint(0, 1 << TINY_K, (16, TINY_INPUT_NUM), generator=gen, dtype=torch.long)

    full = decode(macro, w, x, adc_active_bits=max_bits)
    for bits in range(1, max_bits + 1):
        lowered = decode(macro, w, x, adc_active_bits=bits)
        expected = torch.sign(full) * (full.abs() >> (max_bits - bits))
        assert torch.equal(lowered, expected), (
            f"bits {bits}: {lowered.tolist()} != right-shifted max-bits codes {expected.tolist()}"
        )


def test_sign_recovery_independent_of_bits(device: torch.device) -> None:
    macro = build_calibrated_macro(device=device)
    x = torch.full((TINY_INPUT_NUM,), (1 << TINY_K) - 1, dtype=torch.long)
    w = torch.zeros((TINY_INPUT_NUM, TINY_OUTPUT_NUM), dtype=torch.long)
    w[:, 0] = 3  # saturating positive column
    w[:, 1] = -3  # saturating negative column
    for bits in range(1, macro.adc_bits + 1):
        out = decode(macro, w, x, adc_active_bits=bits)
        assert int(out[0]) == (1 << bits) - 1
        assert int(out[1]) == -((1 << bits) - 1)
