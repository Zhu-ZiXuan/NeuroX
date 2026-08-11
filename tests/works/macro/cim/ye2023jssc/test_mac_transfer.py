"""Eager integer-MAC transfer breadth for the ye2023jssc WH-2T1R CIM macro.

The analytic witness (``_utils.build_config``) pairs a drive-point HRS leakage equal
to the V_X = 0 floor with a seated PH0 compensation, so the RS-CSA sees exactly
``(I_unit - floor) * MAC`` and its injected reference is that same step: the code equals
the UNSIGNED integer MAC bit-exactly (``clamp(sum_in w * x, 0, 2**adc_bits - 1)``)
even though every physical column — the redundant SUBA4 plane included — carries
a nonzero leakage floor. This file adds the transfer breadth the per-module smoke
tests and ``test_macro.py`` do not cover — random input batches, zero-weight /
zero-input decode, code-max saturation (a widened witness so the MAC can exceed
the 4-bit ceiling), mixed per-row input patterns, and a per-magnitude LSB-first
place-value sweep (a reversed digit convention would decode a weight-4 as a
weight-1).

Determinism comes from the ``all_off`` policy (every kernel nonideality off) plus
a scheme that wires no stochastic source at all. Weights are UNSIGNED radix-2
digits over ``weight_radix = (1, 2, 4)`` so a single weight lives in ``[0, 7]``.
Runs eagerly (dynamo disabled) so the ``@torch.compile`` solver leaf is not
unrolled.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest
import torch
import torch._dynamo
from torch import Tensor

from ._utils import (
    MAG_MAX,
    TINY_INPUT_NUM,
    TINY_OUTPUT_NUM,
    W_MAX,
    Ye2023JsscCimMacro,
    build_config,
    build_macro,
    decode,
    ideal_mac,
)


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is ``@torch.compile``; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


def _assert_decode_matches_ideal(macro: Ye2023JsscCimMacro, w: Tensor, x: Tensor) -> Tensor:
    """Decode ``(w, x)`` and assert codes == clamped UNSIGNED integer MAC; return codes."""
    out = decode(macro, w, x)
    expected = ideal_mac(w, x)
    assert torch.equal(out.long(), expected), f"MAC decode mismatch:\n{out.tolist()}\nvs ideal\n{expected.tolist()}"
    return out


# ---------------------------------------------------------------------------
# Random + structured batches
# ---------------------------------------------------------------------------


def test_random_batch_bit_exact(device: torch.device) -> None:
    """Random unsigned weights x a random 1-bit input batch decode the clamped MAC bit-exactly."""
    macro = build_macro(build_config(), device=device)
    torch.manual_seed(7)
    w = torch.randint(0, W_MAX + 1, (TINY_INPUT_NUM, TINY_OUTPUT_NUM), dtype=torch.long, device=device)
    x = torch.randint(0, 2, (4, TINY_INPUT_NUM), dtype=torch.long, device=device)
    out = _assert_decode_matches_ideal(macro, w, x)
    assert tuple(out.shape) == (4, TINY_OUTPUT_NUM)


def test_mixed_input_patterns_bit_exact(device: torch.device) -> None:
    """Hand-picked mixed weight / input patterns decode the exact per-output MAC."""
    macro = build_macro(build_config(), device=device)
    w = torch.tensor(
        [[1, 3], [7, 0], [2, 5], [4, 4]],
        dtype=torch.long,
        device=device,
    ).transpose(-1, -2)
    x = torch.tensor([[1, 1], [1, 0], [0, 1]], dtype=torch.long, device=device)  # batch (3,)
    out = _assert_decode_matches_ideal(macro, w, x)
    # Cross-check the hand-computed MAC explicitly (no clip needed; all <= 15).
    assert out.tolist() == [[4, 7, 7, 8], [1, 7, 2, 4], [3, 0, 5, 4]]


# ---------------------------------------------------------------------------
# Zero decode
# ---------------------------------------------------------------------------


def test_zero_weight_and_zero_input_decode_zero(device: torch.device) -> None:
    """Zero weight (any input) and zero input (any weight) both decode all-zero."""
    macro = build_macro(build_config(), device=device)

    # No active weight, non-zero input: every weight cell is HRS, so the drive-point
    # leakage matches the floor the seated PH0 removes.
    w_zero = torch.zeros((TINY_INPUT_NUM, TINY_OUTPUT_NUM), dtype=torch.long, device=device)
    out = _assert_decode_matches_ideal(macro, w_zero, torch.tensor([1, 1], dtype=torch.long, device=device))
    assert int(out.abs().sum()) == 0

    # Active weight, zero input: no BL drive, so every column sits on the floor.
    w_some = torch.tensor(
        [[1, 3], [7, 0], [2, 5], [4, 4]],
        dtype=torch.long,
        device=device,
    ).transpose(-1, -2)
    out = _assert_decode_matches_ideal(macro, w_some, torch.zeros(TINY_INPUT_NUM, dtype=torch.long, device=device))
    assert int(out.abs().sum()) == 0


# ---------------------------------------------------------------------------
# Saturation at the code ceiling
# ---------------------------------------------------------------------------


def test_saturation_clips_at_code_max(device: torch.device) -> None:
    """A MAC beyond ``2**adc_bits - 1`` saturates the unsigned code at the 4-bit ceiling.

    The default witness has only two inputs (max MAC 14 < 15), so this uses a
    widened witness (four inputs) whose true MAC (4 * 7 = 28) overruns the ceiling.
    """
    input_num = 4
    cfg = dataclasses.replace(build_config(), max_active_num=input_num)
    macro = build_macro(cfg, input_num=input_num, device=device)

    w = torch.full((input_num, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)
    x = torch.ones(input_num, dtype=torch.long, device=device)
    out = _assert_decode_matches_ideal(macro, w, x)  # ideal_mac clamps to MAG_MAX
    assert bool((out == MAG_MAX).all()), out.tolist()


# ---------------------------------------------------------------------------
# LSB-first place-value guard (per magnitude)
# ---------------------------------------------------------------------------


def test_place_value_magnitude_sweep_lsb_first(device: torch.device) -> None:
    """A single active row of weight ``m`` with input 1 decodes to ``m`` for every ``m`` in ``[1, 7]``.

    Each magnitude exercises a distinct digit-plane combination over
    ``weight_radix = (1, 2, 4)``; a reversed-but-consistent digit convention swaps
    the place values (e.g. reads a weight-4, digits ``[0, 0, 1]`` LSB-first, as a
    weight-1), so these exact matches pin the LSB-first mapping across the ladder.
    """
    macro = build_macro(build_config(), device=device)
    x = torch.tensor([1, 0], dtype=torch.long, device=device)  # drive input 0 only
    for m in range(1, W_MAX + 1):
        w = torch.zeros((TINY_INPUT_NUM, TINY_OUTPUT_NUM), dtype=torch.long, device=device)
        w[0, 0] = m  # output 0 carries weight m on input 0
        out = _assert_decode_matches_ideal(macro, w, x)
        assert int(out[0]) == m, f"weight {m} decoded {int(out[0])}"
