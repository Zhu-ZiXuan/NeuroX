"""Xue2020 integer-MAC transfer tests."""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.works.macro.cim.xue2020jssc import Xue2020JsscCimMacro
from tests.works.macro.cim.xue2020jssc.macro._utils import (
    TINY_INPUT_NUM,
    TINY_K,
    TINY_OUTPUT_NUM,
    build_calibrated_macro,
    decode,
    ideal_mac,
)


def _assert_decode_matches_ideal(macro: Xue2020JsscCimMacro, w: Tensor, x: Tensor) -> Tensor:
    """Decode `(w, x)` and assert codes == clamped ideal integer MAC; return codes."""
    out = decode(macro, w, x)
    expected = ideal_mac(w, x)
    assert torch.equal(out, expected), f"MAC decode mismatch:\n{out.tolist()}\nvs ideal\n{expected.tolist()}"
    return out


def test_saturation_clips_at_magnitude_max(device: torch.device) -> None:
    macro = build_calibrated_macro(device=device)
    x_full = torch.full((TINY_INPUT_NUM,), (1 << TINY_K) - 1, dtype=torch.int32)

    weight = torch.zeros((TINY_INPUT_NUM, TINY_OUTPUT_NUM), dtype=torch.int32)
    weight[:, 0] = 3
    weight[:, 1] = -3
    for bits in range(1, macro.adc_bits + 1):
        out = decode(macro, weight, x_full, adc_active_bits=bits)
        limit = (1 << bits) - 1
        assert out.tolist() == [limit, -limit, 0, 0]


def test_zero_weight_and_zero_input_decode_zero(device: torch.device) -> None:
    macro = build_calibrated_macro(device=device)

    w_zero = torch.zeros((TINY_INPUT_NUM, TINY_OUTPUT_NUM), dtype=torch.int32)
    x_some = torch.tensor([1, 2, 3, 1], dtype=torch.int32)
    out = _assert_decode_matches_ideal(macro, w_zero, x_some)
    assert int(out.abs().sum()) == 0

    w_some = torch.tensor(
        [[1, 1, -1, 0], [-2, 1, 0, 0], [3, -1, 0, 0], [0, 0, 0, 0]],
        dtype=torch.int32,
    ).transpose(-1, -2)
    out = _assert_decode_matches_ideal(macro, w_some, torch.zeros((TINY_INPUT_NUM,), dtype=torch.int32))
    assert int(out.abs().sum()) == 0


def test_w_digit_num_1_ternary_transfer(device: torch.device) -> None:
    macro = build_calibrated_macro(device=device, w_digit_num=1)
    assert macro.w_value_range == (-1, 1)

    torch.manual_seed(11)
    w = torch.randint(-1, 2, (TINY_INPUT_NUM, TINY_OUTPUT_NUM), dtype=torch.int32)
    x = torch.randint(0, 1 << TINY_K, (5, TINY_INPUT_NUM), dtype=torch.int32)
    out = _assert_decode_matches_ideal(macro, w, x)
    assert tuple(out.shape) == (5, TINY_OUTPUT_NUM)
    assert int(out.min()) < 0 < int(out.max())


def test_asymmetric_weight_regression_lsb_first(device: torch.device) -> None:
    macro = build_calibrated_macro(device=device)
    x = torch.tensor([1, 2, 2, 2], dtype=torch.int32)
    w = torch.tensor(
        [
            [3, -2, 1, 0],
            [-3, 2, -1, 0],
            [2, 0, 3, -1],
            [1, -1, 2, 3],
        ],
        dtype=torch.int32,
    ).transpose(-1, -2)
    out = _assert_decode_matches_ideal(macro, w, x)
    assert out.tolist() == [1, -1, 6, 7]


def test_random_batch_matches_integer_mac_at_every_adc_width(device: torch.device) -> None:
    macro = build_calibrated_macro(device=device)
    max_bits = macro.adc_bits
    gen = torch.Generator().manual_seed(0)
    w = torch.randint(-3, 4, (TINY_INPUT_NUM, TINY_OUTPUT_NUM), generator=gen, dtype=torch.int32)
    x = torch.randint(0, 1 << TINY_K, (16, TINY_INPUT_NUM), generator=gen, dtype=torch.int32)

    full = decode(macro, w, x, adc_active_bits=max_bits)
    assert tuple(full.shape) == (16, TINY_OUTPUT_NUM)
    assert torch.equal(full, ideal_mac(w, x))
    for bits in range(1, max_bits):
        lowered = decode(macro, w, x, adc_active_bits=bits)
        expected = torch.sign(full) * (full.abs() >> (max_bits - bits))
        assert torch.equal(lowered, expected), (
            f"bits {bits}: {lowered.tolist()} != right-shifted max-bits codes {expected.tolist()}"
        )
