"""Tests for value-domain transcoders."""

from __future__ import annotations

import pytest
import torch

from neurox.encoding import (
    CanonicalTranscoder,
    ComplementTranscoder,
    Transcoder,
    TrueFormTranscoder,
    UnsignedTranscoder,
)


@pytest.mark.parametrize(
    ("radix", "digit_count", "expected"),
    [
        (2, 3, (-4, 3)),
        (4, 4, (-128, 127)),
        (3, 2, (-3, 5)),
    ],
)
def test_complement_value_range(radix: int, digit_count: int, expected: tuple[int, int]) -> None:
    transcoder = ComplementTranscoder(radix=radix, digit_count=digit_count)
    assert transcoder.value_range == expected


@pytest.mark.parametrize(
    ("radix", "digit_count", "expected"),
    [
        (4, 4, (-204, 204)),
        (2, 3, (-5, 5)),
        (3, 5, (-182, 182)),
    ],
)
def test_canonical_value_range(radix: int, digit_count: int, expected: tuple[int, int]) -> None:
    transcoder = CanonicalTranscoder(radix=radix, digit_count=digit_count)
    assert transcoder.value_range == expected


def test_true_form_digits_are_lsb_first() -> None:
    transcoder = TrueFormTranscoder(radix=10, digit_count=3)
    x = torch.tensor([321, -321], dtype=torch.int32)
    digits = transcoder.encode(x)
    expected = torch.tensor([[1, 2, 3], [-1, -2, -3]], dtype=torch.int32)
    assert torch.equal(digits, expected)


def test_complement_digits_are_lsb_first_with_folded_msb() -> None:
    transcoder = ComplementTranscoder(radix=2, digit_count=4)
    x = torch.tensor([-1, -4, 3], dtype=torch.int32)
    digits = transcoder.encode(x)
    expected = torch.tensor(
        [
            [1, 1, 1, -1],
            [0, 0, 1, -1],
            [1, 1, 0, 0],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(digits, expected)


@pytest.mark.parametrize(
    "transcoder",
    [
        TrueFormTranscoder(radix=2, digit_count=3),
        UnsignedTranscoder(radix=2, digit_count=3),
        UnsignedTranscoder(radix=3, digit_count=2),
        ComplementTranscoder(radix=2, digit_count=3),
        CanonicalTranscoder(radix=2, digit_count=3),
        CanonicalTranscoder(radix=4, digit_count=3),
    ],
)
def test_encode_decode_roundtrip_inside_value_range(transcoder: Transcoder) -> None:
    lo, hi = transcoder.value_range
    x = torch.arange(lo, hi + 1, dtype=torch.int32)
    digits = transcoder.encode(x)
    decoded = transcoder.decode(digits)
    assert torch.equal(decoded, x)


@pytest.mark.parametrize("transcoder_type", [TrueFormTranscoder, CanonicalTranscoder])
def test_encode_inserts_digit_axis_at_requested_dim(transcoder_type: type[Transcoder]) -> None:
    transcoder = transcoder_type(radix=4, digit_count=3)
    x = torch.arange(8, dtype=torch.int32).reshape(2, 4)
    digits = transcoder.encode(x, dim=1)
    assert digits.shape == (2, 3, 4)
    assert torch.equal(transcoder.decode(digits, dim=1), x)


def _canonical_representable_range(radix: int, digits: int) -> tuple[int, int]:
    max_abs = sum((radix - 1) * (radix**power) for power in range(digits - 1, -1, -2))
    return -max_abs, max_abs


class TestCanonicalEncoder:
    def test_known_case_radix4(self) -> None:
        min_val, max_val = _canonical_representable_range(radix=4, digits=4)
        x = torch.tensor([63], dtype=torch.int32)

        assert min_val <= int(x.item()) <= max_val, "input outside representable range"

        tc = CanonicalTranscoder(radix=4, digit_count=4)
        encoded = tc.encode(x)
        expected = torch.tensor([[-1, 0, 0, 1]], dtype=torch.int32)

        assert torch.equal(encoded, expected), f"unexpected encoding: expected {expected}, got {encoded}"
        assert torch.equal(tc.decode(encoded), x), "decode does not match original input"

    def test_zero_negatives_and_boundaries(self) -> None:
        radix = 4
        digits = 4
        min_val, max_val = _canonical_representable_range(radix=radix, digits=digits)
        x = torch.tensor([0, -1, -63, -15, min_val, max_val], dtype=torch.int32)

        assert torch.all((x >= min_val) & (x <= max_val)), "input outside representable range"

        tc = CanonicalTranscoder(radix=radix, digit_count=digits)
        encoded = tc.encode(x)
        decoded_x = tc.decode(encoded)
        assert torch.equal(decoded_x, x), "round-trip failed for zero / negative inputs"
        assert torch.all(encoded[0] == 0), "encoding of 0 must be all zeros"

    @pytest.mark.parametrize("radix", [2, 3, 4, 8])
    @pytest.mark.parametrize("digits", [4, 8])
    def test_full_range_fuzzing(self, radix: int, digits: int) -> None:
        min_val, max_val = _canonical_representable_range(radix=radix, digits=digits)
        x = torch.randint(min_val, max_val + 1, size=(1000,), dtype=torch.int32)

        tc = CanonicalTranscoder(radix=radix, digit_count=digits)
        encoded = tc.encode(x)
        decoded_x = tc.decode(encoded)

        assert torch.equal(decoded_x, x.long()), f"fuzzing failed at radix={radix}, digits={digits}"
