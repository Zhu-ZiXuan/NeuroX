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


def test_binary_complement_keeps_the_highest_digit_signed() -> None:
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
    assert transcoder.place_values == (1, 2, 4, 8)
    assert transcoder.has_signed_digits
    assert torch.equal(transcoder.decode(digits), x)


@pytest.mark.parametrize("radix", [2, 3, 5, 8, 10])
@pytest.mark.parametrize("digit_count", [1, 2, 3])
@pytest.mark.parametrize("dim", [0, -1])
def test_complement_decoder_preserves_fixed_count_and_radix_interval(radix: int, digit_count: int, dim: int) -> None:
    transcoder = ComplementTranscoder(radix=radix, digit_count=digit_count)
    lo = -(radix // 2) * radix ** (digit_count - 1)
    period = radix**digit_count
    values = torch.arange(lo - period, lo + 2 * period, dtype=torch.int64).reshape(3, period)
    digits = transcoder.encode(values, dim=dim)
    assert digits.shape[dim] == digit_count
    expected = (values - lo).remainder(period) + lo
    torch.testing.assert_close(transcoder.decode(digits, dim=dim), expected)
    assert transcoder.has_signed_digits
    assert transcoder.place_values == tuple(radix**i for i in range(digit_count))


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


def test_canonical_carry_reaches_a_nonadjacent_high_digit() -> None:
    transcoder = CanonicalTranscoder(radix=4, digit_count=4)
    values = torch.tensor([63, -63, 0], dtype=torch.int32)
    encoded = transcoder.encode(values)
    expected = torch.tensor([[-1, 0, 0, 1], [1, 0, 0, -1], [0, 0, 0, 0]], dtype=torch.int32)
    assert torch.equal(encoded, expected)
    assert torch.equal(transcoder.decode(encoded), values)


@pytest.mark.parametrize("radix", [2, 3, 4, 8])
@pytest.mark.parametrize("digits", [4, 8])
def test_canonical_roundtrip_includes_signed_boundaries(radix: int, digits: int) -> None:
    transcoder = CanonicalTranscoder(radix=radix, digit_count=digits)
    lo, hi = transcoder.value_range
    generator = torch.Generator().manual_seed(31)
    values = torch.cat(
        (
            torch.tensor([lo, hi, 0, -1], dtype=torch.int32),
            torch.randint(lo, hi + 1, size=(1000,), dtype=torch.int32, generator=generator),
        )
    )
    assert torch.equal(transcoder.decode(transcoder.encode(values)), values)
