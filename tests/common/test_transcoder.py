"""Tests for value-domain transcoders."""

from __future__ import annotations

import pytest
import torch

from neurox.common.encoding import (
    CanonicalTranscoder,
    ComplementTranscoder,
    Encoding,
    Transcoder,
    TrueFormTranscoder,
    UnsignedTranscoder,
)


@pytest.mark.parametrize(
    ("radix", "digit_count", "expected"),
    [
        (2, 3, (0, 7)),
        (4, 2, (0, 15)),
    ],
)
def test_unsigned_value_range(radix: int, digit_count: int, expected: tuple[int, int]) -> None:
    transcoder = UnsignedTranscoder(radix=radix, digit_count=digit_count)
    assert transcoder.value_range == expected


@pytest.mark.parametrize(
    ("radix", "digit_count", "expected"),
    [
        (2, 3, (-7, 7)),
        (4, 2, (-15, 15)),
    ],
)
def test_true_form_value_range(radix: int, digit_count: int, expected: tuple[int, int]) -> None:
    transcoder = TrueFormTranscoder(radix=radix, digit_count=digit_count)
    assert transcoder.value_range == expected


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


@pytest.mark.parametrize(
    ("encoding", "expected_type"),
    [
        (Encoding.UNSIGNED, UnsignedTranscoder),
        (Encoding.TRUE_FORM, TrueFormTranscoder),
        (Encoding.COMPLEMENT, ComplementTranscoder),
        (Encoding.CANONICAL, CanonicalTranscoder),
    ],
)
def test_from_encoding_dispatches_enum(encoding: Encoding, expected_type: type[Transcoder]) -> None:
    transcoder = Transcoder.from_encoding(encoding=encoding, radix=2, digit_count=3)
    assert isinstance(transcoder, expected_type)


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


def test_encode_inserts_digit_axis_at_requested_dim() -> None:
    transcoder = TrueFormTranscoder(radix=2, digit_count=3)
    x = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    digits = transcoder.encode(x, dim=1)
    assert digits.shape == (2, 3, 3)
    assert torch.equal(transcoder.decode(digits, dim=1), x)
