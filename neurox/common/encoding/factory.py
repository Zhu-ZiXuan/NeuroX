"""Construct transcoders from closed encoding identifiers."""

from __future__ import annotations

from typing import assert_never

from .base import Encoding, Transcoder
from .canonical import CanonicalTranscoder
from .complement import ComplementTranscoder
from .true_form import TrueFormTranscoder


def create_transcoder(*, encoding: Encoding, radix: int, digit_count: int) -> Transcoder:
    """Construct the transcoder one encoding identifier selects.

    Returns:
        Transcoder realizing `encoding` over `(radix, digit_count)`.

    Raises:
        ValueError: `radix < 2` or `digit_count < 1`.
    """
    match encoding:
        case Encoding.TRUE_FORM:
            return TrueFormTranscoder(radix=radix, digit_count=digit_count)
        case Encoding.COMPLEMENT:
            return ComplementTranscoder(radix=radix, digit_count=digit_count)
        case Encoding.CANONICAL:
            return CanonicalTranscoder(radix=radix, digit_count=digit_count)
    assert_never(encoding)
