"""Construct transcoders from closed encoding identifiers.

See also:
    docs/internals/common/encoding/encodings.md
"""

from __future__ import annotations

from typing import assert_never

from .base import Encoding, Transcoder
from .canonical import CanonicalTranscoder
from .complement import ComplementTranscoder
from .true_form import TrueFormTranscoder


def create_transcoder(*, encoding: Encoding, radix: int, digit_count: int) -> Transcoder:
    """Construct the transcoder selected by ``encoding``."""
    match encoding:
        case Encoding.TRUE_FORM:
            return TrueFormTranscoder(radix=radix, digit_count=digit_count)
        case Encoding.COMPLEMENT:
            return ComplementTranscoder(radix=radix, digit_count=digit_count)
        case Encoding.CANONICAL:
            return CanonicalTranscoder(radix=radix, digit_count=digit_count)
    assert_never(encoding)
