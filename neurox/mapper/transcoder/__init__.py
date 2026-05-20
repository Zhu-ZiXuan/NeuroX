"""Signed-digit transcoders: integer ↔ digit-list conversion.

See also:
    docs/dev/modules/mapper/transcoder/README.md
"""

from .base import Encoding, Transcoder
from .canonical import CanonicalTranscoder
from .complement import ComplementTranscoder
from .true_form import TrueFormTranscoder

__all__ = [
    "CanonicalTranscoder",
    "ComplementTranscoder",
    "Encoding",
    "Transcoder",
    "TrueFormTranscoder",
]
