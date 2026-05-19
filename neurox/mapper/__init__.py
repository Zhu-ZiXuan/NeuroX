"""Macro-internal digital-domain mapping strategies.

Top-level exposes the shared digit-encoder primitive; macro-specific
mapper strategies live in subpackages (e.g. :mod:`neurox.mapper.xbar`).
"""

from .transcoder import Encoding, SignedDigitTranscoder, Transcoder

__all__ = [
    "Encoding",
    "SignedDigitTranscoder",
    "Transcoder",
]
