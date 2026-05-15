"""Macro-internal digital-domain mapping strategies.

The package's top level exposes only the shared digit-encoder
primitive (:class:`Transcoder` + :class:`SignedDigitTranscoder`).
Macro-specific mapper strategies live in subpackages and are
imported from there directly — today :mod:`neurox.mapper.xbar` for
the xbar macro's activation / weight pair, e.g.::

    from neurox.mapper.xbar import XbarXMapper, XbarWMapper

This keeps the top-level namespace small and lets future macro
types add their own mapping subpackages without polluting the
shared root.
"""

from .transcoder import Encoding, SignedDigitTranscoder, Transcoder

__all__ = [
    "Encoding",
    "SignedDigitTranscoder",
    "Transcoder",
]
