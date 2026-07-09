"""Shared base for digital, integer-exact circuit modules.

See also:
    docs/reference/primitive/digital/README.md
"""

from __future__ import annotations

from typing import TypeVar

from neurox.primitive.circuit import CircuitBase, CircuitConfig

DigitalConfigT = TypeVar("DigitalConfigT", bound=CircuitConfig)


class DigitalCircuit(CircuitBase[DigitalConfigT]):
    """Base for digital, integer-exact circuit blocks.

    A digital block computes an exact integer function and holds no analog
    device state, so it has no static manufacturing variation to resample.
    This base states that once for the whole family by implementing the
    fabricate hook as an explicit no-op; each digital leaf inherits "no
    mismatch here" instead of restating it, while the root ``FabricateMixin``
    keeps no default so a node that forgets the hook still fails loudly.
    """

    def _sample_fabricate_mismatch(self) -> None:
        pass  # digital logic carries no static analog mismatch
