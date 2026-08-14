"""The shared root of every data class that carries tensors.

What a tensor field costs a data class — equality, freezing, the keyword-only
call — is settled here once, so no such class hand-writes its dataclass
decoration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import dataclass_transform


@dataclass_transform(eq_default=False, frozen_default=True, kw_only_default=True)
@dataclass(eq=False, frozen=True, kw_only=True)
class TensorDataClassBase:
    """Base for a data class that carries tensors.

    A tensor compares elementwise, so a field holding one leaves value `==`
    ill-defined: the comparison returns a tensor rather than a verdict, and no
    reduction of it is the one a caller means. Every class here therefore
    equals only itself, and `==` and `hash()` are identity throughout the
    hierarchy. Value equality stays with the configuration and policy value
    objects, which carry no tensors and are compared by what they hold.

    A subclass declares its fields as annotated class attributes carrying at
    most a plain default, and must not apply `@dataclass`; this base supplies a
    frozen, keyword-only dataclass to every descendant, however deep.

    Being a dataclass is also what `TensorGroupMixin` requires of its host, but
    the two are orthogonal: this base settles equality, freezing, and the
    keyword-only call, while the mixin adds per-tensor-field shape operations
    and demands nothing of them. A class wanting both inherits both side by
    side, this base first.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        dataclass(eq=False, frozen=True, kw_only=True)(cls)
