"""State-binding helpers used by ``load_neurox_state``.

Carries the per-call diagnostic that :func:`neurox.replace.load_neurox_state`
returns alongside its in-place mutation of the model, plus the dedicated
exception raised when strict validation fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class NeuroxStateError(ValueError):
    """Raised when a NeuroX-flat checkpoint fails strict schema validation.

    Carries the unresolved ``missing`` keys (and optionally ``unexpected``
    extras) so callers can render a precise diagnostic before re-raising
    or downgrading to a warning.
    """

    def __init__(self, message: str, *, missing: list[str], unexpected: list[str]) -> None:
        super().__init__(message)
        self.missing = list(missing)
        self.unexpected = list(unexpected)


@dataclass(frozen=True)
class StateBindingReport:
    """Diagnostic returned by :func:`load_neurox_state`.

    Attributes:
        loaded: Qualified module names that received their full 12-buffer
            state-dict slice.
        missing: ``"<qualified>.<suffix>"`` keys NeuroX operators expected
            but the checkpoint did not contain.  Strict mode raises when
            this list is non-empty; non-strict callers can use it to
            decide whether to warn.
        unexpected: Checkpoint keys that did not match any operator buffer
            in the replaced model.  Includes float-pass-through tensors
            (BN params, biases on float layers, …) so it is informational
            unless the caller is enforcing a closed schema.
    """

    loaded: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
