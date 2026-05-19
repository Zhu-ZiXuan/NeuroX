"""State-binding helpers used by ``load_neurox_state``."""

from __future__ import annotations

from dataclasses import dataclass, field


class NeuroxStateError(ValueError):
    """Raised when a NeuroX-flat checkpoint fails strict schema validation.

    Attributes:
        missing: Required keys absent from the checkpoint.
        unexpected: Checkpoint keys not matched against any operator buffer.
    """

    def __init__(self, message: str, *, missing: list[str], unexpected: list[str]) -> None:
        super().__init__(message)
        self.missing = list(missing)
        self.unexpected = list(unexpected)


@dataclass(frozen=True)
class StateBindingReport:
    """Diagnostic returned by :func:`load_neurox_state`.

    Attributes:
        loaded: Qualified module names that received their full state slice.
        missing: ``"<qualified>.<suffix>"`` keys expected but absent.
        unexpected: Checkpoint keys not matched to any operator buffer.
    """

    loaded: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
