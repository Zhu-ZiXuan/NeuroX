"""Side-channel payload prober for calibration and diagnostics.

See also:
    docs/internals/common/prober.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Generic, Protocol, Self, TypeVar

import torch


class SupportsDetach(Protocol):
    """Probe payload that can return a detached copy."""

    def detach(self) -> Self: ...


PayloadT = TypeVar("PayloadT", bound=SupportsDetach)


class Prober(Generic[PayloadT], ABC):
    """Context manager capturing one observation link's per-call payloads.

    Read captured payloads after leaving the context::

        with SomeLinkProber() as prober:
            model(...)
        for payload in prober.records:
            ...
    """

    @classmethod
    @abstractmethod
    def _stack(cls) -> list[Prober[PayloadT]]:
        """Return the concrete observation link's active-prober stack."""
        raise NotImplementedError

    def __init__(self) -> None:
        self.records: list[PayloadT] = []

    def __enter__(self) -> Self:
        type(self)._stack().append(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        popped = type(self)._stack().pop()
        if popped is not self:
            raise RuntimeError("Prober exited out of LIFO order; the active stack is corrupted")

    @classmethod
    @torch.compiler.disable
    def active(cls) -> bool:
        """Return whether this probe link has an active context."""
        return bool(cls._stack())

    @classmethod
    @torch.compiler.disable
    def submit(cls, payload: PayloadT) -> None:
        """Submit a detached payload to every active context for this link."""
        stack = cls._stack()
        if not stack:
            return
        detached = payload.detach()
        for prober in stack:
            prober.records.append(detached)
