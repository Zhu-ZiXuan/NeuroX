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
    """A probe payload: a value able to return a detached copy of itself.

    A payload is a small frozen dataclass of per-call tensors (plus plain
    scalar metadata); ``detach`` returns an equivalent payload whose tensor
    fields are detached, so a probe never keeps a live compute graph alive.
    """

    def detach(self) -> Self: ...


PayloadT = TypeVar("PayloadT", bound=SupportsDetach)


class Prober(Generic[PayloadT], ABC):
    """Context manager capturing one observation link's per-call payloads.

    Pure mechanism: each concrete subclass binds ONE observation link (an
    emitter site + its payload type) and is the sole capture point for that
    link. A subclass provides its OWN typed active stack through :meth:`_stack`,
    so entering a prober of one subclass never captures an emission a prober of
    another subclass submits.
    That per-subclass isolation is the link routing: an emitter names its
    link's subclass at the call site and reaches exactly the probers of that
    subclass.

    Canonical usage — record inside the ``with`` block, read the stream
    after it::

        with SomeLinkProber() as prober:
            model(...)
        for payload in prober.records:
            ...

    Probers of the same subclass stack: entering pushes onto that subclass's
    active stack, exiting pops. An emission reaches every stacked prober of
    the link's subclass, so an outer session-scoped prober keeps recording
    while an inner one captures a narrow window. A record is the payload
    alone: no emitting module or name is stored. Payloads are detached once
    at submission (via :meth:`SupportsDetach.detach`) and the same frozen
    object is shared across every active prober — safe because payloads are
    frozen dataclasses — so a probing run never keeps a compute graph alive.
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
        """True iff at least one prober of this subclass is currently active.

        Emitters gate their diagnostic computation, payload construction, and
        emission on this: ``if SomeLinkProber.active(): <build payload>;
        SomeLinkProber.submit(payload)``. Without an active prober the emitter
        never pays for a payload it would drop. ``@torch.compiler.disable``
        keeps the predicate — and the guarded branch it collapses at trace
        time — out of any caller's compiled graph.

        """
        return bool(cls._stack())

    @classmethod
    @torch.compiler.disable
    def submit(cls, payload: PayloadT) -> None:
        """Submit ``payload`` to every active prober of this subclass.

        A no-op with an empty stack (returns before touching the payload).
        Otherwise the payload is detached ONCE and the same frozen object is
        appended to every active prober's records — sharing is safe because
        payloads are frozen. ``@torch.compiler.disable`` keeps the hook out of
        any caller's compiled graph.

        """
        stack = cls._stack()
        if not stack:
            return
        detached = payload.detach()
        for prober in stack:
            prober.records.append(detached)
