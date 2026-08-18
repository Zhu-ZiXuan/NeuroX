"""Shared side-channel record collection primitives.

Each recorder family shares one active slot, held as a plain class attribute.
Collection is a pure side channel across every family: a run computes the same
numbers whether or not a recorder is active, and an emit site builds what it
submits only behind its family's active gate, so an uncollected run pays the
gate read and nothing beyond it. Collection is assumed single-threaded: the slot
is not thread-local, and two threads recording at once would share one book.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from types import TracebackType
from typing import Any, ClassVar, Self, cast, final

import torch
from torch import Tensor

from .tensor_dataclass import TensorDataClassBase
from .tensor_fields import walk_tensor_fields


class RecordBase(TensorDataClassBase):
    """One item a side channel collects.

    `TensorDataClassBase` fixes how a subclass declares its fields and settles
    identity equality for the whole hierarchy. Override `detach` or `to` only
    for a record whose tensors need what a walk of the declared fields cannot
    express.
    """

    def detach(self) -> Self:
        """Return this record with every tensor field detached from autograd.

        Returns:
            `self` when no field would change, a copy otherwise.
        """
        return self._map_tensors(lambda tensor: tensor.detach() if tensor.requires_grad else tensor)

    def to(self, device: torch.device) -> Self:
        """Return this record with every tensor field parked on one device.

        Returns:
            `self` when no field would change, a copy otherwise.
        """
        return self._map_tensors(lambda tensor: tensor.to(device))

    def _map_tensors(self, transform: Callable[[Tensor], Tensor]) -> Self:
        """Rebuild through every tensor field, recursing into nested dataclasses.

        `transform` returns its argument itself where a field already holds what
        was asked for, so an all-unchanged walk gives `self` back.
        """
        changed = False

        def track(tensor: Tensor) -> Tensor:
            nonlocal changed
            result = transform(tensor)
            changed = changed or result is not tensor
            return result

        rebuilt = walk_tensor_fields(self, track)
        return rebuilt if changed else self


class RecorderBase[RecordT: RecordBase](ABC):
    """Collect one family's records for as long as its context is open.

    Subclass this base directly to open a family, binding the family's record
    type as `RecorderBase[SomeRecord]`; activation, accumulation, and
    finalization belong here, so the only collection logic such a subclass adds
    is a `_submit_impl` hook deciding which of the family's records it keeps. A
    family is that direct subclass together with everything below it, sharing
    the one active slot it owns: at most one recorder of a family collects at a
    time, and entering a second raises `RuntimeError`. An exception frees the
    slot but skips the parking. An emit site reads the family's slot through
    `active` or `current` and hands records to `submit`.

    Re-entering one instance accumulates into the same book; a fresh book is a
    fresh instance.

    Args:
        sync_device: Device a clean exit parks the collected records on. The
            default `None` leaves each record where it was recorded, so a book
            parks on one device only when a caller names it.
    """

    _family_root: ClassVar[type[RecorderBase[Any]] | None] = None
    # The family root's active slot. Which record type it holds is a per-family
    # property that no annotation on the shared base can express.
    _active_recorder: ClassVar[RecorderBase[Any] | None]

    def __init__(self, *, sync_device: torch.device | None = None) -> None:
        self._root()  # a bare RecorderBase instance owns no slot to collect into
        self._sync_device = sync_device
        self.__records: list[RecordT] = []

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if RecorderBase in cls.__bases__:
            cls._family_root = cls
            cls._active_recorder = None

    @property
    @final
    def records(self) -> tuple[RecordT, ...]:
        """The book so far, in submission order; each access returns a snapshot."""
        return tuple(self.__records)

    def __enter__(self) -> Self:
        root = self._root()
        if root._active_recorder is not None:  # noqa: SLF001
            raise RuntimeError(f"only one {root.__name__} may be active at a time")
        root._active_recorder = self  # noqa: SLF001
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._root()._active_recorder = None  # noqa: SLF001
        if exc_type is None:
            self._finalize()

    @classmethod
    def _root(cls) -> type[RecorderBase[Any]]:
        """Return the family root holding this class's active slot.

        Raises:
            TypeError: The class opens no family, i.e. it is `RecorderBase`
                itself.
        """
        root = cls._family_root
        if root is None:
            raise TypeError(f"{cls.__qualname__} opens no recorder family; subclass RecorderBase directly to open one")
        return root

    @classmethod
    @torch.compiler.disable
    def current(cls) -> Self | None:
        """Return the family's active recorder, or `None` outside a context.

        The read is kept out of every graph on purpose: the slot is Python state
        a trace cannot guard on. Dynamo folds an empty slot into the trace as a
        constant, so a region first compiled outside any context would stay
        pinned to "inactive" and silently collect nothing ever after.
        """
        return cast(Self, cls._root()._active_recorder)  # noqa: SLF001

    @classmethod
    @torch.compiler.disable
    def active(cls) -> bool:
        """Return whether the family has an active recorder.

        The read is kept out of every graph, so an emit site gating its billing
        work on it breaks the graph there and gets the live answer on every
        call.
        """
        return cls.current() is not None

    @classmethod
    @final
    @torch.compiler.disable
    def submit(cls, record: RecordT) -> None:
        """Hand one record across the graph boundary to its family hook.

        Everything the emitter computes to build the record stays in the
        caller's graph; only the hand-over leaves it. Under CUDA-graph capture
        (`torch.compile(mode="reduce-overhead")`) a submitted tensor may live in
        cudagraph-owned memory that a later replay overwrites, so an emitter
        inside such a region clones before it submits.

        Args:
            record: Record to collect; dropped when no recorder is active.
        """
        cls._submit_impl(record)

    @classmethod
    def _submit_impl(cls, record: RecordT) -> None:
        """Apply the family's admission rule and submit what survives."""
        cls._submit_record(record)

    @classmethod
    @final
    @torch.compiler.disable
    def _submit_record(cls, record: RecordT) -> None:
        """Append one record to the family's active recorder, detached.

        Collection's invariant machinery: a family hook decides admission and
        calls this. The record is dropped when no recorder is active.
        """
        recorder = cls.current()
        if recorder is None:
            return
        recorder.__records.append(record.detach())  # noqa: SLF001

    def _finalize(self) -> None:
        """Park every collected record on this recorder's `sync_device`.

        Without one the records stay where they were recorded. The sweep visits
        the whole book on every clean exit; a record already on that device is
        returned unchanged, and each record parks itself, so a cross-device book
        of `N` records costs `N` transfers.
        """
        if self._sync_device is None:
            return
        self.__records = [record.to(self._sync_device) for record in self.__records]
