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

from .tensor_dataclass_mixin import TensorDataClassMixin, walk_single_tensor_fields
from .torch_compat import torch_compiler_disable


class RecordBase(TensorDataClassMixin):
    """One item a side channel collects.

    `TensorDataClassMixin` fixes how a subclass declares its fields and settles
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

        rebuilt = walk_single_tensor_fields(track, self)
        return rebuilt if changed else self


class RecorderBase[RecordT: RecordBase](ABC):
    """Collect one family's records while its context is open.

    Each direct subclass opens a family and binds its record type. The family
    and its descendants share one active slot. `_submit_impl` decides admission
    and calls `_submit_record` to retain a detached record. Emitters use `active`
    or `current` to guard record construction, then call `submit`.

    Context entry activates collection; clean exit moves retained records to
    `sync_device`. Exceptional exit frees the active slot without moving records.
    Re-entering an instance appends to its existing collection.

    Args:
        sync_device: Destination for records on clean exit. `None` leaves each
            record on its original device.

    Raises:
        RuntimeError: Another recorder of this family is active on entry.
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

    @final
    def __enter__(self) -> Self:
        root = self._root()
        if root._active_recorder is not None:  # noqa: SLF001
            raise RuntimeError(f"only one {root.__name__} may be active at a time")
        root._active_recorder = self  # noqa: SLF001
        return self

    @final
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
    @final
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
    @torch_compiler_disable
    @final
    def current(cls) -> Self | None:
        """Return the family's active recorder, or `None` outside a context.

        This access crosses a compiler boundary so every call reads the live
        Python slot rather than a value captured during tracing.
        """
        return cast(Self, cls._root()._active_recorder)  # noqa: SLF001

    @classmethod
    @torch_compiler_disable
    @final
    def active(cls) -> bool:
        """Return whether the family has an active recorder.

        This access crosses a compiler boundary so emitters can gate record
        construction on the live Python slot.
        """
        return cls.current() is not None

    @classmethod
    @final
    @torch_compiler_disable
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
    @torch_compiler_disable
    def _submit_record(cls, record: RecordT) -> None:
        """Append one record to the family's active recorder, detached.

        Collection's invariant machinery: a family hook decides admission and
        calls this. The record is dropped when no recorder is active.
        """
        recorder = cls.current()
        if recorder is None:
            return
        recorder.__records.append(record.detach())  # noqa: SLF001

    @final
    def _finalize(self) -> None:
        """Move retained records to `sync_device` on clean context exit.

        `None` leaves records in place. Each record's `to` method handles its own
        transfer; records already on the destination device remain unchanged.
        """
        if self._sync_device is None:
            return
        self.__records = [record.to(self._sync_device) for record in self.__records]
