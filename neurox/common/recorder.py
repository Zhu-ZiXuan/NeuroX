"""Shared side-channel record collection primitives.

Each recorder family shares one active slot, held as a plain class attribute.
Collection is a pure side channel across every family: a run computes the same
numbers whether or not a recorder is active, and an emit site builds what it
submits only behind its family's active gate, so an uncollected run pays the
gate read and nothing beyond it. Collection is assumed single-threaded: the slot
is not thread-local, and two threads recording at once would share one book.
"""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import Any, ClassVar, Self, cast, final

import torch
from torch import Tensor

from .base_only_mixin import BaseOnlyMixin
from .dataclass_mixin import TensorDataClassMixin, map_single_tensor_fields


class RecordBase(TensorDataClassMixin, BaseOnlyMixin, base_only=True):
    """One item a side channel collects.

    `TensorDataClassMixin` fixes how a subclass declares its fields and settles
    identity equality for the whole hierarchy. Override `detach` or `to` only
    for a record whose tensors need what a walk of the declared fields cannot
    express.
    """

    # === Public API ===

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

    # === Tools for subclass and internal use ===

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

        rebuilt = map_single_tensor_fields(track, self)
        return rebuilt if changed else self


class RecorderBase[RecordT: RecordBase](BaseOnlyMixin, base_only=True):
    """Collect one family's records while its context is open.

    Each direct subclass opens a family and binds its record type. The family
    and its descendants share one active slot. `_submit_impl` decides admission
    and calls `_submit_record` to retain a detached record. Emitters use `active`
    or `current` to guard record construction, then call `submit`.

    Each context collects into an empty batch. Exit appends that batch to the
    instance's history and frees the active slot, including on exceptions.
    Clean exit also moves retained records to `sync_device`.

    Enter and exit contexts outside compiled functions. Compiled emitters
    append only to the current batch; repeating the same call order and record
    counts keeps its structure stable at each compiled entry. Read `records`
    outside compiled functions so accumulated history stays outside tracing.

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
        self._sync_device = sync_device
        self.__records: list[RecordT] = []
        self.__pending_records: list[RecordT] = []

    def __init_subclass__(cls, *, base_only: bool = False, **kwargs: object) -> None:
        super().__init_subclass__(base_only=base_only, **kwargs)
        if RecorderBase in cls.__bases__:
            cls._family_root = cls
            cls._active_recorder = None

    # === Public API ===

    @property
    @final
    def records(self) -> tuple[RecordT, ...]:
        """Snapshot of history and current-context records, in submission order."""
        return (*self.__records, *self.__pending_records)

    @final
    def __enter__(self) -> Self:
        root = self._root()
        if root._active_recorder is not None:  # noqa: SLF001
            raise RuntimeError(f"only one {root.__name__} may be active at a time")
        self.__pending_records = []
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
        self.__records.extend(self.__pending_records)
        self.__pending_records = []
        if exc_type is None:
            self._finalize()

    @classmethod
    @final
    def current(cls) -> Self | None:
        """Return the family's active recorder, or `None` outside a context."""
        return cast(Self, cls._root()._active_recorder)  # noqa: SLF001

    @classmethod
    @final
    def active(cls) -> bool:
        """Return whether the family has an active recorder."""
        return cls.current() is not None

    @classmethod
    @final
    def submit(cls, record: RecordT) -> None:
        """Pass one record to the family's admission hook.

        Collection detaches tensors without copying their storage. When using
        CUDA Graph replay, preserve retained tensor values outside compiled
        calls before a later replay can overwrite them.

        Args:
            record: Record to collect; dropped when no recorder is active.
        """
        cls._submit_impl(record)

    # === For subclass to implement or override ===

    @classmethod
    def _submit_impl(cls, record: RecordT) -> None:
        """Apply the family's admission rule and submit what survives."""
        cls._submit_record(record)

    # === Tools for subclass and internal use ===

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
    @final
    def _submit_record(cls, record: RecordT) -> None:
        """Append one record to the family's active recorder, detached.

        Collection's invariant machinery: a family hook decides admission and
        calls this. The record is dropped when no recorder is active.
        """
        recorder = cls.current()
        if recorder is None:
            return
        recorder.__pending_records.append(record.detach())  # noqa: SLF001

    @final
    def _finalize(self) -> None:
        """Move retained records to `sync_device` on clean context exit.

        `None` leaves records in place. Each record's `to` method handles its own
        transfer; records already on the destination device remain unchanged.
        """
        if self._sync_device is None:
            return
        self.__records = [record.to(self._sync_device) for record in self.__records]
