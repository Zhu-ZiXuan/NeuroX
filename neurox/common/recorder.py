"""Shared side-channel record collection primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from types import TracebackType
from typing import Any, ClassVar, Self, cast, final

import torch

from .base_only_mixin import BaseOnlyMixin


class RecorderBase[RecordT, HistoryT, ResultT](BaseOnlyMixin, ABC, base_only=True):
    """Collect one family's submissions and merge records after each context.

    Direct subclasses open independent families and bind their submission,
    history-entry, and result types; descendants share their family's active
    slot. Collection is single-threaded: the slot is not thread-local. Emitters
    use `current` or `active` and prepare submissions only when active, without
    changing numerical behavior. Each family's public submission method
    detaches its payload before passing it to `_submit_record`.

    The base owns `_current_records` and `_history_records`; subclasses expose
    completed data through `result`. Each context starts with an empty queue.

    Enter and exit outside compiled functions. Compiled emitters access only the
    current batch. Exit releases the active slot and queue; failed execution
    discards the batch and propagates the exception. Clean execution calls
    `_merge_records`, then `_sync_history`, then appends the new entries to
    history. Errors propagate without recovery.

    Args:
        sync_device: Destination for retained results after merging records.
            `None` keeps each result on its current device.

    Raises:
        RuntimeError: Another recorder of this family is active on entry.
    """

    # Family slots cover different submission, history, and result types; each family's
    # public API retains its bound types and current-instance type.
    __family_root: ClassVar[type[RecorderBase[Any, Any, Any]]]
    __active_recorder: ClassVar[RecorderBase[Any, Any, Any] | None]

    def __init__(self, *, sync_device: torch.device | None = None) -> None:
        self._sync_device = sync_device
        self._current_records: list[RecordT] = []
        self._history_records: list[HistoryT] = []

    def __init_subclass__(cls, *, base_only: bool = False, **kwargs: object) -> None:
        super().__init_subclass__(base_only=base_only, **kwargs)
        if RecorderBase in cls.__bases__:
            cls.__family_root = cls
            cls.__active_recorder = None

    # === Public API ===

    @final
    def __enter__(self) -> Self:
        family_root = self.__family_root
        if family_root.__active_recorder is not None:  # noqa: SLF001
            raise RuntimeError(f"only one {family_root.__name__} may be active at a time")
        family_root.__active_recorder = self  # noqa: SLF001
        self._current_records = []
        return self

    @final
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        family_root = self.__family_root
        family_root.__active_recorder = None  # noqa: SLF001
        records = self._current_records
        self._current_records = []
        if exc_type is not None:
            return
        history_records = self._merge_records(records)
        history_records = self._sync_history(history_records)
        self._history_records.extend(history_records)

    @property
    @abstractmethod
    def result(self) -> ResultT:
        """Expose completed data for analysis outside the collection lifecycle.

        Read after collection, outside compiled functions, and treat the data as
        read-only. Current submissions are excluded. Containers and tensors may
        be shared with the recorder; exporting does not end or clear collection.
        """
        raise NotImplementedError

    @classmethod
    @final
    def current(cls) -> Self | None:
        """Return the family's active recorder, or `None` outside a context."""
        family_root = cls.__family_root
        return cast(Self | None, family_root.__active_recorder)  # noqa: SLF001

    @classmethod
    @final
    def active(cls) -> bool:
        """Return whether the family has an active recorder."""
        return cls.current() is not None

    # === For subclass to implement or override ===

    @abstractmethod
    def _merge_records(self, records: Sequence[RecordT]) -> Sequence[HistoryT]:
        """Return new history entries from one context's submissions.

        Runs outside compilation after clean execution, including empty contexts.
        The active slot and submission queue have already been released. The
        returned sequence contains only this context's entries; the base owns
        appending them to history.
        """
        raise NotImplementedError

    @abstractmethod
    def _sync_history(self, records: Sequence[HistoryT]) -> Sequence[HistoryT]:
        """Return the new history entries with their tensors on `_sync_device`.

        Receives only the current context's merged entries, before the base
        appends them to history. Returns entries unchanged when `_sync_device`
        is `None`. Otherwise may update mutable entries in place or return
        replacements, preserving their order and meaning.
        """
        raise NotImplementedError

    # === Tools for subclass and internal use ===

    @final
    def _submit_record(self, record: RecordT) -> None:
        """Append a prepared record to this instance's current batch.

        The family's submission method owns detachment and payload preparation.
        When using CUDA Graph replay, preserve retained values outside compiled
        calls before a later replay can overwrite shared tensor storage.

        Raises:
            RuntimeError: This instance is not the family's active recorder.
        """
        if self.current() is not self:
            raise RuntimeError("records can only be submitted to the active recorder")
        self._current_records.append(record)
