"""Shared side-channel record collection for the profiler and the probers.

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
from dataclasses import Field
from types import TracebackType
from typing import Any, ClassVar, Self, cast, final

import torch
from torch import Tensor

from .tensor_dataclass import TensorDataClassBase
from .tensor_fields import walk_tensor_fields

# Where a recorder parks its book by default. A recording device is a harness
# knob rather than a physical quantity, so it carries a code default.
_DEFAULT_DEVICE = torch.device("cpu")


class RecordBase(TensorDataClassBase):
    """One item a side channel collects.

    A record carries no value equality: it equals only itself, so `==` and
    `hash()` are identity throughout the hierarchy.

    A subclass declares its fields as annotated class attributes carrying at
    most a plain default, and must not apply `@dataclass` or define `__init__`
    or `__post_init__`; `TensorDataClassBase` supplies a frozen, keyword-only
    dataclass. Override `detach` or `to` only for a record whose tensors need
    what a walk of the declared fields cannot express.
    """

    def __init_subclass__(cls) -> None:
        if "__init__" in cls.__dict__:
            raise TypeError(f"{cls.__qualname__} must declare dataclass fields, not __init__()")
        if "__post_init__" in cls.__dict__:
            raise TypeError(f"{cls.__qualname__} carries data only; it declares fields, not __post_init__()")
        for name, value in cls.__dict__.items():
            if isinstance(value, Field):
                raise TypeError(
                    f"{cls.__qualname__}.{name} is a field() specifier; a record declares plain fields only"
                )
        # After the guards: the base's decoration consumes every field()
        # specifier a subclass declared, leaving nothing for the walk to find.
        super().__init_subclass__()

    def detach(self) -> Self:
        """Return this record with every tensor field detached from autograd.

        Returns:
            `self` when no field would change, a copy otherwise.
        """
        return self._map_tensors(lambda tensor: tensor.detach() if tensor.requires_grad else tensor)

    def to(self, device: torch.device) -> Self:
        """Return this record with every tensor field parked on one device.

        Args:
            device: Destination device.

        Returns:
            `self` when no field would change, a copy otherwise.
        """
        return self._map_tensors(lambda tensor: tensor.to(device))

    def _map_tensors(self, transform: Callable[[Tensor], Tensor]) -> Self:
        """Rebuild through every tensor field, recursing into nested dataclasses.

        Args:
            transform: Per-tensor-field transform, returning its argument
                itself for a field already holding what was asked for.

        Returns:
            `self` when every field came back unchanged, a rebuilt record
            otherwise.
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
    type as `RecorderBase[SomeRecord]`; such a subclass adds no collection logic
    of its own, since activation, accumulation, and finalization belong here. A
    family is that direct subclass together with everything below it, sharing
    the one active slot it owns: at most one recorder of a family collects at a
    time, and entering a second raises `RuntimeError`. An exception frees the
    slot but skips the parking. An emit site reads the family's slot through
    `active` or `current` and hands records to `submit`.

    Re-entering one instance accumulates into the same book; a fresh book is a
    fresh instance.

    Args:
        device: Where a clean exit parks the collected records. `None` leaves
            each record on the device it was recorded on.
    """

    _family_root: ClassVar[type[RecorderBase[Any]] | None] = None
    # The family root's active slot. Which record type it holds is a per-family
    # property that no annotation on the shared base can express.
    _active_recorder: ClassVar[RecorderBase[Any] | None]

    def __init__(self, *, device: torch.device | None = _DEFAULT_DEVICE) -> None:
        self._root()  # a bare RecorderBase instance owns no slot to collect into
        self._device = device
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
    @torch.compiler.disable
    def submit(cls, record: RecordT) -> None:
        """Hand one record to the family's active recorder, detached.

        Everything the emitter computes to build the record stays in the
        caller's graph; only the hand-over leaves it. Under CUDA-graph capture
        (`torch.compile(mode="reduce-overhead")`) a submitted tensor may live in
        cudagraph-owned memory that a later replay overwrites, so an emitter
        inside such a region clones before it submits.

        Args:
            record: Record to collect; dropped when no recorder is active.
        """
        recorder = cls.current()
        if recorder is None:
            return
        recorder.__records.append(record.detach())  # noqa: SLF001

    def _finalize(self) -> None:
        """Park every collected record on this recorder's device.

        The sweep visits the whole book on every clean exit; a record already on
        the device is returned unchanged, so re-sweeping what an earlier
        activation collected moves nothing. Each record parks itself, so a
        cross-device book of `N` records costs `N` transfers rather than one
        batched copy.
        """
        if self._device is None:
            return
        self.__records = [record.to(self._device) for record in self.__records]
