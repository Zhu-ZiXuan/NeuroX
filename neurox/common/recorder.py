"""Shared side-channel record collection for the profiler and the probers.

A recorder family is a direct subclass of :class:`RecorderBase` together with
everything below it. The family shares one active slot held as a plain class
attribute, so at most one recorder of a family collects at a time. Reading the
slot and submitting to it both stay out of the caller's graph under
``torch.compile``: the slot is Python state a trace cannot guard on, so an emit
site's demand gate and its submission are graph breaks by design. Collection is
assumed single-threaded: the slot is not thread-local, and two threads
recording at once would share one book.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from dataclasses import Field, dataclass
from types import TracebackType
from typing import Any, ClassVar, Generic, Self, TypeVar, dataclass_transform

import torch
from torch import Tensor

from .tensor_fields import walk_tensor_fields

# Where a recorder parks its book by default. A recording device is a harness
# knob rather than a physical quantity, so it carries a code default.
_DEFAULT_DEVICE = torch.device("cpu")


@dataclass_transform(eq_default=False, frozen_default=True, kw_only_default=True)
@dataclass(eq=False, frozen=True, kw_only=True)
class RecordBase:
    """One item a side channel collects.

    A record carries no value equality: it equals only itself, so ``==`` and
    ``hash()`` are identity throughout the hierarchy.

    Subclass requirements:
        - Declare fields as annotated class attributes, with at most a plain
          default; a ``dataclasses.field()`` specifier is rejected at class
          definition.
        - Do not apply ``@dataclass`` or define ``__init__`` or
          ``__post_init__``; this base supplies a frozen, keyword-only
          dataclass. :meth:`detach` and :meth:`to` rebuild the record through
          those declared fields, recursing into nested dataclasses.
        - Override either method only for a record whose tensors need what the
          field walk cannot express.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if "__init__" in cls.__dict__:
            raise TypeError(f"{cls.__qualname__} must declare dataclass fields, not __init__()")
        if "__post_init__" in cls.__dict__:
            raise TypeError(f"{cls.__qualname__} carries data only; it declares fields, not __post_init__()")
        for name, value in cls.__dict__.items():
            if isinstance(value, Field):
                raise TypeError(
                    f"{cls.__qualname__}.{name} is a field() specifier; a record declares plain fields only"
                )
        dataclass(eq=False, frozen=True, kw_only=True)(cls)

    def detach(self) -> Self:
        """Return this record with every tensor field detached from autograd.

        Returns:
            ``self`` when no field would change, a copy otherwise.
        """
        return self._map_tensors(lambda tensor: tensor.detach() if tensor.requires_grad else tensor)

    def to(self, device: torch.device) -> Self:
        """Return this record with every tensor field on ``device``.

        Args:
            device: Device the record's tensors are parked on.

        Returns:
            ``self`` when no field would change, a copy otherwise.
        """
        return self._map_tensors(lambda tensor: tensor.to(device))

    def _map_tensors(self, transform: Callable[[Tensor], Tensor]) -> Self:
        """Rebuild through every tensor field, keeping ``self`` when none moves.

        Args:
            transform: Per-tensor-field transform, returning its argument
                itself for a field already holding what was asked for.

        Returns:
            ``self`` when every field came back unchanged, a rebuilt record
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


RecordT = TypeVar("RecordT", bound=RecordBase)


class RecorderBase(Generic[RecordT], ABC):
    """Collect one family's records for as long as its context is open.

    Subclassing this base directly opens a family: that class owns the active
    slot every class below it shares, so a family root and one of its own
    subclasses are mutually exclusive collectors. An emit site reads the slot
    through :meth:`active` or :meth:`current` and hands records to
    :meth:`submit`; a record submitted with no active recorder is dropped.

    Subclass requirements:
        - Subclass this base directly to open a family, binding the family's
          record type as ``RecorderBase[SomeRecord]``.
        - Add no collection logic of its own: activation, accumulation, and
          finalization belong to this base.

    Args:
        device: Where a clean exit parks the collected records. ``None`` leaves
            each record on the device it was recorded on.

    Attributes:
        records: Records collected so far, in submission order. Re-entering one
            instance accumulates into the same list; a fresh book is a fresh
            instance.
    """

    _family_root: ClassVar[type[RecorderBase[Any]] | None] = None
    # The family root's active slot. Which record type it holds is a per-family
    # property that no annotation on the shared base can express.
    _active_recorder: ClassVar[Any]

    def __init__(self, *, device: torch.device | None = _DEFAULT_DEVICE) -> None:
        self._root()  # a bare RecorderBase instance owns no slot to collect into
        self._device = device
        self.records: list[RecordT] = []

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if RecorderBase in cls.__bases__:
            cls._family_root = cls
            cls._active_recorder = None

    def __enter__(self) -> Self:
        root = self._root()
        if root._active_recorder is not None:
            raise RuntimeError(f"only one {root.__name__} may be active at a time")
        root._active_recorder = self
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._root()._active_recorder = None
        if exc_type is None:
            self._finalize()

    @classmethod
    def _root(cls) -> type[RecorderBase[Any]]:
        """Return the family root holding this class's active slot.

        Raises:
            TypeError: The class opens no family, i.e. it is
                :class:`RecorderBase` itself.
        """
        root = cls._family_root
        if root is None:
            raise TypeError(f"{cls.__qualname__} opens no recorder family; subclass RecorderBase directly to open one")
        return root

    @classmethod
    @torch.compiler.disable
    def current(cls) -> Self | None:
        """Return the family's active recorder, or ``None`` outside a context.

        Note:
            The read is kept out of every graph on purpose. Dynamo folds an
            empty slot into the trace as a constant and installs no guard on
            it, so a region first compiled outside any context would stay
            pinned to "inactive" and silently collect nothing ever after.
        """
        return cls._root()._active_recorder

    @classmethod
    @torch.compiler.disable
    def active(cls) -> bool:
        """Return whether the family has an active recorder.

        Note:
            Kept out of every graph for the reason :meth:`current` states: a
            traced guard would freeze an emit site's branch at whatever the
            slot held when the region was first compiled.
        """
        return cls.current() is not None

    @classmethod
    @torch.compiler.disable
    def submit(cls, record: RecordT) -> None:
        """Hand one record to the family's active recorder, detached.

        Everything the emitter computes to build the record stays in the
        caller's graph; only the hand-over leaves it, alongside the demand
        gate :meth:`active` reads.

        Args:
            record: Record to collect; dropped when no recorder is active.

        Note:
            Under CUDA-graph capture (``torch.compile(mode="reduce-overhead")``)
            a submitted tensor may live in cudagraph-owned memory that a later
            replay overwrites. An emitter inside such a region clones before it
            submits.
        """
        recorder = cls.current()
        if recorder is None:
            return
        recorder.records.append(record.detach())

    def _finalize(self) -> None:
        """Park every collected record on this recorder's device.

        The sweep visits the whole book on every clean exit; a record already
        on the device is returned unchanged, so re-sweeping what an earlier
        activation collected moves nothing, though it still costs one visit
        per record. Each record parks itself, so a cross-device book of ``N``
        records costs ``N`` transfers rather than one batched copy: a consumer
        collecting a large book on an accelerator, or one that post-processes
        records where they were recorded, passes ``device=None`` and keeps
        them in place.
        """
        if self._device is None:
            return
        self.records = [record.to(self._device) for record in self.records]
