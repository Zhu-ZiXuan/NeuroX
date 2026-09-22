"""Shared side-channel record collection primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import wraps
from inspect import Parameter, signature
from itertools import count
from types import TracebackType
from typing import Any, ClassVar, Concatenate, Self, cast, final
from weakref import WeakValueDictionary

import torch
from torch import Tensor
from torch._library.effects import EffectType

from .base_only_mixin import BaseOnlyMixin

_identities = count()
_submissions = count()
_recorders: WeakValueDictionary[int, RecorderBase[Any, Any, Any]] = WeakValueDictionary()


class RecorderBase[RecordT, HistoryT, ResultT](BaseOnlyMixin, ABC, base_only=True):
    """Collect one family's submissions and merge records after each context.

    Direct subclasses open independent families and bind their submission,
    history-entry, and result types; descendants share their family's active
    slot. Collection is single-threaded: the slot is not thread-local. Emitters
    use `current` or `active` and prepare submissions only when active, without
    changing numerical behavior. Families decorate their `submit_*` methods
    with `submission`; those methods validate inputs, prepare retained payloads,
    construct typed records, and pass them to `_submit_record`. Each family
    owns detachment and storage preservation inside its runtime submission.

    The base owns `_current_records` and `_history_records`; subclasses expose
    completed data through `result`. Each context starts with an empty queue.

    Enter and exit outside compiled functions. Ordered runtime submission keeps
    collection state and submission method bodies outside tracing. A CPU
    identity tensor routes each call to its recorder. Submission is
    excluded from CUDA Graph capture so every invocation reaches the recorder.
    Exit releases the active slot and queue; failed execution discards the batch
    and propagates the exception. Clean execution calls `_merge_records`, then
    `_sync_history`, then appends the new entries to history. Errors propagate
    without recovery.

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
        self._register_identity()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._register_identity()

    def __init_subclass__(cls, *, base_only: bool = False, **kwargs: object) -> None:
        super().__init_subclass__(base_only=base_only, **kwargs)
        if RecorderBase in cls.__bases__:
            cls.__family_root = cls
            cls.__active_recorder = None

    # === Public API ===

    @staticmethod
    def submission[OwnerT: RecorderBase[Any, Any, Any], **P](
        method: Callable[Concatenate[OwnerT, P], None],
    ) -> Callable[Concatenate[OwnerT, P], None]:
        """Run a submission method eagerly through an ordered graph operation.

        Decorated methods return `None`, leave argument tensors unchanged, and
        annotate their named arguments after `self` with types supported by
        `torch.library.custom_op`. Their signatures supply the operator schemas.
        Tensor values stay dynamic; Python scalar and string arguments
        specialize compiled callers. The method body runs
        once per invocation, including under `fullgraph=True`, and is skipped
        during fake execution. It may read recorder state and construct records.
        """
        method_signature = signature(method, eval_str=True)
        parameters = list(method_signature.parameters.values())
        owner_name = parameters.pop(0).name
        if any(
            parameter.kind not in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY) for parameter in parameters
        ):
            raise TypeError("submission arguments must be named parameters")
        # Custom ops require positional tensor parameters. Preserve the public
        # method's binding and normalize only the operator's internal signature.
        parameters.sort(key=lambda parameter: parameter.default is not Parameter.empty)
        parameters = [parameter.replace(kind=Parameter.POSITIONAL_OR_KEYWORD) for parameter in parameters]
        argument_names = tuple(parameter.name for parameter in parameters)

        def deliver(identity: Tensor, *args: object, **kwargs: object) -> None:
            # The CPU identity is routing metadata, never accelerator data.
            arguments = dict(zip(argument_names, args, strict=False))
            arguments.update(kwargs)
            cast("Callable[..., None]", method)(_recorders[int(identity.item())], **arguments)

        # Infer the schema from the original method, replacing only its owner.
        identity_parameter = Parameter("identity", Parameter.POSITIONAL_OR_KEYWORD, annotation=Tensor)
        cast("Any", deliver).__signature__ = method_signature.replace(parameters=[identity_parameter, *parameters])
        operation = torch.library.custom_op(
            f"neurox::record_{method.__name__}_{next(_submissions)}",
            deliver,
            mutates_args=(),
            tags=(torch.Tag.cudagraph_unsafe,),
        )

        @operation.register_fake
        def fake(*args: object, **kwargs: object) -> None:
            pass

        operation.register_effect(EffectType.ORDERED)

        @wraps(method)
        def submit(self: OwnerT, /, *args: P.args, **kwargs: P.kwargs) -> None:
            # Capture an opaque call; eager execution uses the same method body.
            if torch.compiler.is_compiling():
                bound = method_signature.bind(self, *args, **kwargs)
                arguments = {name: value for name, value in bound.arguments.items() if name != owner_name}
                operation(self._identity, **arguments)
            else:
                method(self, *args, **kwargs)

        return submit

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
        return cast("Self | None", family_root.__active_recorder)  # noqa: SLF001

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

    def _register_identity(self) -> None:
        identity = next(_identities)
        self._identity = torch.tensor(identity, dtype=torch.int64, device="cpu")
        _recorders[identity] = self

    @final
    def _submit_record(self, record: RecordT) -> None:
        """Append a prepared record to this instance's current batch.

        Call inside a decorated submission method, after that method has
        detached and preserved any tensor fields it retains.

        Raises:
            RuntimeError: This instance is not the family's active recorder.
        """
        if self.current() is not self:
            raise RuntimeError("records can only be submitted to the active recorder")
        self._current_records.append(record)
