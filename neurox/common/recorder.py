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
    explicitly passes its tensor payloads through `_export_tensor`, which
    detaches, copies and exports them to CPU before they are stored in a record.
    The base never traverses or interprets record fields.

    The base owns `_current_records` and `_history_records`; subclasses expose
    completed data through `result`. Each context starts with an empty queue.

    Enter and exit outside compiled functions. Ordered runtime submission keeps
    collection state and submission method bodies outside tracing. A CPU
    identity tensor routes each call to its recorder. Submission is
    excluded from CUDA Graph capture so every invocation reaches the recorder.
    CUDA-to-CPU exports use pinned destinations and a separate stream per source
    device. Destinations belong to the records and remain pinned while retained;
    their storage is never recycled while a record still references it. Other
    device inputs use ordinary CPU copies. No CUDA resources are created
    until the first CUDA-to-CPU export.

    Exit releases the active slot and waits for pending exports before merging
    records. Failed execution drains transfers and discards the batch. Clean
    execution calls `_merge_records` and appends its entries to history. Copying
    or serializing an active recorder raises `RuntimeError`; completed copies
    retain their history and create fresh transfer resources on demand.

    Raises:
        RuntimeError: Another recorder of this family is active on entry.
    """

    # Family slots cover different submission, history, and result types; each family's
    # public API retains its bound types and current-instance type.
    __family_root: ClassVar[type[RecorderBase[Any, Any, Any]]]
    __active_recorder: ClassVar[RecorderBase[Any, Any, Any] | None]

    def __init__(self) -> None:
        self._current_records: list[RecordT] = []
        self._history_records: list[HistoryT] = []
        self._export_streams: dict[torch.device, tuple[torch.cuda.Stream, torch.cuda.Event]] = {}
        self._pending_export_devices: set[torch.device] = set()
        self._register_identity()

    def __getstate__(self) -> dict[str, Any]:
        if self.current() is self:
            raise RuntimeError("copy or serialize a recorder outside its collection context")
        return {
            name: value
            for name, value in self.__dict__.items()
            if name not in ("_identity", "_export_streams", "_pending_export_devices")
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._export_streams = {}
        self._pending_export_devices = set()
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
        parameters = list(method_signature.parameters.values())[1:]
        submission_signature = method_signature.replace(parameters=parameters)
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
                arguments = submission_signature.bind(*args, **kwargs).arguments
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
        self._wait_for_exports()
        if exc_type is None:
            self._history_records.extend(self._merge_records(records))

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
        Exported tensors are ready on CPU. The active slot and submission queue
        have already been released. The returned sequence
        contains only this context's entries; the base owns appending them to history.
        """
        raise NotImplementedError

    # === Tools for subclass and internal use ===

    @final
    def _export_tensor(self, tensor: Tensor) -> Tensor:
        """Capture a detached snapshot and export it to CPU.

        Call inside a runtime submission. The snapshot owns independent storage
        even when no device transfer is needed, so later caller mutations and
        compiled buffer reuse cannot change the recorded values.

        CUDA-to-CPU copies are enqueued on a separate stream after the producing
        stream's work. The returned tensor must only be stored in the record;
        its CPU values become readable at `_merge_records` or after context exit.
        Source storage remains protected until the copy completes. Destination
        storage is owned by the returned tensor, including across later contexts.

        Args:
            tensor: Values to retain without their autograd history.
                Shape and dtype are preserved.

        Raises:
            RuntimeError: This instance is not the active recorder of its family.
        """
        if self.current() is not self:
            raise RuntimeError("tensors can only be exported by the active recorder")
        if not tensor.is_cuda:
            return tensor.detach().to(device="cpu", copy=True)

        tensor = tensor.detach().clone()
        source_device = tensor.device
        transfer = self._export_streams.get(source_device)
        if transfer is None:
            transfer = (torch.cuda.Stream(device=source_device), torch.cuda.Event())
            self._export_streams[source_device] = transfer
        stream, ready = transfer
        exported = torch.empty_like(tensor, device="cpu", pin_memory=True)
        self._pending_export_devices.add(source_device)
        ready.record(torch.cuda.current_stream(source_device))
        stream.wait_event(ready)
        # The snapshot can leave Python scope before the copy finishes. Register
        # its consumer stream so the allocator cannot reuse its storage early.
        tensor.record_stream(stream)
        with torch.cuda.stream(stream):
            exported.copy_(tensor, non_blocking=True)
        return exported

    def _wait_for_exports(self) -> None:
        pending = self._pending_export_devices
        self._pending_export_devices = set()
        failure: RuntimeError | None = None
        for device in pending:
            try:
                self._export_streams[device][0].synchronize()
            except RuntimeError as error:
                # Drain the remaining devices even if one transfer failed.
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure

    def _register_identity(self) -> None:
        identity = next(_identities)
        self._identity = torch.tensor(identity, dtype=torch.int64, device="cpu")
        _recorders[identity] = self

    @final
    def _submit_record(self, record: RecordT) -> None:
        """Append a prepared record to this instance's current batch.

        Call inside a decorated submission method, after passing every retained
        tensor field through `_export_tensor`.
        Exported CPU values are not read until the context has drained transfers.

        Raises:
            RuntimeError: This instance is not the family's active recorder.
        """
        if self.current() is not self:
            raise RuntimeError("records can only be submitted to the active recorder")
        self._current_records.append(record)
