"""Submission metadata shared by recorders and compiled execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from inspect import Signature

import torch
from torch import Tensor
from torch._library.effects import EffectType
from torch.fx.node import Target


@dataclass(frozen=True)
class Submission:
    signature: Signature
    deliver: Callable[..., None]
    payloads: tuple[str, ...]
    leading_rank: str | None
    additive: bool
    allow_loop: bool


_submissions: dict[Target, Submission] = {}
_active_count = 0


def register_submission(operation: Target, submission: Submission) -> None:
    """Register a recorder operation before tracing its callers."""
    _submissions[operation] = submission


def submission_for(operation: Target) -> Submission | None:
    return _submissions.get(operation)


def enter_recording() -> None:
    global _active_count
    _active_count += 1


def exit_recording() -> None:
    global _active_count
    _active_count -= 1


def recording_active() -> bool:
    return _active_count > 0


def trace_recording_active() -> bool:
    """Query diagnostic collection without coupling kernels to the API role."""
    from neurox.api.trace import TraceCollector

    return TraceCollector.active()


def record_trace(values: dict[str, Tensor], *, source: str) -> None:
    """Route a completed diagnostic payload to the active trace recorder."""
    from neurox.api.trace import TraceCollector

    collector = TraceCollector.current()
    if collector is not None:
        collector.submit(values, source=source)


@torch.library.custom_op("neurox::record_scope_begin", mutates_args=())
def scope_begin(kind: str, shape: list[int], axis: int) -> Tensor:
    raise RuntimeError("Recorded execution requires neurox.execution.compilation.compile_recording")


@scope_begin.register_fake
def _scope_begin_fake(kind: str, shape: list[int], axis: int) -> Tensor:
    return torch.empty(0)


@torch.library.custom_op("neurox::record_scope_end", mutates_args=())
def scope_end(token: Tensor) -> None:
    raise RuntimeError("Recorded execution requires neurox.execution.compilation.compile_recording")


@scope_end.register_fake
def _scope_end_fake(token: Tensor) -> None:
    pass


scope_begin.register_effect(EffectType.ORDERED)
scope_end.register_effect(EffectType.ORDERED)
