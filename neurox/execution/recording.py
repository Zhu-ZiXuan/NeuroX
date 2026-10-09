"""Explicit record-layout boundaries owned by NeuroX execution helpers."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from neurox.common.recording import recording_active, scope_begin, scope_end


def begin_record_scope(
    kind: Literal["scan", "map", "chunk", "group", "energy_sum", "energy_access"],
    *,
    shape: tuple[int, ...] = (),
    axis: int = 0,
) -> Tensor | None:
    """Mark a record transformation without changing numerical operands.

    Execution helpers own scan/map and positional chunk scopes. A physical
    owner may use `energy_sum` with its phase dimensions in `shape`; only
    additive energy submissions are reduced. Traces retain those positions.
    Scopes must nest and close within the same captured graph.
    """
    # Collection and compilation are Python controls specialized during tracing.
    if recording_active() and torch.compiler.is_compiling():
        token: Tensor = scope_begin(kind, list(shape), axis)
        return token
    return None


def end_record_scope(token: Tensor | None) -> None:
    # An absent token is the no-collection path, fixed during tracing.
    if token is not None:
        scope_end(token)
