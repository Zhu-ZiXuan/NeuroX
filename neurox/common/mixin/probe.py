"""Shared mixin for probe-record emission."""

from __future__ import annotations

import torch
from torch import Tensor


class ProbeMixin:
    """Emit a host's per-call probe tensors to every active prober.

    One emit hook — ``_probe_record`` — hands named tensors from a host's
    primary method to the probe side channel. The hook is presence-gated
    only: it iterates the active prober stack and submits to each member;
    channel selection is the prober's business (its allowlist), never the
    emitter's. Every emit is a pure side channel — a no-op without an
    active prober that never alters host numerics. The collector half —
    storage, channel gating, pairing, name resolution — lives in the
    prober.

    Host requirements:
        - Inherit ``nn.Module`` alongside this mixin so
          ``Prober.resolve_names`` can name the emitter from a root's
          traversal.
        - Call ``_probe_record`` with the channel string a prober selects
          on and the named tensors that channel carries, at most once per
          logical operation per channel.
    """

    @torch.compiler.disable
    def _probe_record(self, channel: str, /, **tensors: Tensor) -> None:
        """Submit named tensors on ``channel`` to every active prober.

        Fast path: with an empty prober stack the call returns after one
        list check — no tensor is touched. ``@torch.compiler.disable``
        keeps the hook out of any caller's compiled graph.

        Args:
            channel: Probe channel name the record is submitted on.
            **tensors: Named per-call tensors; each prober stores them
                detached.
        """
        from neurox.common.prober import Prober  # local import: avoid cycle

        stack = Prober._active_stack
        if not stack:
            return
        for prober in stack:
            prober.submit(channel, self, tensors)
