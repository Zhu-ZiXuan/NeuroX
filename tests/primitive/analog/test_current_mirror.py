"""CurrentMirror is a pure transport primitive.

The mirror does not self-account rail energy: the current-domain consumer
that owns the rail tallies dissipation. ``CurrentMirror.replicate`` is a
pure ratio copy ``mirror_ratio·i_in`` (all-off) and logs NO dynamic energy
and NO latency. Events are captured under :class:`NeuroxProfiler`.
"""

from __future__ import annotations

import torch

from neurox.common.mixin import ProfileMixin
from neurox.common.profiler import NeuroxProfiler
from neurox.primitive.analog.current_mirror import (
    CurrentMirror,
    CurrentMirrorConfig,
    CurrentMirrorPolicy,
)


def _energy_total(events: list, module: ProfileMixin) -> float:
    """Sum the logged dynamic energy [fJ] of the events ``module`` emitted."""
    return sum(e.dynamic_energy__fJ for e in events if e.module is module)


def test_pure_copy_no_energy_no_latency() -> None:
    """``replicate`` is a pure ratio copy and logs no dynamic energy, no latency."""
    mirror_ratio = 2.0
    mirror = CurrentMirror(
        config=CurrentMirrorConfig(
            mirror_ratio=mirror_ratio,
            ratio_sigma_relative=0.1,
        ),
        policy=CurrentMirrorPolicy(mismatch=False),
        inst_shape=(),
        dtype=torch.float64,
        T__K=300.0,
    )
    mirror.eval()

    i_in__uA = torch.tensor([10.0, -4.0], dtype=torch.float64)
    i_out__uA = mirror_ratio * i_in__uA

    with NeuroxProfiler() as p:
        out = mirror.replicate(i_in__uA)
    torch.testing.assert_close(out, i_out__uA)

    # Pure transport primitive: no dynamic energy and no latency self-log.
    assert _energy_total(p.energy_events, mirror) == 0.0
    assert p.latency_events == []
