"""CurrentMux is a pure transport primitive.

The mux does not self-account rail energy: the current-domain consumer
that owns the rail tallies dissipation. ``CurrentMux.transport`` is a pure
``mux_gain·i`` copy that logs NO dynamic energy and NO latency. Events are
captured under :class:`NeuroxProfiler`.
"""

from __future__ import annotations

import pytest
import torch

from neurox.common.mixin import ProfileMixin
from neurox.common.profiler import NeuroxProfiler
from neurox.primitive.analog.current_mux import (
    CurrentMux,
    CurrentMuxConfig,
    CurrentMuxPolicy,
)


def _energy_total(events: list, module: ProfileMixin) -> float:
    """Sum the logged dynamic energy [fJ] of the events ``module`` emitted."""
    return sum(e.dynamic_energy__fJ for e in events if e.module is module)


@pytest.mark.parametrize("mux_gain", [1.0, 2.0])
def test_pure_copy_no_energy_no_latency(mux_gain: float) -> None:
    """``transport`` is a pure ``mux_gain·i`` copy: no energy, no latency."""
    mux = CurrentMux(
        config=CurrentMuxConfig(
            select_num=4,
            mux_gain=mux_gain,
        ),
        policy=CurrentMuxPolicy(),
        inst_shape=(),
        dtype=torch.float64,
        T__K=300.0,
    )
    mux.eval()

    i__uA = torch.tensor([10.0, -4.0], dtype=torch.float64)
    i_out__uA = mux_gain * i__uA
    with NeuroxProfiler() as p:
        out = mux.transport(i__uA)
    torch.testing.assert_close(out, i_out__uA)

    # Pure transport primitive: no dynamic energy and no latency self-log.
    assert _energy_total(p.energy_events, mux) == 0.0
    assert p.latency_events == []
