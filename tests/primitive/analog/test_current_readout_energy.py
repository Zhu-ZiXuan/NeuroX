"""Regression: CurrentMirror / CurrentMux are pure transport primitives.

Neither block self-accounts rail energy: the current-domain consumer that
owns the rail tallies dissipation. These blocks only transform the current.

- ``CurrentMirror.replicate`` is a pure ratio copy
  ``mirror_ratio·ratio_mismatch·i_in`` and logs NO dynamic energy.
- ``CurrentMux.transport`` is a pure ``mux_gain·i`` copy that logs NO dynamic
  energy and NO latency.

Events are captured under :class:`NeuroxProfiler`.
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
from neurox.primitive.analog.current_mux import (
    CurrentMux,
    CurrentMuxConfig,
    CurrentMuxPolicy,
)


def _energy_total(events: list, module: ProfileMixin) -> float:
    """Sum the logged dynamic energy [fJ] of the events ``module`` emitted."""
    return sum(e.dynamic_energy__fJ for e in events if e.module is module)


def test_current_mirror_pure_copy_no_energy() -> None:
    """``replicate`` is a pure ratio copy and logs no dynamic energy."""
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

    # Pure transport primitive: no dynamic energy self-log.
    assert _energy_total(p.energy_events, mirror) == 0.0


def test_current_mirror_copy_carries_ratio_mismatch() -> None:
    """With ``mismatch`` on the copy scales by the held per-instance ratio."""
    mirror = CurrentMirror(
        config=CurrentMirrorConfig(
            mirror_ratio=2.0,
            ratio_sigma_relative=0.1,
        ),
        policy=CurrentMirrorPolicy(mismatch=True),
        inst_shape=(3,),
        dtype=torch.float64,
        T__K=300.0,
    )
    mirror.eval()
    mirror._sample_fabricate_mismatch()

    i_in__uA = torch.tensor([10.0, -4.0, 7.0], dtype=torch.float64)
    expected = 2.0 * mirror.ratio_mismatch * i_in__uA

    out = mirror.replicate(i_in__uA)
    torch.testing.assert_close(out, expected)


def test_current_mux_pure_copy_no_energy_no_latency() -> None:
    """``transport`` is a pure ``mux_gain·i`` copy: no energy, no latency."""
    select_num = 4
    i__uA = torch.tensor([10.0, -4.0], dtype=torch.float64)

    for mux_gain in (1.0, 2.0):
        mux = CurrentMux(
            config=CurrentMuxConfig(
                select_num=select_num,
                mux_gain=mux_gain,
            ),
            policy=CurrentMuxPolicy(),
            inst_shape=(),
            dtype=torch.float64,
            T__K=300.0,
        )
        mux.eval()

        i_out__uA = mux_gain * i__uA
        with NeuroxProfiler() as p:
            out = mux.transport(i__uA)
        torch.testing.assert_close(out, i_out__uA)

        # Pure transport primitive: no dynamic energy and no latency self-log.
        assert _energy_total(p.energy_events, mux) == 0.0
        assert p.latency_events == []
