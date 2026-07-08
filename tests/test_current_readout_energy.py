"""Regression: CurrentMirror / CurrentMux are pure transport primitives.

Neither block self-accounts rail energy: the current-domain consumer that
owns the rail tallies dissipation. These blocks only transform the current.

- ``CurrentMirror.replicate`` is a pure ratio copy
  ``mirror_ratio·ratio_mismatch·i_in`` and logs NO dynamic energy.
- ``CurrentMux.transport`` is a pure ``mux_gain·i`` copy that logs NO dynamic
  energy but KEEPS its transport-latency self-log.

Events are captured under :class:`NeuroxProfiler`.
"""

from __future__ import annotations

import torch

from neurox.analog.current_mirror import (
    CurrentMirror,
    CurrentMirrorConfig,
    CurrentMirrorPolicy,
)
from neurox.analog.current_mux import (
    CurrentMux,
    CurrentMuxConfig,
    CurrentMuxPolicy,
)
from neurox.common.profiler import NeuroxProfiler


def _energy_total(events: list, name: str) -> float:
    """Sum the logged dynamic energy [fJ] of events emitted by ``name``."""
    return sum(e.dynamic_energy__fJ for e in events if e.qualified_name == name)


def _latency_total(events: list, name: str) -> float:
    """Sum the logged latency [ns] of events emitted by ``name``."""
    return sum(e.latency__ns for e in events if e.qualified_name == name)


def test_current_mirror_pure_copy_no_energy() -> None:
    """``replicate`` is a pure ratio copy and logs no dynamic energy."""
    mirror_ratio = 2.0
    mirror = CurrentMirror(
        config=CurrentMirrorConfig(
            mirror_ratio=mirror_ratio,
            ratio_sigma_relative=0.1,
            area_per_inst__um2=1.0,
            leakage_per_inst__uW=1.0,
        ),
        policy=CurrentMirrorPolicy(mismatch=False),
        name="mirror",
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
    assert _energy_total(p.energy_events, "mirror") == 0.0


def test_current_mirror_copy_carries_ratio_mismatch() -> None:
    """With ``mismatch`` on the copy scales by the held per-instance ratio."""
    mirror = CurrentMirror(
        config=CurrentMirrorConfig(
            mirror_ratio=2.0,
            ratio_sigma_relative=0.1,
            area_per_inst__um2=1.0,
            leakage_per_inst__uW=1.0,
        ),
        policy=CurrentMirrorPolicy(mismatch=True),
        name="mirror",
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


def test_current_mux_pure_copy_no_energy_keeps_latency() -> None:
    """``transport`` is a pure ``mux_gain·i`` copy: no energy, latency kept."""
    select_num = 4
    latency_per_op__ns = 5.0
    i__uA = torch.tensor([10.0, -4.0], dtype=torch.float64)

    for mux_gain in (1.0, 2.0):
        mux = CurrentMux(
            config=CurrentMuxConfig(
                select_num=select_num,
                mux_gain=mux_gain,
                latency_per_op__ns=latency_per_op__ns,
                area_per_inst__um2=1.0,
                leakage_per_inst__uW=1.0,
            ),
            policy=CurrentMuxPolicy(),
            name="mux",
            inst_shape=(),
            dtype=torch.float64,
            T__K=300.0,
        )
        mux.eval()

        i_out__uA = mux_gain * i__uA
        with NeuroxProfiler() as p:
            out = mux.transport(i__uA)
        torch.testing.assert_close(out, i_out__uA)

        # No dynamic energy self-log.
        assert _energy_total(p.energy_events, "mux") == 0.0
        # Latency self-log is KEPT: inst_shape=() → inst_count 1, so
        # serial_op_count = numel = 2 visits.
        expected_latency__ns = latency_per_op__ns * i__uA.numel()
        assert _latency_total(p.latency_events, "mux") == expected_latency__ns
