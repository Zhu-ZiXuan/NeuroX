"""Regression: CurrentMirror / CurrentMux log OUTPUT-ONLY dynamic energy.

Both blocks attribute rail dissipation to the output current only:

- ``CurrentMirror.replicate`` logs ``v_supply·|mirror_ratio·i_in|·t`` —
  the input branch term is excluded, so the total is strictly below the
  both-branch sum ``v_supply·(|i_in| + |i_out|)·t``.
- ``CurrentMux.transport`` logs ``v_supply·|mux_gain·i|·t`` on the
  shared output lane.

Energy is captured under :class:`NeuroxProfiler`; the profiler reduces
each per-element energy tensor with ``.detach().sum()`` at record time,
so the total for one block is the sum of its events' ``dynamic_energy__fJ``.
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


def test_current_mirror_logs_output_only_energy() -> None:
    """``replicate`` logs ``v_supply·|i_out|·t`` (output branch only)."""
    mirror_ratio = 2.0
    v_supply__V = 0.9
    read_pulse__ns = 3.0
    mirror = CurrentMirror(
        config=CurrentMirrorConfig(
            mirror_ratio=mirror_ratio,
            v_supply__V=v_supply__V,
            ratio_sigma_relative=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        policy=CurrentMirrorPolicy(mismatch=False),
        name="mirror",
        inst_shape=(),
        dtype=torch.float64,
        T__K=300.0,
        read_pulse__ns=read_pulse__ns,
    )
    mirror.eval()

    i_in__uA = torch.tensor([10.0, -4.0], dtype=torch.float64)
    i_out__uA = mirror_ratio * i_in__uA

    with NeuroxProfiler() as p:
        out = mirror.replicate(i_in__uA)
    torch.testing.assert_close(out, i_out__uA)

    logged = _energy_total(p.energy_events, "mirror")
    output_only = (v_supply__V * i_out__uA.abs() * read_pulse__ns).sum().item()
    both_branch = (v_supply__V * (i_in__uA.abs() + i_out__uA.abs()) * read_pulse__ns).sum().item()

    assert logged == output_only
    # Output-only excludes the input branch term, so it is strictly below
    # the both-branch dissipation.
    assert logged < both_branch


def test_current_mux_logs_output_only_energy() -> None:
    """``transport`` logs ``v_supply·|mux_gain·i|·t`` on the output lane."""
    select_num = 4
    v_supply__V = 0.9
    read_pulse__ns = 3.0
    i__uA = torch.tensor([10.0, -4.0], dtype=torch.float64)

    for mux_gain in (1.0, 2.0):
        mux = CurrentMux(
            config=CurrentMuxConfig(
                select_num=select_num,
                mux_gain=mux_gain,
                v_supply__V=v_supply__V,
                latency_per_op__ns=0.0,
                area_per_inst__um2=0.0,
                leakage_per_inst__uW=0.0,
            ),
            policy=CurrentMuxPolicy(),
            name="mux",
            inst_shape=(),
            dtype=torch.float64,
            T__K=300.0,
            read_pulse__ns=read_pulse__ns,
        )
        mux.eval()

        i_out__uA = mux_gain * i__uA
        with NeuroxProfiler() as p:
            out = mux.transport(i__uA)
        torch.testing.assert_close(out, i_out__uA)

        logged = _energy_total(p.energy_events, "mux")
        output_only = (v_supply__V * i_out__uA.abs() * read_pulse__ns).sum().item()
        assert logged == output_only
