"""Tests that every profiled name is derived from the reported root's traversal.

A module never knows its own name, so an event carries its emitter and the
report resolves the name against the root it is handed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neurox.common.mixin import ProfileMixin
from neurox.common.profiler import NeuroxProfiler


class _Leaf(nn.Module, ProfileMixin):
    """Minimal emitting host: same base order as ``ModuleBase``."""

    def __init__(self, *, energy__fJ: float = 0.0, latency__ns: float = 0.0) -> None:
        nn.Module.__init__(self)
        self._area_per_inst__um2 = 2.0
        self._leakage_per_inst__uW = 0.5
        self._energy__fJ = energy__fJ
        self._latency__ns = latency__ns

    @property
    def inst_count(self) -> int:
        return 1

    def run(self, *, channel: str | None = None) -> None:
        if self._energy__fJ:
            self._log_dynamic_energy(torch.tensor(self._energy__fJ), channel=channel)
        if self._latency__ns:
            self._log_latency(torch.tensor(self._latency__ns))


class _Owner(nn.Module):
    def __init__(self, leaf: _Leaf) -> None:
        super().__init__()
        self.leaf = leaf


def test_child_is_named_by_the_tree() -> None:
    """The name is the attribute path the root's traversal composes."""
    owner = _Owner(_Leaf(energy__fJ=4.0))
    with NeuroxProfiler() as p:
        owner.leaf.run()
    assert p.report(owner).energy_by_name == {"leaf": 4.0}


def test_reported_root_is_named_empty_string() -> None:
    """``named_modules`` names a root ``""``; the root's own events key on it.

    The empty name is falsy but valid — it is not the unrooted case.
    """
    leaf = _Leaf(energy__fJ=4.0)
    with NeuroxProfiler() as p:
        leaf.run()
    by_name = p.report(leaf).energy_by_name
    assert by_name == {"": 4.0}


def test_emitter_outside_the_reported_root_is_labelled_unrooted() -> None:
    """An event the root cannot name stays visible under an ``<unrooted>`` label."""
    inside, outside = _Leaf(energy__fJ=4.0), _Leaf(energy__fJ=7.0)
    owner = _Owner(inside)
    with NeuroxProfiler() as p:
        inside.run()
        outside.run()
    by_name = p.report(owner).energy_by_name
    assert by_name == {"leaf": 4.0, "<unrooted>._Leaf": 7.0}


def test_unrooted_event_still_counts_toward_the_total() -> None:
    """Grouping never loses an event: the per-name sum is the total."""
    inside, outside = _Leaf(energy__fJ=4.0, latency__ns=1.0), _Leaf(energy__fJ=7.0, latency__ns=2.0)
    owner = _Owner(inside)
    with NeuroxProfiler() as p:
        inside.run()
        outside.run()
    report = p.report(owner)
    assert sum(report.energy_by_name.values()) == p.total_dynamic_energy__fJ == 11.0
    assert sum(report.latency_by_name.values()) == p.total_latency__ns == 3.0


def test_name_follows_a_post_construction_swap() -> None:
    """A replaced child takes the role name of where it lands, not where it was built."""
    owner = _Owner(_Leaf())
    probe = _Leaf(energy__fJ=9.0)
    owner.leaf = probe  # the probe-install shape: swap after the tree exists
    with NeuroxProfiler() as p:
        probe.run()
    assert p.report(owner).energy_by_name == {"leaf": 9.0}


def test_events_carry_the_emitter_so_identity_needs_no_name() -> None:
    """A caller holding a module selects its events without naming anything."""
    a, b = _Leaf(energy__fJ=4.0), _Leaf(energy__fJ=7.0)
    owner = _Owner(a)
    owner.other = b
    with NeuroxProfiler() as p:
        a.run()
        b.run()
    assert [e.dynamic_energy__fJ for e in p.energy_events if e.module is b] == [7.0]


def test_static_record_name_comes_from_the_walk() -> None:
    """The static walk names each host as it reaches it."""
    owner = _Owner(_Leaf())
    records = NeuroxProfiler.collect_static(owner)
    assert [r.qualified_name for r in records] == ["leaf"]
    assert records[0].module_type == "_Leaf"


def test_channelled_energy_groups_under_module_dot_channel() -> None:
    """A channelled event's report row reads ``<module dotted name>.<channel>``."""
    owner = _Owner(_Leaf(energy__fJ=4.0))
    with NeuroxProfiler() as p:
        owner.leaf.run(channel="cablc")
    assert p.report(owner).energy_by_name == {"leaf.cablc": 4.0}


def test_distinct_channels_on_the_same_module_stay_separate_rows() -> None:
    """One emitter billing multiple branches per op keeps each channel its own row."""
    owner = _Owner(_Leaf(energy__fJ=4.0))
    with NeuroxProfiler() as p:
        owner.leaf.run(channel="cablc")
        owner.leaf.run(channel="dswct")
        owner.leaf.run()  # un-channelled event on the same emitter
    by_name = p.report(owner).energy_by_name
    assert by_name == {"leaf.cablc": 4.0, "leaf.dswct": 4.0, "leaf": 4.0}
    assert sum(by_name.values()) == p.total_dynamic_energy__fJ == 12.0


def test_unrooted_channelled_event_keeps_the_channel_suffix() -> None:
    """An unrooted emitter's channel still appends onto the ``<unrooted>`` label."""
    outside = _Leaf(energy__fJ=7.0)
    owner = _Owner(_Leaf())
    with NeuroxProfiler() as p:
        outside.run(channel="dswct")
    assert p.report(owner).energy_by_name == {"<unrooted>._Leaf.dswct": 7.0}


def test_default_channel_is_none_and_matches_unchannelled_behavior() -> None:
    """``channel=None`` (the default) is byte-identical to the un-channelled call."""
    owner = _Owner(_Leaf(energy__fJ=4.0))
    with NeuroxProfiler() as p:
        owner.leaf.run(channel=None)
    report = p.report(owner)
    assert report.energy_by_name == {"leaf": 4.0}
    assert report.energy_events[0].channel is None
