"""A record carries the name its tree stamped, read at emission, and never the module itself."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neurox import Profiler, Reporter, stamp_names
from neurox.common import ConfigBase, ModuleBase, PolicyBase


class _Config(ConfigBase):
    pass


class _Policy(PolicyBase):
    pass


class _Leaf(ModuleBase[_Config, _Policy]):
    """Minimal emitting module."""

    def __init__(self, *, energy__fJ: float = 0.0) -> None:
        super().__init__(config=_Config(), policy=_Policy(), inst_shape=())
        self._energy__fJ = energy__fJ

    @property
    def _area_per_inst__um2(self) -> float:
        return 2.0

    @property
    def _leakage_per_inst__uW(self) -> float:
        return 0.5

    def run(self, *, channel: str | None = None) -> None:
        if self._energy__fJ:
            self._record_dynamic_energy(torch.tensor(self._energy__fJ), channel=channel)


class _Owner(nn.Module):
    """A plain container binding one emitting child under a role name."""

    def __init__(self, leaf: _Leaf) -> None:
        super().__init__()
        self.leaf = leaf


def test_a_record_carries_the_stamped_name() -> None:
    leaf = _Leaf(energy__fJ=4.0)
    stamp_names(_Owner(leaf))
    with Profiler() as profiler:
        leaf.run()
    (record,) = profiler.records
    assert record.qualified_name == "leaf"


def test_one_emitters_records_are_selected_by_its_name_alone() -> None:
    a, b = _Leaf(energy__fJ=4.0), _Leaf(energy__fJ=7.0)
    owner = _Owner(a)
    owner.other = b
    stamp_names(owner)
    with Profiler() as profiler:
        a.run()
        b.run()
    energies = [float(r.dynamic_energy__fJ.sum()) for r in profiler.records if r.qualified_name == "other"]
    assert energies == [7.0]


def test_an_unstamped_emitter_fails_in_its_own_frame() -> None:
    leaf = _Leaf(energy__fJ=4.0)
    with Profiler(), pytest.raises(RuntimeError, match="carries no name stamp"):
        leaf.run()


def test_an_unstamped_emitter_is_free_to_run_unprofiled() -> None:
    _Leaf(energy__fJ=4.0).run()  # must not raise


def test_a_name_follows_a_post_construction_swap() -> None:
    """A replaced child takes the role name of where it lands, once the tree is walked again."""
    owner = _Owner(_Leaf())
    probe = _Leaf(energy__fJ=9.0)
    owner.leaf = probe  # the probe-install shape: swap after the tree exists
    stamp_names(owner)
    with Profiler() as profiler:
        probe.run()
    assert Reporter(owner).by_name(profiler) == {"leaf": 9.0}


def test_a_stamp_is_read_at_emission_not_at_report_time() -> None:
    """A later walk renaming the module cannot rewrite what the record already froze."""
    leaf = _Leaf(energy__fJ=4.0)
    owner = _Owner(leaf)
    stamp_names(owner)
    reporter = Reporter(owner)  # the canonical order: bind the reporter before the run
    with Profiler() as profiler:
        leaf.run()
    stamp_names(leaf)  # the same module, renamed by a walk of its own
    assert profiler.records[0].qualified_name == "leaf"
    assert reporter.by_name(profiler) == {"leaf": 4.0}


def test_a_channel_is_stored_verbatim_and_named_only_at_report_time() -> None:
    leaf = _Leaf(energy__fJ=4.0)
    owner = _Owner(leaf)
    stamp_names(owner)
    with Profiler() as profiler:
        leaf.run(channel="cablc")
        leaf.run(channel=None)
    assert [r.channel for r in profiler.records] == ["cablc", None]
    assert Reporter(owner).by_name(profiler) == {"leaf.cablc": 4.0, "leaf": 4.0}


def test_the_ledger_rejects_a_second_active_profiler() -> None:
    with Profiler(), pytest.raises(RuntimeError, match="only one Profiler"), Profiler():
        pass
