"""Report ownership, virtual paths, aggregation, and measurement names."""

from __future__ import annotations

import torch
import torch.nn as nn

from neurox import Profiler, Reporter, stamp_names
from neurox.common.module import ConfigBase, NonProfileModule, PolicyBase, ProfileModule

_AREA_PER_INST__UM2 = 2.0
_LEAKAGE_PER_INST__UW = 0.5


class _Config(ConfigBase):
    pass


class _Policy(PolicyBase):
    pass


class _Leaf(ProfileModule):
    def __init__(self, *, energy__fJ: float = 0.0, inst_count: int = 1) -> None:
        super().__init__(config=_Config(), policy=_Policy(), inst_shape=(inst_count,))
        self._energy__fJ = energy__fJ

    @property
    def _area_per_inst__um2(self) -> float:
        return _AREA_PER_INST__UM2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return _LEAKAGE_PER_INST__UW

    def run(self, *, channel: str | None = None) -> None:
        self._record_dynamic_energy(torch.tensor(self._energy__fJ), channel=channel)


class _Untargeted(NonProfileModule):
    def __init__(self) -> None:
        super().__init__(config=_Config(), policy=_Policy(), inst_shape=())


class _Owner(nn.Module):
    def __init__(self, leaf: nn.Module) -> None:
        super().__init__()
        self.leaf = leaf


def test_static_ownership_crosses_non_profile_roots_and_plain_containers() -> None:
    owner = _Untargeted()
    owner.branch = _Owner(_Leaf(energy__fJ=4.0, inst_count=3))
    owner.other = _Leaf(inst_count=2)
    stamp_names(owner)
    reporter = Reporter(owner)
    with Profiler() as profiler:
        owner.branch.leaf.run()

    assert [entry.qualified_name for entry in reporter.static_entries] == ["branch.leaf", "other"]
    assert reporter.static.area__um2 == 5 * _AREA_PER_INST__UM2
    assert reporter.static.leakage__uW == 5 * _LEAKAGE_PER_INST__UW
    assert reporter.by_name(profiler) == {"branch.leaf": 4.0}


def test_root_records_and_channels_do_not_shadow_real_children() -> None:
    root = _Leaf(energy__fJ=4.0)
    root.cablc = _Leaf(energy__fJ=3.0)
    stamp_names(root)
    reporter = Reporter(root)
    with Profiler() as profiler:
        root.run()
        root.run(channel="cablc")
        root.cablc.run()

    assert [entry.qualified_name for entry in reporter.static_entries] == ["", "cablc"]
    assert reporter.by_name(profiler) == {"": 4.0, ".cablc": 4.0, "cablc": 3.0}
    assert {entry.qualified_name: entry.path for entry in reporter.dynamic_entries(profiler)} == {
        "": (),
        ".cablc": ("", "cablc"),
        "cablc": ("cablc",),
    }


def test_repeated_records_and_channels_aggregate_consistently_across_views() -> None:
    owner = _Owner(_Leaf(energy__fJ=2.0))
    owner.other = _Leaf(energy__fJ=5.0)
    owner.unused = _Leaf(energy__fJ=9.0)
    stamp_names(owner)
    with Profiler() as profiler:
        owner.leaf.run(channel="rail")
        owner.other.run()
        owner.leaf.run()
        owner.leaf.run(channel="rail")
        owner.leaf.run(channel="control")
    reporter = Reporter(owner)

    by_name = reporter.by_name(profiler)
    assert by_name == {"leaf.rail": 4.0, "other": 5.0, "leaf": 2.0, "leaf.control": 2.0}
    entries = reporter.dynamic_entries(profiler)
    assert [entry.qualified_name for entry in entries[:2]] == ["other", "leaf.rail"]
    assert {entry.qualified_name: entry.dynamic_energy__fJ for entry in entries} == by_name
    assert next(entry.path for entry in entries if entry.qualified_name == "leaf.rail") == ("leaf", "rail")

    groups = {
        "other": "digital",
        "unused": "unused",
        "leaf.rail": "analog",
        "leaf": "analog",
        "leaf.control": "digital",
    }
    by_group = reporter.by_group(profiler, groups)
    assert by_group == {"analog": 6.0, "digital": 7.0}
    assert list(by_group) == ["analog", "digital"]
    total = reporter.total_dynamic_energy__fJ(profiler)
    assert total == sum(by_name.values()) == sum(by_group.values()) == 13.0


def test_restamping_does_not_rewrite_an_existing_measurement() -> None:
    leaf = _Leaf(energy__fJ=4.0)
    owner = _Owner(leaf)
    stamp_names(owner)
    reporter = Reporter(owner)
    with Profiler() as profiler:
        leaf.run()
    stamp_names(leaf)
    assert profiler.records[0].qualified_name == "leaf"
    assert reporter.by_name(profiler) == {"leaf": 4.0}
