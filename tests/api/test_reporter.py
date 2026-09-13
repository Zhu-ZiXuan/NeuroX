"""Tests for the reporter: the one place a module or a record becomes a named row of the bound model's walk."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neurox import Profiler, Reporter, stamp_names
from neurox.common.module import ConfigBase, ModuleBase, PolicyBase

_AREA_PER_INST__UM2 = 2.0
_LEAKAGE_PER_INST__UW = 0.5


class _Config(ConfigBase):
    pass


class _Policy(PolicyBase):
    pass


class _Leaf(ModuleBase):
    """Minimal emitting module."""

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


class _Other(_Leaf):
    """A second emitter class, so a report covers more than one kind of host."""


class _Untargeted(ModuleBase):
    """A module whose silicon is counted at its owner, so it declares none."""

    is_profile_target = False

    def __init__(self) -> None:
        super().__init__(config=_Config(), policy=_Policy(), inst_shape=())


class _Owner(nn.Module):
    """A plain container binding one emitting child under a role name."""

    def __init__(self, leaf: nn.Module) -> None:
        super().__init__()
        self.leaf = leaf


# === Static entries ===


def test_the_bound_model_is_its_own_static_row() -> None:
    """`named_modules` names a root `""`; a target root holds that row."""
    leaf = _Leaf()
    stamp_names(leaf)
    (entry,) = Reporter(leaf).static_entries
    assert entry.qualified_name == ""


def test_a_non_target_holds_no_static_row() -> None:
    owner = _Owner(_Leaf())
    owner.shadow = _Untargeted()
    stamp_names(owner)
    assert [entry.qualified_name for entry in Reporter(owner).static_entries] == ["leaf"]


def test_static_entries_keep_the_models_traversal_order() -> None:
    owner = _Owner(_Leaf())
    owner.other = _Other()
    stamp_names(owner)
    assert [entry.qualified_name for entry in Reporter(owner).static_entries] == ["leaf", "other"]


def test_static_totals_sum_every_targets_row() -> None:
    owner = _Owner(_Leaf(inst_count=2))
    owner.other = _Other(inst_count=3)
    stamp_names(owner)
    static = Reporter(owner).static
    assert static.area__um2 == 5 * _AREA_PER_INST__UM2
    assert static.leakage__uW == 5 * _LEAKAGE_PER_INST__UW


# === The naming gate ===


def test_a_stamp_from_another_tree_stops_the_reporter_at_construction() -> None:
    """A stamp the current walk disagrees with is stale, and a stale name is refused."""
    leaf = _Leaf()
    stamp_names(_Owner(leaf))
    with pytest.raises(ValueError, match="stale or belongs to another tree"):
        Reporter(leaf)


def test_restamping_against_the_reported_model_clears_the_gate() -> None:
    """Reporting a subtree is naming that subtree, which is one more walk."""
    leaf = _Leaf()
    stamp_names(_Owner(leaf))
    stamp_names(leaf)
    (entry,) = Reporter(leaf).static_entries
    assert entry.qualified_name == ""


def test_an_instance_bound_at_a_second_location_stops_the_reporter() -> None:
    owner = _Owner(_Leaf())
    stamp_names(owner)
    owner.alias = owner.leaf  # bound a second time, after the walk that named the model
    with pytest.raises(ValueError, match="bound at both locations"):
        Reporter(owner)


def test_a_non_target_is_held_to_the_same_stamp() -> None:
    """The gate covers every NeuroX module, whether or not it holds a row."""
    owner = _Owner(_Leaf())
    stamp_names(owner)
    owner.shadow = _Untargeted()  # bound after the walk that named the model
    with pytest.raises(ValueError, match="carries no name stamp"):
        Reporter(owner)


# === Row names ===


def test_a_child_is_named_by_the_bound_model() -> None:
    """The name is the attribute path the model's traversal composes."""
    leaf = _Leaf(energy__fJ=4.0)
    owner = _Owner(leaf)
    stamp_names(owner)
    with Profiler() as profiler:
        leaf.run()
    assert Reporter(owner).by_name(profiler) == {"leaf": 4.0}


def test_the_bound_models_own_records_key_on_the_empty_name() -> None:
    """The empty name is falsy but valid — it is the model itself, not a missing name."""
    leaf = _Leaf(energy__fJ=4.0)
    stamp_names(leaf)
    with Profiler() as profiler:
        leaf.run()
    assert Reporter(leaf).by_name(profiler) == {"": 4.0}


def test_a_channel_reads_as_a_virtual_submodule_of_its_emitter() -> None:
    leaf = _Leaf(energy__fJ=4.0)
    owner = _Owner(leaf)
    stamp_names(owner)
    with Profiler() as profiler:
        leaf.run(channel="cablc")
    (entry,) = Reporter(owner).dynamic_entries(profiler)
    assert entry.path == ("leaf", "cablc")
    assert entry.qualified_name == "leaf.cablc"


def test_a_channel_on_the_bound_model_hangs_off_its_empty_name() -> None:
    """The model is named `""`, so what it bills itself reads under that empty segment."""
    leaf = _Leaf(energy__fJ=4.0)
    stamp_names(leaf)
    with Profiler() as profiler:
        leaf.run(channel="cablc")
    (entry,) = Reporter(leaf).dynamic_entries(profiler)
    assert entry.path == ("", "cablc")
    assert entry.qualified_name == ".cablc"


def test_a_channel_of_the_bound_model_never_shadows_a_top_level_child() -> None:
    """The leading empty segment separates a self-billed branch from a real child row."""
    root = _Leaf(energy__fJ=4.0)
    root.cablc = _Other(energy__fJ=3.0)
    stamp_names(root)
    with Profiler() as profiler:
        root.run(channel="cablc")
        root.cablc.run()
    assert Reporter(root).by_name(profiler) == {".cablc": 4.0, "cablc": 3.0}


def test_distinct_channels_on_one_module_stay_separate_rows() -> None:
    leaf = _Leaf(energy__fJ=4.0)
    owner = _Owner(leaf)
    stamp_names(owner)
    with Profiler() as profiler:
        leaf.run(channel="cablc")
        leaf.run(channel="dswct")
        leaf.run()  # an un-channelled branch of the same emitter
    reporter = Reporter(owner)
    assert reporter.by_name(profiler) == {"leaf.cablc": 4.0, "leaf.dswct": 4.0, "leaf": 4.0}
    assert reporter.total_dynamic_energy__fJ(profiler) == 12.0


# === Fail-loud naming ===


def test_a_channel_colliding_with_a_real_module_is_an_error() -> None:
    """Real and virtual children share one namespace."""
    leaf = _Leaf(energy__fJ=4.0)
    leaf.sub = nn.Identity()
    owner = _Owner(leaf)
    stamp_names(owner)
    with Profiler() as profiler:
        leaf.run(channel="sub")
    with pytest.raises(ValueError, match="collides with the real module"):
        Reporter(owner).by_name(profiler)


# === Grouping ===


def test_dynamic_entries_merge_by_path_and_lead_with_the_largest() -> None:
    small = _Leaf(energy__fJ=1.0)
    large = _Other(energy__fJ=5.0)
    owner = _Owner(small)
    owner.other = large
    stamp_names(owner)
    with Profiler() as profiler:
        small.run()
        large.run()
        small.run()
    entries = Reporter(owner).dynamic_entries(profiler)
    assert [entry.qualified_name for entry in entries] == ["other", "leaf"]
    assert [entry.dynamic_energy__fJ for entry in entries] == [5.0, 2.0]
    assert Reporter(owner).by_name(profiler) == {"leaf": 2.0, "other": 5.0}


def test_by_group_folds_the_mapped_rows_into_one_figure_each() -> None:
    """The caller states what belongs together; a channel row is a row like any other."""
    leaf = _Leaf(energy__fJ=4.0)
    other = _Other(energy__fJ=5.0)
    owner = _Owner(leaf)
    owner.other = other
    stamp_names(owner)
    with Profiler() as profiler:
        leaf.run(channel="rail")
        leaf.run()
        other.run()
    groups = {"leaf.rail": "analog", "leaf": "analog", "other": "digital"}
    assert Reporter(owner).by_group(profiler, groups) == {"analog": 8.0, "digital": 5.0}


def test_by_group_orders_labels_by_first_contribution() -> None:
    leaf = _Leaf(energy__fJ=4.0)
    other = _Other(energy__fJ=5.0)
    owner = _Owner(leaf)
    owner.other = other
    stamp_names(owner)
    with Profiler() as profiler:
        other.run()
        leaf.run()
        other.run()
    groups = {"leaf": "analog", "other": "digital"}
    assert list(Reporter(owner).by_group(profiler, groups)) == ["digital", "analog"]


def test_a_grouped_row_no_record_used_contributes_nothing() -> None:
    """One grouping outlives one measurement: the rows it maps need not all be exercised."""
    leaf = _Leaf(energy__fJ=4.0)
    owner = _Owner(leaf)
    owner.other = _Other(energy__fJ=5.0)
    stamp_names(owner)
    with Profiler() as profiler:
        leaf.run()
    groups = {"leaf": "analog", "leaf.rail": "analog", "other": "digital"}
    assert Reporter(owner).by_group(profiler, groups) == {"analog": 4.0}


def test_every_view_totals_the_same_measurement() -> None:
    leaf = _Leaf(energy__fJ=4.0)
    other = _Other(energy__fJ=5.0)
    owner = _Owner(leaf)
    owner.other = other
    stamp_names(owner)
    with Profiler() as profiler:
        leaf.run(channel="rail")
        leaf.run()
        other.run()
    reporter = Reporter(owner)
    groups = {"leaf.rail": "analog", "leaf": "analog", "other": "digital"}
    total = reporter.total_dynamic_energy__fJ(profiler)
    assert total == 13.0
    assert sum(reporter.by_name(profiler).values()) == total
    assert sum(reporter.by_group(profiler, groups).values()) == total
    assert sum(entry.dynamic_energy__fJ for entry in reporter.dynamic_entries(profiler)) == total


def test_a_measurement_that_recorded_nothing_totals_zero() -> None:
    """An empty book is a valid report: zero total, no rows, no error."""
    owner = _Owner(_Leaf())
    stamp_names(owner)
    reporter = Reporter(owner)
    with Profiler() as profiler:
        pass
    assert reporter.total_dynamic_energy__fJ(profiler) == 0.0
    assert reporter.by_name(profiler) == {}
    assert reporter.dynamic_entries(profiler) == ()


# === Render ===


def test_render_dumps_both_tables_in_canonical_units() -> None:
    """The dump is one text: dynamic rows and total, then static rows and totals."""
    leaf = _Leaf(energy__fJ=4.0)
    owner = _Owner(leaf)
    stamp_names(owner)
    with Profiler() as profiler:
        leaf.run(channel="rail")
    text = Reporter(owner).render(profiler)
    assert "dynamic" in text
    assert "static" in text
    assert "energy [fJ]" in text
    assert "area [um2]" in text
    assert "leakage [uW]" in text
    assert "leaf.rail" in text
    assert "total" in text


def test_render_without_a_profiler_is_the_static_table_alone() -> None:
    owner = _Owner(_Leaf())
    stamp_names(owner)
    text = Reporter(owner).render()
    assert "static" in text
    assert "dynamic" not in text
    assert "leaf" in text
