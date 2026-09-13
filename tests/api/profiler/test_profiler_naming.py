"""A record carries the name its tree stamped, read at emission, and never the module itself."""

from __future__ import annotations

import torch
import torch.nn as nn

from neurox import Profiler, Reporter, stamp_names
from neurox.common.module import ConfigBase, ModuleBase, PolicyBase


class _Config(ConfigBase):
    pass


class _Policy(PolicyBase):
    pass


class _Leaf(ModuleBase):
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


def test_an_unstamped_emitter_is_free_to_run_unprofiled() -> None:
    _Leaf(energy__fJ=4.0).run()  # must not raise


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
