"""Per-unit-operation energy layout: a record keeps the caller's leading dims and sums every axis past them."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from neurox import Profiler, stamp_names
from neurox.common.module import ConfigBase, ModuleBase, PolicyBase

# Pairwise-distinct extents: every axis is identifiable from a reduced shape
# alone, so a mis-grouped reduction cannot hide behind an aliased size.
_LEADING_NUM = 2  # caller unit operations
_ROUND_NUM = 5  # an emitter work axis
_INST_NUM = 3  # a second emitter work axis
_DETAIL_NUM = 7  # an emitter detail axis


class _Config(ConfigBase):
    pass


class _Policy(PolicyBase):
    pass


class _Emitter(ModuleBase):
    """Minimal emitting module."""

    def __init__(self, inst_count: int = 1) -> None:
        super().__init__(config=_Config(), policy=_Policy(), inst_shape=(inst_count,))

    @property
    def _area_per_inst__um2(self) -> float:
        return 0.0

    @property
    def _leakage_per_inst__uW(self) -> float:
        return 0.0

    def emit(self, energy: Tensor, *, channel: str | None = None) -> None:
        self._record_dynamic_energy(energy, channel=channel)


def _billed_energy() -> Tensor:
    """A distinct-valued `[2, 5, 3, 7]` energy, so any mis-reduction shows."""
    shape = (_LEADING_NUM, _ROUND_NUM, _INST_NUM, _DETAIL_NUM)
    return torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)


# === Reduction ===


@pytest.mark.parametrize("rank", [None, 0, 1, 2, 3, 4])
def test_reduction_preserves_the_caller_prefix_and_total(rank: int | None) -> None:
    x = _billed_energy()
    emitter = _Emitter()
    stamp_names(emitter)
    profiler = Profiler() if rank is None else Profiler(leading_rank=rank)
    expected_rank = 0 if rank is None else rank
    with profiler:
        emitter.emit(x)
    (record,) = profiler.records
    energy = record.dynamic_energy__fJ
    assert energy.shape == x.shape[:expected_rank]
    expected = x if expected_rank == x.ndim else x.sum(dim=tuple(range(expected_rank, x.ndim)))
    torch.testing.assert_close(energy, expected)
