"""Per-unit-operation energy layout: a record keeps the caller's leading dims and sums every axis past them."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from neurox import Profiler, Reporter, stamp_names
from neurox.common import ConfigBase, ModuleBase, PolicyBase

_E_OP__FJ = 0.5

# Pairwise-distinct extents: every axis is identifiable from a reduced shape
# alone, so a mis-grouped reduction cannot hide behind an aliased size.
_LEADING_NUM = 2  # caller unit operations
_ROUND_NUM = 5  # an emitter work axis
_INST_NUM = 3  # a second emitter work axis
_DETAIL_NUM = 7  # an emitter detail axis

_DELTA__FJ = 1024.0  # one perturbation, larger than any billed element


class _Config(ConfigBase):
    pass


class _Policy(PolicyBase):
    pass


class _Emitter(ModuleBase[_Config, _Policy]):
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


class _Other(_Emitter):
    """A second emitter class, so two hosts bill side by side in one book."""


def _billed_energy() -> Tensor:
    """A distinct-valued `[2, 5, 3, 7]` energy, so any mis-reduction shows."""
    shape = (_LEADING_NUM, _ROUND_NUM, _INST_NUM, _DETAIL_NUM)
    return torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)


def _total__fJ(emitter: nn.Module, profiler: Profiler) -> float:
    """The measurement's whole dynamic energy, reported against its one emitter."""
    return Reporter(emitter).total_dynamic_energy__fJ(profiler)


# === Reduction ===


def test_default_profiler_reproduces_a_full_sum_exactly() -> None:
    """`leading_rank=0` collapses every axis."""
    x = _billed_energy()
    emitter = _Emitter()
    stamp_names(emitter)
    with Profiler() as p:
        emitter.emit(x)
    (record,) = p.records
    assert record.dynamic_energy__fJ.shape == ()
    assert record.dynamic_energy__fJ.item() == x.sum().item()
    assert _total__fJ(emitter, p) == x.sum().item()


def test_leading_dims_survive_as_per_unit_operation_energy() -> None:
    x = _billed_energy()
    emitter = _Emitter()
    stamp_names(emitter)
    with Profiler(leading_rank=1) as p:
        emitter.emit(x)
    (record,) = p.records
    assert record.dynamic_energy__fJ.shape == (_LEADING_NUM,)
    torch.testing.assert_close(record.dynamic_energy__fJ, x.sum(dim=(1, 2, 3)))
    assert _total__fJ(emitter, p) == x.sum().item()


@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4])
def test_every_rank_keeps_exactly_its_prefix_and_sums_the_rest(rank: int) -> None:
    x = _billed_energy()
    emitter = _Emitter()
    stamp_names(emitter)
    with Profiler(leading_rank=rank) as p:
        emitter.emit(x)
    energy = p.records[0].dynamic_energy__fJ
    assert energy.shape == x.shape[:rank]
    expected = x if rank == x.ndim else x.sum(dim=tuple(range(rank, x.ndim)))
    torch.testing.assert_close(energy, expected)


@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4])
def test_the_total_is_invariant_across_every_leading_rank(rank: int) -> None:
    x = _billed_energy()
    emitter = _Emitter()
    stamp_names(emitter)
    with Profiler(leading_rank=rank) as p:
        emitter.emit(x)
    assert _total__fJ(emitter, p) == x.sum().item()


def test_an_energy_of_exactly_the_leading_rank_is_kept_whole() -> None:
    x = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    emitter = _Emitter()
    stamp_names(emitter)
    with Profiler(leading_rank=2) as p:
        emitter.emit(x)
    torch.testing.assert_close(p.records[0].dynamic_energy__fJ, x)


# === Caller-leading locality ===


def test_perturbing_one_caller_operation_moves_only_that_operation() -> None:
    emitter = _Emitter()
    stamp_names(emitter)
    base = _billed_energy()
    perturbed = base.clone()
    perturbed[0] += _DELTA__FJ  # caller operation 0 only, everything it touches

    def measure(x: Tensor) -> Tensor:
        with Profiler(leading_rank=1) as p:
            emitter.emit(x)
        return p.records[0].dynamic_energy__fJ

    response = measure(perturbed) - measure(base)
    assert response.shape == (_LEADING_NUM,)
    assert response[1].item() == 0.0
    assert response[0].item() == _DELTA__FJ * _ROUND_NUM * _INST_NUM * _DETAIL_NUM


# === Constructor validation ===


def test_negative_leading_rank_is_rejected() -> None:
    with pytest.raises(ValueError, match="leading_rank must be non-negative"):
        Profiler(leading_rank=-1)


# === Flat per-op lump ===


def _flat_lump__fJ(energy_per_op__fJ: float, shape: tuple[int, ...]) -> Tensor:
    """The emitter-side idiom for a constant bill: a 0-dim constant, expanded."""
    return torch.full((), energy_per_op__fJ, dtype=torch.float32).expand(shape)


def test_a_flat_lump_carries_the_constants_dtype_not_the_billed_layouts() -> None:
    emitter = _Emitter()
    stamp_names(emitter)
    with Profiler() as p:
        emitter.emit(_flat_lump__fJ(_E_OP__FJ, (2, 3, 4)))
    energy = p.records[0].dynamic_energy__fJ
    assert energy.dtype == torch.float32
    assert energy.item() == 12.0


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_a_flat_lump_matches_the_materialized_energy(rank: int) -> None:
    shape = (2, 3, 4, 5)
    expanded, materialized = _Emitter(), _Other()
    stamp_names(expanded)
    stamp_names(materialized)
    with Profiler(leading_rank=rank) as p:
        expanded.emit(_flat_lump__fJ(_E_OP__FJ, shape))
        materialized.emit(torch.full(shape, _E_OP__FJ, dtype=torch.float32))
    a, b = (record.dynamic_energy__fJ for record in p.records)
    assert a.shape == b.shape == shape[:rank]
    torch.testing.assert_close(a, b)


def test_a_flat_lump_is_reduced_without_ever_being_materialized() -> None:
    emitter = _Emitter()
    stamp_names(emitter)
    lump = _flat_lump__fJ(1.0, (4, 1024, 1024))
    assert lump.untyped_storage().size() == lump.element_size()
    with Profiler(leading_rank=1) as p:
        emitter.emit(lump)
    energy = p.records[0].dynamic_energy__fJ
    assert energy.shape == (4,)
    torch.testing.assert_close(energy, torch.full((4,), float(1024 * 1024)))


def test_a_per_op_constant_may_vary_over_the_caller_block() -> None:
    emitter = _Emitter()
    stamp_names(emitter)
    with Profiler(leading_rank=1) as p:
        emitter.emit(torch.tensor([1.0, 2.0]).unsqueeze(-1).expand(2, 3))
    torch.testing.assert_close(p.records[0].dynamic_energy__fJ, torch.tensor([3.0, 6.0]))


# === Channels and record identity ===


def test_two_channels_on_one_module_bill_independently() -> None:
    emitter = _Emitter()
    stamp_names(emitter)
    with Profiler(leading_rank=1) as p:
        emitter.emit(torch.ones(2, 4), channel="rail")
        emitter.emit(torch.ones(2, 3), channel="cap")
        emitter.emit(torch.ones(2, 3))  # un-channelled branch alongside the channelled ones
    assert [record.channel for record in p.records] == ["rail", "cap", None]
    assert _total__fJ(emitter, p) == 20.0


def test_branches_of_one_module_may_carry_different_work_axes() -> None:
    """Only the caller block is shared; past it each branch keeps its own layout."""
    emitter = _Emitter()
    stamp_names(emitter)
    with Profiler(leading_rank=1) as p:
        emitter.emit(torch.ones(2, 4), channel="rail")
        emitter.emit(torch.ones(2, 3, 5, 7), channel="cap")
    a, b = (record.dynamic_energy__fJ for record in p.records)
    assert a.shape == b.shape == (2,)


def test_two_indistinguishable_emissions_stay_two_records() -> None:
    """A record equals only itself, so re-billing one branch never folds the book."""
    emitter = _Emitter()
    stamp_names(emitter)
    with Profiler(leading_rank=1) as p:
        emitter.emit(torch.ones(2, 3, 4), channel="rail")
        emitter.emit(torch.ones(2, 3, 4), channel="rail")
    first, second = p.records
    assert first is not second
    assert first != second
    assert first in p.records


# === Parking the records ===


def test_a_record_holds_the_laid_out_energy_after_the_ledger_closes() -> None:
    """The fold happens at emission; the exit sweep only parks what it produced, detached and on CPU."""
    emitter = _Emitter()
    stamp_names(emitter)
    with Profiler(leading_rank=1) as p:
        emitter.emit(_billed_energy())
    (record,) = p.records
    assert record.dynamic_energy__fJ.device == torch.device("cpu")
    assert record.dynamic_energy__fJ.requires_grad is False
