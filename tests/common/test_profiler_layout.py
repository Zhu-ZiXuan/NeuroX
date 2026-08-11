"""Tests for the profiler's per-unit-operation energy layout.

The recorded payload follows one rule: sum every axis past the caller's leading
dims, keep the caller's leading dims. One event element is therefore the energy
of one caller unit operation, and ``leading_rank=0`` collapses everything to a
scalar -- the bit-exactness anchor every accounting test rests on.

The emitter declares nothing about its own axes; ``leading_rank`` is a property
of the measurement site alone. There is no runtime check on it: getting it
wrong never changes a total, only how finely the per-unit-operation view
resolves.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.mixin import ProfileMixin
from neurox.common.profiler import NeuroxProfiler

_E_OP__FJ = 0.5

# Pairwise-distinct extents: every axis is identifiable from a reduced shape
# alone, so a mis-grouped reduction cannot hide behind an aliased size.
_LEADING_NUM = 2  # caller unit operations
_ROUND_NUM = 5  # an emitter work axis
_INST_NUM = 3  # a second emitter work axis
_DETAIL_NUM = 7  # an emitter detail axis

_DELTA__FJ = 1024.0  # one perturbation, larger than any payload element


class _Emitter(nn.Module, ProfileMixin):
    """Minimal emitting host."""

    def __init__(self, inst_count: int = 1) -> None:
        nn.Module.__init__(self)
        self._inst_count = inst_count

    @property
    def _area_per_inst__um2(self) -> float:
        return 0.0

    @property
    def _leakage_per_inst__uW(self) -> float:
        return 0.0

    @property
    def inst_count(self) -> int:
        return self._inst_count

    def emit(self, energy: Tensor, *, channel: str | None = None) -> None:
        self._record_dynamic_energy(energy, channel=channel)


class _Other(_Emitter):
    """A second emitter class, so a disagreement names two module types."""


def _payload() -> Tensor:
    """A distinct-valued ``[2, 5, 3, 7]`` payload, so any mis-reduction shows."""
    shape = (_LEADING_NUM, _ROUND_NUM, _INST_NUM, _DETAIL_NUM)
    return torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)


# === Reduction ===


def test_default_profiler_reproduces_a_full_sum_exactly() -> None:
    """``leading_rank=0`` collapses every axis: the anchor for existing accounting."""
    x = _payload()
    emitter = _Emitter()
    with NeuroxProfiler() as p:
        emitter.emit(x)
    (event,) = p.energy_events
    assert event.dynamic_energy__fJ.shape == ()
    assert event.dynamic_energy__fJ.item() == x.sum().item()
    assert p.total_dynamic_energy__fJ == x.sum().item()


def test_leading_dims_survive_as_per_unit_operation_energy() -> None:
    """Each kept leading element is the energy of one caller unit operation."""
    x = _payload()
    emitter = _Emitter()
    with NeuroxProfiler(leading_rank=1) as p:
        emitter.emit(x)
    (event,) = p.energy_events
    assert event.dynamic_energy__fJ.shape == (_LEADING_NUM,)
    torch.testing.assert_close(event.dynamic_energy__fJ, x.sum(dim=(1, 2, 3)))
    assert p.total_dynamic_energy__fJ == x.sum().item()


@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4])
def test_every_rank_keeps_exactly_its_prefix_and_sums_the_rest(rank: int) -> None:
    """The reduction is positional: the first ``rank`` dims survive, nothing else."""
    x = _payload()
    emitter = _Emitter()
    with NeuroxProfiler(leading_rank=rank) as p:
        emitter.emit(x)
    energy = p.energy_events[0].dynamic_energy__fJ
    assert energy.shape == x.shape[:rank]
    expected = x if rank == x.ndim else x.sum(dim=tuple(range(rank, x.ndim)))
    torch.testing.assert_close(energy, expected)


@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4])
def test_the_total_is_invariant_across_every_leading_rank(rank: int) -> None:
    """Keeping axes never changes the total; only which axes remain addressable."""
    x = _payload()
    emitter = _Emitter()
    with NeuroxProfiler(leading_rank=rank) as p:
        emitter.emit(x)
    assert p.total_dynamic_energy__fJ == x.sum().item()


def test_a_payload_of_exactly_the_leading_rank_is_kept_whole() -> None:
    """With nothing past the caller block there is nothing to sum."""
    x = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    emitter = _Emitter()
    with NeuroxProfiler(leading_rank=2) as p:
        emitter.emit(x)
    torch.testing.assert_close(p.energy_events[0].dynamic_energy__fJ, x)


# === Caller-leading locality ===


def test_perturbing_one_caller_operation_moves_only_that_operation() -> None:
    """The per-unit-operation law: a change in operation 0 reaches row 0 and no other."""
    emitter = _Emitter()
    base = _payload()
    perturbed = base.clone()
    perturbed[0] += _DELTA__FJ  # caller operation 0 only, everything it touches

    def measure(x: Tensor) -> Tensor:
        with NeuroxProfiler(leading_rank=1) as p:
            emitter.emit(x)
        return p.energy_events[0].dynamic_energy__fJ

    response = measure(perturbed) - measure(base)
    assert response.shape == (_LEADING_NUM,)
    assert response[1].item() == 0.0
    assert response[0].item() == _DELTA__FJ * _ROUND_NUM * _INST_NUM * _DETAIL_NUM


# === Constructor validation ===


def test_negative_leading_rank_is_rejected() -> None:
    with pytest.raises(ValueError, match="leading_rank must be non-negative"):
        NeuroxProfiler(leading_rank=-1)


# === Flat per-op lump ===


def _flat_lump__fJ(energy_per_op__fJ: float, shape: tuple[int, ...]) -> Tensor:
    """The emitter-side idiom for a constant bill: a 0-dim constant, expanded."""
    return torch.full((), energy_per_op__fJ, dtype=torch.float32).expand(shape)


def test_a_flat_lump_carries_the_constants_dtype_not_the_billed_layouts() -> None:
    """The energy dtype is the emitter's to fix.

    The billed layout is typically an integer code or a reduced-precision
    signal, which would truncate a sub-unit per-op energy to zero.
    """
    emitter = _Emitter()
    with NeuroxProfiler() as p:
        emitter.emit(_flat_lump__fJ(_E_OP__FJ, (2, 3, 4)))
    energy = p.energy_events[0].dynamic_energy__fJ
    assert energy.dtype == torch.float32
    assert energy.item() == 12.0


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_a_flat_lump_matches_the_materialized_payload(rank: int) -> None:
    """The expanded view is an optimization: it must agree elementwise."""
    shape = (2, 3, 4, 5)
    expanded, materialized = _Emitter(), _Other()
    with NeuroxProfiler(leading_rank=rank) as p:
        expanded.emit(_flat_lump__fJ(_E_OP__FJ, shape))
        materialized.emit(torch.full(shape, _E_OP__FJ, dtype=torch.float32))
    a, b = (e.dynamic_energy__fJ for e in p.energy_events)
    assert a.shape == b.shape == shape[:rank]
    torch.testing.assert_close(a, b)


def test_a_flat_lump_is_reduced_without_ever_being_materialized() -> None:
    """Only the caller's leading dims are built, whatever the summed extents cost."""
    emitter = _Emitter()
    payload = _flat_lump__fJ(1.0, (4, 1024, 1024))
    assert payload.untyped_storage().size() == payload.element_size()
    with NeuroxProfiler(leading_rank=1) as p:
        emitter.emit(payload)
    energy = p.energy_events[0].dynamic_energy__fJ
    assert energy.shape == (4,)
    torch.testing.assert_close(energy, torch.full((4,), float(1024 * 1024)))


def test_a_per_op_constant_may_vary_over_the_caller_block() -> None:
    """One expansion per caller operation, when the per-op energy is not flat."""
    emitter = _Emitter()
    with NeuroxProfiler(leading_rank=1) as p:
        emitter.emit(torch.tensor([1.0, 2.0]).unsqueeze(-1).expand(2, 3))
    torch.testing.assert_close(p.energy_events[0].dynamic_energy__fJ, torch.tensor([3.0, 6.0]))


# === Channels and event identity ===


def test_two_channels_on_one_module_bill_independently() -> None:
    """A composite bills structurally different branches under distinct labels."""
    emitter = _Emitter()
    with NeuroxProfiler(leading_rank=1) as p:
        emitter.emit(torch.ones(2, 4), channel="rail")
        emitter.emit(torch.ones(2, 3), channel="cap")
        emitter.emit(torch.ones(2, 3))  # un-channelled branch alongside the channelled ones
    assert [e.channel for e in p.energy_events] == ["rail", "cap", None]
    assert p.total_dynamic_energy__fJ == 20.0


def test_branches_of_one_module_may_carry_different_work_axes() -> None:
    """Only the caller block is shared; past it each branch has its own layout."""
    emitter = _Emitter()
    with NeuroxProfiler(leading_rank=1) as p:
        emitter.emit(torch.ones(2, 4), channel="rail")
        emitter.emit(torch.ones(2, 3, 5, 7), channel="cap")
    a, b = (e.dynamic_energy__fJ for e in p.energy_events)
    assert a.shape == b.shape == (2,)


def test_scalar_views_are_pure_field_reads_after_exit() -> None:
    """One batched sync at exit; repeated scalar views never re-reduce an event."""
    emitter = _Emitter()
    with NeuroxProfiler() as p:
        emitter.emit(torch.ones(4))
        emitter.emit(torch.ones(6))
    stale = p.energy_events[0].dynamic_energy__fJ
    assert p.total_dynamic_energy__fJ == 10.0
    assert p.energy_by_type == {"_Emitter": 10.0}
    stale.zero_()  # a post-exit mutation cannot reach an already-synced total
    assert p.total_dynamic_energy__fJ == 10.0
    assert p.energy_by_type == {"_Emitter": 10.0}
    assert p.report(emitter).energy_by_name == {"": 10.0}


def test_events_compare_without_evaluating_their_tensors() -> None:
    """``EnergyEvent`` equality is emitter identity plus channel, not payload values."""
    emitter = _Emitter()
    with NeuroxProfiler(leading_rank=1) as p:
        emitter.emit(torch.ones(2, 3, 4))
        emitter.emit(torch.ones(2, 3, 4) * 7.0)
    first, second = p.energy_events
    assert first == second
    assert first in p.energy_events
