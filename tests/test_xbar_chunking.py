"""Unit tests for :class:`Offset1T1RXbar`'s solver chunking scheduler.

Covers the chunking contract in ``vec_mat_mul``. A single
``solve_chunk_size`` budget bounds the per-chunk broadcast-leading
instance count directly, independent of which leading axes are serial
(A: x-batch / M / Sa) or inst (B: Sw / Tc / Tr):
  - ``solve_chunk_size == 0`` runs the single-block (full-broadcast) path.
  - any positive chunk size yields bit-exact ADC codes vs the unchunked
    path under deterministic (noise-off) policies, including chunk
    boundaries that straddle the A/B axis split.
  - ``chunk_size > total leading`` collapses to a single chunk.
  - the per-VMM profiler events (core energy / latency, WL DAC) stay
    single-count and value-invariant under chunking, and the core /
    DAC latency scales with the A-side (x-batch) serial count only —
    never with the B-side (inst) parallel count.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import Tensor

from neurox.analog.adc import AdcOperationPoint
from neurox.analog.dac import GeneralDACConfig
from neurox.common.profiler import NeuroxProfiler
from neurox.tools.xbar_adc._sampling import (
    build_offset_1t1r_xbar_all_off,
    load_distribution,
    make_generator,
    sample_w,
    sample_x_batches,
)
from neurox.xbar import Offset1T1RXbar

REPO_ROOT = Path(__file__).resolve().parent.parent
XBAR_CONFIG = REPO_ROOT / "example" / "config" / "1t1r_28nm.toml"


def _build(
    chunk_size: int,
    device: torch.device,
    *,
    inst: int = 4,
) -> Offset1T1RXbar:
    return build_offset_1t1r_xbar_all_off(
        XBAR_CONFIG,
        device=device,
        inst_shape=(inst,),
        solve_chunk_size=chunk_size,
    )


def _run(
    chunk_size: int,
    device: torch.device,
    *,
    x_batch: int = 8,
    inst: int = 4,
) -> torch.Tensor:
    xbar = _build(chunk_size, device, inst=inst)
    distribution = load_distribution(None, xbar)
    g = make_generator(0, device)
    w = next(
        iter(
            sample_w(distribution, xbar, n=xbar._inst_shape[0], batch_w=xbar._inst_shape[0], device=device, generator=g)
        )
    )
    xbar.program(w)
    x = next(
        iter(sample_x_batches(distribution, xbar, n_total=x_batch, batch_size=x_batch, device=device, generator=g))
    )
    # x.shape = (x_batch, row_num); add inst-broadcast slot at -2 so the
    # leading classifies cleanly into (x_batch, inst-broadcast) before
    # core.cim_read's internal WL-fanout unsqueeze. Broadcast leading is
    # then (x_batch, inst) → x_batch·inst flat instances to chunk over.
    x_with_inst_slot = x.unsqueeze(-2)
    op = AdcOperationPoint(adc_mode=0, adc_bits=8)
    result = xbar.vec_mat_mul(x_with_inst_slot, adc_operation_point=op)
    assert isinstance(result, Tensor)
    return result


@pytest.fixture(scope="module")
def fixture_config() -> Iterator[Path]:
    """Skip the suite if the chip-fixture TOML isn't present."""
    if not XBAR_CONFIG.is_file():
        pytest.skip(f"missing test fixture: {XBAR_CONFIG}")
    yield XBAR_CONFIG


@pytest.fixture(scope="module")
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def test_chunk_zero_runs_single_block(fixture_config: Path, device: torch.device) -> None:
    """``0`` disables chunking and matches an over-sized chunk clipped to extent."""
    codes_disabled = _run(0, device)
    codes_big = _run(100, device, x_batch=8)
    assert torch.equal(codes_disabled, codes_big), "single-block path and over-sized chunk must agree bit-exactly"


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 7, 8, 16, 32])
def test_chunk_size_bit_exact(fixture_config: Path, device: torch.device, chunk_size: int) -> None:
    """Any chunk size is bit-exact vs single-block under deterministic policy.

    Broadcast leading is ``(x_batch=8, inst=4)`` → 32 flat instances. The
    chunk sizes span exact divisors (1, 2, 4, 8, 16, 32) and remainder /
    boundary-straddling values (3, 7); since the budget is axis-agnostic,
    these naturally exercise chunk boundaries that cross the inst-axis
    grouping, which is the whole point of the merged knob.
    """
    codes_full = _run(0, device, x_batch=8)
    codes_chunked = _run(chunk_size, device, x_batch=8)
    assert torch.equal(codes_full, codes_chunked), (
        f"chunk_size={chunk_size} produced different codes "
        f"(max abs diff: {(codes_full.float() - codes_chunked.float()).abs().max().item():.4e})"
    )


def test_chunk_size_larger_than_extent_is_noop(fixture_config: Path, device: torch.device) -> None:
    """Over-sized chunks clip to the extent and match the single-block result."""
    codes_full = _run(0, device, x_batch=8)
    codes_big = _run(100, device, x_batch=8)
    assert torch.equal(codes_full, codes_big)


def _vmm_under_profiler(
    chunk_size: int,
    device: torch.device,
) -> tuple[int, int, float]:
    """Run one VMM under the profiler.

    Returns:
        ``(core_energy_event_count, core_latency_event_count, total_core_energy)``.
    """
    xbar = _build(chunk_size, device, inst=4)
    distribution = load_distribution(None, xbar)
    g = make_generator(0, device)
    w = next(iter(sample_w(distribution, xbar, n=4, batch_w=4, device=device, generator=g)))
    xbar.program(w)
    x = next(iter(sample_x_batches(distribution, xbar, n_total=8, batch_size=8, device=device, generator=g)))
    x_with_inst_slot = x.unsqueeze(-2)
    op = AdcOperationPoint(adc_mode=0, adc_bits=8)
    with NeuroxProfiler() as p:
        xbar.vec_mat_mul(x_with_inst_slot, adc_operation_point=op)
    # __exit__ auto-calls _finalize(); reading after the block sees float events.
    energy_events = [e for e in p.energy_events if e.qualified_name.endswith(".core")]
    latency_events = [e for e in p.latency_events if e.qualified_name.endswith(".core")]
    total = sum(e.dynamic_energy__fJ for e in energy_events)
    return len(energy_events), len(latency_events), total


def test_profiler_single_event_under_chunking(fixture_config: Path, device: torch.device) -> None:
    """Chunking must emit exactly ONE core energy event + ONE latency event
    per VMM, with total energy equal to the single-block path (bit-exact within fp)."""
    single_energy_count, single_latency_count, single_energy = _vmm_under_profiler(0, device)
    assert single_energy_count == 1, f"single-block emitted {single_energy_count} core energy events"
    assert single_latency_count == 1, f"single-block emitted {single_latency_count} core latency events"
    for cs in (2, 3, 5, 7):
        chunk_energy_count, chunk_latency_count, chunk_energy = _vmm_under_profiler(cs, device)
        assert chunk_energy_count == 1, (
            f"chunk_size={cs} emitted {chunk_energy_count} core energy events; "
            "the chunked path must aggregate per-chunk contributions into a single event"
        )
        assert chunk_latency_count == 1, f"chunk_size={cs} emitted {chunk_latency_count} core latency events"
        assert abs(chunk_energy - single_energy) < 1e-3 * max(abs(single_energy), 1.0), (
            f"chunk_size={cs} energy {chunk_energy:.4f} vs single-block {single_energy:.4f}"
        )


def _dac_events_under_profiler(
    chunk_size: int,
    device: torch.device,
) -> tuple[int, int, float, float]:
    """Run one VMM under the profiler; return (energy_event_count,
    latency_event_count, total_dac_energy__fJ, total_dac_latency__ns)
    filtered to the WL DAC alone.
    """
    xbar = _build(chunk_size, device, inst=4)
    distribution = load_distribution(None, xbar)
    g = make_generator(0, device)
    w = next(iter(sample_w(distribution, xbar, n=4, batch_w=4, device=device, generator=g)))
    xbar.program(w)
    x = next(iter(sample_x_batches(distribution, xbar, n_total=8, batch_size=8, device=device, generator=g)))
    x_with_inst_slot = x.unsqueeze(-2)
    op = AdcOperationPoint(adc_mode=0, adc_bits=8)
    with NeuroxProfiler() as p:
        xbar.vec_mat_mul(x_with_inst_slot, adc_operation_point=op)
    e_events = [e for e in p.energy_events if e.qualified_name.endswith(".wl_dac")]
    l_events = [e for e in p.latency_events if e.qualified_name.endswith(".wl_dac")]
    e_total = sum(e.dynamic_energy__fJ for e in e_events)
    l_total = sum(e.latency__ns for e in l_events)
    return len(e_events), len(l_events), e_total, l_total


def test_wl_dac_events_invariant_under_chunking(fixture_config: Path, device: torch.device) -> None:
    """WL DAC must emit exactly ONE energy + ONE latency event per VMM,
    and both VALUES must be invariant across chunk sizes — a regression
    net against DAC-side over-counting of inst dims as serial when the
    WL-fanout singleton slot is fed into ``convert``. (Core-side serial
    mis-attribution has its own dedicated test —
    :func:`test_core_latency_scales_with_a_side_only`.)
    """
    base_e_count, base_l_count, base_e, base_l = _dac_events_under_profiler(0, device)
    assert base_e_count == 1, f"single-block emitted {base_e_count} wl_dac energy events"
    assert base_l_count == 1, f"single-block emitted {base_l_count} wl_dac latency events"
    for cs in (2, 3, 5, 7):
        e_count, l_count, e_val, l_val = _dac_events_under_profiler(cs, device)
        assert e_count == 1, f"chunk_size={cs} emitted {e_count} wl_dac energy events"
        assert l_count == 1, f"chunk_size={cs} emitted {l_count} wl_dac latency events"
        # Bit-exact value invariance — DAC must not be touched by chunk
        # bookkeeping at all.
        assert e_val == base_e, f"chunk_size={cs} wl_dac energy {e_val:.6f} != base {base_e:.6f}"
        assert l_val == base_l, f"chunk_size={cs} wl_dac latency {l_val:.6f} != base {base_l:.6f}"


# ---------------------------------------------------------------------------
# Core latency must depend only on A-side (x_batch), not on B-side (inst)
# ---------------------------------------------------------------------------


def _core_latency_at(
    *,
    x_batch: int,
    inst: int,
    device: torch.device,
    core_latency__ns: float,
    chunk_size: int = 0,
) -> float:
    """Build a fresh xbar with the core's ``latency_per_op__ns`` patched to a
    non-zero value, run one VMM under the given chunk size, and return the
    single core latency event's value.
    """
    xbar = _build(chunk_size, device, inst=inst)
    # The chip preset's core latency_per_op__ns is 0; substitute a fresh
    # frozen config with the test-only override via dataclasses.replace,
    # then rebind ``xbar.core.config`` to the new instance. This respects
    # the frozen-dataclass invariant (existing instance is never mutated)
    # and re-runs ``__post_init__`` validation on the new instance.
    xbar.core.config = replace(xbar.core.config, latency_per_op__ns=core_latency__ns)
    distribution = load_distribution(None, xbar)
    g = make_generator(0, device)
    w = next(iter(sample_w(distribution, xbar, n=inst, batch_w=inst, device=device, generator=g)))
    xbar.program(w)
    x = next(
        iter(sample_x_batches(distribution, xbar, n_total=x_batch, batch_size=x_batch, device=device, generator=g))
    )
    x_with_inst_slot = x.unsqueeze(-2)
    op = AdcOperationPoint(adc_mode=0, adc_bits=8)
    with NeuroxProfiler() as p:
        xbar.vec_mat_mul(x_with_inst_slot, adc_operation_point=op)
    core_l = [e for e in p.latency_events if e.qualified_name.endswith(".core")]
    assert len(core_l) == 1
    return core_l[0].latency__ns


def test_core_latency_scales_with_a_side_only(fixture_config: Path, device: torch.device) -> None:
    """Core latency must scale linearly with A-side (x_batch) and be
    invariant to B-side (inst).

    The serial-op count for ``cim_read`` is
    ``prod(leading[p] for p in a_positions)``. B-side (inst) positions
    are parallel physical xbars and must NOT enter the per-op-latency
    multiplier — multiplying by them would double-count against
    ``leakage_energy = leakage_power_total × total_latency`` since
    ``leakage_power_total`` already aggregates per-inst leakage.
    """
    T = 5.0  # ns; arbitrary non-zero
    # A-side scaling: holding inst, doubling x_batch must double core latency.
    l_x4 = _core_latency_at(x_batch=4, inst=4, device=device, core_latency__ns=T)
    l_x8 = _core_latency_at(x_batch=8, inst=4, device=device, core_latency__ns=T)
    assert l_x4 == T * 4, f"x_batch=4 inst=4 core latency = {l_x4}, expected {T * 4}"
    assert l_x8 == T * 8, f"x_batch=8 inst=4 core latency = {l_x8}, expected {T * 8}"
    # B-side invariance: holding x_batch, changing inst must NOT change latency.
    l_inst2 = _core_latency_at(x_batch=4, inst=2, device=device, core_latency__ns=T)
    l_inst4 = _core_latency_at(x_batch=4, inst=4, device=device, core_latency__ns=T)
    assert l_inst2 == l_inst4 == T * 4, (
        f"inst=2 gave {l_inst2}; inst=4 gave {l_inst4}; expected both = {T * 4} "
        "— core latency must NOT scale with inst count (parallel hardware)"
    )
    # Chunking is an internal memory schedule. It must not perturb the
    # logical serial-op count attached to the single core latency event.
    for cs in (2, 3, 5, 7):
        l_chunked = _core_latency_at(
            x_batch=4,
            inst=4,
            device=device,
            core_latency__ns=T,
            chunk_size=cs,
        )
        assert l_chunked == T * 4, f"chunk_size={cs} core latency = {l_chunked}, expected {T * 4}"


def test_core_latency_invariant_under_chunking(fixture_config: Path, device: torch.device) -> None:
    """Core latency value must be bit-exact invariant across chunk sizes at
    fixed (x_batch, inst). Chunking is an internal memory-bounding
    decomposition; the per-VMM event's latency tensor builds from
    ``self.config.latency_per_op__ns × serial_op_count`` where
    ``serial_op_count`` is determined purely by the broadcast leading's
    A-side classification — independent of how the chunk loop partitions
    positions within that leading.
    """
    T = 5.0  # ns; arbitrary non-zero
    x_batch, inst = 8, 4
    expected = T * x_batch
    base = _core_latency_at(
        x_batch=x_batch,
        inst=inst,
        device=device,
        core_latency__ns=T,
        chunk_size=0,
    )
    assert base == expected, f"base (no chunking) core latency = {base}, expected {expected}"
    for cs in (1, 2, 3, 5, 7):
        chunked = _core_latency_at(
            x_batch=x_batch,
            inst=inst,
            device=device,
            core_latency__ns=T,
            chunk_size=cs,
        )
        assert chunked == base, (
            f"chunk_size={cs} core latency = {chunked} != base {base}; "
            "chunking must not perturb the per-VMM core latency value"
        )


# ---------------------------------------------------------------------------
# WL DAC latency mirrors the core's A-side-only / chunk-invariance contract.
# The fixture chip preset has DAC ``latency_per_op__ns = 0`` so absolute
# value-correctness is invisible without a non-zero override — the existing
# ``test_wl_dac_events_invariant_under_chunking`` only proves event count
# + bit-exact value invariance, both of which pass trivially at T=0. These
# tests patch the DAC config to non-zero T to give the assertions real teeth.
# ---------------------------------------------------------------------------


def _dac_latency_at(
    *,
    x_batch: int,
    inst: int,
    device: torch.device,
    dac_latency__ns: float,
    chunk_size: int = 0,
) -> float:
    """Mirror of :func:`_core_latency_at` for the WL DAC."""
    xbar = _build(chunk_size, device, inst=inst)
    dac_cfg = xbar.core.wl_dac.config
    assert isinstance(dac_cfg, GeneralDACConfig)
    xbar.core.wl_dac.config = replace(dac_cfg, latency_per_op__ns=dac_latency__ns)
    distribution = load_distribution(None, xbar)
    g = make_generator(0, device)
    w = next(iter(sample_w(distribution, xbar, n=inst, batch_w=inst, device=device, generator=g)))
    xbar.program(w)
    x = next(
        iter(sample_x_batches(distribution, xbar, n_total=x_batch, batch_size=x_batch, device=device, generator=g))
    )
    x_with_inst_slot = x.unsqueeze(-2)
    op = AdcOperationPoint(adc_mode=0, adc_bits=8)
    with NeuroxProfiler() as p:
        xbar.vec_mat_mul(x_with_inst_slot, adc_operation_point=op)
    dac_l = [e for e in p.latency_events if e.qualified_name.endswith(".wl_dac")]
    assert len(dac_l) == 1
    return dac_l[0].latency__ns


def test_wl_dac_latency_scales_with_a_side_only(fixture_config: Path, device: torch.device) -> None:
    """DAC latency must scale with A-side (x_batch) and be invariant
    to B-side (inst) — same contract as ``cim_read`` itself. Catches
    any regression that pushes inst dims into the DAC's own serial
    count (e.g. via the WL-fanout slot being fed into ``convert``
    under a future positional-inference change).
    """
    T = 2.0
    l_x4 = _dac_latency_at(x_batch=4, inst=4, device=device, dac_latency__ns=T)
    l_x8 = _dac_latency_at(x_batch=8, inst=4, device=device, dac_latency__ns=T)
    assert l_x4 == T * 4, f"x_batch=4 inst=4 dac latency = {l_x4}, expected {T * 4}"
    assert l_x8 == T * 8, f"x_batch=8 inst=4 dac latency = {l_x8}, expected {T * 8}"
    l_inst2 = _dac_latency_at(x_batch=4, inst=2, device=device, dac_latency__ns=T)
    l_inst4 = _dac_latency_at(x_batch=4, inst=4, device=device, dac_latency__ns=T)
    assert l_inst2 == l_inst4 == T * 4, (
        f"inst=2 gave {l_inst2}; inst=4 gave {l_inst4}; expected both = {T * 4} "
        "— DAC latency must NOT scale with inst count"
    )


def test_wl_dac_latency_invariant_under_chunking(fixture_config: Path, device: torch.device) -> None:
    """DAC latency value must be bit-exact invariant across chunk sizes.
    The WL DAC runs once outside the chunk loop, so its profile event
    should never depend on the chunk partitioning.
    """
    T = 2.0
    x_batch, inst = 8, 4
    expected = T * x_batch
    base = _dac_latency_at(
        x_batch=x_batch,
        inst=inst,
        device=device,
        dac_latency__ns=T,
        chunk_size=0,
    )
    assert base == expected, f"base (no chunking) dac latency = {base}, expected {expected}"
    for cs in (1, 2, 3, 5, 7):
        chunked = _dac_latency_at(
            x_batch=x_batch,
            inst=inst,
            device=device,
            dac_latency__ns=T,
            chunk_size=cs,
        )
        assert chunked == base, (
            f"chunk_size={cs} dac latency = {chunked} != base {base}; "
            "chunking must not perturb the WL DAC's per-VMM latency value"
        )
