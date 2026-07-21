"""Sub-phase plane contract tests: shape, per-instance decode, latency scaling.

The macro consumes pre-expanded zero-masked WL planes (primitive trailing
``[row_num]``, at most ``max_active_rows`` live rows each) and emits one ADC
code per column per plane with the leading order preserved and primitive
trailing ``[col_num]`` — no phase axis of the macro's own. The caller's
sub-phase axis rides the anonymous leading through every readout stage.
Because it matches no fabricated ``inst_shape`` axis it is time-serial on
all hardware, which pins down the observable contracts asserted here on the
hand-built tiny witness tile:

  * shape: ``vec_mat_mul`` on planes ``[P, row_num]`` returns
    ``[P, col_num]`` (leading preserved, no trailing P); a single full-row
    plane on the ``active_row_num == row_num`` variant returns ``[col_num]``
    and equals the ``sum(P)`` of its size-1 masked expansion;
  * inst prefix: with a non-empty ``inst_shape`` each instance's per-plane
    codes come from that instance's weights alone, with the sub-phase axis
    inserted left of the inst-aligned span;
  * latency: the profiler's total latency scales exactly with the plane
    count between one full-row plane and P = 2 masked planes of the same
    drive (every serial-op multiplier — core, readout chain, ADC — counts
    solved planes).

Runs eagerly (dynamo disabled); tiny geometry, seconds of numerics.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest
import torch
import torch._dynamo

from neurox.common.profiler import NeuroxProfiler
from neurox.works.macro.cim.isub_iadc_1t1r.macro import IsubIadc1t1rCimMacro
from tests.works.macro.cim.isub_iadc_1t1r._utils import (
    ADC_OP,
    TINY_ACTIVE_ROW_NUM,
    TINY_COL_NUM,
    TINY_PHASE_NUM,
    TINY_ROW_NUM,
    build_calibrated_tile,
    build_tile,
    build_tiny_config,
    masked_planes,
    per_phase_clamp_reference,
    tile_device,
)


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is ``@torch.compile``; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


def _build_variant(device: torch.device, *, active_row_num: int = TINY_ACTIVE_ROW_NUM) -> IsubIadc1t1rCimMacro:
    """Tiny witness tile at ``active_row_num`` (placeholder ladder — shape / latency only)."""
    config = build_tiny_config()
    if active_row_num != config.active_row_num:
        config = dataclasses.replace(config, active_row_num=active_row_num)
    return build_tile(config, device=device)


def _program_plus_column(xbar: IsubIadc1t1rCimMacro) -> None:
    device = tile_device(xbar)
    w = torch.zeros((TINY_COL_NUM, 1, xbar.config.row_num), dtype=torch.long, device=device)
    w[0, 0, :] = 1
    xbar.program(w)


# ---------------------------------------------------------------------------
# Shape contract
# ---------------------------------------------------------------------------


def test_plane_leading_preserved_no_trailing_phase(device: torch.device) -> None:
    """Planes ``[P, row]`` -> codes ``[P, col]``; leading order preserved."""
    x = torch.ones((TINY_ROW_NUM,), dtype=torch.long, device=device)

    two_phase = _build_variant(device)
    _program_plus_column(two_phase)
    # Shape: [row_num] -> [P, row_num]   zero-masked WL planes
    planes = masked_planes(x, row_num=TINY_ROW_NUM, max_active_rows=TINY_ACTIVE_ROW_NUM)
    with torch.no_grad():
        out2 = two_phase.vec_mat_mul(planes, adc_operation_point=ADC_OP)
    assert tuple(out2.shape) == (TINY_PHASE_NUM, TINY_COL_NUM)

    # Batched leading dims ride through untouched (leading order preserved).
    xb = torch.ones((3, 2, TINY_ROW_NUM), dtype=torch.long, device=device)
    # Shape: [3, 2, row_num] -> [3, 2, P, row_num]
    planes_b = masked_planes(xb, row_num=TINY_ROW_NUM, max_active_rows=TINY_ACTIVE_ROW_NUM)
    with torch.no_grad():
        outb = two_phase.vec_mat_mul(planes_b, adc_operation_point=ADC_OP)
    assert tuple(outb.shape) == (3, 2, TINY_PHASE_NUM, TINY_COL_NUM)


def test_full_row_plane_matches_masked_expansion_sum(device: torch.device) -> None:
    """``active_row_num == row_num``: a bare full-row plane returns ``[col]``
    and equals the ``sum(P)`` of its size-1 masked expansion."""
    x = torch.ones((TINY_ROW_NUM,), dtype=torch.long, device=device)

    single_phase = _build_variant(device, active_row_num=TINY_ROW_NUM)
    assert single_phase.config.row_num // single_phase.max_active_rows == 1
    _program_plus_column(single_phase)
    with torch.no_grad():
        out_direct = single_phase.vec_mat_mul(x, adc_operation_point=ADC_OP)
    assert tuple(out_direct.shape) == (TINY_COL_NUM,)  # no axis of the macro's own

    # Shape: [row_num] -> [1, row_num]   size-1 masked expansion
    planes = masked_planes(x, row_num=TINY_ROW_NUM, max_active_rows=TINY_ROW_NUM)
    with torch.no_grad():
        out_planes = single_phase.vec_mat_mul(planes, adc_operation_point=ADC_OP)
    assert tuple(out_planes.shape) == (1, TINY_COL_NUM)
    # Shape: [1, col_num] -> [col_num]   caller-side sum over P
    assert torch.equal(out_planes.sum(dim=0), out_direct)


# ---------------------------------------------------------------------------
# Non-empty inst prefix: per-instance decode
# ---------------------------------------------------------------------------


def _per_instance_patterns(n_inst: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Distinct ternary plane + distinct WL plane per instance, |MAC_p| <= 7."""
    a = TINY_ACTIVE_ROW_NUM
    w = torch.zeros((n_inst, TINY_COL_NUM, 1, TINY_ROW_NUM), dtype=torch.long)
    x = torch.ones((n_inst, TINY_ROW_NUM), dtype=torch.long)
    for i in range(n_inst):
        w[i, 0, 0, : 3 + i] = 1  # phase-0 positive block
        w[i, 0, 0, a : a + 1 + i] = -1  # phase-1 negative block
        w[i, 1, 0, 1 : 3 + 2 * i] = -1  # mixed column
        x[i, i] = 0  # per-instance WL plane
    return w, x


@pytest.mark.parametrize("n_inst", [2, 3])  # (2,): trailing inst dim == P; (3,): != P
def test_inst_prefix_per_instance_decode(device: torch.device, n_inst: int) -> None:
    """Bit-exact per-instance decode with a non-empty inst prefix.

    Each instance carries its own ternary plane and WL plane; the caller's
    sub-phase axis inserts left of the inst-alignment span, so instance
    ``i``'s per-plane codes must come from instance ``i``'s weights alone —
    including at ``inst_shape = (2,)`` where the trailing inst dim equals P.
    """
    xbar = build_calibrated_tile(device, inst_shape=(n_inst,))
    w, x = _per_instance_patterns(n_inst)
    xbar.program(w.to(device))
    # Shape: [n_inst, row_num] -> [P, n_inst, row_num]   P left of the inst span
    planes = masked_planes(x.to(device), row_num=TINY_ROW_NUM, max_active_rows=TINY_ACTIVE_ROW_NUM, inst_rank=1)
    with torch.no_grad():
        out = xbar.vec_mat_mul(planes, adc_operation_point=ADC_OP)
    assert tuple(out.shape) == (TINY_PHASE_NUM, n_inst, TINY_COL_NUM)
    for i in range(n_inst):
        expected = per_phase_clamp_reference(w[i], x[i], active_row_num=TINY_ACTIVE_ROW_NUM)
        assert torch.equal(out[:, i].cpu(), expected), f"instance {i} decode mismatch"

    # Batched leading rides left of the [..., P, inst, col] axes.
    xb = x.unsqueeze(0).expand(2, n_inst, TINY_ROW_NUM)
    # Shape: [2, n_inst, row_num] -> [2, P, n_inst, row_num]
    planes_b = masked_planes(xb.to(device), row_num=TINY_ROW_NUM, max_active_rows=TINY_ACTIVE_ROW_NUM, inst_rank=1)
    with torch.no_grad():
        outb = xbar.vec_mat_mul(planes_b, adc_operation_point=ADC_OP)
    assert tuple(outb.shape) == (2, TINY_PHASE_NUM, n_inst, TINY_COL_NUM)
    assert torch.equal(outb[0], out)


# ---------------------------------------------------------------------------
# Latency scales with the plane count
# ---------------------------------------------------------------------------


def test_latency_scales_with_plane_count(device: torch.device) -> None:
    """Total profiled latency of one VMM scales exactly with the plane count.

    Every latency event multiplies a per-op circuit property by a serial op
    count that counts solved WL planes (core plane count, readout-chain and
    ADC column-serial counts), so P = 2 masked planes cost exactly twice one
    full-row plane of the same drive.
    """
    x = torch.ones((TINY_ROW_NUM,), dtype=torch.long, device=device)
    totals: dict[int, float] = {}
    for active_row_num in (TINY_ROW_NUM, TINY_ACTIVE_ROW_NUM):  # 1 plane, 2 planes
        xbar = _build_variant(device, active_row_num=active_row_num)
        _program_plus_column(xbar)
        # Shape: [row_num] -> [P, row_num]   P = row_num / active_row_num
        planes = masked_planes(x, row_num=TINY_ROW_NUM, max_active_rows=active_row_num)
        with NeuroxProfiler() as profiler, torch.no_grad():
            xbar.vec_mat_mul(planes, adc_operation_point=ADC_OP)
        totals[TINY_ROW_NUM // active_row_num] = profiler.total_latency__ns

    assert totals[1] > 0.0
    assert totals[2] == pytest.approx(2.0 * totals[1], rel=1e-6), f"latency totals: {totals}"
