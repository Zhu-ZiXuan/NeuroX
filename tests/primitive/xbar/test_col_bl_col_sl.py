"""Unit tests for `solve_col_bl_col_sl_dc`.

The subject is the resistor-network IR-drop solve: the harness
(`tests.utils.standalone_solver_fixture.build_solver_harness`)
builds a fully LINEAR tiny tile — table-driven linear cells and ideal
`r_out = 0` rail clamps — so the exact DCOP is the solution of a dense
KCL conductance system with Dirichlet boundaries at the reference taps.

Covers:
  * dense-oracle correctness: node voltages and driver currents match a
    `torch.linalg.solve` float64 assembly of the same network.
  * trajectory probing: the `ColBlColSlRecord` stream submitted to
    `ColBlColSlProber` carries the residual pair of every outer clamp
    event and of every inner Newton step, plus one terminal record
    holding the DCOP.
  * open-end law: each rail is a uniform ladder that stops at the last
    row, so that node balances on one rail link where an interior node
    balances on two — pinned on a five-row tile and, in closed form, on
    a one-row one.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
import torch._dynamo
from torch import Tensor

from neurox.primitive.xbar.cell import XbarCellDcop
from neurox.primitive.xbar.solver import (
    ColBlColSlDcop,
    ColBlColSlProber,
    ColBlColSlRecord,
    ColBlColSlSolverConfig,
    solve_col_bl_col_sl_dc,
)
from tests.utils.standalone_solver_fixture import COL_NUM, ROW_NUM, SolverHarness, build_solver_harness

_N_OUTER = 3
_N_INNER = 3


@pytest.fixture(autouse=True)
def _eager_solver() -> Iterator[None]:
    """Run the solver eagerly for these tests.

    The numerical leaf is `@torch.compile(dynamic=False)`; fully unrolling
    it would spend minutes compiling for no benefit to what is asserted.
    """
    with torch._dynamo.config.patch(disable=True):
        yield


def _dense_kcl_solution(harness: SolverHarness) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Solve the harness network as one dense KCL system per column.

    Assembly derives from the solver's own KCL convention
    (`_wire_kcl.f_kcl__uA`): the ladder is uniform, so at
    wire node `k` the residual is `i_inject__uA + (v[k] - v[k-1]) g +
    (v[k] - v[k+1]) g` with the driver at `v[-1]` and the second term
    absent at the open end `k = R-1`; the linear cell injects
    `i = g_cell (v_bl - v_sl)` drained from BL and pushed into SL; the
    ideal drivers pin the rail boundaries at the reference taps
    (Dirichlet). Unknowns per column are `[v_bl(0..R-1), v_sl(0..R-1)]`.

    Returns `(v_bl_node__V, v_sl_node__V, i_bl_driver__uA, i_sl_driver__uA)`:
    the two node grids followed by the two per-column driver currents.
    """
    cfg = harness.cell_config
    idx = harness.w_state_idx
    dtype = harness.v_wl_drive__V.dtype
    device = harness.v_wl_drive__V.device

    # WL-switched per-cell chord conductance from the config tables
    # (batch dim is 1: drop it for the per-column assembly).
    g_on = torch.tensor(cfg.g_cell_on_table__uS, device=device, dtype=dtype)[idx]
    g_off = torch.tensor(cfg.g_cell_off_table__uS, device=device, dtype=dtype)[idx]
    on = harness.v_wl_drive__V[0] > cfg.v_wl_on_threshold__V
    # Shape: [col, row]
    g_cell = torch.where(on, g_on, g_off)

    g_bl = 1.0 / harness.bl_segment_r__MOhm
    g_sl = 1.0 / harness.sl_segment_r__MOhm
    v_bl_ref = harness.bl_v_ref__V.to(device=device, dtype=dtype)
    v_sl_ref = harness.sl_v_ref__V.to(device=device, dtype=dtype)

    col_num, row_num = g_cell.shape
    n = row_num
    lhs = torch.zeros(col_num, 2 * n, 2 * n, device=device, dtype=dtype)
    rhs = torch.zeros(col_num, 2 * n, device=device, dtype=dtype)
    for k in range(n):
        # The open end at k = R-1 has no link onward, every other node has one.
        g_right = 0.0 if k + 1 == n else 1.0
        # BL node k: +i_cell drained.
        lhs[:, k, k] = g_bl * (1.0 + g_right) + g_cell[:, k]
        lhs[:, k, n + k] = -g_cell[:, k]
        if k > 0:
            lhs[:, k, k - 1] = -g_bl
        if k + 1 < n:
            lhs[:, k, k + 1] = -g_bl
        # SL node k: -i_cell injected.
        lhs[:, n + k, n + k] = g_sl * (1.0 + g_right) + g_cell[:, k]
        lhs[:, n + k, k] = -g_cell[:, k]
        if k > 0:
            lhs[:, n + k, n + k - 1] = -g_sl
        if k + 1 < n:
            lhs[:, n + k, n + k + 1] = -g_sl
    rhs[:, 0] = g_bl * v_bl_ref
    rhs[:, n] = g_sl * v_sl_ref

    x = torch.linalg.solve(lhs, rhs)
    v_bl_node__V = x[:, :n]
    v_sl_node__V = x[:, n:]
    i_bl_driver__uA = (v_bl_ref - v_bl_node__V[:, 0]) * g_bl
    i_sl_driver__uA = (v_sl_ref - v_sl_node__V[:, 0]) * g_sl
    return v_bl_node__V, v_sl_node__V, i_bl_driver__uA, i_sl_driver__uA


def test_dcop_matches_dense_kcl_oracle(device: torch.device) -> None:
    """The solver reproduces the dense KCL solve of the linear network."""
    harness = build_solver_harness(
        solver_config=ColBlColSlSolverConfig(n_outer=_N_OUTER, n_inner=_N_INNER),
        device=device,
    )
    dcop = solve_col_bl_col_sl_dc(**harness.solve_kwargs())
    v_bl_exp, v_sl_exp, i_bl_exp, i_sl_exp = _dense_kcl_solution(harness)

    tol = {"rtol": 0.0, "atol": 1e-9}
    # Ideal drivers: the clamp boundaries settle exactly at the taps.
    torch.testing.assert_close(dcop.v_bl_clamp__V, harness.bl_v_ref__V.expand_as(dcop.v_bl_clamp__V), **tol)
    torch.testing.assert_close(dcop.v_sl_drive__V, harness.sl_v_ref__V.expand_as(dcop.v_sl_drive__V), **tol)
    # Node-voltage profiles and driver currents (batch dim is 1).
    torch.testing.assert_close(dcop.v_bl_node__V[0], v_bl_exp, **tol)
    torch.testing.assert_close(dcop.v_sl_node__V[0], v_sl_exp, **tol)
    torch.testing.assert_close(dcop.i_bl_driver__uA[0], i_bl_exp, **tol)
    torch.testing.assert_close(dcop.i_sl_driver__uA[0], i_sl_exp, **tol)


@pytest.mark.parametrize(
    ("col_num", "row_num"),
    [(1, ROW_NUM), (COL_NUM, 1), (1, 1)],
    ids=["single_column", "single_row", "single_cell"],
)
def test_a_degenerate_axis_is_well_defined(device: torch.device, col_num: int, row_num: int) -> None:
    """A tile of one column, one row, or one cell solves like any other.

    Neither axis carries a lower bound: a single column is one independent
    ladder, a single row is a ladder of one node, and the dense KCL oracle
    settles both alike, so the solve must reproduce it rather than refuse
    the shape.
    """
    harness = build_solver_harness(
        solver_config=ColBlColSlSolverConfig(n_outer=_N_OUTER, n_inner=_N_INNER),
        device=device,
        col_num=col_num,
        row_num=row_num,
    )
    dcop = solve_col_bl_col_sl_dc(**harness.solve_kwargs())
    v_bl_exp, v_sl_exp, i_bl_exp, i_sl_exp = _dense_kcl_solution(harness)

    tol = {"rtol": 0.0, "atol": 1e-9}
    assert dcop.v_bl_node__V.shape[-2:] == (col_num, row_num)
    torch.testing.assert_close(dcop.v_bl_node__V[0], v_bl_exp, **tol)
    torch.testing.assert_close(dcop.v_sl_node__V[0], v_sl_exp, **tol)
    torch.testing.assert_close(dcop.i_bl_driver__uA[0], i_bl_exp, **tol)
    torch.testing.assert_close(dcop.i_sl_driver__uA[0], i_sl_exp, **tol)


# ---------------------------------------------------------------------------
# Trajectory probing
# ---------------------------------------------------------------------------


def _tensors_of(record: ColBlColSlRecord[XbarCellDcop]) -> list[Tensor]:
    """Every tensor the record carries, reaching inside a nested DCOP."""
    found: list[Tensor] = []
    for name in ("f_bl_clamp__V", "f_sl_clamp__V", "f_bl_kcl__uA", "f_sl_kcl__uA"):
        value = getattr(record, name)
        if value is not None:
            found.append(value)
    dcop = record.dcop
    if dcop is not None:
        found += [
            dcop.i_bl_driver__uA,
            dcop.i_sl_driver__uA,
            dcop.v_bl_node__V,
            dcop.v_sl_node__V,
            dcop.v_bl_clamp__V,
            dcop.v_sl_drive__V,
            dcop.cell.i__uA,
            dcop.cell.di_dvbl__uS,
            dcop.cell.di_dvsl__uS,
        ]
    return found


def _probe(
    harness: SolverHarness, *, min_outer: int = 0
) -> tuple[tuple[ColBlColSlRecord[XbarCellDcop], ...], ColBlColSlDcop[XbarCellDcop]]:
    """Run one probed solve, returning `(records, dcop)`."""
    with ColBlColSlProber(min_outer=min_outer) as prober:
        dcop = solve_col_bl_col_sl_dc(**harness.solve_kwargs())
    return prober.records, dcop


def test_one_solve_emits_its_whole_trajectory(device: torch.device) -> None:
    """TRAJECTORY LAW: every outer step emits its clamp event plus `n_inner` steps.

    The trajectory is the loop laid out flat, so its length and its
    `(outer, inner)` coordinates follow the iteration counts alone, and it
    closes with exactly one terminal record.
    """
    harness = build_solver_harness(
        solver_config=ColBlColSlSolverConfig(n_outer=_N_OUTER, n_inner=_N_INNER),
        device=device,
    )
    records, _dcop = _probe(harness)

    assert len(records) == _N_OUTER * (1 + _N_INNER) + 1
    expected = [(k, j) for k in range(_N_OUTER) for j in range(1 + _N_INNER)]
    expected.append((_N_OUTER, 0))
    assert [(r.outer, r.inner) for r in records] == expected
    # Exactly one record closes the solve, and it is the last.
    terminals = [r for r in records if r.dcop is not None]
    assert len(terminals) == 1
    assert terminals[0] is records[-1]


def test_each_record_kind_fills_only_its_own_residuals(device: torch.device) -> None:
    """KIND LAW: `inner` selects the residual pair; the other pair stays absent."""
    harness = build_solver_harness(
        solver_config=ColBlColSlSolverConfig(n_outer=_N_OUTER, n_inner=_N_INNER),
        device=device,
    )
    records, _dcop = _probe(harness)

    for record in records[:-1]:
        assert record.dcop is None
        if record.inner == 0:
            assert record.f_bl_clamp__V is not None
            assert record.f_sl_clamp__V is not None
            assert record.f_bl_kcl__uA is None
            assert record.f_sl_kcl__uA is None
        else:
            assert record.f_bl_kcl__uA is not None
            assert record.f_sl_kcl__uA is not None
            assert record.f_bl_clamp__V is None
            assert record.f_sl_clamp__V is None

    terminal = records[-1]
    assert terminal.dcop is not None
    assert terminal.f_bl_kcl__uA is None
    assert terminal.f_sl_kcl__uA is None
    assert terminal.f_bl_clamp__V is None
    assert terminal.f_sl_clamp__V is None


def _bl_kcl_at__uA(harness: SolverHarness, dcop: ColBlColSlDcop[XbarCellDcop]) -> Tensor:
    """Rebuild the BL wire KCL residual of a settled state, node by node.

    Assembly follows the solver's own convention (`_wire_kcl.f_kcl__uA`): at
    node `k` the residual is the cell current drained there plus the drop
    across the link back towards the clamp plus the drop across the link
    onward, the latter absent at the open end `k = R-1`.
    """
    g_bl = 1.0 / harness.bl_segment_r__MOhm
    # Shape: [..., col, row]
    v__V = dcop.v_bl_node__V
    # The clamp hangs off index 0 through one link of the same pitch.
    left__V = torch.cat((dcop.v_bl_clamp__V.unsqueeze(-1), v__V[..., :-1]), dim=-1)
    # The open end has no link onward, which is a zero drop.
    right__V = torch.cat((v__V[..., 1:], v__V[..., -1:]), dim=-1)
    return dcop.cell.i__uA + (v__V - left__V) * g_bl + (v__V - right__V) * g_bl


def test_residuals_belong_to_their_own_records_state(device: torch.device) -> None:
    """OFF-BY-ONE LAW: a record's residual is the residual AT its own iterate.

    The last inner record is the state its own step was taken FROM, so the
    residual it carries is not the one the settled state produces — the step
    in between moved the solve towards zero. That is why the terminal record,
    which is that settled state, carries no paired residual of its own: one
    step separates it from the last residual on the stream.
    """
    harness = build_solver_harness(
        solver_config=ColBlColSlSolverConfig(n_outer=1, n_inner=1),
        device=device,
    )
    records, _dcop = _probe(harness)
    last_inner = records[-2]
    terminal = records[-1]
    assert last_inner.inner == 1
    assert last_inner.f_bl_kcl__uA is not None
    assert terminal.dcop is not None
    assert terminal.f_bl_kcl__uA is None

    # The post-step residual, rebuilt from the terminal state.
    settled__uA = _bl_kcl_at__uA(harness, terminal.dcop)
    assert settled__uA.abs().max().item() < last_inner.f_bl_kcl__uA.abs().max().item()


def test_min_outer_filters_collection_side(device: torch.device) -> None:
    """MIN_OUTER LAW: only records from `outer >= min_outer` are collected.

    The emitter is blind to the filter — it emits the whole trajectory
    either way — so the collected book is a suffix of the unfiltered one and
    the terminal record, at `outer == n_outer`, always survives.
    """
    harness = build_solver_harness(
        solver_config=ColBlColSlSolverConfig(n_outer=_N_OUTER, n_inner=_N_INNER),
        device=device,
    )
    full, _dcop = _probe(harness, min_outer=0)
    tail, _dcop_tail = _probe(harness, min_outer=_N_OUTER - 1)

    assert [(r.outer, r.inner) for r in tail] == [(r.outer, r.inner) for r in full if r.outer >= _N_OUTER - 1]
    assert tail[-1].dcop is not None
    assert min(r.outer for r in tail) == _N_OUTER - 1

    # A threshold past the last outer step keeps the terminal record alone.
    terminal_only, _dcop_only = _probe(harness, min_outer=_N_OUTER)
    assert len(terminal_only) == 1
    assert terminal_only[0].dcop is not None


def test_every_recorded_tensor_is_detached(device: torch.device) -> None:
    """The book never reaches back into the graph the solve built."""
    harness = build_solver_harness(
        solver_config=ColBlColSlSolverConfig(n_outer=_N_OUTER, n_inner=_N_INNER),
        device=device,
    )
    records, _dcop = _probe(harness)
    for record in records:
        tensors = _tensors_of(record)
        assert tensors
        for tensor in tensors:
            assert tensor.grad_fn is None
            assert tensor.requires_grad is False


def test_terminal_record_carries_the_converged_dcop(device: torch.device) -> None:
    """The terminal record's DCOP is the one the call returned."""
    harness = build_solver_harness(
        solver_config=ColBlColSlSolverConfig(n_outer=_N_OUTER, n_inner=_N_INNER),
        device=device,
    )
    records, dcop = _probe(harness)
    terminal = records[-1].dcop
    assert terminal is not None
    assert terminal.v_bl_node__V.shape == dcop.v_bl_node__V.shape
    assert torch.isfinite(terminal.v_bl_node__V).all()
    assert torch.isfinite(terminal.cell.i__uA).all()
    # The solve drives both wire KCL residuals to fp64 noise.
    last_inner = records[-2]
    assert last_inner.f_bl_kcl__uA is not None
    assert last_inner.f_sl_kcl__uA is not None
    assert last_inner.f_bl_kcl__uA.abs().max().item() < 1e-9
    assert last_inner.f_sl_kcl__uA.abs().max().item() < 1e-9


def test_solve_output_bit_identical_probed_vs_unprobed(device: torch.device) -> None:
    """The demand gate never perturbs the solve: DCOP is bit-identical.

    The probed run is held to having actually emitted, so the law cannot pass
    by the emit path having gone quiet; and the record's own detach must copy
    rather than reach back into the solution it was built from, which the
    equality on the returned DCOP catches.
    """
    harness = build_solver_harness(
        solver_config=ColBlColSlSolverConfig(n_outer=_N_OUTER, n_inner=_N_INNER),
        device=device,
    )
    unprobed = solve_col_bl_col_sl_dc(**harness.solve_kwargs())
    records, probed = _probe(harness)
    assert len(records) == _N_OUTER * (1 + _N_INNER) + 1
    for field in (
        "v_bl_node__V",
        "v_sl_node__V",
        "v_bl_clamp__V",
        "v_sl_drive__V",
        "i_bl_driver__uA",
        "i_sl_driver__uA",
    ):
        assert torch.equal(getattr(unprobed, field), getattr(probed, field))
    assert torch.equal(unprobed.cell.i__uA, probed.cell.i__uA)


def test_the_far_node_is_the_ladder_open_end(device: torch.device) -> None:
    """OPEN-END LAW: the last row balances on one rail link, interiors on two.

    A rail is a uniform ladder that stops at the last row — that node
    has the link back towards the driver and nothing onward, while every
    interior node has both. The residual the solver drives to zero and the
    Jacobian it drives it with must agree on that, so the converged profile
    closes the one-link balance at the far node and refuses it one row in.
    """
    harness = build_solver_harness(
        solver_config=ColBlColSlSolverConfig(n_outer=_N_OUTER, n_inner=_N_INNER),
        device=device,
        row_num=5,
    )
    dcop = solve_col_bl_col_sl_dc(**harness.solve_kwargs())
    g_bl = 1.0 / harness.bl_segment_r__MOhm
    v_bl__V = dcop.v_bl_node__V
    # Shape: [..., col_num, row_num]
    i_cell__uA = dcop.cell.i__uA

    # The far node: its cell current returns through the single link back.
    open_end__uA = i_cell__uA[..., -1] + (v_bl__V[..., -1] - v_bl__V[..., -2]) * g_bl
    assert open_end__uA.abs().max().item() < 1e-9

    # One row in: the same one-link balance is off by the onward link's own
    # current, and only counting both links closes it.
    onward__uA = (v_bl__V[..., -2] - v_bl__V[..., -1]) * g_bl
    interior_one_link__uA = i_cell__uA[..., -2] + (v_bl__V[..., -2] - v_bl__V[..., -3]) * g_bl
    assert interior_one_link__uA.abs().min().item() > 1e-3
    assert (interior_one_link__uA + onward__uA).abs().max().item() < 1e-9


def test_a_single_row_tile_matches_the_one_link_closed_form(device: torch.device) -> None:
    """A one-row tile is one node per rail, reached through exactly one link.

    Nothing is left to iterate: each rail is the reference tap in series with
    a single link, so the cell branch sees `dV / (1 + g_cell (r_BL + r_SL))`
    and the whole DCOP follows in closed form. It is the sharpest statement of
    the open-end rule — a node counted as interior would carry twice the rail
    conductance and land somewhere else entirely.
    """
    harness = build_solver_harness(
        solver_config=ColBlColSlSolverConfig(n_outer=_N_OUTER, n_inner=_N_INNER),
        device=device,
        row_num=1,
    )
    dcop = solve_col_bl_col_sl_dc(**harness.solve_kwargs())

    dtype = harness.v_wl_drive__V.dtype
    cfg = harness.cell_config
    # Every access device is on at the harness WL drive.
    # Shape: [col_num, row_num=1]
    g_cell__uS = torch.tensor(cfg.g_cell_on_table__uS, device=device, dtype=dtype)[harness.w_state_idx]
    r_bl__MOhm = harness.bl_segment_r__MOhm
    r_sl__MOhm = harness.sl_segment_r__MOhm

    dv__V = harness.bl_v_ref__V - harness.sl_v_ref__V
    i_expected__uA = g_cell__uS * dv__V / (1.0 + g_cell__uS * (r_bl__MOhm + r_sl__MOhm))
    v_bl_expected__V = harness.bl_v_ref__V - i_expected__uA * r_bl__MOhm
    v_sl_expected__V = harness.sl_v_ref__V + i_expected__uA * r_sl__MOhm

    tol = {"rtol": 0.0, "atol": 1e-12}
    torch.testing.assert_close(dcop.v_bl_node__V[0], v_bl_expected__V, **tol)
    torch.testing.assert_close(dcop.v_sl_node__V[0], v_sl_expected__V, **tol)
    torch.testing.assert_close(dcop.i_bl_driver__uA[0], i_expected__uA.squeeze(-1), **tol)
    torch.testing.assert_close(dcop.i_sl_driver__uA[0], -i_expected__uA.squeeze(-1), **tol)
