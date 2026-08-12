"""Unit tests for :class:`NestedParallelRailSolver`.

The subject is the resistor-network IR-drop solve: the harness
(:func:`tests.utils.standalone_solver_fixture.build_solver_harness`)
builds a fully LINEAR tiny tile — table-driven linear cells and ideal
``r_out = 0`` rail clamps — so the exact DCOP is the solution of a dense
KCL conductance system with Dirichlet boundaries at the reference taps.

Covers:
  * dense-oracle correctness: node voltages and driver currents match a
    ``torch.linalg.solve`` float64 assembly of the same network.
  * wire-residual decay: the :class:`SolverRecord` submitted to
    :class:`SolverProber` carries BL / SL wire KCL residuals at fp64
    round-off, alongside the converged DCOP.
  * inner-only entry point: ``solve_array_fixed_clamp`` is a direct
    solve for the linear network (the coupled block-2x2 wire Newton's
    Jacobian is exact, so one full step lands on the solution).
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
    NestedParallelRailSolver,
    NestedParallelRailSolverConfig,
    SolverProber,
    SolverRecord,
)
from tests.utils.standalone_solver_fixture import COL_NUM, ROW_NUM, SolverHarness, build_solver_harness


@pytest.fixture(autouse=True)
def _eager_solver() -> Iterator[None]:
    """Run the solver eagerly for these tests.

    ``solve_dc`` is ``@torch.compile(dynamic=False)``; fully unrolling it
    would spend minutes compiling for no benefit to what is asserted.
    """
    with torch._dynamo.config.patch(disable=True):
        yield


def _dense_kcl_solution(harness: SolverHarness) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Solve the harness network as one dense KCL system per column.

    Assembly derives from the solver's own KCL convention
    (``_wire_kcl.col_wire_kcl_residual``): the ladder is uniform, so at
    wire node ``k`` the residual is ``i_inject + (v[k] - v[k-1]) g +
    (v[k] - v[k+1]) g`` with the driver at ``v[-1]`` and the second term
    absent at the open end ``k = R-1``; the linear cell injects
    ``i = g_cell (v_bl - v_sl)`` drained from BL and pushed into SL; the
    ideal drivers pin the rail boundaries at the reference taps
    (Dirichlet). Unknowns per column are ``[v_bl(0..R-1), v_sl(0..R-1)]``.

    Returns ``(v_bl_node, v_sl_node, i_bl_driver, i_sl_driver)``: the two node
    grids followed by the two per-column driver currents.
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
    v_bl_node = x[:, :n]
    v_sl_node = x[:, n:]
    i_bl_driver = (v_bl_ref - v_bl_node[:, 0]) * g_bl
    i_sl_driver = (v_sl_ref - v_sl_node[:, 0]) * g_sl
    return v_bl_node, v_sl_node, i_bl_driver, i_sl_driver


def test_dcop_matches_dense_kcl_oracle(device: torch.device) -> None:
    """Nested solver reproduces the dense KCL solve of the linear network."""
    harness = build_solver_harness(
        solver_config=NestedParallelRailSolverConfig(n_outer=3, n_inner=3),
        device=device,
    )
    dcop = harness.solver.solve_dc(**harness.solver_kwargs())
    v_bl_exp, v_sl_exp, i_bl_exp, i_sl_exp = _dense_kcl_solution(harness)

    tol = {"rtol": 0.0, "atol": 1e-9}
    # Ideal drivers: the clamp boundaries settle exactly at the taps.
    torch.testing.assert_close(dcop.v_bl_clamp, harness.bl_v_ref__V.expand_as(dcop.v_bl_clamp), **tol)
    torch.testing.assert_close(dcop.v_sl_drive, harness.sl_v_ref__V.expand_as(dcop.v_sl_drive), **tol)
    # Node-voltage profiles and driver currents (batch dim is 1).
    torch.testing.assert_close(dcop.v_bl_node[0], v_bl_exp, **tol)
    torch.testing.assert_close(dcop.v_sl_node[0], v_sl_exp, **tol)
    torch.testing.assert_close(dcop.i_bl_driver[0], i_bl_exp, **tol)
    torch.testing.assert_close(dcop.i_sl_driver[0], i_sl_exp, **tol)


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
        solver_config=NestedParallelRailSolverConfig(n_outer=3, n_inner=3),
        device=device,
        col_num=col_num,
        row_num=row_num,
    )
    dcop = harness.solver.solve_dc(**harness.solver_kwargs())
    v_bl_exp, v_sl_exp, i_bl_exp, i_sl_exp = _dense_kcl_solution(harness)

    tol = {"rtol": 0.0, "atol": 1e-9}
    assert dcop.v_bl_node.shape[-2:] == (col_num, row_num)
    torch.testing.assert_close(dcop.v_bl_node[0], v_bl_exp, **tol)
    torch.testing.assert_close(dcop.v_sl_node[0], v_sl_exp, **tol)
    torch.testing.assert_close(dcop.i_bl_driver[0], i_bl_exp, **tol)
    torch.testing.assert_close(dcop.i_sl_driver[0], i_sl_exp, **tol)


def _assert_fully_detached(record: SolverRecord[XbarCellDcop]) -> None:
    """Every tensor on the record — including inside its DCOP — is detached."""
    for tensor in (
        record.wire_bl__uA,
        record.wire_sl__uA,
        record.clamp_bl__V,
        record.clamp_sl__V,
        record.dcop.i_bl_driver,
        record.dcop.i_sl_driver,
        record.dcop.v_bl_node,
        record.dcop.v_sl_node,
        record.dcop.v_bl_clamp,
        record.dcop.v_sl_drive,
        record.dcop.cell.i__uA,
        record.dcop.cell.di_dvbl__uS,
        record.dcop.cell.di_dvsl__uS,
    ):
        assert tensor.grad_fn is None
        assert tensor.requires_grad is False


def test_record_carries_dcop_and_residuals(device: torch.device) -> None:
    """A probed solve emits one detached record: DCOP + wire residuals.

    Nested solver drives both wire KCL residuals to fp64 noise; the record
    also carries the converged DCOP (finite tensors, expected leading shape),
    all tensors fully detached including inside ``dcop`` and ``dcop.cell``.
    """
    harness = build_solver_harness(
        solver_config=NestedParallelRailSolverConfig(n_outer=3, n_inner=3),
        device=device,
    )
    with SolverProber() as prober:
        dcop = harness.solver.solve_dc(**harness.solver_kwargs())
    records = prober.records
    # One public solve_dc invocation emits exactly one record.
    assert len(records) == 1
    record = records[0]
    # A solver is no module, so the record names its own solve entry point.
    assert record.emitter == "NestedParallelRailSolver.solve_dc"

    assert record.wire_bl__uA.max().item() < 1e-9
    assert record.wire_sl__uA.max().item() < 1e-9

    # The record carries the converged DCOP with the same leading shape.
    assert record.dcop.v_bl_node.shape == dcop.v_bl_node.shape
    assert torch.isfinite(record.dcop.v_bl_node).all()
    assert torch.isfinite(record.dcop.cell.i__uA).all()
    _assert_fully_detached(record)


def test_solve_output_bit_identical_probed_vs_unprobed(device: torch.device) -> None:
    """The demand gate never perturbs the solve: DCOP is bit-identical.

    The probed run is held to having actually emitted, so the law cannot pass
    by the emit path having gone quiet; and the record's own detach must copy
    rather than reach back into the solution it was built from, which the
    equality on the returned DCOP catches.
    """
    harness = build_solver_harness(
        solver_config=NestedParallelRailSolverConfig(n_outer=3, n_inner=3),
        device=device,
    )
    unprobed = harness.solver.solve_dc(**harness.solver_kwargs())
    with SolverProber() as prober:
        probed = harness.solver.solve_dc(**harness.solver_kwargs())
    assert len(prober.records) == 1
    for field in ("v_bl_node", "v_sl_node", "v_bl_clamp", "v_sl_drive", "i_bl_driver", "i_sl_driver"):
        assert torch.equal(getattr(unprobed, field), getattr(probed, field))
    assert torch.equal(unprobed.cell.i__uA, probed.cell.i__uA)


def test_inner_direct_solve_exactness(device: torch.device) -> None:
    """``solve_array_fixed_clamp`` solves the linear network directly.

    With a linear cell the inner coupled block-2x2 wire Newton has an
    exact Jacobian, so it is a direct solve: the pinned-clamp solution
    matches the dense KCL oracle (the pinned values equal the reference
    taps, i.e. the same Dirichlet boundary) and the wire residuals sit
    at fp64 noise.
    """
    harness = build_solver_harness(
        solver_config=NestedParallelRailSolverConfig(n_outer=1, n_inner=3),
        device=device,
    )
    solver = harness.solver
    assert isinstance(solver, NestedParallelRailSolver)
    v_wl = harness.v_wl_drive__V
    *batch, _row = v_wl.shape
    col_num = harness.w_state_idx.shape[-2]
    dtype = v_wl.dtype
    v_bl_clamp = harness.bl_v_ref__V.to(device=device, dtype=dtype).expand(*batch, col_num)
    v_sl_drive = harness.sl_v_ref__V.to(device=device, dtype=dtype).expand(*batch, col_num)
    with SolverProber() as prober:
        dcop = solver.solve_array_fixed_clamp(
            v_bl_clamp__V=v_bl_clamp,
            v_sl_drive__V=v_sl_drive,
            bl_segment_r__MOhm=harness.bl_segment_r__MOhm,
            sl_segment_r__MOhm=harness.sl_segment_r__MOhm,
            cell=harness.cell,
            cell_snap=harness.cell_snapshot(),
        )
    records = prober.records
    assert len(records) == 1
    record = records[0]
    assert record.emitter == "NestedParallelRailSolver.solve_array_fixed_clamp"
    assert record.wire_bl__uA.max().item() < 1e-9
    assert record.wire_sl__uA.max().item() < 1e-9
    # Pinned clamps → clamp residual identically zero.
    assert torch.equal(record.clamp_bl__V, torch.zeros_like(record.clamp_bl__V))
    assert torch.equal(record.clamp_sl__V, torch.zeros_like(record.clamp_sl__V))

    v_bl_exp, v_sl_exp, _i_bl_exp, _i_sl_exp = _dense_kcl_solution(harness)
    tol = {"rtol": 0.0, "atol": 1e-9}
    torch.testing.assert_close(dcop.v_bl_node[0], v_bl_exp, **tol)
    torch.testing.assert_close(dcop.v_sl_node[0], v_sl_exp, **tol)


def test_the_far_node_is_the_ladder_open_end(device: torch.device) -> None:
    """OPEN-END LAW: the last row balances on one rail link, interiors on two.

    A rail is a uniform ladder that simply stops at the last row — that node
    has the link back towards the driver and nothing onward, while every
    interior node has both. The residual the solver drives to zero and the
    Jacobian it drives it with must agree on that, so the converged profile
    closes the one-link balance at the far node and refuses it one row in.
    """
    harness = build_solver_harness(
        solver_config=NestedParallelRailSolverConfig(n_outer=3, n_inner=3),
        device=device,
        row_num=5,
    )
    dcop = harness.solver.solve_dc(**harness.solver_kwargs())
    g_bl = 1.0 / harness.bl_segment_r__MOhm
    v_bl = dcop.v_bl_node
    # Shape: [..., col_num, row_num]
    i_cell = dcop.cell.i__uA

    # The far node: its cell current returns through the single link back.
    open_end__uA = i_cell[..., -1] + (v_bl[..., -1] - v_bl[..., -2]) * g_bl
    assert open_end__uA.abs().max().item() < 1e-9

    # One row in: the same one-link balance is off by the onward link's own
    # current, and only counting both links closes it.
    onward__uA = (v_bl[..., -2] - v_bl[..., -1]) * g_bl
    interior_one_link__uA = i_cell[..., -2] + (v_bl[..., -2] - v_bl[..., -3]) * g_bl
    assert interior_one_link__uA.abs().min().item() > 1e-3
    assert (interior_one_link__uA + onward__uA).abs().max().item() < 1e-9


def test_a_single_row_tile_matches_the_one_link_closed_form(device: torch.device) -> None:
    """A one-row tile is one node per rail, reached through exactly one link.

    Nothing is left to iterate: each rail is the reference tap in series with
    a single link, so the cell branch sees ``dV / (1 + g_cell (r_BL + r_SL))``
    and the whole DCOP follows in closed form. It is the sharpest statement of
    the open-end rule — a node counted as interior would carry twice the rail
    conductance and land somewhere else entirely.
    """
    harness = build_solver_harness(
        solver_config=NestedParallelRailSolverConfig(n_outer=3, n_inner=3),
        device=device,
        row_num=1,
    )
    dcop = harness.solver.solve_dc(**harness.solver_kwargs())

    dtype = harness.v_wl_drive__V.dtype
    cfg = harness.cell_config
    # Every access device is on at the harness WL drive.
    # Shape: [col_num, row_num=1]
    g_cell = torch.tensor(cfg.g_cell_on_table__uS, device=device, dtype=dtype)[harness.w_state_idx]
    r_bl = harness.bl_segment_r__MOhm
    r_sl = harness.sl_segment_r__MOhm

    dv = harness.bl_v_ref__V - harness.sl_v_ref__V
    i_expected = g_cell * dv / (1.0 + g_cell * (r_bl + r_sl))
    v_bl_expected = harness.bl_v_ref__V - i_expected * r_bl
    v_sl_expected = harness.sl_v_ref__V + i_expected * r_sl

    tol = {"rtol": 0.0, "atol": 1e-12}
    torch.testing.assert_close(dcop.v_bl_node[0], v_bl_expected, **tol)
    torch.testing.assert_close(dcop.v_sl_node[0], v_sl_expected, **tol)
    torch.testing.assert_close(dcop.i_bl_driver[0], i_expected.squeeze(-1), **tol)
    torch.testing.assert_close(dcop.i_sl_driver[0], -i_expected.squeeze(-1), **tol)
