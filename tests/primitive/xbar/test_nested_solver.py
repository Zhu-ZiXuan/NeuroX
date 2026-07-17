"""Unit tests for :class:`NestedParallelRailSolver`.

Covers:
  * residual decay: the nested solver drives every KCL residual class
    down to fp64 round-off.
  * inner-only entry point: ``solve_array_fixed_clamp`` converges the
    inner sub-problem at any pinned clamp.
  * ``compute_residuals=False`` elides the residual algebra.

All tests build a standalone :class:`Solver` harness via
:func:`tests.utils.standalone_solver_fixture.build_solver_harness` and
call ``solver.solve_dc`` directly — they do not depend on
:class:`XbarArray1t1r` or :class:`Offset1t1rCimMacro`. Convergence
properties of the solver hold under any in-range inputs; the harness
uses a uniform mid-range RRAM g pattern and a uniform WL drive so the
tests are reproducible without a workload sampler.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import torch._dynamo

from neurox.primitive.xbar.solver import NestedParallelRailSolver, NestedParallelRailSolverConfig
from tests.utils.standalone_solver_fixture import build_solver_harness

REPO_ROOT = Path(__file__).resolve().parents[3]
XBAR_CONFIG = REPO_ROOT / "works" / "offset_1t1r" / "config" / "1t1r_28nm.toml"


@pytest.fixture(scope="module")
def fixture_config() -> Iterator[Path]:
    if not XBAR_CONFIG.is_file():
        pytest.skip(f"missing test fixture: {XBAR_CONFIG}")
    yield XBAR_CONFIG


@pytest.fixture(autouse=True)
def _eager_solver() -> Iterator[None]:
    """Run the solver eagerly for these tests.

    ``solve_dc`` is ``@torch.compile(dynamic=False)``; at the large
    iteration counts these tests use, fully unrolling it would spend
    minutes compiling for no benefit to what is asserted.
    """
    with torch._dynamo.config.patch(disable=True):
        yield


def test_nested_residuals_at_machine_precision(fixture_config: Path, device: torch.device) -> None:
    """Nested solver drives all three residuals to fp64 noise."""
    harness = build_solver_harness(
        config_path=XBAR_CONFIG,
        solver_config=NestedParallelRailSolverConfig(n_outer=20, n_inner=10),
        inst_shape=(4,),
        x_batch=4,
        device=device,
    )
    dcop = harness.solver.solve_dc(**harness.solver_kwargs(), compute_residuals=True)
    residuals = dcop.residuals
    assert residuals is not None
    cell_residuals = dcop.cell.residuals
    assert cell_residuals is not None
    assert cell_residuals.cell__uA.max().item() < 1e-9
    assert residuals.wire_bl__uA.max().item() < 1e-6
    assert residuals.wire_sl__uA.max().item() < 1e-9


def test_nested_inner_only_converges(fixture_config: Path, device: torch.device) -> None:
    """``solve_array_fixed_clamp`` converges the inner sub-problem to
    fp64 noise — inner system is M-matrix monotone, no multi-equilibrium."""
    harness = build_solver_harness(
        config_path=XBAR_CONFIG,
        solver_config=NestedParallelRailSolverConfig(n_outer=1, n_inner=50),
        inst_shape=(4,),
        x_batch=4,
        device=device,
    )
    solver = harness.solver
    assert isinstance(solver, NestedParallelRailSolver)
    # Pinned clamps at the clamp-reference voltages — same shape as the
    # solver's port-current tensors. The reference is injected per snapshot,
    # so the resolved taps come from the harness.
    v_wl = harness.v_wl_drive__V
    *batch, phys_col, _row = v_wl.shape
    dtype = v_wl.dtype
    v_bl_clamp = harness.bl_v_ref__V.to(device=device, dtype=dtype).expand(*batch, phys_col)
    v_sl_drive = harness.sl_v_ref__V.to(device=device, dtype=dtype).expand(*batch, phys_col)
    dcop = solver.solve_array_fixed_clamp(
        v_bl_clamp__V=v_bl_clamp,
        v_sl_drive__V=v_sl_drive,
        bl_segment_r__MOhm=harness.bl_segment_r__MOhm,
        sl_segment_r__MOhm=harness.sl_segment_r__MOhm,
        bl_segment_g__uS=harness.bl_segment_g__uS,
        sl_segment_g__uS=harness.sl_segment_g__uS,
        cell=harness.cell,
        cell_snap=harness.cell_snapshot(),
        compute_residuals=True,
    )
    assert dcop.residuals is not None
    assert dcop.cell.residuals is not None
    assert dcop.cell.residuals.cell__uA.max().item() < 1e-9
    assert dcop.residuals.wire_bl__uA.max().item() < 1e-9
    assert dcop.residuals.wire_sl__uA.max().item() < 1e-9


def test_nested_residuals_none_on_hot_path(fixture_config: Path, device: torch.device) -> None:
    """``compute_residuals=False`` elides the residual algebra."""
    harness = build_solver_harness(
        config_path=XBAR_CONFIG,
        solver_config=NestedParallelRailSolverConfig(n_outer=10, n_inner=10),
        inst_shape=(4,),
        x_batch=4,
        device=device,
    )
    dcop = harness.solver.solve_dc(**harness.solver_kwargs(), compute_residuals=False)
    assert dcop.residuals is None
