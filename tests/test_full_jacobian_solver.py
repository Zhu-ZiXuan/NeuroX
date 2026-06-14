"""Unit tests for :class:`FullJacobianSolver1T1R`.

Covers:
  * residual decay: full-Jacobian Newton drives all 5 residual classes
    to fp64 noise within a few iterations.
  * quadratic convergence on synthetic chip-preset inputs.
  * physical solution agreement with the nested solver — both converge
    to the same fixed point.
  * ``compute_residuals=False`` yields ``residuals=None``.
  * xbar batch chunking is bit-exact under FullJacobian (integration).

Solver-only tests build a standalone harness via
:func:`tests.utils.standalone_solver_fixture.build_solver_harness` and
call ``solver.solve_dc`` directly. The chunking test stays at the
xbar level because chunking is xbar's concern, not the solver's.

Symbolic-derivative correctness (analytical vs FD Jacobian) is
verified separately by :mod:`tests.test_full_jacobian_fd_verify`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neurox.analog.adc import AdcOperationPoint
from neurox.tools.solver_calibrate._common import build_xbar_for_calibration
from neurox.tools.xbar_adc._sampling import (
    load_distribution,
    make_generator,
    sample_w,
    sample_x_batches,
)
from neurox.xbar._1t1r import (
    FullJacobianSolver1T1RConfig,
    NestedSolver1T1RConfig,
)
from tests.utils.standalone_solver_fixture import build_solver_harness

REPO_ROOT = Path(__file__).resolve().parent.parent
XBAR_CONFIG = REPO_ROOT / "example" / "config" / "1t1r_28nm.toml"


@pytest.fixture(scope="module")
def fixture_config():
    if not XBAR_CONFIG.is_file():
        pytest.skip(f"missing test fixture: {XBAR_CONFIG}")
    yield XBAR_CONFIG


@pytest.fixture(scope="module")
def device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def test_full_jacobian_residuals_machine_precision(fixture_config, device):
    """A handful of Newton iters drive all 5 residual classes to fp64 noise."""
    harness = build_solver_harness(
        config_path=XBAR_CONFIG,
        solver_config=FullJacobianSolver1T1RConfig(n_newton=5),
        inst_shape=(4,),
        x_batch=4,
        device=device,
    )
    dcop = harness.solver.solve_dc(**harness.solver_kwargs(), compute_residuals=True)
    residuals = dcop.residuals
    assert residuals is not None
    assert residuals.cell__uA.max().item() < 1e-9
    assert residuals.wire_bl__uA.max().item() < 1e-6
    assert residuals.wire_sl__uA.max().item() < 1e-9
    assert residuals.clamp_bl__V.max().item() < 1e-9
    assert residuals.clamp_sl__V.max().item() < 1e-9


def test_full_jacobian_quadratic_convergence(fixture_config, device):
    """Newton residuals roughly square each iteration (classic quadratic)."""
    cell_max_by_iter = []
    for n in (1, 2, 3, 5):
        harness = build_solver_harness(
            config_path=XBAR_CONFIG,
            solver_config=FullJacobianSolver1T1RConfig(n_newton=n),
            inst_shape=(8,),
            x_batch=4,
            device=device,
        )
        dcop = harness.solver.solve_dc(**harness.solver_kwargs(), compute_residuals=True)
        assert dcop.residuals is not None
        cell_max_by_iter.append(dcop.residuals.cell__uA.max().item())
    assert cell_max_by_iter[1] < cell_max_by_iter[0] / 10, (
        f"expected ~10× drop from n_newton=1→2: {cell_max_by_iter[:2]}"
    )
    assert cell_max_by_iter[2] < cell_max_by_iter[1] / 100, (
        f"expected ~100× drop from n_newton=2→3 (quadratic): {cell_max_by_iter[:3]}"
    )


def test_full_jacobian_matches_nested(fixture_config, device):
    """Both solvers converge to the same physical operating point (v_bl_node)."""
    full_h = build_solver_harness(
        config_path=XBAR_CONFIG,
        solver_config=FullJacobianSolver1T1RConfig(n_newton=8),
        inst_shape=(4,),
        x_batch=4,
        device=device,
    )
    nested_h = build_solver_harness(
        config_path=XBAR_CONFIG,
        solver_config=NestedSolver1T1RConfig(n_outer=8, n_inner=2),
        inst_shape=(4,),
        x_batch=4,
        device=device,
    )
    dcop_full = full_h.solver.solve_dc(**full_h.solver_kwargs(), compute_residuals=False)
    dcop_nested = nested_h.solver.solve_dc(**nested_h.solver_kwargs(), compute_residuals=False)
    max_diff = (dcop_full.v_bl_clamp - dcop_nested.v_bl_clamp).abs().max().item()
    assert max_diff < 1e-9, f"v_bl_clamp differs by {max_diff:.3e} V"


def test_full_jacobian_residuals_none_on_hot_path(fixture_config, device):
    harness = build_solver_harness(
        config_path=XBAR_CONFIG,
        solver_config=FullJacobianSolver1T1RConfig(n_newton=5),
        inst_shape=(4,),
        x_batch=4,
        device=device,
    )
    dcop = harness.solver.solve_dc(**harness.solver_kwargs(), compute_residuals=False)
    assert dcop.residuals is None


def test_full_jacobian_chunking_bit_exact(fixture_config, device):
    """Integration test: xbar batch chunking is bit-exact under FullJacobian.

    Stays at the xbar level because chunking is the xbar's concern; this
    test verifies the chunking scheduler doesn't perturb the result, not
    the solver itself.
    """
    config = FullJacobianSolver1T1RConfig(n_newton=5)

    def run(chunk_size: int) -> torch.Tensor:
        xbar = build_xbar_for_calibration(
            XBAR_CONFIG,
            device=device,
            inst_shape=(4,),
            dtype=torch.float64,
            solver_config=config,
            solve_chunk_size_inst=chunk_size,
        )
        distribution = load_distribution(None, xbar)
        g = make_generator(0, device)
        w = next(iter(sample_w(distribution, xbar, n=4, batch_w=4, device=device, generator=g)))
        xbar.program(w)
        x = next(iter(sample_x_batches(distribution, xbar, n_total=8, batch_size=8, device=device, generator=g)))
        x_with_inst_slot = x.unsqueeze(-2)  # (x_batch, 1, row): inst-broadcast slot
        op = AdcOperationPoint(adc_mode=0, adc_bits=8)
        return xbar.vec_mat_mul(x_with_inst_slot, adc_operation_point=op)

    full = run(0)
    for cs in (1, 2, 3, 8):
        chunked = run(cs)
        assert torch.equal(full, chunked), (
            f"chunk_size={cs} differs: max abs diff {(full.float() - chunked.float()).abs().max().item():.3e}"
        )
