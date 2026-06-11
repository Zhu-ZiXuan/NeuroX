"""Unit tests for :class:`FullJacobianSolver1T1R`.

Covers:
  * residual decay: full-Jacobian Newton drives all 5 residual classes to
    fp64 noise within a few iterations (quadratic convergence).
  * physical solution agreement with nested solver on small arrays — both
    converge to the same fixed point.
  * chunking bit-exact (uses ``Offset1T1RXbar`` batch chunking).
  * ``compute_residuals=False`` yields ``residuals=None``.

Symbolic-derivative correctness (analytical Jacobian vs FD Jacobian on
each of the 25 partial-derivative entries) is verified separately by
:mod:`tests.test_full_jacobian_fd_verify`, which builds a tiny
standalone harness so it can sidestep the production xbar's snapshot
plumbing.
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


def _build(solver_config, device, *, inst: int = 4, chunk_size: int = 0):
    return build_xbar_for_calibration(
        XBAR_CONFIG,
        device=device,
        inst_shape=(inst,),
        dtype=torch.float64,
        solver_config=solver_config,
        batch_chunk_size=chunk_size,
    )


def _program_and_solve(xbar, *, x_batch: int = 4, seed: int = 0, compute_residuals: bool = True):
    device = xbar.core.bl_segment_g__uS.device
    distribution = load_distribution(None, xbar)
    g = make_generator(seed, device)
    w = next(iter(sample_w(distribution, xbar, n=xbar._inst_shape[0], batch_w=xbar._inst_shape[0],
                           device=device, generator=g)))
    xbar.program(w)
    x = next(iter(sample_x_batches(distribution, xbar, n_total=x_batch, batch_size=x_batch,
                                    device=device, generator=g)))
    return x, xbar.core.solve_dc(x.unsqueeze(-2), compute_residuals=compute_residuals)


def test_full_jacobian_residuals_machine_precision(fixture_config, device):
    """A handful of Newton iters must drive all 5 residual classes to fp64 noise."""
    config = FullJacobianSolver1T1RConfig(n_newton=5)
    xbar = _build(config, device)
    _, dcop = _program_and_solve(xbar)
    assert dcop.residuals is not None
    assert dcop.residuals.cell__uA.max().item() < 1e-9
    assert dcop.residuals.wire_bl__uA.max().item() < 1e-6
    assert dcop.residuals.wire_sl__uA.max().item() < 1e-9
    assert dcop.residuals.clamp_bl__V.max().item() < 1e-9
    assert dcop.residuals.clamp_sl__V.max().item() < 1e-9


def test_full_jacobian_quadratic_convergence(fixture_config, device):
    """Newton residuals must roughly square each iteration (classic quadratic)."""
    cell_max_by_iter = []
    for n in (1, 2, 3, 5):
        config = FullJacobianSolver1T1RConfig(n_newton=n)
        xbar = _build(config, device, inst=8)
        _, dcop = _program_and_solve(xbar, x_batch=4)
        cell_max_by_iter.append(dcop.residuals.cell__uA.max().item())
    # Roughly: 1 → 2 → 3 should square the residual each step.
    assert cell_max_by_iter[1] < cell_max_by_iter[0] / 10, (
        f"expected ~10× drop from n_newton=1→2: {cell_max_by_iter[:2]}"
    )
    assert cell_max_by_iter[2] < cell_max_by_iter[1] / 100, (
        f"expected ~100× drop from n_newton=2→3 (quadratic): {cell_max_by_iter[:3]}"
    )


def test_full_jacobian_matches_nested(fixture_config, device):
    """Both solvers must converge to the same physical operating point.

    Compares v_out_phys (the readout-side voltage) on identical (w, x);
    fp64 tolerance is 1e-9.
    """
    full_xbar = _build(FullJacobianSolver1T1RConfig(n_newton=8), device, inst=4)
    nested_xbar = _build(NestedSolver1T1RConfig(n_outer=8, n_inner=2), device, inst=4)
    _, dcop_full = _program_and_solve(full_xbar)
    _, dcop_nested = _program_and_solve(nested_xbar)
    max_diff = (dcop_full.v_out_phys - dcop_nested.v_out_phys).abs().max().item()
    assert max_diff < 1e-9, f"v_out_phys differs by {max_diff:.3e} V"


def test_full_jacobian_residuals_none_on_hot_path(fixture_config, device):
    config = FullJacobianSolver1T1RConfig(n_newton=5)
    xbar = _build(config, device)
    _, dcop = _program_and_solve(xbar, compute_residuals=False)
    assert dcop.residuals is None


def test_full_jacobian_chunking_bit_exact(fixture_config, device):
    """xbar batch chunking is bit-exact under FullJacobian, just like nested."""
    config = FullJacobianSolver1T1RConfig(n_newton=5)

    def run(chunk_size: int) -> torch.Tensor:
        xbar = _build(config, device, inst=4, chunk_size=chunk_size)
        distribution = load_distribution(None, xbar)
        g = make_generator(0, device)
        w = next(iter(sample_w(distribution, xbar, n=4, batch_w=4, device=device, generator=g)))
        xbar.program(w)
        x = next(iter(sample_x_batches(distribution, xbar, n_total=8, batch_size=8,
                                        device=device, generator=g)))
        op = AdcOperationPoint(adc_mode=0, adc_bits=8)
        return xbar.vec_mat_mul(x.unsqueeze(-2), adc_operation_point=op)

    full = run(0)
    for cs in (1, 2, 3, 8):
        chunked = run(cs)
        assert torch.equal(full, chunked), (
            f"chunk_size={cs} differs: max abs diff "
            f"{(full.float() - chunked.float()).abs().max().item():.3e}"
        )


