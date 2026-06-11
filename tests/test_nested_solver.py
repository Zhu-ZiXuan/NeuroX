"""Unit tests for :class:`NestedSolver1T1R`.

Covers:
  * residual decay: the nested solver drives every KCL residual class
    down to fp64 round-off on the chip preset.
  * inner-only entry point: ``solve_array_fixed_clamp`` converges the
    inner sub-problem at any pinned clamp (debug helper used by the
    legacy calibration path; the new step-ratio plateau framework
    uses the full ``solve_dc`` path instead).
  * batch chunking is bit-exact under nested.
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
    NestedSolver1T1R,
    NestedSolver1T1RConfig,
    Solver1T1RConfig,
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


def _build_with_solver(solver_config: Solver1T1RConfig, device: torch.device, inst: int = 4):
    return build_xbar_for_calibration(
        XBAR_CONFIG,
        device=device,
        inst_shape=(inst,),
        dtype=torch.float64,
        solver_config=solver_config,
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
    dcop = xbar.core.solve_dc(x.unsqueeze(-2), compute_residuals=compute_residuals)
    return x, dcop


def test_nested_residuals_at_machine_precision(fixture_config, device):
    """Nested solver should drive all three residuals to fp64 noise."""
    config = NestedSolver1T1RConfig(
        n_outer=20, n_inner=10,
    )
    xbar = _build_with_solver(config, device)
    _, dcop = _program_and_solve(xbar)
    assert dcop.residuals is not None
    assert dcop.residuals.cell__uA.max().item() < 1e-9
    assert dcop.residuals.wire_bl__uA.max().item() < 1e-6
    assert dcop.residuals.wire_sl__uA.max().item() < 1e-9


def test_nested_inner_only_converges(fixture_config, device):
    """``solve_array_fixed_clamp`` must converge the inner sub-problem
    to machine precision when given enough iterations; the inner system
    is M-matrix monotone so there is no multi-equilibrium issue."""
    config = NestedSolver1T1RConfig(
        n_outer=1, n_inner=50,
    )
    xbar = _build_with_solver(config, device)
    solver: NestedSolver1T1R = xbar.core.solver  # narrow
    distribution = load_distribution(None, xbar)
    g = make_generator(0, device)
    w = next(iter(sample_w(distribution, xbar, n=xbar._inst_shape[0], batch_w=xbar._inst_shape[0],
                           device=device, generator=g)))
    xbar.program(w)
    x = next(iter(sample_x_batches(distribution, xbar, n_total=4, batch_size=4,
                                    device=device, generator=g)))
    x_grid = x.unsqueeze(-2)
    full_shape = torch.broadcast_shapes(xbar.core.rram.g__uS.shape, x_grid.shape)
    *batch, _phys_col_num, _row_num = full_shape
    phys_col_num = xbar.core.fabricated_col_num
    rram_snap = xbar.core.rram.snapshot(shape=full_shape)
    nmos_snap = xbar.core.nmos.snapshot(shape=full_shape)
    v_wl_drive = xbar.core.wl_dac.convert(x_grid)
    v_bl_clamp = torch.full((*batch, phys_col_num), solver.bl_driver.v_ref__V,
                             device=device, dtype=xbar.core.dtype)
    v_sl_drive = torch.full((*batch, phys_col_num), solver.sl_driver.v_ref__V,
                             device=device, dtype=xbar.core.dtype)
    dcop = solver.solve_array_fixed_clamp(
        v_wl_drive__V=v_wl_drive,
        v_bl_clamp__V=v_bl_clamp,
        v_sl_drive__V=v_sl_drive,
        bl_segment_r__MOhm=xbar.core.bl_segment_r__MOhm,
        sl_segment_r__MOhm=xbar.core.sl_segment_r__MOhm,
        bl_segment_g__uS=xbar.core.bl_segment_g__uS,
        sl_segment_g__uS=xbar.core.sl_segment_g__uS,
        rram_snapshot=rram_snap,
        nmos_snapshot=nmos_snap,
        compute_residuals=True,
    )
    assert dcop.residuals is not None
    assert dcop.residuals.cell__uA.max().item() < 1e-9
    assert dcop.residuals.wire_bl__uA.max().item() < 1e-9
    assert dcop.residuals.wire_sl__uA.max().item() < 1e-9


def test_nested_residuals_none_on_hot_path(fixture_config, device):
    """``compute_residuals=False`` must elide the residual algebra."""
    config = NestedSolver1T1RConfig(
        n_outer=10, n_inner=10,
    )
    xbar = _build_with_solver(config, device)
    _, dcop = _program_and_solve(xbar, compute_residuals=False)
    assert dcop.residuals is None


def test_nested_chunking_bit_exact(fixture_config, device):
    """The xbar batch-chunking scheduler must produce bit-exact ADC codes
    under the nested solver too."""
    config = NestedSolver1T1RConfig(
        n_outer=10, n_inner=10,
    )

    def run(chunk_size: int) -> torch.Tensor:
        xbar = build_xbar_for_calibration(
            XBAR_CONFIG,
            device=device,
            inst_shape=(4,),
            dtype=torch.float64,
            solver_config=config,
            batch_chunk_size=chunk_size,
        )
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
