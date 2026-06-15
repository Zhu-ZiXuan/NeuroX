"""Independent cross-checks for :class:`FullJacobianSolver1T1R`.

Builds a TINY harness — single column, four rows, no production xbar
machinery — instantiates RRAM / NMOS / TIA / SL driver directly with
the chip-preset configs, and:

  1. Runs :class:`FullJacobianSolver1T1R` from a fresh init to
     convergence; collects ``u_solver*``.
  2. Builds an INDEPENDENT residual function ``F(u)`` from the same
     device objects (no shared internal state with the solver) via
     :mod:`tests._dense_jacobian_reference`.
  3. Asserts ``|F(u_solver*)|.max < 1e-9`` — the solver's converged
     point genuinely satisfies the residual equations (catches: wrong
     fixed point).
  4. Computes the FD Jacobian ``J_FD = ∂F/∂u`` at ``u_solver*`` and
     verifies one dense Newton step at the converged point is small
     (sanity that ``J_FD`` is non-singular and the FD harness itself
     is healthy — this does **not** verify the solver's analytical
     Jacobian matches FD entry-wise; see test docstring).
  5. Verifies the FD Jacobian's *structure* — diagonal 3×3 wire-row
     blocks, sparse 3×3 off-diagonal blocks (BL-BL + SL-SL only), two
     bordered boundary rows — matches the doc's closed-form recipe.

A true element-wise analytical-vs-FD Jacobian comparison would require
exposing the solver's Jacobian-assembly path to tests, which is not
done today. The structural check (step 5) is the strongest signal we
have for the symbolic stamping being correct: it catches wrong
inter-row coupling, missing boundary entries, sign-convention errors
that produce stationary non-zero structure.

The harness is independent of the production xbar / Offset1T1R stack:
all the topology is constructed by hand at tiny size (R=4, single col,
single batch) so neither snapshot slicing nor distribution sampling is
needed.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from neurox.analog import Driver, DriverConfig, DriverPolicy
from neurox.analog.tia import OpAmpTIA, OpAmpTIAConfig, OpAmpTIAPolicy
from neurox.common.load_dump import dataclass_from_file
from neurox.device import NMOS, RRAM, NMOSPolicy, RRAMPolicy
from neurox.xbar import Offset1T1RXbarConfig
from neurox.xbar._1t1r import (
    FullJacobianSolver1T1R,
    FullJacobianSolver1T1RConfig,
)
from tests._dense_jacobian_reference import build_residual_fn, numerical_jacobian

REPO_ROOT = Path(__file__).resolve().parent.parent
XBAR_CONFIG = REPO_ROOT / "example" / "presets" / "xbar" / "1t1r_28nm.toml"

NUM_ROW = 4
NUM_COL = 2  # solver requires > 1; we slice col=0 for the FD reference


@pytest.fixture(scope="module")
def harness(pytestconfig: pytest.Config) -> dict[str, Any]:
    """Build a tiny standalone (RRAM, NMOS, TIA, SL driver, solver) harness."""
    if not XBAR_CONFIG.is_file():
        pytest.skip(f"missing chip preset: {XBAR_CONFIG}")

    # Honor the shared ``--device`` option from tests/conftest.py instead
    # of hard-coding cuda:0 — CPU-only runs should pick cpu, GPU runs
    # should follow whatever the user passed.
    device_name = str(pytestconfig.getoption("device"))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA requested but not available")
    dtype = torch.float64
    xbar_config = dataclass_from_file(Offset1T1RXbarConfig, XBAR_CONFIG, section="xbar")
    core_cfg = xbar_config.core_config

    # Devices at tile=(1, NUM_COL, NUM_ROW) — single batch, single col, NUM_ROW rows.
    rram = RRAM(
        config=core_cfg.rram_config,
        policy=RRAMPolicy(prog_gamma=False, stuck_at=False, read_telegraph=False, read_thermal=False),
        inst_shape=(1, NUM_COL, NUM_ROW),
        dtype=dtype,
        T__K=300.0,
        g_max__uS=core_cfg.rram_g_max__uS,
    )
    nmos = NMOS(
        config=core_cfg.nmos_config,
        policy=NMOSPolicy(A_vt_mismatch=False, A_beta_mismatch=False),
        inst_shape=(1, NUM_COL, NUM_ROW),
        dtype=dtype,
        T__K=300.0,
        W__um=core_cfg.access_nmos_W__um,
        L__um=core_cfg.access_nmos_L__um,
    )

    base_tia_cfg = core_cfg.tia_config
    assert isinstance(base_tia_cfg, OpAmpTIAConfig)
    tia_cfg = replace(base_tia_cfg, n_newton=6)
    tia = OpAmpTIA(
        config=tia_cfg,
        policy=OpAmpTIAPolicy(
            opamp_gain_sigma=False,
            nmos=NMOSPolicy(A_vt_mismatch=False, A_beta_mismatch=False),
        ),
        name="probe",
        inst_shape=(NUM_COL,),
        dtype=dtype,
        T__K=300.0,
    )
    sl_driver_cfg = core_cfg.sl_driver_config
    assert isinstance(sl_driver_cfg, DriverConfig)
    sl_driver = Driver(
        config=sl_driver_cfg,
        policy=DriverPolicy(drive_thermal=False),
        name="probe",
        inst_shape=(NUM_COL,),
        dtype=dtype,
        T__K=300.0,
    )

    for d in (rram, nmos, tia, sl_driver):
        d.to(device)
        d.eval()
        d.fabricate()

    # Program RRAM with mid-range conductance.
    g_target = torch.full(
        (1, NUM_COL, NUM_ROW),
        float(core_cfg.rram_g_max__uS) * 0.4,
        device=device,
        dtype=dtype,
    )
    rram.program(g_target, t_elapsed=0.0)

    # Tile-level segment R/G — choose the same value as a single chip-preset
    # row repeated NUM_ROW times.
    bl_segment_r__MOhm = torch.tensor(
        [core_cfg.bl_first_r__MOhm] + [core_cfg.bl_segment_r__MOhm] * (NUM_ROW - 1),
        device=device,
        dtype=dtype,
    )
    sl_segment_r__MOhm = torch.tensor(
        [core_cfg.sl_first_r__MOhm] + [core_cfg.sl_segment_r__MOhm] * (NUM_ROW - 1),
        device=device,
        dtype=dtype,
    )
    bl_segment_g__uS = 1.0 / bl_segment_r__MOhm
    sl_segment_g__uS = 1.0 / sl_segment_r__MOhm

    # WL drive — mid-range gate voltage.
    v_wl_grid = torch.full(
        (1, NUM_COL, NUM_ROW),
        0.7,
        device=device,
        dtype=dtype,
    )
    v_wl_per_row = v_wl_grid[0, 0, :]

    # Build solver.
    solver = FullJacobianSolver1T1R(
        config=FullJacobianSolver1T1RConfig(n_newton=8),
        rram=rram,
        nmos=nmos,
        bl_driver=tia,
        sl_driver=sl_driver,
    )

    # Sample snapshots once.
    rram_snap = rram.snapshot(shape=(1, NUM_COL, NUM_ROW), multi_coords=None)
    nmos_snap = nmos.snapshot(shape=(1, NUM_COL, NUM_ROW), multi_coords=None)
    bl_drv_snap = tia.snapshot(shape=(NUM_COL,), multi_coords=None)
    sl_drv_snap = sl_driver.snapshot(shape=(NUM_COL,), multi_coords=None)

    # Slice every snapshot to the single (batch=0, col=0) instance for the
    # residual-function harness. RRAM and NMOS snapshots have shape
    # ``[1, 1, NUM_ROW]`` for ``g`` / mismatch tensors; index ``[0, 0]``
    # gives 1-D length-NUM_ROW tensors. Drivers' snapshots are already
    # per-col; ``[0]`` gives scalar per-field tensors.
    def _slice_dataclass_per_col(obj: Any, col_indexer: tuple[int, ...]) -> Any:
        from dataclasses import is_dataclass

        kwargs = {}
        for f in obj.__dataclass_fields__:
            v = getattr(obj, f)
            if isinstance(v, torch.Tensor) and v.dim() >= len(col_indexer):
                kwargs[f] = v[col_indexer]
            elif is_dataclass(v) and hasattr(v, "__dataclass_fields__"):
                kwargs[f] = _slice_dataclass_per_col(v, col_indexer)
            else:
                kwargs[f] = v
        return type(obj)(**kwargs)

    rram_snap_1 = _slice_dataclass_per_col(rram_snap, (0, 0))
    nmos_snap_1 = _slice_dataclass_per_col(nmos_snap, (0, 0))
    bl_drv_snap_1 = _slice_dataclass_per_col(bl_drv_snap, (0,))
    sl_drv_snap_1 = _slice_dataclass_per_col(sl_drv_snap, (0,))

    res_fn = build_residual_fn(
        v_wl_drive_grid__V=v_wl_per_row,
        bl_segment_g__uS=bl_segment_g__uS,
        sl_segment_g__uS=sl_segment_g__uS,
        rram_snapshot=rram_snap_1,
        nmos_snapshot=nmos_snap_1,
        bl_driver=tia,
        bl_driver_snapshot=bl_drv_snap_1,
        sl_driver=sl_driver,
        sl_driver_snapshot=sl_drv_snap_1,
        rram=rram,
        nmos=nmos,
        num_row=NUM_ROW,
    )

    return {
        "device": device,
        "dtype": dtype,
        "rram": rram,
        "nmos": nmos,
        "tia": tia,
        "sl_driver": sl_driver,
        "rram_snap": rram_snap,
        "nmos_snap": nmos_snap,
        "bl_drv_snap": bl_drv_snap,
        "sl_drv_snap": sl_drv_snap,
        "bl_segment_r__MOhm": bl_segment_r__MOhm,
        "sl_segment_r__MOhm": sl_segment_r__MOhm,
        "bl_segment_g__uS": bl_segment_g__uS,
        "sl_segment_g__uS": sl_segment_g__uS,
        "v_wl_grid": v_wl_grid,
        "v_wl_per_row": v_wl_per_row,
        "solver": solver,
        "res_fn": res_fn,
    }


def test_solver_converges_to_zero_residual_per_independent_function(harness: dict[str, Any]) -> None:
    """Solver-converged u* must satisfy the INDEPENDENT residual function ≈ 0.

    Catches: solver bug that converges to a fake fixed point not honouring
    the true KCL equations.
    """
    dcop = harness["solver"].solve_dc(
        v_wl_drive__V=harness["v_wl_grid"],
        bl_segment_r__MOhm=harness["bl_segment_r__MOhm"],
        sl_segment_r__MOhm=harness["sl_segment_r__MOhm"],
        bl_segment_g__uS=harness["bl_segment_g__uS"],
        sl_segment_g__uS=harness["sl_segment_g__uS"],
        rram_snapshot=harness["rram_snap"],
        nmos_snapshot=harness["nmos_snap"],
        bl_driver_snapshot=harness["bl_drv_snap"],
        sl_driver_snapshot=harness["sl_drv_snap"],
        compute_residuals=False,
    )
    # Pack into flat unknown vector u in the order the residual function expects.
    u_star = torch.cat(
        [
            dcop.v_bl_node[0, 0],  # [R]
            dcop.v_sl_node[0, 0],
            dcop.v_x_node[0, 0],
            dcop.v_bl_clamp[0, 0].reshape(1),
            dcop.v_sl_drive[0, 0].reshape(1),
        ]
    )
    f_star = harness["res_fn"](u_star)
    max_residual = f_star.abs().max().item()
    assert max_residual < 1e-9, (
        f"Independent residual at solver-converged u*: max |F| = {max_residual:.3e}. "
        "Either FullJacobian converged to a wrong fixed point OR the residual "
        "function disagrees with the solver's residual definition."
    )


def test_fd_jacobian_well_conditioned_at_solver_converged_point(harness: dict[str, Any]) -> None:
    """The FD Jacobian at u_solver* must be non-singular and well-conditioned.

    Computes ``Δu_FD = −J_FD⁻¹·F(u*)`` and checks it is small. Note that
    since ``F(u*) ≈ 0`` (asserted by the previous test), this would be
    trivially ≈ 0 for ANY invertible operator — so the assertion does
    **not** prove the analytical Jacobian stamping inside FullJacobian
    matches FD. What it does verify is:

      * ``J_FD`` is non-singular (linalg.solve does not blow up);
      * The residual function and FD eps are mutually consistent (no
        catastrophic cancellation amplifies the round-off);
      * The reference harness can perform a Newton step at u_solver*
        without numerical pathology.

    A true element-wise comparison between the solver's analytical
    Jacobian and FD Jacobian would require exposing the solver's
    Jacobian-assembly logic to tests, which is not done today. The
    indirect cross-checks here (independent residual ≈ 0 + block
    structure matching the doc's closed-form recipe in the next test)
    are the strongest signals we currently have for analytical-J
    correctness.
    """
    dcop = harness["solver"].solve_dc(
        v_wl_drive__V=harness["v_wl_grid"],
        bl_segment_r__MOhm=harness["bl_segment_r__MOhm"],
        sl_segment_r__MOhm=harness["sl_segment_r__MOhm"],
        bl_segment_g__uS=harness["bl_segment_g__uS"],
        sl_segment_g__uS=harness["sl_segment_g__uS"],
        rram_snapshot=harness["rram_snap"],
        nmos_snapshot=harness["nmos_snap"],
        bl_driver_snapshot=harness["bl_drv_snap"],
        sl_driver_snapshot=harness["sl_drv_snap"],
        compute_residuals=False,
    )
    u_star = torch.cat(
        [
            dcop.v_bl_node[0, 0],
            dcop.v_sl_node[0, 0],
            dcop.v_x_node[0, 0],
            dcop.v_bl_clamp[0, 0].reshape(1),
            dcop.v_sl_drive[0, 0].reshape(1),
        ]
    )
    f_star = harness["res_fn"](u_star)
    j_fd = numerical_jacobian(harness["res_fn"], u_star, eps=1e-6)
    delta_u = torch.linalg.solve(j_fd, -f_star)
    max_delta = delta_u.abs().max().item()
    assert max_delta < 1e-6, (
        f"FD-Newton step at solver-converged u*: max |Δu| = {max_delta:.3e} V. "
        "Either F(u*) is not ≈ 0 (caught by the previous test) or J_FD is "
        "singular / ill-conditioned at u* — the FD harness is broken."
    )


def test_fd_jacobian_has_expected_block_structure(harness: dict[str, Any]) -> None:
    """FD Jacobian at u* must match the doc's block-tridiagonal layout.

    The 5 residual classes × 5 unknown classes give a (3R+2)×(3R+2)
    matrix. Per the doc:

      * Diagonal 3×3 blocks for each wire row (V_BL[k], V_SL[k], V_X[k])
        are DENSE (cell couplings + wire diagonal).
      * Off-diagonal 3×3 blocks couple adjacent wire rows ONLY through
        BL-BL and SL-SL wire conductances; the V_X column / row in the
        off-diagonal block must be zero.
      * Two bordered rows for V_BL_CL and V_SL_DR couple ONLY to V_BL[0]
        and V_SL[0] respectively (their other columns are zero).

    Catches: wrong inter-row coupling (e.g., V_X cross-row) or missing
    boundary border entries.
    """
    dcop = harness["solver"].solve_dc(
        v_wl_drive__V=harness["v_wl_grid"],
        bl_segment_r__MOhm=harness["bl_segment_r__MOhm"],
        sl_segment_r__MOhm=harness["sl_segment_r__MOhm"],
        bl_segment_g__uS=harness["bl_segment_g__uS"],
        sl_segment_g__uS=harness["sl_segment_g__uS"],
        rram_snapshot=harness["rram_snap"],
        nmos_snapshot=harness["nmos_snap"],
        bl_driver_snapshot=harness["bl_drv_snap"],
        sl_driver_snapshot=harness["sl_drv_snap"],
        compute_residuals=False,
    )
    u_star = torch.cat(
        [
            dcop.v_bl_node[0, 0],
            dcop.v_sl_node[0, 0],
            dcop.v_x_node[0, 0],
            dcop.v_bl_clamp[0, 0].reshape(1),
            dcop.v_sl_drive[0, 0].reshape(1),
        ]
    )
    j_fd = numerical_jacobian(harness["res_fn"], u_star, eps=1e-6)

    # The residual function builds F as [f_bl[0:R], f_sl[0:R], f_x[0:R], f_cl_bl, f_cl_sl].
    # The unknown vector u is [V_BL[0:R], V_SL[0:R], V_X[0:R], V_BL_CL, V_SL_DR].
    # Both are 3R + 2 long.
    R = NUM_ROW
    n = 3 * R + 2
    assert j_fd.shape == (n, n)

    # Index helpers — these match the residual function's layout.
    def idx_f_bl(k: int) -> int:
        return k

    def idx_f_sl(k: int) -> int:
        return R + k

    def idx_f_x(k: int) -> int:
        return 2 * R + k

    idx_f_cl_bl = 3 * R
    idx_f_cl_sl = 3 * R + 1

    def idx_v_bl(k: int) -> int:
        return k

    def idx_v_sl(k: int) -> int:
        return R + k

    def idx_v_x(k: int) -> int:
        return 2 * R + k

    idx_v_bl_cl = 3 * R
    idx_v_sl_dr = 3 * R + 1

    # --- (A) Off-diagonal V_X coupling must be zero ---
    # V_X[k] has no inter-row coupling; ∂F/∂V_X[k'] = 0 for any
    # residual at row k' ≠ k except via the cell (which is per-row, not
    # cross-row). Specifically the F_BL[k] row's V_X column should only
    # have a non-zero entry at k' == k.
    fd_eps = 1e-7
    for k in range(R):
        for k_prime in range(R):
            if k_prime == k:
                continue
            # F_BL[k] vs V_X[k']: no coupling.
            entry = j_fd[idx_f_bl(k), idx_v_x(k_prime)].abs().item()
            assert entry < fd_eps, (
                f"unexpected cross-row coupling F_BL[{k}] vs V_X[{k_prime}]: |J| = {entry:.3e} (should be 0)"
            )
            # F_SL[k] vs V_X[k']: no coupling.
            entry = j_fd[idx_f_sl(k), idx_v_x(k_prime)].abs().item()
            assert entry < fd_eps, (
                f"unexpected cross-row coupling F_SL[{k}] vs V_X[{k_prime}]: |J| = {entry:.3e} (should be 0)"
            )
            # F_X[k] vs V_X[k']: no coupling.
            entry = j_fd[idx_f_x(k), idx_v_x(k_prime)].abs().item()
            assert entry < fd_eps, (
                f"unexpected cross-row coupling F_X[{k}] vs V_X[{k_prime}]: |J| = {entry:.3e} (should be 0)"
            )

    # --- (B) BL/SL cross-coupling in OFF-DIAGONAL row pairs must be zero ---
    # Wire BL only couples V_BL[k]-V_BL[k±1]; not V_BL to V_SL across rows.
    for k in range(R):
        for k_prime in range(R):
            if k_prime == k:
                continue
            # F_BL[k] vs V_SL[k']: no coupling (cell BL→SL coupling is per-row only).
            entry = j_fd[idx_f_bl(k), idx_v_sl(k_prime)].abs().item()
            assert entry < fd_eps, (
                f"unexpected cross-row coupling F_BL[{k}] vs V_SL[{k_prime}]: |J| = {entry:.3e} (should be 0)"
            )
            entry = j_fd[idx_f_sl(k), idx_v_bl(k_prime)].abs().item()
            assert entry < fd_eps, (
                f"unexpected cross-row coupling F_SL[{k}] vs V_BL[{k_prime}]: |J| = {entry:.3e} (should be 0)"
            )

    # --- (C) BL-BL wire coupling sign + magnitude ---
    # ∂F_BL[k]/∂V_BL[k-1] = -bl_segment_g[k] for k >= 1
    g_seg_bl = harness["bl_segment_g__uS"]
    for k in range(1, R):
        expected = -g_seg_bl[k].item()
        actual = j_fd[idx_f_bl(k), idx_v_bl(k - 1)].item()
        rel_err = abs(actual - expected) / (abs(expected) + 1e-30)
        assert rel_err < 1e-5, (
            f"F_BL[{k}] vs V_BL[{k - 1}]: expected {expected:.3e}, got {actual:.3e} (rel err {rel_err:.3e})"
        )

    # --- (D) Boundary rows: F_CL_BL touches V_BL_CL and V_BL[0] only ---
    # All other columns must be 0.
    for col in range(n):
        if col in (idx_v_bl_cl, idx_v_bl(0)):
            continue
        entry = j_fd[idx_f_cl_bl, col].abs().item()
        assert entry < fd_eps, f"F_CL_BL has unexpected coupling to column {col}: |J| = {entry:.3e} (should be 0)"
    # Symmetric for F_CL_SL.
    for col in range(n):
        if col in (idx_v_sl_dr, idx_v_sl(0)):
            continue
        entry = j_fd[idx_f_cl_sl, col].abs().item()
        assert entry < fd_eps, f"F_CL_SL has unexpected coupling to column {col}: |J| = {entry:.3e} (should be 0)"

    # --- (E) Off-diagonal V_X column for cross rows must be zero (column ⇒ no row affects V_X across) ---
    # Symmetric check vs (A): no wire row's F has cross-row V_X coupling.
    # Already checked above; this is the column-wise dual:
    # for any unknown V_X[k_prime] at k_prime ≠ k, NO residual at row k can depend
    # on it except through V_BL[k], V_SL[k] — already checked.
