"""Unit tests for the pluggable crossbar-cell abstraction.

Covers the 1T1R cell (:class:`XbarCell1T1R`): the condensed branch solve
(shapes + signed terminal conductances), the full DC working point
(internal ``V_X`` + access-node KCL residual), and the per-cell
device-capacitance switching energy.

The cell is built standalone from the reference 28nm chip preset's
``[xbar.core_config.cell_config]`` fields — RRAM / NMOS device configs
and the cell-physical sizing / parasitic-cap densities — so the test
exercises realistic device values without depending on the array solver
or core. CPU only.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from neurox.common.load_dump import dataclass_from_file
from neurox.device import NMOSPolicy, RRAMPolicy
from neurox.xbar._1t1r import Offset1T1RXbarConfig
from neurox.xbar._1t1r.cell import (
    XbarCell1T1R,
    XbarCell1T1RConfig,
    XbarCell1T1RDCOP,
    XbarCell1T1RPolicy,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
XBAR_CONFIG = REPO_ROOT / "example" / "config" / "1t1r_28nm.toml"

DTYPE = torch.float64
N_NEWTON = 3  # placeholder count; the per-cell Newton converges fast at fp64
COL = 4
ROW = 5
PREFIX = (2,)
INST_SHAPE = (*PREFIX, COL, ROW)


@pytest.fixture(scope="module")
def cell() -> Iterator[XbarCell1T1R]:
    """Build, fabricate, and program a standalone 1T1R cell on CPU."""
    if not XBAR_CONFIG.is_file():
        pytest.skip(f"missing test fixture: {XBAR_CONFIG}")

    xbar_config = dataclass_from_file(Offset1T1RXbarConfig, XBAR_CONFIG, section="xbar")
    preset_cell = xbar_config.core_config.cell_config

    cell_config = XbarCell1T1RConfig(
        rram_config=preset_cell.rram_config,
        nmos_config=preset_cell.nmos_config,
        rram_g_max__uS=preset_cell.rram_g_max__uS,
        access_nmos_W__um=preset_cell.access_nmos_W__um,
        access_nmos_L__um=preset_cell.access_nmos_L__um,
        c_gs_per_um__fF=preset_cell.c_gs_per_um__fF,
        c_gd_per_um__fF=preset_cell.c_gd_per_um__fF,
        c_db_per_um__fF=preset_cell.c_db_per_um__fF,
        state_to_g_map__uS=preset_cell.state_to_g_map__uS,
        n_newton=N_NEWTON,
    )
    policy = XbarCell1T1RPolicy(
        rram=RRAMPolicy(prog_gamma=False, stuck_at=False, read_telegraph=False, read_thermal=False),
        nmos=NMOSPolicy(A_vt_mismatch=False, A_beta_mismatch=False),
    )
    cell = XbarCell1T1R(
        config=cell_config,
        policy=policy,
        inst_shape=INST_SHAPE,
        dtype=DTYPE,
        T__K=300.0,
    )
    cell.eval()
    cell.fabricate()

    # Program to a mid-state index so RRAM g is well inside the window.
    w_state_idx = torch.ones(INST_SHAPE, dtype=torch.long)
    cell.program(w_state_idx)

    yield cell


def _make_terminals(cell: XbarCell1T1R) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a forward-biased (V_BL > V_SL), WL-on operating point."""
    v_bl = torch.full(INST_SHAPE, 0.30, dtype=DTYPE)
    v_sl = torch.full(INST_SHAPE, 0.05, dtype=DTYPE)
    v_wl = torch.full(INST_SHAPE, 0.90, dtype=DTYPE)
    return v_bl, v_sl, v_wl


def test_solve_branch_shapes_and_signs(cell: XbarCell1T1R) -> None:
    """``solve_branch`` returns the three branch tensors at the cell shape
    with the BL-side conductance non-negative and the SL-side conductance
    non-positive."""
    v_bl, v_sl, v_wl = _make_terminals(cell)
    snap = cell.snapshot(control=v_wl, shape=INST_SHAPE, multi_coords=None, t_elapsed=0.0)

    i__uA, di_dvbl__uS, di_dvsl__uS = cell.solve_branch(v_bl, v_sl, snap)

    assert i__uA.shape == INST_SHAPE
    assert di_dvbl__uS.shape == INST_SHAPE
    assert di_dvsl__uS.shape == INST_SHAPE

    assert torch.all(di_dvbl__uS >= 0.0)
    assert torch.all(di_dvsl__uS <= 0.0)

    # Forward bias with WL on drives a positive BL → SL current.
    assert torch.all(i__uA > 0.0)


def test_solve_dc_vx_and_residual(cell: XbarCell1T1R) -> None:
    """``solve_dc`` returns the condensed ``V_X`` between the terminals and
    a tiny access-node KCL residual once Newton has converged."""
    v_bl, v_sl, v_wl = _make_terminals(cell)
    snap = cell.snapshot(control=v_wl, shape=INST_SHAPE, multi_coords=None, t_elapsed=0.0)

    dcop = cell.solve_dc(v_bl, v_sl, snap, compute_residuals=True)

    assert isinstance(dcop, XbarCell1T1RDCOP)
    assert dcop.v_x__V.shape == INST_SHAPE
    # V_X (NMOS drain / RRAM bottom) sits between the two terminals.
    assert torch.all(dcop.v_x__V <= v_bl + 1e-9)
    assert torch.all(dcop.v_x__V >= v_sl - 1e-9)

    assert dcop.residuals is not None
    # n_newton Newton steps drive the access-node KCL to fp64 noise.
    assert dcop.residuals.cell__uA.max().item() < 1e-6

    # solve_branch and solve_dc agree on the condensed branch current.
    i_branch, _, _ = cell.solve_branch(v_bl, v_sl, snap)
    assert torch.allclose(i_branch, dcop.i__uA)


def test_solve_dc_residual_none_on_hot_path(cell: XbarCell1T1R) -> None:
    """``compute_residuals=False`` leaves the residual bundle as ``None``."""
    v_bl, v_sl, v_wl = _make_terminals(cell)
    snap = cell.snapshot(control=v_wl, shape=INST_SHAPE, multi_coords=None, t_elapsed=0.0)
    dcop = cell.solve_dc(v_bl, v_sl, snap, compute_residuals=False)
    assert dcop.residuals is None


def test_dynamic_energy_shape_nonneg(cell: XbarCell1T1R) -> None:
    """``dynamic_energy`` is a per-cell non-negative [fJ] tensor."""
    v_bl, v_sl, v_wl = _make_terminals(cell)
    snap = cell.snapshot(control=v_wl, shape=INST_SHAPE, multi_coords=None, t_elapsed=0.0)
    dcop = cell.solve_dc(v_bl, v_sl, snap)

    energy__fJ = cell.dynamic_energy(v_bl, v_sl, dcop, snap)
    assert energy__fJ.shape == INST_SHAPE
    assert torch.all(energy__fJ >= 0.0)
