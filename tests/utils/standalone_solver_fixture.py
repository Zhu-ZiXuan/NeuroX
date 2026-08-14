"""Standalone linear solver harness for solver-only tests.

Builds a fully hand-written, fully linear tiny tile: an
`XbarCell1t1rLinear` cell grid (table-driven chord conductance,
empty policy), two IDEAL `VoltageDriver` rail clamps
(`r_out = 0`, so `solve_dc` returns the reference voltage
exactly), and one dedicated single-tap `Vref` per clamp. Every
config value is an explicit in-code witness; no config file is read and
no nonideality toggle is enabled, so the assembled system is an exactly
linear resistor network with Dirichlet rail boundaries — a dense KCL
oracle can reproduce the solver's DCOP to round-off.

Public surface: `build_solver_harness` returns a frozen
`SolverHarness` carrying the constructed solver, the programmed cell,
the two ideal clamp drivers with their snaps, the two rail link
resistances, the WL drive, and the oracle inputs (the cell config, the
programmed state indices, and the resolved rail reference taps). Tests
call `harness.solver.solve_dc(**harness.solver_kwargs(), ...)`; the
per-call cell snap is rebuilt by `SolverHarness.cell_snapshot`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from neurox.primitive.analog import (
    VoltageDriver,
    VoltageDriverConfig,
    VoltageDriverPolicy,
    VoltageDriverSnap,
    Vref,
    VrefConfig,
    VrefPolicy,
)
from neurox.primitive.xbar.cell import (
    XbarCell1t1rLinear,
    XbarCell1t1rLinearConfig,
    XbarCell1t1rLinearPolicy,
    XbarCell1t1rLinearSnap,
)
from neurox.primitive.xbar.solver import ColBlColSlSolver, ColBlColSlSolverConfig

# --- Hand-written harness constants (arbitrary small witnesses) ---

COL_NUM = 3
ROW_NUM = 4
X_BATCH = 1

# Two weight states: WL-off leakage ~5 uS, WL-on chord ~50/100 uS.
G_CELL_OFF_TABLE__uS = (4.0, 5.0)
G_CELL_ON_TABLE__uS = (50.0, 100.0)
VX_RATIO_OFF_TABLE = (0.5, 0.5)
VX_RATIO_ON_TABLE = (0.4, 0.6)
V_WL_ON_THRESHOLD__V = 0.5

# Resistance of one rail link [MOhm] — the lattice is uniform, the clamp
# driver's own link to node 0 included. BL and SL values differ so a rail
# swap cannot cancel; both are large enough against the cell chords above
# that the IR drop along the ladder stays plainly visible.
BL_SEGMENT_R__MOhm = 2e-4
SL_SEGMENT_R__MOhm = 4e-4

# Rail reference taps [V], one dedicated source per clamp driver.
BL_V_REF__V = 0.3
SL_V_REF__V = 0.1


def _linear_cell_config() -> XbarCell1t1rLinearConfig:
    return XbarCell1t1rLinearConfig(
        g_cell_off_table__uS=G_CELL_OFF_TABLE__uS,
        g_cell_on_table__uS=G_CELL_ON_TABLE__uS,
        vx_ratio_off_table=VX_RATIO_OFF_TABLE,
        vx_ratio_on_table=VX_RATIO_ON_TABLE,
        v_wl_on_threshold__V=V_WL_ON_THRESHOLD__V,
    )


def _ideal_driver_config() -> VoltageDriverConfig:
    return VoltageDriverConfig(
        r_out__MOhm=0.0,
        offset_sigma__V=0.0,
        thermal_sigma__V=0.0,
        energy_per_op__fJ=0.0,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
    )


def _single_tap_vref(v_ref__V: float, *, dtype: torch.dtype) -> Vref:
    """Build one dedicated reference source: the degenerate `[[v]]` bank."""
    return Vref(
        config=VrefConfig(
            v_refs__V=((v_ref__V,),),
            tolerance_sigma_relative=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        policy=VrefPolicy(tolerance=False),
        inst_shape=(),
        dtype=dtype,
        T__K=300.0,
    )


@dataclass(frozen=True)
class SolverHarness:
    """All inputs required to call `ColBlColSlSolver.solve_dc` directly.

    Also carries the dense-oracle inputs: the hand-written linear cell
    config, the programmed state-index grid, and the resolved rail
    reference taps (exact clamp targets, since both drivers are ideal).
    """

    solver: ColBlColSlSolver
    cell: XbarCell1t1rLinear
    cell_config: XbarCell1t1rLinearConfig
    w_state_idx: Tensor
    bl_driver: VoltageDriver
    sl_driver: VoltageDriver
    bl_driver_snap: VoltageDriverSnap
    sl_driver_snap: VoltageDriverSnap
    bl_segment_r__MOhm: float
    sl_segment_r__MOhm: float
    v_wl_drive__V: Tensor
    bl_v_ref__V: Tensor
    sl_v_ref__V: Tensor

    def cell_snapshot(self) -> XbarCell1t1rLinearSnap:
        """Build the per-call cell snap at the harness WL drive.

        The cell takes its own gate voltage per cell, so the per-row drive is
        expanded onto the cell grid exactly as an owning array does.
        """
        shape = (*self.v_wl_drive__V.shape[:-1], *self.cell.inst_shape)
        return self.cell.snapshot(
            control=self.v_wl_drive__V.unsqueeze(-2).expand(shape),
            shape=shape,
            t_elapsed=0.0,
        )

    def solver_kwargs(self) -> dict[str, Any]:
        """Pack the per-call kwargs for `solver.solve_dc(...)`.

        Includes the cell and the two clamp drivers — the stateless solver
        takes them per call.
        """
        return {
            "bl_segment_r__MOhm": self.bl_segment_r__MOhm,
            "sl_segment_r__MOhm": self.sl_segment_r__MOhm,
            "cell": self.cell,
            "cell_snap": self.cell_snapshot(),
            "bl_driver": self.bl_driver,
            "bl_driver_snap": self.bl_driver_snap,
            "sl_driver": self.sl_driver,
            "sl_driver_snap": self.sl_driver_snap,
        }


def build_solver_harness(
    *,
    solver_config: ColBlColSlSolverConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    v_wl_drive__V: float = 0.9,
    col_num: int = COL_NUM,
    row_num: int = ROW_NUM,
) -> SolverHarness:
    """Construct the standalone linear solver harness.

    The cell grid is `(col_num, row_num)` with a leading x-batch of
    `X_BATCH`; the programmed state indices alternate over the two
    table states so both table entries are exercised. Both rail clamps
    are ideal `VoltageDriver` instances (`r_out = 0`) whose snaps
    resolve the two hand-built reference taps, so the clamp boundaries
    are exact Dirichlet values and the whole system is linear.

    Args:
        solver_config: Parallel BL/SL solver parameters.
        device: Device the harness buffers are allocated on.
        dtype: Float dtype for device buffers.
        v_wl_drive__V: Uniform WL drive voltage for the harness call
            (default above the on-threshold: every access device on).
        col_num: Number of independent columns; a degenerate `1` is a
            legitimate tile.
        row_num: Number of wire-ladder nodes per column; a degenerate
            `1` is a legitimate tile.
    """
    cell_config = _linear_cell_config()
    grid_shape = (col_num, row_num)

    # --- Cell + ideal boundary drivers (all policies empty / all-off) ---

    cell = XbarCell1t1rLinear(
        config=cell_config,
        policy=XbarCell1t1rLinearPolicy(),
        inst_shape=grid_shape,
        dtype=dtype,
        T__K=300.0,
    )
    driver_config = _ideal_driver_config()
    driver_policy = VoltageDriverPolicy(offset=False, thermal=False)
    bl_driver = VoltageDriver(
        config=driver_config,
        policy=driver_policy,
        inst_shape=(col_num,),
        dtype=dtype,
        T__K=300.0,
    )
    sl_driver = VoltageDriver(
        config=driver_config,
        policy=driver_policy,
        inst_shape=(col_num,),
        dtype=dtype,
        T__K=300.0,
    )
    bl_ref = _single_tap_vref(BL_V_REF__V, dtype=dtype)
    sl_ref = _single_tap_vref(SL_V_REF__V, dtype=dtype)

    for m in (cell, bl_driver, sl_driver, bl_ref, sl_ref):
        m.to(device)
        m.eval()
        m.fabricate()

    # --- Cell programming: alternate the two table states over the grid ---

    w_state_idx = (torch.arange(col_num * row_num, device=device) % 2).reshape(grid_shape)
    cell.program(w_state_idx)

    # --- v_wl_drive — uniform per-row WL control for the cell snap ---

    v_wl_drive = torch.full((X_BATCH, row_num), v_wl_drive__V, device=device, dtype=dtype)

    # --- Clamp references (one dedicated source per clamp) + boundary-driver snaps ---

    # Each source is fabricate-only: its bank is read (mode 0, its sole
    # mode) and broadcast by view onto the full clamp-bank grid, then handed
    # to the driver's own `snapshot`, which expands it onto `shape` again
    # and draws whatever per-position dynamic noise its (here all-off)
    # policy would enable.
    # Shape: [X_BATCH, col_num, tap=1] -> [X_BATCH, col_num]
    bl_ref_full = bl_ref.v_out__V[..., 0, :].expand(X_BATCH, col_num, 1).squeeze(-1)
    sl_ref_full = sl_ref.v_out__V[..., 0, :].expand(X_BATCH, col_num, 1).squeeze(-1)
    bl_drv_snap = bl_driver.snapshot(v_ref__V=bl_ref_full, shape=bl_ref_full.shape)
    sl_drv_snap = sl_driver.snapshot(v_ref__V=sl_ref_full, shape=sl_ref_full.shape)

    # --- Scalar rail-reference taps for the dense oracle ---

    # The tap is uniform over the clamp bank: the dense oracle takes it
    # as one scalar Dirichlet boundary value.
    # Shape: [X_BATCH, col_num] -> []
    bl_v_ref, sl_v_ref = bl_ref_full[0, 0], sl_ref_full[0, 0]

    # --- Solver (stateless: cell + drivers supplied per call) ---

    solver = ColBlColSlSolver(config=solver_config)

    return SolverHarness(
        solver=solver,
        cell=cell,
        cell_config=cell_config,
        w_state_idx=w_state_idx,
        bl_driver=bl_driver,
        sl_driver=sl_driver,
        bl_driver_snap=bl_drv_snap,
        sl_driver_snap=sl_drv_snap,
        bl_segment_r__MOhm=BL_SEGMENT_R__MOhm,
        sl_segment_r__MOhm=SL_SEGMENT_R__MOhm,
        v_wl_drive__V=v_wl_drive,
        bl_v_ref__V=bl_v_ref,
        sl_v_ref__V=sl_v_ref,
    )
