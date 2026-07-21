"""Standalone linear solver harness for solver-only tests.

Builds a fully hand-written, fully linear tiny tile: an
:class:`XbarCell1t1rLinear` cell grid (table-driven chord conductance,
empty policy), two IDEAL :class:`VoltageDriver` rail clamps
(``r_out = 0``, so ``solve_clamp`` returns the reference voltage
exactly), and a hand-built two-tap :class:`VoltageReference`. Every
config value is an explicit in-code witness; no config file is read and
no nonideality toggle is enabled, so the assembled system is an exactly
linear resistor network with Dirichlet rail boundaries — a dense KCL
oracle can reproduce the solver's DCOP to round-off.

Public surface: :func:`build_solver_harness` returns a frozen
``SolverHarness`` carrying the constructed solver, the programmed cell,
the two ideal clamp drivers with their snaps, the wire R/G tensors, the
WL drive, and the oracle inputs (the cell config, the programmed state
indices, and the resolved rail reference taps). Tests call
``harness.solver.solve_dc(**harness.solver_kwargs(), ...)``; the
per-call cell snap is rebuilt by :meth:`SolverHarness.cell_snapshot`.
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
    VoltageReference,
    VoltageReferenceConfig,
    VoltageReferencePolicy,
)
from neurox.primitive.xbar.cell import (
    XbarCell1t1rLinear,
    XbarCell1t1rLinearConfig,
    XbarCell1t1rLinearPolicy,
    XbarCell1t1rLinearSnap,
)
from neurox.primitive.xbar.solver import Solver, SolverConfig

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

C_NODE__fF = 0.1

# Wire segment resistances [MOhm]; index 0 is driver-to-first. BL and SL
# values differ so a rail swap cannot cancel.
BL_FIRST_R__MOhm = 2e-4
BL_SEGMENT_R__MOhm = 1e-4
SL_FIRST_R__MOhm = 4e-4
SL_SEGMENT_R__MOhm = 2e-4

# Rail reference taps [V]: tap 0 = BL clamp, tap 1 = SL drive.
BL_V_REF__V = 0.3
SL_V_REF__V = 0.1


def _linear_cell_config() -> XbarCell1t1rLinearConfig:
    return XbarCell1t1rLinearConfig(
        c_bl__fF=C_NODE__fF,
        c_x__fF=C_NODE__fF,
        c_sl__fF=C_NODE__fF,
        c_wl__fF=C_NODE__fF,
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


@dataclass(frozen=True)
class SolverHarness:
    """All inputs required to call :meth:`Solver.solve_dc` directly.

    Also carries the dense-oracle inputs: the hand-written linear cell
    config, the programmed state-index grid, and the resolved rail
    reference taps (exact clamp targets, since both drivers are ideal).
    """

    solver: Solver
    cell: XbarCell1t1rLinear
    cell_config: XbarCell1t1rLinearConfig
    w_state_idx: Tensor
    bl_driver: VoltageDriver
    sl_driver: VoltageDriver
    bl_driver_snap: VoltageDriverSnap
    sl_driver_snap: VoltageDriverSnap
    bl_segment_r__MOhm: Tensor
    sl_segment_r__MOhm: Tensor
    bl_segment_g__uS: Tensor
    sl_segment_g__uS: Tensor
    v_wl_drive__V: Tensor
    bl_v_ref__V: Tensor
    sl_v_ref__V: Tensor

    def cell_snapshot(self) -> XbarCell1t1rLinearSnap:
        """Build the per-call cell snap at the harness WL drive."""
        return self.cell.snapshot(
            control=self.v_wl_drive__V,
            shape=tuple(self.v_wl_drive__V.shape),
            multi_coords=None,
            t_elapsed=0.0,
        )

    def solver_kwargs(self) -> dict[str, Any]:
        """Pack the per-call kwargs for ``solver.solve_dc(...)``.

        Includes the cell and the two clamp drivers — the stateless solver
        takes them per call.
        """
        return {
            "bl_segment_r__MOhm": self.bl_segment_r__MOhm,
            "sl_segment_r__MOhm": self.sl_segment_r__MOhm,
            "bl_segment_g__uS": self.bl_segment_g__uS,
            "sl_segment_g__uS": self.sl_segment_g__uS,
            "cell": self.cell,
            "cell_snap": self.cell_snapshot(),
            "bl_driver": self.bl_driver,
            "bl_driver_snap": self.bl_driver_snap,
            "sl_driver": self.sl_driver,
            "sl_driver_snap": self.sl_driver_snap,
        }


def _wire_seg_tensor(first: float, segment: float, row_num: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.tensor([first] + [segment] * (row_num - 1), device=device, dtype=dtype)


def build_solver_harness(
    *,
    solver_config: SolverConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    v_wl_drive__V: float = 0.9,
) -> SolverHarness:
    """Construct the standalone linear solver harness.

    The cell grid is ``(COL_NUM, ROW_NUM)`` with a leading x-batch of
    ``X_BATCH``; the programmed state indices alternate over the two
    table states so both table entries are exercised. Both rail clamps
    are ideal ``VoltageDriver`` instances (``r_out = 0``) whose snaps
    resolve the two hand-built reference taps, so the clamp boundaries
    are exact Dirichlet values and the whole system is linear.

    Args:
        solver_config: Concrete ``SolverConfig`` (nested).
        device: Torch device.
        dtype: Float dtype for device buffers.
        v_wl_drive__V: Uniform WL drive voltage for the harness call
            (default above the on-threshold: every access device on).
    """
    cell_config = _linear_cell_config()
    grid_shape = (COL_NUM, ROW_NUM)

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
        inst_shape=(COL_NUM,),
        dtype=dtype,
        T__K=300.0,
    )
    sl_driver = VoltageDriver(
        config=driver_config,
        policy=driver_policy,
        inst_shape=(COL_NUM,),
        dtype=dtype,
        T__K=300.0,
    )
    clamp_ref = VoltageReference(
        config=VoltageReferenceConfig(
            v_refs__V=(BL_V_REF__V, SL_V_REF__V),
            tolerance_sigma_relative=0.0,
            noise_sigma_relative=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        policy=VoltageReferencePolicy(tolerance=False, noise=False),
        inst_shape=(),
        dtype=dtype,
        T__K=300.0,
    )

    for m in (cell, bl_driver, sl_driver, clamp_ref):
        m.to(device)
        m.eval()
        m.fabricate()

    # --- Cell programming: alternate the two table states over the grid ---

    w_state_idx = (torch.arange(COL_NUM * ROW_NUM, device=device) % 2).reshape(grid_shape)
    cell.program(w_state_idx)

    # --- Wire R / G tensors ---

    bl_seg_r = _wire_seg_tensor(BL_FIRST_R__MOhm, BL_SEGMENT_R__MOhm, ROW_NUM, device, dtype)
    sl_seg_r = _wire_seg_tensor(SL_FIRST_R__MOhm, SL_SEGMENT_R__MOhm, ROW_NUM, device, dtype)
    bl_seg_g = 1.0 / bl_seg_r
    sl_seg_g = 1.0 / sl_seg_r

    # --- v_wl_drive — uniform per-cell WL control for the cell snap ---

    v_wl_drive = torch.full((X_BATCH, *grid_shape), v_wl_drive__V, device=device, dtype=dtype)

    # --- Clamp-reference snapshot (once) + boundary-driver snaps ---

    clamp_taps = clamp_ref.v_ref__V(clamp_ref.snapshot())
    bl_v_ref = clamp_taps[0]
    sl_v_ref = clamp_taps[1]
    bl_drv_snap = bl_driver.snapshot(v_ref__V=bl_v_ref, shape=(X_BATCH, COL_NUM), multi_coords=None)
    sl_drv_snap = sl_driver.snapshot(v_ref__V=sl_v_ref, shape=(X_BATCH, COL_NUM), multi_coords=None)

    # --- Solver (stateless: cell + drivers supplied per call) ---

    solver = Solver.from_config(config=solver_config)

    return SolverHarness(
        solver=solver,
        cell=cell,
        cell_config=cell_config,
        w_state_idx=w_state_idx,
        bl_driver=bl_driver,
        sl_driver=sl_driver,
        bl_driver_snap=bl_drv_snap,
        sl_driver_snap=sl_drv_snap,
        bl_segment_r__MOhm=bl_seg_r,
        sl_segment_r__MOhm=sl_seg_r,
        bl_segment_g__uS=bl_seg_g,
        sl_segment_g__uS=sl_seg_g,
        v_wl_drive__V=v_wl_drive,
        bl_v_ref__V=bl_v_ref,
        sl_v_ref__V=sl_v_ref,
    )
