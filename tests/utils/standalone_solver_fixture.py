"""Linear-cell solver fixtures with ideal clamps and dense-oracle inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from neurox.common.module import DcopBase
from neurox.primitive.analog import (
    Reference,
    ReferenceConfig,
    ReferencePolicy,
    VoltageDriver,
    VoltageDriverConfig,
    VoltageDriverPolicy,
    VoltageDriverSnap,
)
from neurox.primitive.xbar.cell import (
    XbarCell1t1rLinear,
    XbarCell1t1rLinearConfig,
    XbarCell1t1rLinearPolicy,
    XbarCell1t1rLinearSnap,
)
from neurox.primitive.xbar.solver import ColBlColSlArraySolver, ColBlColSlArrayState, ColBlColSlArrayTrace
from neurox.primitive.xbar.solver.resistive_cell import ResistiveCellDcop


class SolverDcop(DcopBase):
    """Caller-owned full electrical result for standalone solver assertions."""

    cell_dcop: ResistiveCellDcop
    i_bl_port__uA: Tensor
    i_sl_port__uA: Tensor
    v_bl_node__V: Tensor
    v_sl_node__V: Tensor
    v_bl_port__V: Tensor
    v_sl_port__V: Tensor


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


def _scalar_reference(v_ref__V: float, *, dtype: torch.dtype) -> Reference:
    """Build one dedicated scalar reference source."""
    return Reference(
        config=ReferenceConfig(
            values=v_ref__V,
            tolerance_sigma_relative=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        policy=ReferencePolicy(tolerance=False),
        inst_shape=(),
        dtype=dtype,
        T__K=300.0,
    )


@dataclass(frozen=True)
class SolverHarness:
    """All inputs required to construct and call `ColBlColSlArraySolver` directly.

    Also carries the dense-oracle inputs: the hand-written linear cell
    config, the programmed state-index grid, and the resolved rail
    reference taps (exact clamp targets, since both drivers are ideal).
    """

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
        )

    def solve_kwargs(self) -> dict[str, Any]:
        """Return construction arguments and per-call snapshots for the solver."""
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
        device: Device the harness buffers are allocated on.
        dtype: Float dtype for device buffers.
        v_wl_drive__V: Uniform WL drive voltage for the harness call
            (default above the on-threshold: every access device on).
        col_num: Number of independent columns; a degenerate `1` is a
            legitimate array.
        row_num: Number of wire-ladder nodes per column; a degenerate
            `1` is a legitimate array.
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
    bl_ref = _scalar_reference(BL_V_REF__V, dtype=dtype)
    sl_ref = _scalar_reference(SL_V_REF__V, dtype=dtype)

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

    bl_ref_full = bl_ref.values().expand(X_BATCH, col_num)
    sl_ref_full = sl_ref.values().expand(X_BATCH, col_num)
    bl_drv_snap = bl_driver.snapshot(v_ref__V=bl_ref_full, shape=bl_ref_full.shape)
    sl_drv_snap = sl_driver.snapshot(v_ref__V=sl_ref_full, shape=sl_ref_full.shape)

    # --- Scalar rail-reference taps for the dense oracle ---

    # The tap is uniform over the clamp bank: the dense oracle takes it
    # as one scalar Dirichlet boundary value.
    # Shape: [batch, col] -> []
    bl_v_ref = bl_ref_full[0, 0]
    sl_v_ref = sl_ref_full[0, 0]

    return SolverHarness(
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


def solve_dcop(
    array_solver: ColBlColSlArraySolver[Any, Any, Any, Any],
    *,
    cell_snap: Any,
    bl_driver_snap: Any,
    sl_driver_snap: Any,
    record_trace: bool,
    trace_mask: Tensor | None = None,
) -> tuple[SolverDcop, ColBlColSlArrayTrace | None]:
    """Request a full terminal DCOP for electrical solver assertions."""

    def final_fn(state: ColBlColSlArrayState) -> SolverDcop:
        return SolverDcop(
            cell_dcop=array_solver.cell.solve_dc(state.v_bl_node__V, state.v_sl_node__V, cell_snap),
            i_bl_port__uA=(state.v_bl_port__V - state.v_bl_node__V[..., 0]) * array_solver.bl_g__uS,
            i_sl_port__uA=(state.v_sl_port__V - state.v_sl_node__V[..., 0]) * array_solver.sl_g__uS,
            v_bl_node__V=state.v_bl_node__V,
            v_sl_node__V=state.v_sl_node__V,
            v_bl_port__V=state.v_bl_port__V,
            v_sl_port__V=state.v_sl_port__V,
        )

    return array_solver.solve(
        cell_snap=cell_snap,
        bl_driver_snap=bl_driver_snap,
        sl_driver_snap=sl_driver_snap,
        final_fn=final_fn,
        record_trace=record_trace,
        trace_mask=trace_mask,
    )
