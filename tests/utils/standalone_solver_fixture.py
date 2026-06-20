"""Standalone solver harness for solver-only tests.

Builds a standalone :class:`XbarCell1T1R` (owning fabricated RRAM /
access-NMOS), independent OpAmpTIA / Driver boundary modules, and a
chip-preset-driven stateless ``Solver``, with synthetic mid-range RRAM g
and a configurable ``v_wl_drive`` grid. Does NOT touch ``CircuitCore1T1R``
/ ``Offset1T1RXbar`` — solver tests should depend only on the solver.

Public surface: :func:`build_solver_harness` returns a frozen
``SolverHarness`` carrying the constructed solver, the fabricated cell,
the boundary drivers, sampled boundary snaps, wire R/G tensors, and
the ``v_wl_drive`` tensor. The solver is stateless, so the cell and the
two clamp drivers are packed as per-call kwargs alongside their snaps.
Tests call ``harness.solver.solve_dc(**harness.solver_kwargs(),
compute_residuals=True)`` to exercise the solver; the per-call cell snap
is rebuilt by :meth:`SolverHarness.cell_snapshot`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from neurox.analog import Driver, DriverPolicy
from neurox.analog.tia import OpAmpTIA, OpAmpTIAConfig, OpAmpTIAPolicy
from neurox.common.load_dump import dataclass_from_file
from neurox.device import NMOSPolicy, RRAMPolicy
from neurox.xbar._1t1r import Offset1T1RXbarConfig
from neurox.xbar._1t1r.cell import XbarCell1T1R, XbarCell1T1RPolicy, XbarCell1T1RSnap
from neurox.xbar.solver import Solver, SolverConfig


@dataclass(frozen=True)
class SolverHarness:
    """All inputs required to call :meth:`Solver.solve_dc` directly."""

    solver: Solver
    cell: XbarCell1T1R
    bl_driver: OpAmpTIA
    sl_driver: Driver
    bl_driver_snap: Any
    sl_driver_snap: Any
    bl_segment_r__MOhm: Tensor
    sl_segment_r__MOhm: Tensor
    bl_segment_g__uS: Tensor
    sl_segment_g__uS: Tensor
    v_wl_drive__V: Tensor
    inst_shape: tuple[int, ...] = field(default_factory=tuple)

    def cell_snapshot(self) -> XbarCell1T1RSnap:
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
        takes them per call, not at construction.
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
    config_path: Path,
    solver_config: SolverConfig,
    inst_shape: tuple[int, ...],
    x_batch: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    g_uniform_frac: float = 0.4,
    v_wl_drive__V: float = 0.7,
) -> SolverHarness:
    """Construct a standalone solver test harness from a chip preset.

    Reads only ``[xbar]`` from the TOML for chip constants (the 1T1R cell
    config carrying RRAM / NMOS + sizing + state map, the TIA / Driver
    configs, and wire R/C). The cell and the two boundary drivers are
    built fresh with no nonideality policy and fabricated once; the cell's
    RRAM is programmed to a uniform mid-range conductance derived from the
    cell config's ``rram_g_max__uS`` via a synthetic state-index tensor.
    The solver is built standalone via :meth:`Solver.from_config`; the
    cell + drivers are supplied per call (see :meth:`SolverHarness.solver_kwargs`).

    Args:
        config_path: Path to a chip TOML carrying ``[xbar]`` (Offset1T1RXbarConfig).
        solver_config: Concrete ``SolverConfig`` (nested).
        inst_shape: Tile multiplicity (e.g. ``(4,)`` or ``(2, 1, 2)`` —
            interpreted as the prefix preceding ``(phys_col, row)``).
        x_batch: Leading x-batch size in front of ``inst_shape``.
        device: Torch device.
        dtype: Float dtype for device buffers.
        g_uniform_frac: RRAM conductance as a fraction of the cell's
            ``rram_g_max__uS`` (default 0.4 ≈ mid-range).
        v_wl_drive__V: Uniform WL drive voltage for the harness call.
    """
    xbar_config = dataclass_from_file(Offset1T1RXbarConfig, config_path, section="xbar")
    core_cfg = xbar_config.core_config
    cell_cfg = core_cfg.cell_config
    phys_col_num = xbar_config.col_num * xbar_config.w_digit_count + (
        xbar_config.col_num * xbar_config.w_digit_count // xbar_config.ref_group_size
    )
    row_num = xbar_config.row_num

    inst_full = (*inst_shape, phys_col_num, row_num)

    # --- Cell + boundary drivers (no nonideality) ---

    cell = XbarCell1T1R(
        config=cell_cfg,
        policy=XbarCell1T1RPolicy(
            rram=RRAMPolicy(prog_gamma=False, stuck_at=False, read_telegraph=False, read_thermal=False),
            nmos=NMOSPolicy(A_vt_mismatch=False, A_beta_mismatch=False),
        ),
        inst_shape=inst_full,
        dtype=dtype,
        T__K=300.0,
    )
    tia_cfg = core_cfg.tia_config
    assert isinstance(tia_cfg, OpAmpTIAConfig)
    bl_driver = OpAmpTIA(
        config=tia_cfg,
        policy=OpAmpTIAPolicy(
            opamp_gain_sigma=False,
            nmos=NMOSPolicy(A_vt_mismatch=False, A_beta_mismatch=False),
        ),
        name="harness.bl_driver",
        inst_shape=(*inst_shape, phys_col_num),
        dtype=dtype,
        T__K=300.0,
    )
    sl_driver = Driver(
        config=core_cfg.sl_driver_config,
        policy=DriverPolicy(drive_thermal=False),
        name="harness.sl_driver",
        inst_shape=(*inst_shape, phys_col_num),
        dtype=dtype,
        T__K=300.0,
    )

    for m in (cell, bl_driver, sl_driver):
        m.to(device)
        m.eval()
        m.fabricate()

    # --- Cell programming: uniform mid-range g via a state index ---

    # Pick the state whose mapped conductance is nearest to the requested
    # mid-range fraction so the synthetic program stays inside the window.
    g_target__uS = float(cell_cfg.rram_g_max__uS) * g_uniform_frac
    state_map = torch.tensor(cell_cfg.state_to_g_map__uS, dtype=dtype)
    state_idx = int((state_map - g_target__uS).abs().argmin().item())
    w_state_idx = torch.full(inst_full, state_idx, device=device, dtype=torch.long)
    cell.program(w_state_idx)

    # --- Wire R / G tensors ---

    bl_seg_r = _wire_seg_tensor(
        core_cfg.bl_first_r__MOhm,
        core_cfg.bl_segment_r__MOhm,
        row_num,
        device,
        dtype,
    )
    sl_seg_r = _wire_seg_tensor(
        core_cfg.sl_first_r__MOhm,
        core_cfg.sl_segment_r__MOhm,
        row_num,
        device,
        dtype,
    )
    bl_seg_g = 1.0 / bl_seg_r
    sl_seg_g = 1.0 / sl_seg_r

    # --- v_wl_drive — uniform per-row WL control for the cell snap ---

    full_shape = (x_batch, *inst_shape, phys_col_num, row_num)
    v_wl_drive = torch.full(full_shape, v_wl_drive__V, device=device, dtype=dtype)

    # --- Boundary-driver snaps at the broadcast shape used by the solver ---

    bl_drv_snap = bl_driver.snapshot(shape=(x_batch, *inst_shape, phys_col_num), multi_coords=None)
    sl_drv_snap = sl_driver.snapshot(shape=(x_batch, *inst_shape, phys_col_num), multi_coords=None)

    # --- Solver (stateless: cell + drivers supplied per call) ---

    solver = Solver.from_config(config=solver_config)

    return SolverHarness(
        solver=solver,
        cell=cell,
        bl_driver=bl_driver,
        sl_driver=sl_driver,
        bl_driver_snap=bl_drv_snap,
        sl_driver_snap=sl_drv_snap,
        bl_segment_r__MOhm=bl_seg_r,
        sl_segment_r__MOhm=sl_seg_r,
        bl_segment_g__uS=bl_seg_g,
        sl_segment_g__uS=sl_seg_g,
        v_wl_drive__V=v_wl_drive,
        inst_shape=inst_shape,
    )
