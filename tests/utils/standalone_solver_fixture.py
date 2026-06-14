"""Standalone Solver1T1R harness for solver-only tests.

Builds a complete set of independent RRAM / NMOS / OpAmpTIA / Driver
device modules + a chip-preset-driven ``Solver1T1R`` (nested or full-
Jacobian), with synthetic mid-range RRAM g and a configurable
``v_wl_drive`` grid. Does NOT touch ``CircuitCore1T1R`` /
``Offset1T1RXbar`` — solver tests should depend only on the solver.

Public surface: :func:`build_solver_harness` returns a frozen
``SolverHarness`` carrying the constructed solver, fabricated devices,
sampled snapshots, wire R/G tensors, and the ``v_wl_drive`` tensor.
Tests call ``harness.solver.solve_dc(**harness.solver_kwargs(),
compute_residuals=True)`` to exercise the solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from neurox.analog import Driver, DriverPolicy
from neurox.analog.tia import OpAmpTIA, OpAmpTIAPolicy
from neurox.common.load_dump import dataclass_from_file
from neurox.device import NMOS, RRAM, NMOSPolicy, RRAMPolicy
from neurox.xbar._1t1r import (
    Offset1T1RXbarConfig,
    Solver1T1R,
    Solver1T1RConfig,
)


@dataclass(frozen=True)
class SolverHarness:
    """All inputs required to call :meth:`Solver1T1R.solve_dc` directly."""

    solver: Solver1T1R
    rram: RRAM
    nmos: NMOS
    bl_driver: OpAmpTIA
    sl_driver: Driver
    rram_snapshot: Any
    nmos_snapshot: Any
    bl_driver_snapshot: Any
    sl_driver_snapshot: Any
    bl_segment_r__MOhm: Tensor
    sl_segment_r__MOhm: Tensor
    bl_segment_g__uS: Tensor
    sl_segment_g__uS: Tensor
    v_wl_drive__V: Tensor
    inst_shape: tuple[int, ...] = field(default_factory=tuple)

    def solver_kwargs(self) -> dict[str, Any]:
        """Pack the per-call kwargs for ``solver.solve_dc(...)``."""
        return {
            "v_wl_drive__V": self.v_wl_drive__V,
            "bl_segment_r__MOhm": self.bl_segment_r__MOhm,
            "sl_segment_r__MOhm": self.sl_segment_r__MOhm,
            "bl_segment_g__uS": self.bl_segment_g__uS,
            "sl_segment_g__uS": self.sl_segment_g__uS,
            "rram_snapshot": self.rram_snapshot,
            "nmos_snapshot": self.nmos_snapshot,
            "bl_driver_snapshot": self.bl_driver_snapshot,
            "sl_driver_snapshot": self.sl_driver_snapshot,
        }


def _wire_seg_tensor(first: float, segment: float, row_num: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.tensor([first] + [segment] * (row_num - 1), device=device, dtype=dtype)


def build_solver_harness(
    *,
    config_path: Path,
    solver_config: Solver1T1RConfig,
    inst_shape: tuple[int, ...],
    x_batch: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    g_uniform_frac: float = 0.4,
    v_wl_drive__V: float = 0.7,
) -> SolverHarness:
    """Construct a standalone solver test harness from a chip preset.

    Reads only ``[xbar]`` from the TOML for chip constants (RRAM / NMOS
    / TIA / Driver configs + wire R/C + core PPA + state map). Devices
    are built fresh with no nonideality policy and fabricated once;
    RRAM is programmed to a uniform mid-range conductance derived from
    the chip's ``rram_g_max__uS``. The solver is built standalone via
    :meth:`Solver1T1R.from_config` and bound to the devices.

    Args:
        config_path: Path to a chip TOML carrying ``[xbar]`` (Offset1T1RXbarConfig).
        solver_config: Concrete ``Solver1T1RConfig`` (nested or full-jacobian).
        inst_shape: Tile multiplicity (e.g. ``(4,)`` or ``(2, 1, 2)`` —
            interpreted as the prefix preceding ``(phys_col, row)``).
        x_batch: Leading x-batch size in front of ``inst_shape``.
        device: Torch device.
        dtype: Float dtype for device buffers.
        g_uniform_frac: RRAM conductance as a fraction of
            ``rram_g_max__uS`` (default 0.4 ≈ mid-range).
        v_wl_drive__V: Uniform WL drive voltage for the harness call.
    """
    xbar_config = dataclass_from_file(Offset1T1RXbarConfig, config_path, section="xbar")
    core_cfg = xbar_config.core_config
    phys_col_num = xbar_config.col_num * xbar_config.w_digit_count + (
        xbar_config.col_num * xbar_config.w_digit_count // xbar_config.ref_group_size
    )
    row_num = xbar_config.row_num

    inst_full = (*inst_shape, phys_col_num, row_num)

    # --- Devices (no nonideality) ---

    rram = RRAM(
        config=core_cfg.rram_config,
        policy=RRAMPolicy(prog_gamma=False, stuck_at=False, read_telegraph=False, read_thermal=False),
        inst_shape=inst_full,
        dtype=dtype,
        T__K=300.0,
        g_max__uS=core_cfg.rram_g_max__uS,
    )
    nmos = NMOS(
        config=core_cfg.nmos_config,
        policy=NMOSPolicy(A_vt_mismatch=False, A_beta_mismatch=False),
        inst_shape=inst_full,
        dtype=dtype,
        T__K=300.0,
        W__um=core_cfg.access_nmos_W__um,
        L__um=core_cfg.access_nmos_L__um,
    )
    bl_driver = OpAmpTIA(
        config=core_cfg.tia_config,
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

    for d in (rram, nmos, bl_driver, sl_driver):
        d.to(device)
        d.eval()
        d.fabricate()

    # --- RRAM programming: uniform mid-range g ---

    g_target = torch.full(
        inst_full,
        float(core_cfg.rram_g_max__uS) * g_uniform_frac,
        device=device,
        dtype=dtype,
    )
    rram.program(g_target, t_elapsed=0.0)

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

    # --- v_wl_drive — uniform per-row ---

    full_shape = (x_batch, *inst_shape, phys_col_num, row_num)
    v_wl_drive = torch.full(full_shape, v_wl_drive__V, device=device, dtype=dtype)

    # --- Snapshots at the broadcast shape used by the solver ---

    rram_snap = rram.snapshot(shape=full_shape, multi_coords=None)
    nmos_snap = nmos.snapshot(shape=full_shape, multi_coords=None)
    bl_drv_snap = bl_driver.snapshot(shape=(x_batch, *inst_shape, phys_col_num), multi_coords=None)
    sl_drv_snap = sl_driver.snapshot(shape=(x_batch, *inst_shape, phys_col_num), multi_coords=None)

    # --- Solver ---

    solver = Solver1T1R.from_config(
        config=solver_config,
        rram=rram,
        nmos=nmos,
        bl_driver=bl_driver,
        sl_driver=sl_driver,
    )

    return SolverHarness(
        solver=solver,
        rram=rram,
        nmos=nmos,
        bl_driver=bl_driver,
        sl_driver=sl_driver,
        rram_snapshot=rram_snap,
        nmos_snapshot=nmos_snap,
        bl_driver_snapshot=bl_drv_snap,
        sl_driver_snapshot=sl_drv_snap,
        bl_segment_r__MOhm=bl_seg_r,
        sl_segment_r__MOhm=sl_seg_r,
        bl_segment_g__uS=bl_seg_g,
        sl_segment_g__uS=sl_seg_g,
        v_wl_drive__V=v_wl_drive,
        inst_shape=inst_shape,
    )
