"""Shared utilities for the solver calibration tools.

Provides:

  * :func:`build_xbar_for_calibration` — builds a noise-off
    :class:`Offset1T1RXbar` with a caller-supplied
    :class:`SolverConfig` and TIA iteration count, leaving the rest
    of the chip preset verbatim. Every CLI in this package starts here.

  * :func:`aggregate_xbar_sweep` — streams a ``(w, x)`` workload through
    a list of candidate :class:`Solver` instances and accumulates
    per-candidate step deltas + residuals + workload-derived signal
    scales (``CandidateRow`` / ``WorkloadScale``). The plateau picker
    in :mod:`._plateau` consumes these.

All sampling reuses the ``xbar_adc._sampling`` distribution helpers so
calibration sweeps share the same workload semantics as the ADC stat
tool.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from neurox.analog.tia import OpAmpTIAConfig
from neurox.tools.xbar_adc._sampling import (
    load_distribution,
    make_generator,
    sample_w,
    sample_x_batches,
)
from neurox.xbar import (
    Offset1T1RXbar,
    Offset1T1RXbarConfig,
    Offset1T1RXbarPolicy,
)
from neurox.xbar._1t1r import CircuitCore1T1R
from neurox.xbar.readout import OffsetSwitchCapMuxAdcReadOutConfig
from neurox.xbar.solver import Solver, SolverConfig

from ._plateau import CandidateRow, WorkloadScale


def build_xbar_for_calibration(
    config_path: Path,
    *,
    device: torch.device,
    inst_shape: tuple[int, ...],
    dtype: torch.dtype,
    solver_config: SolverConfig,
    solve_chunk_size: int = 0,
) -> Offset1T1RXbar:
    """Build a noise-off xbar with the supplied solver config.

    Substitutes ``solver_config`` on a copy of the chip preset TOML;
    everything else — including TIA ``n_newton`` — comes verbatim from
    the preset. The discipline for re-calibration after a TIA design
    change is: calibrate TIA first, write the picked ``n_newton`` to
    the preset, then calibrate the solver from the updated preset.

    Args:
        config_path: Chip preset TOML path.
        device: Target torch device.
        inst_shape: Per-instance shape for the xbar fabricator.
        dtype: Tensor dtype.
        solver_config: Concrete :class:`SolverConfig` subclass selecting
            which solver implementation to dispatch via the registry.

    Returns:
        A fabricated noise-off :class:`Offset1T1RXbar`.
    """
    from neurox.common.load_dump import dataclass_from_file, preset_path

    xbar_config = dataclass_from_file(Offset1T1RXbarConfig, config_path, section="xbar")
    readout_config = xbar_config.readout_config
    if not isinstance(readout_config, OffsetSwitchCapMuxAdcReadOutConfig):
        raise TypeError(
            f"calibration tool only supports OffsetSwitchCapMuxAdcReadOut; got {type(readout_config).__name__}"
        )

    core_cfg = xbar_config.core_config
    tia_cfg = core_cfg.tia_config
    if not isinstance(tia_cfg, OpAmpTIAConfig):
        raise TypeError(f"calibration tool requires OpAmpTIAConfig; got {type(tia_cfg).__name__}")
    new_core_cfg = replace(core_cfg, solver_config=solver_config)
    new_xbar_config = replace(xbar_config, core_config=new_core_cfg)

    # Noise-off policy from the shared preset (standard McsSarAdc topology),
    # overriding only the run-specific chunking knob.
    base_policy = dataclass_from_file(Offset1T1RXbarPolicy, preset_path("policy/all_off.toml"), section="xbar")
    policy = replace(
        base_policy,
        core=replace(
            base_policy.core,
            solve_chunk_size=solve_chunk_size,
        ),
    )

    xbar = Offset1T1RXbar(
        config=new_xbar_config,
        policy=policy,
        name="probe",
        inst_shape=inst_shape,
        dtype=dtype,
        T__K=300.0,
    )
    xbar.to(device)
    xbar.eval()
    xbar.fabricate()
    return xbar


# ---------------------------------------------------------------------------
# Step-ratio plateau sweep aggregator (nested SL/BL solver)
# ---------------------------------------------------------------------------


# Step-delta classes (plateau picker). ``v_x`` is the condensed
# access-node voltage carried on the cell DCOP (``SolverDCOP.cell.v_x__V``);
# the rest are solver-owned wire / clamp unknowns read straight off the DCOP.
_XBAR_SOLVER_UNKNOWN_FIELDS: tuple[str, ...] = (
    "v_bl_node",
    "v_sl_node",
    "v_bl_clamp",
    "v_sl_drive",
)
"""SolverDCOP-owned fields tracked for the plateau picker's step deltas."""

_XBAR_CELL_STEP_KEY = "v_x"
"""Step-delta key for the cell's condensed access-node voltage (``cell.v_x__V``)."""

_XBAR_UNKNOWN_FIELDS: tuple[str, ...] = (*_XBAR_SOLVER_UNKNOWN_FIELDS, _XBAR_CELL_STEP_KEY)
"""All step-delta classes (solver wire / clamp unknowns + the cell access node)."""

# Residual classes (safety guard). ``cell__uA`` is the per-cell internal-KCL
# residual on the cell DCOP (``SolverDCOP.cell.residuals.cell__uA``); the
# rest are solver-owned wire / clamp residuals.
_XBAR_SOLVER_RESIDUAL_FIELDS: tuple[str, ...] = (
    "wire_bl__uA",
    "wire_sl__uA",
    "clamp_bl__V",
    "clamp_sl__V",
)
"""SolverResiduals fields tracked for the residual safety guard."""

_XBAR_CELL_RESIDUAL_KEY = "cell__uA"
"""Residual key for the per-cell internal-KCL mismatch (``cell.residuals.cell__uA``)."""

_XBAR_RESIDUAL_FIELDS: tuple[str, ...] = (_XBAR_CELL_RESIDUAL_KEY, *_XBAR_SOLVER_RESIDUAL_FIELDS)
"""All residual classes (per-cell internal KCL + solver wire / clamp)."""


def _solver_inputs_from_core(core: CircuitCore1T1R, x: Tensor) -> dict[str, Any]:
    """Assemble :meth:`Solver.solve_dc` kwargs from an already-fabricated core.

    Tool-local W2 path: the calibrate CLIs need to drive the solver
    under realistic chip context (RRAM g programmed via workload
    sampling, real wire R/G, real DAC drive), but bypass
    :meth:`CircuitCore1T1R.cim_read` so they can inspect raw
    :class:`SolverDCOP` residuals. We rely on the chip-context
    invariants core upholds (fabricated cell devices + cached wire R/G +
    a WL DAC that knows how to convert a code tensor) without going
    through cim_read's chunked loop or energy aggregation.

    The cell and the two clamp drivers are now per-call solve arguments,
    so they are packed here straight off the core alongside their snaps.

    Single-block (unchunked) solve: the calibration sweep runs at
    chip-fixture scale where one solve fits in memory.
    """
    # The sampler yields a raw activation batch ``(batch, row)`` with no
    # weight-instance slots, whereas ``CircuitCore1T1R.cim_read`` is fed an
    # ``x`` that already carries a size-1 placeholder in every g-instance
    # leading position (the macro / ``vec_mat_mul`` caller inserts them, as
    # ``xbar_adc.statistic`` does via ``x.unsqueeze(-2)``). Insert the same
    # placeholders here so the activation batch and the fabricated-instance
    # axes occupy DISJOINT leading positions and broadcast into a combined
    # ``(batch, *inst)`` leading, instead of colliding the batch dim against
    # the instance dim. The core's instance leading is ``g.ndim - 2`` dims.
    g_shape = core.cell.rram.g__uS.shape
    inst_rank = len(g_shape) - 2
    *x_batch, x_row = x.shape
    # ``x_code`` matches cim_read's input contract: ``(*batch, *1_inst, row)``.
    x_code = x.reshape(*x_batch, *(1,) * inst_rank, x_row)

    # Mirror CircuitCore1T1R.cim_read's pre-loop setup: keep ``x_code`` for
    # the DAC convert in its natural ``(*leading, row)`` shape (no synthetic
    # WL-fanout dim that has nothing to do with the DAC's own structure). The
    # fanout slot is added back via unsqueeze(-2) after convert so the cell
    # snap's WL drive is ``(*leading, 1, row)`` as the solver expects.
    x_grid = x_code.unsqueeze(-2)
    full_shape = torch.broadcast_shapes(g_shape, x_grid.shape)
    *batch_list, phys_col_num, row_num = full_shape
    leading = tuple(batch_list)
    cell_trailing = (phys_col_num, row_num)
    tia_trailing = (phys_col_num,)
    x_dac_input = x_code.expand(*leading, row_num)
    v_wl_dac = core.wl_dac.convert(x_dac_input)
    v_wl_drive = v_wl_dac.unsqueeze(-2)
    return {
        "bl_segment_r__MOhm": core.bl_segment_r__MOhm,
        "sl_segment_r__MOhm": core.sl_segment_r__MOhm,
        "bl_segment_g__uS": core.bl_segment_g__uS,
        "sl_segment_g__uS": core.sl_segment_g__uS,
        "cell": core.cell,
        "cell_snap": core.cell.snapshot(
            control=v_wl_drive,
            shape=(*leading, *cell_trailing),
            multi_coords=None,
            t_elapsed=0.0,
        ),
        "bl_driver": core.tia,
        "bl_driver_snap": core.tia.snapshot(shape=(*leading, *tia_trailing), multi_coords=None),
        "sl_driver": core.sl_driver,
        "sl_driver_snap": core.sl_driver.snapshot(shape=(*leading, *tia_trailing), multi_coords=None),
    }


def aggregate_xbar_sweep(
    xbar: Offset1T1RXbar,
    *,
    candidate_solvers: list[tuple[int, Solver]],
    n_weight: int,
    n_input_per_weight: int,
    batch_w: int,
    distribution_path: Path | None,
    device: torch.device,
    seed: int,
) -> tuple[list[CandidateRow], WorkloadScale]:
    """Stream the workload, run every candidate per ``(w, x)``, and aggregate.

    For each ``(w, x)`` batch we run all candidate solvers in order and:

      * accumulate per-candidate ``max |residual|`` over the workload;
      * accumulate per-candidate ``max |u_n − u_{n-1}|`` (step delta) over
        the workload, for the five unknown classes ``V_BL``, ``V_SL``,
        ``V_X``, ``V_BL_clamp``, ``V_SL_drive``;
      * track workload-derived signal scales (``max |I_cell|`` for current
        residuals, ``max |V_BL_node|`` for voltage residuals) from the
        most-converged candidate.

    The leading candidate (``i = 0``) has no predecessor — its
    ``step_max__V`` field is ``None``.

    The solvers are stateless (config-only): every candidate is driven
    against the same per-call cell / driver state assembled once from
    ``xbar.core`` for each ``(w, x)``, so only the solver config differs
    between passes. Total cost ≈ ``len(candidates) × workload_solve_time``.

    Args:
        xbar: A built xbar (e.g. via :func:`build_xbar_for_calibration`).
        candidate_solvers: ``[(iter_count, solver)]`` ordered by ascending
            ``iter_count``.
        n_weight: Number of distinct programmed weights to sweep.
        n_input_per_weight: Inputs per weight (per VMM batch).
        batch_w: Weight-axis chunk size for the sampler.
        distribution_path: Optional workload distribution TOML; ``None``
            falls back to uniform sampling.
        device: Torch device.
        seed: RNG seed for reproducibility.

    Returns:
        ``(rows, scale)``: ``rows`` are per-candidate aggregates (one per
        entry in ``candidate_solvers``) and ``scale`` is the workload-derived
        :class:`WorkloadScale`. Feed both to
        :func:`._plateau.pick_with_plateau_and_guard`.
    """
    n_candidates = len(candidate_solvers)
    if n_candidates == 0:
        raise ValueError("candidate_solvers must not be empty")

    distribution = load_distribution(distribution_path, xbar)
    g = make_generator(seed, device)

    step_per_class: list[dict[str, float]] = [dict.fromkeys(_XBAR_UNKNOWN_FIELDS, 0.0) for _ in range(n_candidates)]
    residual_max: list[dict[str, float]] = [dict.fromkeys(_XBAR_RESIDUAL_FIELDS, 0.0) for _ in range(n_candidates)]
    i_cell_typ__uA = 0.0
    v_node_typ__V = 0.0

    for w in sample_w(distribution, xbar, n=n_weight, batch_w=batch_w, device=device, generator=g):
        xbar.program(w)
        for x in sample_x_batches(
            distribution,
            xbar,
            n_total=n_input_per_weight,
            batch_size=n_input_per_weight,
            device=device,
            generator=g,
        ):
            solver_inputs = _solver_inputs_from_core(xbar.core, x)
            prev_fields: dict[str, Tensor] | None = None
            for ci, (_iter_count, solver) in enumerate(candidate_solvers):
                solver_dcop = solver.solve_dc(**solver_inputs, compute_residuals=True)
                assert solver_dcop.residuals is not None
                assert solver_dcop.cell.residuals is not None
                # Per-candidate residual maxima. The per-cell internal-KCL
                # residual lives on the cell DCOP; the wire / clamp residuals
                # on the solver DCOP.
                cell_res = float(solver_dcop.cell.residuals.cell__uA.abs().max().item())
                if cell_res > residual_max[ci][_XBAR_CELL_RESIDUAL_KEY]:
                    residual_max[ci][_XBAR_CELL_RESIDUAL_KEY] = cell_res
                for f in _XBAR_SOLVER_RESIDUAL_FIELDS:
                    val = float(getattr(solver_dcop.residuals, f).abs().max().item())
                    if val > residual_max[ci][f]:
                        residual_max[ci][f] = val
                # Step delta vs the predecessor candidate at the SAME (w, x).
                # The access-node voltage is condensed on the cell DCOP.
                curr_fields = {f: getattr(solver_dcop, f) for f in _XBAR_SOLVER_UNKNOWN_FIELDS}
                curr_fields[_XBAR_CELL_STEP_KEY] = solver_dcop.cell.v_x__V
                if prev_fields is not None:
                    for f in _XBAR_UNKNOWN_FIELDS:
                        val = float((curr_fields[f] - prev_fields[f]).abs().max().item())
                        if val > step_per_class[ci][f]:
                            step_per_class[ci][f] = val
                prev_fields = curr_fields
                # Workload scale: read off the most-converged candidate so the
                # signal-scale denominator is at the true operating point.
                if ci == n_candidates - 1:
                    val_i = float(solver_dcop.cell.i__uA.abs().max().item())
                    if val_i > i_cell_typ__uA:
                        i_cell_typ__uA = val_i
                    val_v = float(solver_dcop.v_bl_node.abs().max().item())
                    if val_v > v_node_typ__V:
                        v_node_typ__V = val_v

    rows: list[CandidateRow] = []
    for ci, (iter_count, _solver) in enumerate(candidate_solvers):
        step_max = max(step_per_class[ci].values()) if ci > 0 else None
        rows.append(
            CandidateRow(
                iter_count=iter_count,
                step_max__V=step_max,
                step_per_class__V=dict(step_per_class[ci]),
                residual_max=dict(residual_max[ci]),
            )
        )
    scale = WorkloadScale(
        i_cell_typ__uA=i_cell_typ__uA,
        v_node_typ__V=v_node_typ__V,
    )
    return rows, scale
