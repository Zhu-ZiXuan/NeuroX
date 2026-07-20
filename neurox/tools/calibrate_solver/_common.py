"""Shared xbar builder + workload-streaming sweep aggregator for the solver
calibration tools.

Registry-driven and scheme-agnostic: the tool TOML carries an abstract-typed
macro config + policy pair (concrete classes selected by ``_neurox_class``,
scheme fragments pulled in via ``_neurox_use``), and the built macro must
expose the structural surface the solver drive reads — ``core`` (an
:class:`~neurox.primitive.xbar.array.XbarArray1t1r`), ``wl_dac``,
``clamp_ref``, ``bl_clamp``, and ``sl_driver``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

# Registers the scheme classes so CimMacro.from_config / the config
# `_neurox_class` discriminators can resolve works-defined subclasses.
import neurox.works  # noqa: F401
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy
from neurox.primitive.xbar.array import XbarArray1t1r, XbarArray1t1rConfig, XbarArray1t1rPolicy
from neurox.primitive.xbar.solver import Solver, SolverConfig
from neurox.tools._plateau import CandidateRow, WorkloadScale

from ._sampling import load_distribution, make_generator, sample_w, sample_x_batches


def _calibration_core(xbar: CimMacro) -> XbarArray1t1r:
    """The macro's owned pure array, type-checked for the solver drive."""
    core = getattr(xbar, "core", None)
    if not isinstance(core, XbarArray1t1r):
        raise TypeError(
            f"solver calibration requires a macro exposing an XbarArray1t1r 'core'; "
            f"{type(xbar).__name__} has {type(core).__name__}"
        )
    return core


def build_xbar_for_calibration(
    config: CimMacroConfig,
    policy: CimMacroPolicy,
    *,
    device: torch.device,
    inst_shape: tuple[int, ...],
    dtype: torch.dtype,
    solver_config: SolverConfig,
    solve_chunk_size: int = 0,
) -> CimMacro:
    """Build a noise-off xbar with the supplied solver config.

    Substitutes ``solver_config`` on the macro's array config and the
    run-specific ``solve_chunk_size`` chunking knob on its array policy;
    everything else comes verbatim from ``config`` / ``policy`` (a noise-off
    policy preset for calibration).

    Args:
        config: Concrete macro config; must carry an
            :class:`XbarArray1t1rConfig` ``array_config`` field.
        policy: Matching macro policy; must carry an
            :class:`XbarArray1t1rPolicy` ``array`` field.
        device: Target torch device.
        inst_shape: Per-instance shape for the xbar fabricator.
        dtype: Tensor dtype.
        solver_config: Concrete :class:`SolverConfig` subclass selecting
            which solver implementation to dispatch via the registry.
        solve_chunk_size: Array chunking knob override; ``0`` disables
            chunking (single-block solve).

    Returns:
        A fabricated macro resolved through the :class:`CimMacro` registry.
    """
    array_config = getattr(config, "array_config", None)
    if not isinstance(array_config, XbarArray1t1rConfig):
        raise TypeError(
            f"solver calibration requires a macro config with an XbarArray1t1rConfig "
            f"'array_config' field; {type(config).__name__} has {type(array_config).__name__}"
        )
    array_policy = getattr(policy, "array", None)
    if not isinstance(array_policy, XbarArray1t1rPolicy):
        raise TypeError(
            f"solver calibration requires a macro policy with an XbarArray1t1rPolicy "
            f"'array' field; {type(policy).__name__} has {type(array_policy).__name__}"
        )

    new_config = replace(config, array_config=replace(array_config, solver_config=solver_config))
    new_policy = replace(policy, array=replace(array_policy, solve_chunk_size=solve_chunk_size))

    xbar = CimMacro.from_config(
        config=new_config,
        policy=new_policy,
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
# access-node voltage carried on the cell DCOP (``SolverDcop.cell.v_x__V``);
# the rest are solver-owned wire / clamp unknowns read straight off the DCOP.
_XBAR_SOLVER_UNKNOWN_FIELDS: tuple[str, ...] = (
    "v_bl_node",
    "v_sl_node",
    "v_bl_clamp",
    "v_sl_drive",
)
"""SolverDcop-owned fields tracked for the plateau picker's step deltas."""

_XBAR_CELL_STEP_KEY = "v_x"
"""Step-delta key for the cell's condensed access-node voltage (``cell.v_x__V``)."""

_XBAR_UNKNOWN_FIELDS: tuple[str, ...] = (*_XBAR_SOLVER_UNKNOWN_FIELDS, _XBAR_CELL_STEP_KEY)
"""All step-delta classes (solver wire / clamp unknowns + the cell access node)."""

# Residual classes (safety guard). ``cell__uA`` is the per-cell internal-KCL
# residual on the cell DCOP (``SolverDcop.cell.residuals.cell__uA``); the
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


def _solver_inputs_from_xbar(xbar: CimMacro, x: Tensor) -> dict[str, Any]:
    """Assemble :meth:`Solver.solve_dc` kwargs from an already-fabricated xbar.

    Drives the solver under realistic chip context (cell conductances
    programmed via workload sampling, real wire R/G, real DAC drive) while
    bypassing the macro's ``vec_mat_mul`` so raw :class:`SolverDcop`
    residuals are inspectable. Single-block (unchunked) solve. Reads the
    macro's structural surface (``core`` / ``wl_dac`` / ``clamp_ref`` /
    ``bl_clamp`` / ``sl_driver``).
    """
    core = _calibration_core(xbar)
    # The sampler yields a raw activation batch ``(batch, row)`` with no
    # weight-instance slots. Insert size-1 placeholders in every g-instance
    # leading position so the activation batch and the fabricated-instance
    # axes occupy DISJOINT leading positions and broadcast into a combined
    # ``(batch, *inst)`` leading, instead of colliding batch against instance.
    g_shape = core.weight_grid_shape
    inst_rank = len(g_shape) - 2
    *x_batch, x_row = x.shape
    # ``x_code`` matches the forward input contract: ``(*batch, *1_inst, row)``.
    x_code = x.reshape(*x_batch, *(1,) * inst_rank, x_row)

    # Keep ``x_code`` for the DAC convert in its natural ``(*leading, row)``
    # shape; the fanout slot is added back via unsqueeze(-2) after convert so
    # the cell snap's WL drive is ``(*leading, 1, row)`` as the solver expects.
    x_grid = x_code.unsqueeze(-2)
    full_shape = torch.broadcast_shapes(g_shape, x_grid.shape)
    *batch_list, phys_col_num, row_num = full_shape
    leading = tuple(batch_list)
    cell_trailing = (phys_col_num, row_num)
    line_trailing = (phys_col_num,)
    x_dac_input = x_code.expand(*leading, row_num)
    v_wl_dac = xbar.wl_dac.convert(x_dac_input)
    v_wl_drive = v_wl_dac.unsqueeze(-2)

    # Mirror the forward boundary-clamp reference injection: snapshot the
    # xbar-owned clamp reference once and thread its 0-d scalar taps
    # (tap 0 = BL clamp, tap 1 = SL drive) into each driver snapshot.
    clamp_taps = xbar.clamp_ref.v_ref__V(xbar.clamp_ref.snapshot())
    bl_v_ref = clamp_taps[0]
    sl_v_ref = clamp_taps[1]
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
        "bl_driver": xbar.bl_clamp,
        "bl_driver_snap": xbar.bl_clamp.snapshot(
            v_ref__V=bl_v_ref, shape=(*leading, *line_trailing), multi_coords=None
        ),
        "sl_driver": xbar.sl_driver,
        "sl_driver_snap": xbar.sl_driver.snapshot(
            v_ref__V=sl_v_ref, shape=(*leading, *line_trailing), multi_coords=None
        ),
    }


def aggregate_xbar_sweep(
    xbar: CimMacro,
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
        :func:`neurox.tools._plateau.pick_with_plateau_and_guard`.
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
            solver_inputs = _solver_inputs_from_xbar(xbar, x)
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
