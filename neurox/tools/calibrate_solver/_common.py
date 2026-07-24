"""Host-agnostic builder + workload-streaming sweep aggregator for the solver
calibration tools.

Registry-driven and scheme-agnostic: the tool TOML names a macro config /
policy file pair (concrete classes selected by ``_neurox_class``, scheme
fragments pulled in via ``_neurox_use``) built through
:meth:`~neurox.primitive.macro.cim.CimMacro.from_config`, exactly the way
:mod:`neurox.tools.calibrate_adc` builds its tile. The tool binds only to its
calibration target — the nested parallel-rail solver family and the 1T1R
cell-family observation it consumes — plus the abstract
:class:`~neurox.primitive.macro.cim.CimMacro` surface; it never reaches through
a concrete host topology.

Candidate iteration counts are swept in config space: per candidate the macro
config file is loaded as a plain dict (serialization machinery), the nested
solver table located by a dotted ``solver_section`` path is patched, and a
FRESH macro is built from the patched dict. The workload rides the macro's
public ``vec_mat_mul`` (row planes serialized over the sub-phase axis); the
calibration data is captured by the solver / cell probers UPSTREAM of the ADC,
so the discarded ADC codes never matter.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from neurox.common.serialize import load_config_dict
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy
from neurox.primitive.physical_constant import T_ROOM__K
from neurox.primitive.xbar.cell import XbarCell1t1rDcop, XbarCell1t1rDetailProber
from neurox.primitive.xbar.solver import (
    NestedParallelRailSolverConfig,
    SolverObservation,
    SolverProber,
)
from neurox.tools._config import resolve_relative_path
from neurox.tools._plateau import CandidateRow, WorkloadScale

from ._sampling import load_distribution, make_generator, sample_w, sample_x_batches

# ---------------------------------------------------------------------------
# Macro section (TOML schema fragment)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MacroSection:
    """``[macro]`` section: which tile to build + where its solver table lives.

    Attributes:
        config_files: Macro config TOML paths in descending merge priority
            (first-wins deep merge, e.g. a geometry overlay on top of the
            scheme default), relative to the tool TOML.
        config_section: Section name inside the config files holding the
            ``_neurox_class``-tagged macro config.
        policy_file: Nonideality policy TOML path (the all-off preset for
            calibration), relative to the tool TOML.
        policy_section: Section name inside ``policy_file``.
        solver_section: Dotted path, RELATIVE to ``config_section``, locating
            the nested-solver config table inside the macro config
            (e.g. ``"array_config.solver_config"``). The candidate sweep
            patches ``n_outer`` / ``n_inner`` here; a path that does not
            resolve to a nested-solver config raises.
    """

    config_files: tuple[Path, ...]
    config_section: str
    policy_file: Path
    policy_section: str
    solver_section: str

    def __post_init__(self) -> None:
        if not self.config_files:
            raise ValueError("require: [macro].config_files non-empty")
        if not self.solver_section:
            raise ValueError("require: [macro].solver_section non-empty")


def resolve_macro_files(section: MacroSection, *, base: Path) -> tuple[list[Path], Path]:
    """Resolve the ``[macro]`` config / policy file references against ``base``."""
    config_paths: list[Path] = []
    for file in section.config_files:
        resolved = resolve_relative_path(file, base)
        assert resolved is not None
        config_paths.append(resolved)
    policy_path = resolve_relative_path(section.policy_file, base)
    assert policy_path is not None
    return config_paths, policy_path


def load_macro_config_dict(config_paths: list[Path], *, config_section: str) -> dict[str, Any]:
    """Load the fully-resolved macro config as a plain dict.

    Reuses the serialization machinery: each file is parsed, its
    ``_neurox_use`` / ``_neurox_use_preset`` directives are expanded, the
    ``config_section`` table is plucked, and the per-file results are merged.
    The returned dict is exactly what :meth:`CimMacroConfig.from_dict` coerces,
    so patching a value in it and rebuilding is equivalent to editing the TOML.
    """
    return load_config_dict(*config_paths, section=config_section)


def _fabricated_macro(
    config: CimMacroConfig,
    policy: CimMacroPolicy,
    *,
    device: torch.device,
    inst_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> CimMacro:
    """Build, move, eval-freeze, and fabricate a macro through the registry."""
    macro = CimMacro.from_config(
        config=config,
        policy=policy,
        inst_shape=inst_shape,
        dtype=dtype,
        T__K=T_ROOM__K,
    )
    macro = macro.to(device)
    macro.eval()
    macro.fabricate()
    return macro


def build_calibration_macro(
    config: CimMacroConfig,
    policy: CimMacroPolicy,
    *,
    device: torch.device,
    inst_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> CimMacro:
    """Build the reference tile (geometry query + workload sampling host)."""
    return _fabricated_macro(config, policy, device=device, inst_shape=inst_shape, dtype=dtype)


def resolve_solver_table(root: dict[str, Any], solver_section: str) -> dict[str, Any]:
    """Descend ``solver_section`` and validate the table is a nested-solver config.

    Walks the dotted path table by table (plain-dict navigation over the
    already-resolved macro config dict), then checks the resolved table
    plausibly IS a :class:`NestedParallelRailSolverConfig`: its
    ``_neurox_class`` discriminator must name that class when present, and in
    its absence the ``n_outer`` / ``n_inner`` keys must be present.

    Returns the LIVE sub-dict (a reference into ``root``), so a caller
    patching keys on it mutates ``root``.

    Raises:
        ValueError: A path segment is missing / not a table, or the table
            carries neither the discriminator nor the swept keys.
        TypeError: The discriminator names a class other than
            ``NestedParallelRailSolverConfig``.
    """
    node: Any = root
    for part in solver_section.split("."):
        if not isinstance(node, dict) or part not in node:
            available = sorted(node) if isinstance(node, dict) else "<not a table>"
            raise ValueError(
                f"solver_section {solver_section!r}: segment {part!r} not found (available keys: {available})"
            )
        node = node[part]
    if not isinstance(node, dict):
        raise ValueError(f"solver_section {solver_section!r} resolves to a {type(node).__name__}, not a table")
    discriminator = node.get("_neurox_class")
    if discriminator is not None:
        if discriminator != NestedParallelRailSolverConfig.__name__:
            raise TypeError(
                f"solver_section {solver_section!r} resolves to _neurox_class {discriminator!r}, "
                f"not {NestedParallelRailSolverConfig.__name__}"
            )
    elif not all(key in node for key in ("n_outer", "n_inner")):
        raise ValueError(
            f"solver_section {solver_section!r} table declares no _neurox_class and lacks the "
            f"'n_outer' / 'n_inner' keys — it does not look like a nested-solver config"
        )
    return node


def build_candidate_macro(
    base_macro_dict: dict[str, Any],
    *,
    solver_section: str,
    overrides: dict[str, int],
    policy: CimMacroPolicy,
    device: torch.device,
    inst_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> CimMacro:
    """Build a fresh macro with ``overrides`` applied to its solver table.

    Deep-copies ``base_macro_dict`` (leaving the caller's shared dict
    untouched), patches the swept iteration counts on the dotted-located
    nested-solver table, then builds the macro config from the patched dict
    with the SAME builder the serialization machinery provides
    (:meth:`CimMacroConfig.from_dict`) and fabricates the tile.
    """
    patched = copy.deepcopy(base_macro_dict)
    table = resolve_solver_table(patched, solver_section)
    table.update(overrides)
    config = CimMacroConfig.from_dict(patched)
    return _fabricated_macro(config, policy, device=device, inst_shape=inst_shape, dtype=dtype)


# ---------------------------------------------------------------------------
# Row-block serialized drive (engine sub-phase mirror)
# ---------------------------------------------------------------------------


def unroll_sub_phase(x: Tensor, *, row_num: int, active_rows: int, inst_rank: int) -> Tensor:
    """Serialize dense WL planes over the sub-phase axis (engine-layer mirror).

    Local, explicitly-labelled mirror of the runtime engine serialization: the
    sub-phase axis ``P = ceil(row_num / active_rows)`` is inserted immediately
    LEFT of the macro's inst-alignment span (``inst_rank`` size-1 slots), and
    rows outside a plane's active window are zeroed (WL off), so every
    conversion drives at most ``active_rows`` live rows — the per-conversion
    drive context the ``vec_mat_mul`` contract requires. Parameterized by
    ``active_rows`` (the calibration knob) rather than the macro's
    ``max_active_rows``.

    Args:
        x: Dense WL plane tensor with trailing ``[row_num]`` and anonymous
            leading batch (no inst slots).
        row_num: Macro row count.
        active_rows: Simultaneously active word lines per plane; ``1 <=
            active_rows <= row_num``. Any in-range value is legal — the plane
            partition uses a ceil count so the rows are always fully covered.
        inst_rank: Rank of the macro's fabricated ``inst_shape``.

    Returns:
        Masked plane tensor trailing ``[P, *(1,) * inst_rank, row_num]``;
        dtype and device follow ``x``.
    """
    n_planes = -(-row_num // active_rows)
    # Static row -> sub-phase ownership; plane p owns rows
    # [p * active_rows, (p + 1) * active_rows). Shape: [P, row_num]
    plane_of_row = torch.arange(row_num, device=x.device) // active_rows
    mask = plane_of_row == torch.arange(n_planes, device=x.device).unsqueeze(-1)
    # Shape: [P, row_num] -> [P, *(1,) * inst_rank, row_num]
    mask = mask.reshape(n_planes, *(1,) * inst_rank, row_num)
    # Insert the P slot + inst-span size-1 slots just left of the row axis so
    # x broadcasts against the mask. Shape: [*batch, row] ->
    # [*batch, 1, *(1,) * inst_rank, row].
    x_expanded = x.reshape(*x.shape[:-1], 1, *(1,) * inst_rank, x.shape[-1])
    # Shape: [*batch, P, *(1,) * inst_rank, row]; zero-fill = WL off.
    return torch.where(mask, x_expanded, x.new_zeros(()))


# ---------------------------------------------------------------------------
# Step-delta / residual classes (plateau picker)
# ---------------------------------------------------------------------------

# ``v_x`` is the condensed access-node voltage carried on the cell DCOP
# (``SolverDcop.cell.v_x__V``); the rest are solver-owned wire / clamp
# unknowns read straight off the DCOP.
_SOLVER_UNKNOWN_FIELDS: tuple[str, ...] = (
    "v_bl_node",
    "v_sl_node",
    "v_bl_clamp",
    "v_sl_drive",
)
"""SolverDcop-owned fields tracked for the plateau picker's step deltas."""

_CELL_STEP_KEY = "v_x"
"""Step-delta key for the cell's condensed access-node voltage (``cell.v_x__V``)."""

# The wire / clamp residuals ride :class:`SolverProber`; ``cell__uA``
# (the per-cell internal-KCL residual) rides :class:`XbarCell1t1rDetailProber`
# and is tracked only when the built cell emits it.
_SOLVER_RESIDUAL_FIELDS: tuple[str, ...] = (
    "wire_bl__uA",
    "wire_sl__uA",
    "clamp_bl__V",
    "clamp_sl__V",
)
"""SolverObservation fields tracked for the residual safety guard."""

_CELL_RESIDUAL_KEY = "cell__uA"
"""Residual key for the per-cell internal-KCL mismatch (``XbarCell1t1rDetailProber``)."""


def solver_step_fields(observation: SolverObservation[Any]) -> dict[str, Tensor]:
    """Extract the step-delta unknown tensors from one solver observation.

    The four solver-owned wire / clamp unknowns are always present; the cell
    access-node ``v_x__V`` is added only when the cell DCOP is an
    :class:`XbarCell1t1rDcop` (the down-drill the calibration target sanctions).
    """
    fields = {name: getattr(observation.dcop, name) for name in _SOLVER_UNKNOWN_FIELDS}
    cell_dcop = observation.dcop.cell
    if isinstance(cell_dcop, XbarCell1t1rDcop):
        fields[_CELL_STEP_KEY] = cell_dcop.v_x__V
    return fields


def step_delta_over_streams(
    prev: list[SolverObservation[Any]],
    curr: list[SolverObservation[Any]],
) -> dict[str, float]:
    """Per-unknown-class ``max |u_curr - u_prev|`` over two 1:1-aligned streams.

    Both streams must carry the same record count (adjacent candidates driven
    by the identical workload); each aligned pair contributes its per-field
    max-abs difference and the per-class result is the max over pairs.
    """
    if len(prev) != len(curr):
        raise ValueError(f"record streams misaligned: prev {len(prev)} vs curr {len(curr)}")
    if not curr:
        return {}
    # Seed every tracked class at zero so an unchanged field reports 0.0 (a
    # genuine plateau) rather than dropping out of the max.
    step: dict[str, float] = dict.fromkeys(solver_step_fields(curr[0]), 0.0)
    for prev_obs, curr_obs in zip(prev, curr, strict=True):
        prev_fields = solver_step_fields(prev_obs)
        curr_fields = solver_step_fields(curr_obs)
        for name, curr_val in curr_fields.items():
            delta = float((curr_val - prev_fields[name]).abs().max().item())
            if delta > step[name]:
                step[name] = delta
    return step


def solver_residual_max(records: list[SolverObservation[Any]]) -> dict[str, float]:
    """Per-class ``max |residual|`` over a solver observation stream."""
    residual: dict[str, float] = dict.fromkeys(_SOLVER_RESIDUAL_FIELDS, 0.0)
    for observation in records:
        for name in _SOLVER_RESIDUAL_FIELDS:
            val = float(getattr(observation, name).abs().max().item())
            if val > residual[name]:
                residual[name] = val
    return residual


# ---------------------------------------------------------------------------
# Candidate sweep aggregator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DriveResult:
    """Pooled records of one candidate's drive over the whole workload.

    Attributes:
        solver_records: Ordered :class:`SolverProber` stream (one per
            plane per chunk).
        cell_count: Total :class:`XbarCell1t1rDetailProber` records; ``0`` for
            a closed-form cell family that never emits.
        cell_residual__uA: ``max |cell__uA|`` over the cell stream (``0.0``
            when no cell records were emitted).
    """

    solver_records: list[SolverObservation[Any]]
    cell_count: int
    cell_residual__uA: float


def _drive_candidate(
    macro: CimMacro,
    workload: list[tuple[Tensor, Tensor]],
    *,
    active_rows: int,
    device: torch.device,
) -> _DriveResult:
    """Program + drive the whole workload; pool the solver / cell records.

    Each ``(w, x)`` is programmed once, serialized into row planes, and driven
    through the macro's public ``vec_mat_mul`` under probers capturing the
    solver / cell observation links. The returned ADC codes are DISCARDED —
    the calibration data rides :class:`SolverProber` upstream of ADC
    conversion, so code clipping at a conservative operating point is
    irrelevant. One drive yields ``n_planes x n_chunks`` solver records (array
    chunking runs inside the real forward path).
    """
    inst_rank = len(macro.inst_shape)
    solver_records: list[SolverObservation[Any]] = []
    cell_count = 0
    cell_residual__uA = 0.0
    for w, x in workload:
        macro.program(w.to(device))
        planes = unroll_sub_phase(x.to(device), row_num=macro.row_num, active_rows=active_rows, inst_rank=inst_rank)
        with SolverProber() as sp, XbarCell1t1rDetailProber() as cp, torch.no_grad():
            macro.vec_mat_mul(planes, adc_mode=0, adc_bits=macro.adc_max_bits)
        batch_solver = sp.records
        batch_cell = cp.records
        if batch_cell and len(batch_cell) != len(batch_solver):
            raise ValueError(
                f"cell / solver record counts differ ({len(batch_cell)} vs {len(batch_solver)}) — "
                "the cell must emit exactly one internal-KCL residual per solver solve"
            )
        solver_records.extend(batch_solver)
        cell_count += len(batch_cell)
        for cell_observation in batch_cell:
            val = float(cell_observation.cell__uA.abs().max().item())
            if val > cell_residual__uA:
                cell_residual__uA = val
    return _DriveResult(solver_records=solver_records, cell_count=cell_count, cell_residual__uA=cell_residual__uA)


@dataclass(frozen=True, kw_only=True)
class SolverSweepContext:
    """Stage-invariant sweep context: everything a sweep pass needs beyond its axis.

    Bundles the resolved run inputs (patched-config source, built sampling
    host, workload dimensions, runtime knobs) that are identical for every
    sweep stage; the per-stage axis (``swept_key`` / ``candidates`` /
    ``fixed_overrides``) stays a direct argument of
    :func:`aggregate_solver_sweep`.
    """

    base_macro_dict: dict[str, Any]
    solver_section: str
    policy: CimMacroPolicy
    sampling_host: CimMacro
    inst_shape: tuple[int, ...]
    dtype: torch.dtype
    active_rows: int
    n_weight: int
    n_input_per_weight: int
    batch_w: int
    distribution_path: Path | None
    device: torch.device
    seed: int


def aggregate_solver_sweep(
    *,
    swept_key: str,
    candidates: list[int],
    fixed_overrides: dict[str, int],
    context: SolverSweepContext,
) -> tuple[list[CandidateRow], WorkloadScale]:
    """Sweep one iteration-count axis, rebuilding a fresh macro per candidate.

    The ``(w, x)`` workload is sampled ONCE (seeded) from the context's sampling host and
    reused identically across every candidate, so only the swept solver knob
    differs between passes; calibration presumes an all-off policy, which makes
    the per-candidate macro rebuilds comparable. For each candidate we

      * patch ``{**fixed_overrides, swept_key: value}`` onto the macro's
        nested-solver table and build a fresh tile;
      * drive the workload and pool the solver / cell observation records;
      * accumulate per-candidate ``max |residual|`` over the pooled stream;
      * accumulate per-candidate ``max |u_n - u_{n-1}|`` (step delta) against
        the predecessor candidate on the 1:1-aligned record streams.

    The leading candidate has no predecessor — its ``step_max__V`` is ``None``.
    Workload signal scales (``max |I_cell|``, ``max |V_BL_node|``) are read off
    the most-converged (last) candidate.

    Raises:
        ValueError: ``candidates`` is empty, aligned record counts disagree, or
            EVERY step delta across the sweep is exactly zero — the signature
            of a ``solver_section`` that does not point at the solver the macro
            actually uses (the patched knob had no effect).

    Returns:
        ``(rows, scale)`` ready for
        :func:`neurox.tools._plateau.pick_with_plateau_and_guard`.
    """
    n_candidates = len(candidates)
    if n_candidates == 0:
        raise ValueError("candidates must not be empty")

    host = context.sampling_host
    device = context.device
    distribution = load_distribution(context.distribution_path, host)
    generator = make_generator(context.seed, device)
    workload: list[tuple[Tensor, Tensor]] = [
        (w, x)
        for w in sample_w(
            distribution, host, n=context.n_weight, batch_w=context.batch_w, device=device, generator=generator
        )
        for x in sample_x_batches(
            distribution,
            host,
            n_total=context.n_input_per_weight,
            batch_size=context.n_input_per_weight,
            device=device,
            generator=generator,
        )
    ]

    step_per_class: list[dict[str, float]] = [{} for _ in range(n_candidates)]
    residual_max: list[dict[str, float]] = [{} for _ in range(n_candidates)]
    i_cell_typ__uA = 0.0
    v_node_typ__V = 0.0

    prev_records: list[SolverObservation[Any]] | None = None
    for ci, value in enumerate(candidates):
        macro = build_candidate_macro(
            context.base_macro_dict,
            solver_section=context.solver_section,
            overrides={**fixed_overrides, swept_key: value},
            policy=context.policy,
            device=device,
            inst_shape=context.inst_shape,
            dtype=context.dtype,
        )
        drive = _drive_candidate(macro, workload, active_rows=context.active_rows, device=device)
        records = drive.solver_records

        residual = solver_residual_max(records)
        if drive.cell_count:
            if drive.cell_count != len(records):
                raise ValueError(f"cell record count ({drive.cell_count}) != solver record count ({len(records)})")
            residual[_CELL_RESIDUAL_KEY] = drive.cell_residual__uA
        residual_max[ci] = residual

        if prev_records is not None:
            step_per_class[ci] = step_delta_over_streams(prev_records, records)
        prev_records = records

        if ci == n_candidates - 1:
            for observation in records:
                val_i = float(observation.dcop.cell.i__uA.abs().max().item())
                if val_i > i_cell_typ__uA:
                    i_cell_typ__uA = val_i
                val_v = float(observation.dcop.v_bl_node.abs().max().item())
                if val_v > v_node_typ__V:
                    v_node_typ__V = val_v

    if n_candidates >= 2 and all(
        not step_per_class[ci] or max(step_per_class[ci].values()) == 0.0 for ci in range(1, n_candidates)
    ):
        raise ValueError(
            f"every step delta over the '{swept_key}' sweep is exactly zero — the patched knob had no "
            f"effect, so solver_section {context.solver_section!r} likely does not point at the solver the macro uses"
        )

    rows: list[CandidateRow] = []
    for ci, value in enumerate(candidates):
        step_max = max(step_per_class[ci].values()) if ci > 0 and step_per_class[ci] else None
        rows.append(
            CandidateRow(
                iter_count=value,
                step_max__V=step_max,
                step_per_class__V=dict(step_per_class[ci]),
                residual_max=dict(residual_max[ci]),
            )
        )
    scale = WorkloadScale(i_cell_typ__uA=i_cell_typ__uA, v_node_typ__V=v_node_typ__V)
    return rows, scale
