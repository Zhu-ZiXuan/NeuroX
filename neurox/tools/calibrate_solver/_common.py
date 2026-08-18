"""Host-agnostic builder + workload-streaming sweep aggregator for the solver calibration tools.

Registry-driven and scheme-agnostic: the tool TOML names a macro config /
policy file pair — concrete classes selected by `_neurox_class`, scheme
fragments pulled in via `_neurox_use` — built through `CimMacro.from_config`.
The tool binds only to its calibration target, the nested parallel-rail solver
family and the 1T1R cell-family record it consumes, plus the abstract
`CimMacro` surface; it never reaches through a concrete host topology.

Candidate iteration counts are swept in config space: per candidate the macro
config file is loaded as a plain dict, the nested solver table located by a
dotted `solver_section` path is patched, and a fresh macro is built from the
patched dict. The workload rides the macro's public `vec_mat_mul` over row
planes serialized on the sub-phase axis; the calibration data is captured by
the solver / cell probers upstream of the ADC, so the discarded ADC codes never
matter.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from neurox.common.serialize import load_config_dict
from neurox.primitive import T_ROOM__K
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy
from neurox.primitive.xbar.cell import XbarCell1t1rDcop, XbarCell1t1rDetailProber
from neurox.primitive.xbar.solver import (
    ColBlColSlProber,
    ColBlColSlRecord,
    ColBlColSlSolverConfig,
)
from neurox.tools._config import resolve_relative_path
from neurox.tools._plateau import CandidateRow, WorkloadScale

from ._sampling import load_distribution, make_generator, sample_w, sample_x_batches


@dataclass(frozen=True)
class MacroSection:
    """`[macro]` section: which tile to build + where its solver table lives."""

    input_num: int
    """Logical input-vector length passed to the macro constructor."""
    output_num: int
    """Logical output-vector length passed to the macro constructor."""
    config_files: tuple[Path, ...]
    """Macro config TOML paths in descending merge priority (first-wins deep
    merge, e.g. a geometry overlay on top of the scheme default), relative to
    the tool TOML."""
    config_section: str
    """Section name inside the config files holding the `_neurox_class`-tagged
    macro config."""
    policy_file: Path
    """Nonideality policy TOML path — the all-off preset for calibration —
    relative to the tool TOML."""
    policy_section: str
    """Section name inside `policy_file`."""
    solver_section: str
    """Dotted path, relative to `config_section`, locating the nested-solver
    config table inside the macro config, e.g.
    `array_config.solver_config`. The candidate sweep patches `n_outer` /
    `n_inner` there; a path that does not resolve to a nested-solver config
    raises."""

    def __post_init__(self) -> None:
        if self.input_num < 1:
            raise ValueError(f"require: [macro].input_num ({self.input_num}) >= 1")
        if self.output_num < 1:
            raise ValueError(f"require: [macro].output_num ({self.output_num}) >= 1")
        if not self.config_files:
            raise ValueError("require: [macro].config_files non-empty")
        if not self.solver_section:
            raise ValueError("require: [macro].solver_section non-empty")


def resolve_macro_files(section: MacroSection, *, base: Path) -> tuple[list[Path], Path]:
    """Resolve the `[macro]` config / policy file references against `base`."""
    config_paths = [resolve_relative_path(file, base) for file in section.config_files]
    policy_path = resolve_relative_path(section.policy_file, base)
    return config_paths, policy_path


def load_macro_config_dict(config_paths: list[Path], *, config_section: str) -> dict[str, Any]:
    """Load the fully-resolved macro config as a plain dict.

    Each file is parsed, its `_neurox_use` / `_neurox_use_preset` directives
    are expanded, the `config_section` table is plucked, and the per-file
    results are merged. The returned dict is exactly what
    `CimMacroConfig.from_dict` coerces, so patching a value in it and
    rebuilding is equivalent to editing the TOML.
    """
    return load_config_dict(*config_paths, section=config_section)


def _fabricated_macro(
    config: CimMacroConfig,
    policy: CimMacroPolicy,
    *,
    input_num: int,
    output_num: int,
    device: torch.device,
    inst_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> CimMacro[CimMacroConfig, CimMacroPolicy]:
    macro = CimMacro.from_config(
        config=config,
        policy=policy,
        input_num=input_num,
        output_num=output_num,
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
    input_num: int,
    output_num: int,
    device: torch.device,
    inst_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> CimMacro[CimMacroConfig, CimMacroPolicy]:
    """Build the reference tile (geometry query + workload sampling host)."""
    return _fabricated_macro(
        config,
        policy,
        input_num=input_num,
        output_num=output_num,
        device=device,
        inst_shape=inst_shape,
        dtype=dtype,
    )


def resolve_solver_table(root: dict[str, Any], solver_section: str) -> dict[str, Any]:
    """Descend `solver_section` and validate the table is a nested-solver config.

    The dotted path is walked table by table over the already-resolved macro
    config dict, then the resolved table is checked to plausibly be a
    `ColBlColSlSolverConfig`: its `_neurox_class` discriminator must
    name that class when present, and in its absence the `n_outer` / `n_inner`
    keys must be.

    Returns:
        The live sub-dict — a reference into `root`, so patching keys on it
        mutates `root`.

    Raises:
        TypeError: A path segment reaches inside a non-table, the path resolves
            to a non-table, or the discriminator names another class.
        KeyError: A path segment is missing.
        ValueError: The table carries neither the discriminator nor the swept
            keys.
    """
    node: Any = root
    for part in solver_section.split("."):
        if not isinstance(node, dict):
            raise TypeError(
                f"solver_section {solver_section!r}: segment {part!r} sits inside a {type(node).__name__}, not a table"
            )
        if part not in node:
            raise KeyError(
                f"solver_section {solver_section!r}: segment {part!r} not found (available keys: {sorted(node)})"
            )
        node = node[part]
    if not isinstance(node, dict):
        raise TypeError(f"solver_section {solver_section!r} resolves to a {type(node).__name__}, not a table")
    discriminator = node.get("_neurox_class")
    if discriminator is not None:
        if discriminator != ColBlColSlSolverConfig.__name__:
            raise TypeError(
                f"solver_section {solver_section!r} resolves to _neurox_class {discriminator!r}, "
                f"not {ColBlColSlSolverConfig.__name__}"
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
    input_num: int,
    output_num: int,
    device: torch.device,
    inst_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> CimMacro[CimMacroConfig, CimMacroPolicy]:
    """Build a fresh macro with `overrides` applied to its solver table.

    `base_macro_dict` is deep-copied, leaving the caller's shared dict
    untouched; the swept iteration counts are patched onto the dotted-located
    nested-solver table, and the macro config is built from the patched dict
    through `CimMacroConfig.from_dict` before the tile is fabricated.
    """
    patched = copy.deepcopy(base_macro_dict)
    table = resolve_solver_table(patched, solver_section)
    table.update(overrides)
    config = CimMacroConfig.from_dict(patched)
    return _fabricated_macro(
        config,
        policy,
        input_num=input_num,
        output_num=output_num,
        device=device,
        inst_shape=inst_shape,
        dtype=dtype,
    )


def unroll_sub_phase(x: Tensor, *, row_num: int, active_rows: int, inst_rank: int) -> Tensor:
    """Serialize dense WL planes over the sub-phase axis, mirroring the engine.

    The sub-phase axis `P = ceil(row_num / active_rows)` is inserted immediately
    left of the macro's inst-alignment span (`inst_rank` size-1 slots), and rows
    outside a plane's active window are zeroed (WL off), so every conversion
    drives at most `active_rows` live rows — the per-conversion drive context
    the `vec_mat_mul` contract requires. The active count is the calibration
    knob rather than the macro's own `max_active_num`.

    Args:
        x: Dense WL plane tensor with an anonymous leading batch, no inst
            slots.
            Shape: `[..., row_num]`.
        row_num: Macro row count.
        active_rows: Simultaneously active word lines per plane, `1 <=
            active_rows <= row_num`. Any in-range value is legal — the plane
            partition uses a ceil count, so the rows are always fully covered.
        inst_rank: Rank of the macro's fabricated `inst_shape`.

    Returns:
        Masked plane tensor; dtype and device follow `x`.
        Shape: `[..., P, *inst_shape=1, row_num]`.
    """
    n_planes = -(-row_num // active_rows)
    # Static row -> sub-phase ownership; plane p owns rows
    # [p * active_rows, (p + 1) * active_rows).
    # Shape: [row_num] -> [P, row_num]
    plane_of_row = torch.arange(row_num, device=x.device) // active_rows
    mask = plane_of_row == torch.arange(n_planes, device=x.device).unsqueeze(-1)
    # Shape: [P, row_num] -> [P, *inst_shape=1, row_num]
    mask = mask.reshape(n_planes, *(1,) * inst_rank, row_num)
    # Insert the P slot + inst-span size-1 slots just left of the row axis so
    # x broadcasts against the mask.
    # Shape: [..., row_num] -> [..., P=1, *inst_shape=1, row_num]
    x_expanded = x.reshape(*x.shape[:-1], 1, *(1,) * inst_rank, x.shape[-1])
    # Zero-fill = WL off.
    # Shape: [..., P, *inst_shape=1, row_num]
    return torch.where(mask, x_expanded, x.new_zeros(()))


# `v_x__V` is the condensed access-node voltage carried on the cell DCOP
# (`ColBlColSlDcop.cell.v_x__V`); the rest are solver-owned wire / clamp
# unknowns read straight off the DCOP. Every key is the field name it reads,
# so a table row names the tensor a reader can go and look at.
_SOLVER_UNKNOWN_FIELDS: tuple[str, ...] = (
    "v_bl_node__V",
    "v_sl_node__V",
    "v_bl_clamp__V",
    "v_sl_drive__V",
)
"""ColBlColSlDcop-owned fields tracked for the plateau picker's step deltas."""

_CELL_STEP_KEY = "v_x__V"
"""Step-delta key for the cell's condensed access-node voltage `cell.v_x__V`."""

# The wire residuals ride the inner Newton steps of `ColBlColSlProber`'s
# trajectory and the clamp residuals its outer clamp events; `cell__uA` (the
# per-cell internal-KCL residual) rides `XbarCell1t1rDetailProber` and is
# tracked only when the built cell emits it.
_WIRE_RESIDUAL_FIELDS: tuple[str, ...] = ("f_bl_kcl__uA", "f_sl_kcl__uA")
"""Inner-step ColBlColSlRecord fields tracked for the residual safety guard."""

_CLAMP_RESIDUAL_FIELDS: tuple[str, ...] = ("f_bl_clamp__V", "f_sl_clamp__V")
"""Outer-event ColBlColSlRecord fields tracked for the residual safety guard."""

_SOLVER_RESIDUAL_FIELDS: tuple[str, ...] = (*_WIRE_RESIDUAL_FIELDS, *_CLAMP_RESIDUAL_FIELDS)
"""Every solver residual class the guard reports, wire first."""

_CELL_RESIDUAL_KEY = "cell__uA"
"""Residual key for the per-cell internal-KCL mismatch."""


def terminal_records(records: list[ColBlColSlRecord[Any]]) -> list[ColBlColSlRecord[Any]]:
    """Keep the one record each solve closes with, in solve order.

    A solve emits its whole iteration trajectory and then exactly one terminal
    record carrying the converged DCOP, so selecting on that DCOP counts solves
    rather than iterations.
    """
    return [record for record in records if record.dcop is not None]


def solver_step_fields(record: ColBlColSlRecord[Any]) -> dict[str, Tensor]:
    """Extract the step-delta unknown tensors from one terminal solver record.

    The four solver-owned wire / clamp unknowns are always present; the cell
    access-node `v_x__V` is added only when the cell DCOP carries one.

    Raises:
        ValueError: The record carries no DCOP, so it is an iteration record
            rather than the terminal one a step delta compares.
    """
    dcop = record.dcop
    if dcop is None:
        raise ValueError(
            f"step fields need a terminal record carrying a DCOP; "
            f"got an iteration record at (outer={record.outer}, inner={record.inner})"
        )
    fields = {name: getattr(dcop, name) for name in _SOLVER_UNKNOWN_FIELDS}
    cell_dcop = dcop.cell
    if isinstance(cell_dcop, XbarCell1t1rDcop):
        fields[_CELL_STEP_KEY] = cell_dcop.v_x__V
    return fields


def step_delta_over_streams(
    prev: list[ColBlColSlRecord[Any]],
    curr: list[ColBlColSlRecord[Any]],
) -> dict[str, float]:
    """Per-unknown-class `max |u_curr - u_prev|` over two 1:1-aligned streams.

    Both streams are terminal-record streams and must carry the same record
    count, adjacent candidates driven by the identical workload; each aligned
    pair contributes its per-field max-abs difference and the per-class result
    is the max over pairs.

    Raises:
        ValueError: The two streams disagree in record count.
    """
    if len(prev) != len(curr):
        raise ValueError(f"record streams misaligned: prev {len(prev)} vs curr {len(curr)}")
    if not curr:
        return {}
    # Seed every tracked class at zero so an unchanged field reports 0.0 (a
    # genuine plateau) rather than dropping out of the max.
    step: dict[str, float] = dict.fromkeys(solver_step_fields(curr[0]), 0.0)
    for prev_record, curr_record in zip(prev, curr, strict=True):
        prev_fields = solver_step_fields(prev_record)
        curr_fields = solver_step_fields(curr_record)
        for name, curr_val in curr_fields.items():
            delta = float((curr_val - prev_fields[name]).abs().max().item())
            if delta > step[name]:
                step[name] = delta
    return step


def solver_residual_max(records: list[ColBlColSlRecord[Any]]) -> dict[str, float]:
    """Per-class `max |residual|` over the LAST iterate of every solve.

    Convergence is a statement about where a solve stopped, so each solve
    contributes the wire residuals of its final inner Newton step and the
    clamp residuals of its final outer clamp event; the descent behind them is
    the trajectory, not the verdict, and never enters the max. A solve's
    trajectory closes at its terminal record, which is where the pending pair
    is folded in.
    """
    residual: dict[str, float] = dict.fromkeys(_SOLVER_RESIDUAL_FIELDS, 0.0)
    last_inner: ColBlColSlRecord[Any] | None = None
    last_outer: ColBlColSlRecord[Any] | None = None

    def fold(record: ColBlColSlRecord[Any] | None, names: tuple[str, ...]) -> None:
        if record is None:
            return
        for name in names:
            field = getattr(record, name)
            val = float(field.abs().max().item())
            if val > residual[name]:
                residual[name] = val

    for record in records:
        if record.dcop is not None:
            fold(last_inner, _WIRE_RESIDUAL_FIELDS)
            fold(last_outer, _CLAMP_RESIDUAL_FIELDS)
            last_inner = None
            last_outer = None
        elif record.inner == 0:
            last_outer = record
        else:
            last_inner = record
    # A stream cut short of its terminal record still carries a last iterate.
    fold(last_inner, _WIRE_RESIDUAL_FIELDS)
    fold(last_outer, _CLAMP_RESIDUAL_FIELDS)
    return residual


@dataclass(frozen=True)
class _DriveResult:
    """Pooled records of one candidate's drive over the whole workload."""

    solver_records: list[ColBlColSlRecord[Any]]
    """Ordered solver stream: every solve's whole iteration trajectory followed
    by its terminal record, over every plane and chunk."""
    cell_count: int
    """Cell records emitted; `0` for a closed-form cell family that never
    emits."""
    cell_residual__uA: float
    """`max |cell__uA|` over the cell stream, `0.0` when no cell record was
    emitted."""


def _drive_candidate(
    macro: CimMacro[CimMacroConfig, CimMacroPolicy],
    workload: list[tuple[Tensor, Tensor]],
    *,
    input_num: int,
    active_rows: int,
    device: torch.device,
) -> _DriveResult:
    """Program + drive the whole workload; pool the solver / cell records.

    Each `(w, x)` is programmed once, serialized into row planes, and driven
    through the macro's public `vec_mat_mul` under probers capturing the solver
    / cell record links. The returned ADC codes are discarded — the calibration
    data rides the solver prober upstream of ADC conversion, so code clipping
    at a conservative operating point is irrelevant. One drive yields
    `n_planes x n_chunks` solves, each contributing its whole iteration
    trajectory plus one terminal record, array chunking running inside the real
    forward path.
    """
    inst_rank = len(macro.inst_shape)
    solver_records: list[ColBlColSlRecord[Any]] = []
    cell_count = 0
    cell_residual__uA = 0.0
    for w, x in workload:
        macro.program(w.to(device))
        planes = unroll_sub_phase(x.to(device), row_num=input_num, active_rows=active_rows, inst_rank=inst_rank)
        # min_outer=0: the guard reads each solve's LAST iterate, and only the
        # whole trajectory tells the aggregator which iterate that was. The
        # records stay where they were solved: every reduction below and in the
        # plateau / residual passes is device-agnostic, so a hot loop pays no
        # per-drive host transfer and pools nothing in host memory.
        with (
            ColBlColSlProber(min_outer=0) as sp,
            XbarCell1t1rDetailProber() as cp,
            torch.no_grad(),
        ):
            macro.vec_mat_mul(planes, quantization_mode=0, adc_bits=macro.adc_max_bits)
        batch_solver = sp.records
        batch_cell = cp.records
        batch_solve_count = len(terminal_records(list(batch_solver)))
        if batch_cell and len(batch_cell) != batch_solve_count:
            raise ValueError(
                f"cell / solver record counts differ ({len(batch_cell)} vs {batch_solve_count}) — "
                "the cell must emit exactly one internal-KCL residual per solver solve"
            )
        solver_records.extend(batch_solver)
        cell_count += len(batch_cell)
        for cell_record in batch_cell:
            val = float(cell_record.cell__uA.abs().max().item())
            if val > cell_residual__uA:
                cell_residual__uA = val
    return _DriveResult(solver_records=solver_records, cell_count=cell_count, cell_residual__uA=cell_residual__uA)


@dataclass(frozen=True, kw_only=True)
class SolverSweepContext:
    """Stage-invariant sweep context: everything a sweep pass needs beyond its axis.

    It bundles the resolved run inputs that are identical for every sweep
    stage — patched-config source, built sampling host, workload dimensions,
    runtime knobs — while the per-stage axis stays a direct argument of the
    sweep call.
    """

    base_macro_dict: dict[str, Any]
    """Resolved macro config the per-candidate patch is applied to."""
    solver_section: str
    """Dotted path locating the nested-solver table inside that config."""
    policy: CimMacroPolicy
    sampling_host: CimMacro[CimMacroConfig, CimMacroPolicy]
    """Tile the workload is sampled from; it never enters a candidate pass."""
    input_num: int
    output_num: int
    inst_shape: tuple[int, ...]
    dtype: torch.dtype
    active_rows: int
    """Simultaneously active word lines per serialized plane."""
    n_weight: int
    """Distinct programmed weights in the workload."""
    n_input_per_weight: int
    """Input vectors driven per programmed weight."""
    batch_w: int
    """Weight-axis chunk size of the sampler; it equals `inst_shape[0]`."""
    distribution_path: Path | None
    """Synthetic-workload distribution TOML; `None` samples uniformly."""
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

    The `(w, x)` workload is sampled once, seeded, from the context's sampling
    host and reused identically across every candidate, so only the swept
    solver knob differs between passes; calibration presumes an all-off policy,
    which makes the per-candidate macro rebuilds comparable. Per candidate the
    pass patches `{**fixed_overrides, swept_key: value}` onto the macro's
    nested-solver table, builds a fresh tile, drives the workload, and
    accumulates both `max |residual|` over the pooled stream and the step delta
    `max |u_n - u_{n-1}|` against the predecessor candidate on the 1:1-aligned
    record streams. The leading candidate has no predecessor, so it carries no
    step. The workload signal scales are read off the most-converged, last,
    candidate.

    Raises:
        ValueError: `candidates` is empty, aligned record counts disagree, or
            every step delta across the sweep is exactly zero — the signature
            of a `solver_section` that does not point at the solver the macro
            actually uses, so the patched knob had no effect.

    Returns:
        `(rows, scale)` ready for the plateau picker.
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
            distribution,
            host,
            input_num=context.input_num,
            output_num=context.output_num,
            n=context.n_weight,
            batch_w=context.batch_w,
            device=device,
            generator=generator,
        )
        for x in sample_x_batches(
            distribution,
            host,
            input_num=context.input_num,
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

    prev_terminals: list[ColBlColSlRecord[Any]] | None = None
    for ci, value in enumerate(candidates):
        macro = build_candidate_macro(
            context.base_macro_dict,
            solver_section=context.solver_section,
            overrides={**fixed_overrides, swept_key: value},
            policy=context.policy,
            input_num=context.input_num,
            output_num=context.output_num,
            device=device,
            inst_shape=context.inst_shape,
            dtype=context.dtype,
        )
        drive = _drive_candidate(
            macro,
            workload,
            input_num=context.input_num,
            active_rows=context.active_rows,
            device=device,
        )
        records = drive.solver_records
        terminals = terminal_records(records)

        residual = solver_residual_max(records)
        if drive.cell_count:
            if drive.cell_count != len(terminals):
                raise ValueError(f"cell record count ({drive.cell_count}) != solve count ({len(terminals)})")
            residual[_CELL_RESIDUAL_KEY] = drive.cell_residual__uA
        residual_max[ci] = residual

        if prev_terminals is not None:
            step_per_class[ci] = step_delta_over_streams(prev_terminals, terminals)
        prev_terminals = terminals

        if ci == n_candidates - 1:
            for record in terminals:
                dcop = record.dcop
                if dcop is None:
                    continue
                val_i = float(dcop.cell.i__uA.abs().max().item())
                if val_i > i_cell_typ__uA:
                    i_cell_typ__uA = val_i
                val_v = float(dcop.v_bl_node__V.abs().max().item())
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
