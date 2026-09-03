"""Host-agnostic builder + workload-streaming sweep aggregator for the solver calibration tools.

Registry-driven and scheme-agnostic: the tool TOML names a macro config /
policy file pair — concrete classes selected by `_neurox_class`, scheme
fragments pulled in via `_neurox_use` — built through `CimMacro.from_config`.
The tool binds only to its calibration target, the nested parallel-rail solver
family and its cell-residual channel, plus the abstract `CimMacro` surface; it
never reaches through a concrete host topology.

Candidate iteration counts are swept in config space: per candidate the macro
config file is loaded as a plain dict, the nested solver table located by a
dotted `solver_section` path is patched, and a fresh macro is built from the
patched dict. The workload rides the macro's public `vec_mat_mul` over row
planes serialized on the sub-phase axis; the calibration data is captured by
the solver and cell probers upstream of the ADC, so the discarded ADC codes
never matter.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from neurox.common import ValidateMixin
from neurox.common.serialize import load_config_dict
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy
from neurox.primitive.physics import T_ROOM__K
from neurox.primitive.xbar.cell import (
    XbarCell1t1rDcop,
    XbarCell1t1rDetailProber,
    XbarCell1t1rDetailRecord,
)
from neurox.primitive.xbar.solver import (
    ColBlColSlProber,
    ColBlColSlRecord,
    ColBlColSlSolverConfig,
)
from neurox.tools._config import resolve_relative_path
from neurox.tools._plateau import CandidateRow, WorkloadScale
from neurox.tools._sampling import load_distribution, make_generator, sample_w, sample_x_batches


@dataclass(frozen=True)
class MacroSection(ValidateMixin):
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
        self._require_non_empty(self.config_files, "[macro].config_files")
        self._require_non_empty(self.solver_section, "[macro].solver_section")


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
    device: torch.device,
    inst_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> CimMacro[CimMacroConfig, CimMacroPolicy]:
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
) -> CimMacro[CimMacroConfig, CimMacroPolicy]:
    """Build the reference macro for geometry queries and workload sampling."""
    return _fabricated_macro(
        config,
        policy,
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
    device: torch.device,
    inst_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> CimMacro[CimMacroConfig, CimMacroPolicy]:
    """Build a fresh macro with `overrides` applied to its solver table.

    `base_macro_dict` is deep-copied, leaving the caller's shared dict
    untouched; the swept iteration counts are patched onto the dotted-located
    nested-solver table, and the macro config is built from the patched dict
    through `CimMacroConfig.from_dict` before the macro is fabricated.
    """
    patched = copy.deepcopy(base_macro_dict)
    table = resolve_solver_table(patched, solver_section)
    table.update(overrides)
    config = CimMacroConfig.from_dict(patched)
    return _fabricated_macro(
        config,
        policy,
        device=device,
        inst_shape=inst_shape,
        dtype=dtype,
    )


def unroll_sub_phase(
    x: Tensor,
    *,
    input_num: int,
    active_inputs: int,
    inst_shape: tuple[int, ...],
) -> Tensor:
    """Serialize dense input planes over the sub-phase axis, mirroring the engine.

    The sub-phase axis `P = ceil(input_num / active_inputs)` is inserted immediately
    left of the macro's inst-alignment span (`inst_rank` size-1 slots), and inputs
    outside a plane's active window are zeroed, so every conversion drives at most
    `active_inputs` live inputs — the per-conversion drive context
    the `vec_mat_mul` contract requires. The active count is the calibration
    knob rather than the macro's own `max_active_num`.

    Args:
        x: Dense logical-input tensor with an anonymous leading batch, no inst
            slots.
            Shape: `[..., input]`.
        input_num: Macro logical input count.
        active_inputs: Simultaneously active logical inputs per plane, `1 <=
            active_inputs <= input_num`. Any in-range value is legal — the plane
            partition uses a ceil count, so the inputs are always fully covered.
        inst_shape: Fabricated instance shape of the called macro.

    Returns:
        Masked plane tensor; dtype and device follow `x`.
        Shape: `[..., P, *inst_shape, input]`.
    """
    inst_rank = len(inst_shape)
    n_planes = -(-input_num // active_inputs)
    # Static input -> sub-phase ownership; plane p owns inputs
    # [p * active_inputs, (p + 1) * active_inputs).
    # Shape: [input] -> [P, input]
    plane_of_input = torch.arange(input_num, device=x.device) // active_inputs
    mask = plane_of_input == torch.arange(n_planes, device=x.device).unsqueeze(-1)
    # Shape: [P, input] -> [P, *inst_shape=1, input]
    mask_shape = (mask.shape[0], *(1,) * inst_rank, mask.shape[-1])
    mask = mask.view(mask_shape)
    # Insert the P slot + inst-span size-1 slots just left of the input axis so
    # x broadcasts against the mask.
    # Shape: [..., input] -> [..., P=1, *inst_shape=1, input]
    expanded_shape = (*x.shape[:-1], *(1,) * (inst_rank + 1), x.shape[-1])
    x_expanded = x.view(expanded_shape)
    # Shape: [..., P, *inst_shape=1, input]
    planes = torch.where(mask, x_expanded, x.new_zeros(()))
    # Shape: [..., P, *inst_shape=1, input] -> [..., P, *inst_shape, input]
    return planes.expand(*planes.shape[: -(inst_rank + 1)], *inst_shape, input_num)


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
# trajectory and the clamp residuals its outer clamp events.
_WIRE_RESIDUAL_FIELDS: tuple[str, ...] = ("f_bl_kcl__uA", "f_sl_kcl__uA")
"""Inner-step ColBlColSlRecord fields tracked for the residual safety guard."""

_CLAMP_RESIDUAL_FIELDS: tuple[str, ...] = ("f_bl_clamp__V", "f_sl_clamp__V")
"""Outer-event ColBlColSlRecord fields tracked for the residual safety guard."""

_SOLVER_RESIDUAL_FIELDS: tuple[str, ...] = (*_WIRE_RESIDUAL_FIELDS, *_CLAMP_RESIDUAL_FIELDS)
"""Every solver residual class the guard reports, wire first."""


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
    """Per-class `max |residual|` from every solve's final recorded updates.

    Each residual belongs to the pre-step iterate that drove an update, so the
    final inner and outer records precede their respective terminal updates by
    one step. They form the conservative residual guard; plateau selection
    compares the returned terminal states themselves. Earlier trajectory
    residuals never enter the max. A solve's terminal record closes its
    trajectory and triggers folding of the pending pair.
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
    # A stream cut short of its terminal record still carries final updates.
    fold(last_inner, _WIRE_RESIDUAL_FIELDS)
    fold(last_outer, _CLAMP_RESIDUAL_FIELDS)
    return residual


@dataclass(frozen=True)
class SolveResidualTrajectory:
    """Complete residual streams from one array solve."""

    solver_records: tuple[ColBlColSlRecord[Any], ...]
    """Outer, inner, and terminal solver records in emission order."""
    cell_records: tuple[XbarCell1t1rDetailRecord, ...]
    """Cell residuals from the seed, every iterative evaluation, and the terminal evaluation."""


def build_residual_trajectories(
    solver_records: tuple[ColBlColSlRecord[Any], ...],
    cell_records: tuple[XbarCell1t1rDetailRecord, ...],
    *,
    n_outer: int,
    n_inner: int,
) -> tuple[SolveResidualTrajectory, ...]:
    """Split the two recorder streams into aligned per-solve trajectories."""
    solver_trajectories: list[tuple[ColBlColSlRecord[Any], ...]] = []
    pending: list[ColBlColSlRecord[Any]] = []
    for record in solver_records:
        pending.append(record)
        if record.dcop is not None:
            solver_trajectories.append(tuple(pending))
            pending = []
    if pending:
        raise ValueError("solver record stream ended without a terminal record")

    if not cell_records:
        return tuple(SolveResidualTrajectory(records, ()) for records in solver_trajectories)

    cell_records_per_solve = 2 + n_outer * n_inner
    expected = len(solver_trajectories) * cell_records_per_solve
    if len(cell_records) != expected:
        raise ValueError(
            f"cell record stream has {len(cell_records)} records for {len(solver_trajectories)} solves; "
            f"expected {cell_records_per_solve} per solve ({expected} total)"
        )
    return tuple(
        SolveResidualTrajectory(
            solver_records=records,
            cell_records=cell_records[index : index + cell_records_per_solve],
        )
        for records, index in zip(
            solver_trajectories,
            range(0, len(cell_records), cell_records_per_solve),
            strict=True,
        )
    )


def current_residual_max(trajectories: tuple[SolveResidualTrajectory, ...]) -> dict[str, float]:
    """Maximum residual at the stopping point of every solve."""
    solver_records = [record for trajectory in trajectories for record in trajectory.solver_records]
    residual = solver_residual_max(solver_records)
    terminal_cell_records = [trajectory.cell_records[-1] for trajectory in trajectories if trajectory.cell_records]
    if terminal_cell_records:
        residual["cell__uA"] = max(float(record.cell__uA.abs().max().item()) for record in terminal_cell_records)
    return residual


@dataclass(frozen=True)
class SolverResidualPoint:
    outer: int
    inner: int
    f_bl_clamp__V: float | None
    f_sl_clamp__V: float | None
    f_bl_kcl__uA: float | None
    f_sl_kcl__uA: float | None


@dataclass(frozen=True)
class CellResidualPoint:
    stage: str
    outer: int | None
    inner: int | None
    cell__uA: float


def aggregate_residual_trajectory(
    trajectories: tuple[SolveResidualTrajectory, ...],
    *,
    n_outer: int,
    n_inner: int,
) -> tuple[tuple[SolverResidualPoint, ...], tuple[CellResidualPoint, ...]]:
    """Reduce aligned workload trajectories to per-iteration max-abs series."""
    solver_coordinates = [(outer, inner) for outer in range(n_outer) for inner in range(n_inner + 1)]
    clamp_bl = [0.0] * n_outer
    clamp_sl = [0.0] * n_outer
    wire_bl = [0.0] * (n_outer * n_inner)
    wire_sl = [0.0] * (n_outer * n_inner)

    for trajectory in trajectories:
        iteration_records = trajectory.solver_records[:-1]
        if [(record.outer, record.inner) for record in iteration_records] != solver_coordinates:
            raise ValueError("solver trajectory coordinates do not match n_outer / n_inner")
        for record in iteration_records:
            if record.inner == 0:
                f_bl_clamp__V = record.f_bl_clamp__V
                f_sl_clamp__V = record.f_sl_clamp__V
                if f_bl_clamp__V is None or f_sl_clamp__V is None:
                    raise ValueError("outer solver record carries no clamp residual")
                clamp_bl[record.outer] = max(
                    clamp_bl[record.outer],
                    float(f_bl_clamp__V.abs().max().item()),
                )
                clamp_sl[record.outer] = max(
                    clamp_sl[record.outer],
                    float(f_sl_clamp__V.abs().max().item()),
                )
            else:
                f_bl_kcl__uA = record.f_bl_kcl__uA
                f_sl_kcl__uA = record.f_sl_kcl__uA
                if f_bl_kcl__uA is None or f_sl_kcl__uA is None:
                    raise ValueError("inner solver record carries no wire residual")
                index = record.outer * n_inner + record.inner - 1
                wire_bl[index] = max(wire_bl[index], float(f_bl_kcl__uA.abs().max().item()))
                wire_sl[index] = max(wire_sl[index], float(f_sl_kcl__uA.abs().max().item()))

    solver_points = tuple(
        SolverResidualPoint(
            outer=outer,
            inner=inner,
            f_bl_clamp__V=clamp_bl[outer] if inner == 0 else None,
            f_sl_clamp__V=clamp_sl[outer] if inner == 0 else None,
            f_bl_kcl__uA=wire_bl[outer * n_inner + inner - 1] if inner > 0 else None,
            f_sl_kcl__uA=wire_sl[outer * n_inner + inner - 1] if inner > 0 else None,
        )
        for outer, inner in solver_coordinates
    )

    if not trajectories or not trajectories[0].cell_records:
        return solver_points, ()
    cell_coordinates = [
        ("seed", None, None),
        *(("iteration", outer, inner) for outer in range(n_outer) for inner in range(1, n_inner + 1)),
        ("terminal", None, None),
    ]
    cell_residual = [0.0] * len(cell_coordinates)
    for trajectory in trajectories:
        if len(trajectory.cell_records) != len(cell_coordinates):
            raise ValueError("cell trajectory length does not match n_outer / n_inner")
        for index, cell_record in enumerate(trajectory.cell_records):
            cell_residual[index] = max(cell_residual[index], float(cell_record.cell__uA.abs().max().item()))
    cell_points = tuple(
        CellResidualPoint(
            stage=stage,
            outer=outer,
            inner=inner,
            cell__uA=cell_residual[index],
        )
        for index, (stage, outer, inner) in enumerate(cell_coordinates)
    )
    return solver_points, cell_points


@dataclass(frozen=True)
class _DriveResult:
    """Pooled records of one candidate's drive over the whole workload."""

    trajectories: tuple[SolveResidualTrajectory, ...]
    """Every solve trajectory over every plane and chunk, in drive order."""


def _drive_candidate(
    macro: CimMacro[CimMacroConfig, CimMacroPolicy],
    workload: list[tuple[Tensor, Tensor]],
    *,
    active_inputs: int,
    device: torch.device,
    n_outer: int,
    n_inner: int,
) -> _DriveResult:
    """Program + drive the whole workload; retain every residual record.

    Each `(w, x)` is programmed once, serialized into input planes, and driven
    through the macro's public `vec_mat_mul` under the solver and cell probers.
    The returned ADC codes are discarded — the calibration data rides those
    probers upstream of ADC conversion, so code clipping at a conservative
    operating point is irrelevant. One drive yields
    `n_planes x n_chunks` solves, each contributing its whole iteration
    trajectory plus one terminal record, array chunking running inside the real
    forward path.
    """
    trajectories: list[SolveResidualTrajectory] = []
    for w, x in workload:
        macro.program(w.to(device))
        planes = unroll_sub_phase(
            x.to(device),
            input_num=macro.input_num,
            active_inputs=active_inputs,
            inst_shape=macro.inst_shape,
        )
        # min_outer=0: the guard reads the residuals that drove each solve's
        # final recorded updates, and only the whole trajectory identifies
        # them. The records stay where they were solved: every reduction below
        # and in the plateau / residual passes is device-agnostic, so a hot
        # loop pays no per-drive host transfer and pools nothing in host memory.
        with (
            ColBlColSlProber(min_outer=0) as solver_prober,
            XbarCell1t1rDetailProber() as cell_prober,
            torch.no_grad(),
        ):
            macro.vec_mat_mul(planes, quantization_mode=0, adc_active_bits=None)
        trajectories.extend(
            build_residual_trajectories(
                solver_prober.records,
                cell_prober.records,
                n_outer=n_outer,
                n_inner=n_inner,
            )
        )
    return _DriveResult(trajectories=tuple(trajectories))


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
    """Macro the workload is sampled from; it never enters a candidate pass."""
    inst_shape: tuple[int, ...]
    dtype: torch.dtype
    active_inputs: int
    """Simultaneously active logical inputs per serialized plane."""
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


@dataclass(frozen=True)
class CandidateResidualTrajectory:
    """Per-iteration worst-case residual trajectory for one candidate."""

    iter_count: int
    n_outer: int
    n_inner: int
    solver: tuple[SolverResidualPoint, ...]
    cell: tuple[CellResidualPoint, ...]


@dataclass(frozen=True)
class SolverSweepResult:
    rows: list[CandidateRow]
    scale: WorkloadScale
    trajectories: tuple[CandidateResidualTrajectory, ...]


def aggregate_solver_sweep(
    *,
    swept_key: str,
    candidates: list[int],
    fixed_overrides: dict[str, int],
    context: SolverSweepContext,
) -> SolverSweepResult:
    """Sweep one iteration-count axis, rebuilding a fresh macro per candidate.

    The `(w, x)` workload is sampled once, seeded, from the context's sampling
    host and reused identically across every candidate, so only the swept
    solver knob differs between passes; calibration presumes an all-off policy,
    which makes the per-candidate macro rebuilds comparable. Per candidate the
    pass patches `{**fixed_overrides, swept_key: value}` onto the macro's
    nested-solver table, builds a fresh macro, drives the workload, groups every
    solver and cell residual record by solve, and reduces aligned solves to a
    per-iteration worst-case trajectory. Candidate acceptance uses the current
    residual where each solve stopped, while the full descent remains available
    for later inspection. The step delta
    `max |u_n - u_{n-1}|` compares terminal states against the predecessor
    candidate on 1:1-aligned streams. The leading candidate has no predecessor,
    so it carries no step. The workload signal scales are read off the
    most-converged, last, candidate.

    Raises:
        ValueError: `candidates` is empty, aligned record counts disagree, or
            every step delta across the sweep is exactly zero — the signature
            of a `solver_section` that does not point at the solver the macro
            actually uses, so the patched knob had no effect.

    Returns:
        Candidate rows and workload scale together with every collected
        residual trajectory.
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
            input_num=host.input_num,
            output_num=host.output_num,
            n=context.n_weight,
            batch_w=context.batch_w,
            device=device,
            generator=generator,
        )
        for x in sample_x_batches(
            distribution,
            host,
            input_num=host.input_num,
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
    candidate_trajectories: list[CandidateResidualTrajectory] = []

    prev_terminals: list[ColBlColSlRecord[Any]] | None = None
    for ci, value in enumerate(candidates):
        overrides = {**fixed_overrides, swept_key: value}
        n_outer = overrides["n_outer"]
        n_inner = overrides["n_inner"]
        macro = build_candidate_macro(
            context.base_macro_dict,
            solver_section=context.solver_section,
            overrides=overrides,
            policy=context.policy,
            device=device,
            inst_shape=context.inst_shape,
            dtype=context.dtype,
        )
        drive = _drive_candidate(
            macro,
            workload,
            active_inputs=context.active_inputs,
            device=device,
            n_outer=n_outer,
            n_inner=n_inner,
        )
        trajectories = drive.trajectories
        solver_trajectory, cell_trajectory = aggregate_residual_trajectory(
            trajectories,
            n_outer=n_outer,
            n_inner=n_inner,
        )
        candidate_trajectories.append(
            CandidateResidualTrajectory(
                iter_count=value,
                n_outer=n_outer,
                n_inner=n_inner,
                solver=solver_trajectory,
                cell=cell_trajectory,
            )
        )
        terminals = [trajectory.solver_records[-1] for trajectory in trajectories]

        residual = current_residual_max(trajectories)
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
    return SolverSweepResult(rows=rows, scale=scale, trajectories=tuple(candidate_trajectories))
