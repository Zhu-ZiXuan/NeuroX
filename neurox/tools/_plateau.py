"""Shared step-ratio plateau detection + workload-relative residual guard.

Both criteria are chip-parameter-free:

  * The primary convergence criterion is the ratio of consecutive solver
    step sizes `|u_n - u_{n-1}|`. Once the ratio crosses `ratio_threshold`
    the solver has stopped making meaningful progress — further iterations
    only oscillate within fp round-off.

  * The sanity residual guard is purely relative — `|F(u_n)|.max <
    reltol × signal_scale` with `signal_scale` derived from the workload
    itself (`max |I_cell|` for current residuals, `max |V_node|` for
    voltage residuals), never from an absolute target tied to a specific
    chip's electrical envelope.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateRow:
    """Per-candidate aggregate stats over the full workload."""

    iter_count: int
    """Logical iteration count of the candidate, e.g. a local nonlinear
    solve's `newton_iter_num` or the axis currently swept in a nested solve."""
    step_max__V: float | None
    """`max |u_n - u_{n-1}|` across every unknown class and every (batch, col,
    row) — the primary convergence indicator. `None` for the leading
    candidate, which has no previous iterate."""
    step_per_class__V: dict[str, float]
    """Per-unknown-class breakdown of `step_max__V`; the keys depend on the
    solver family."""
    residual_max: dict[str, float]
    """`residual name → max(|residual|)` over the workload, the unit baked
    into the key (e.g. `cell__uA`)."""


@dataclass(frozen=True)
class WorkloadScale:
    """Signal scales derived from the workload itself.

    They are the denominators of the relative residual guard, so the guard
    threshold carries the same chip-independent meaning across solvers and
    workloads.
    """

    i_cell_typ__uA: float
    """`max |I_cell|` over the full workload — the denominator of the cell
    and wire current residuals."""
    v_node_typ__V: float
    """`max |V_BL_node|` over the full workload — the denominator of the
    clamp voltage residuals."""


@dataclass(frozen=True)
class PickResult:
    """Outcome of running plateau detection + residual guard on a sweep."""

    iter_count: int | None
    """The smallest candidate at the plateau; `None` if it was not reached."""
    reason: str
    """Human-readable explanation of the pick or the rejection."""
    residual_guard_ratios: dict[str, float]
    """Per-residual-class ratio `residual / scale` at the picked candidate."""


# ---------------------------------------------------------------------------
# Plateau detection
# ---------------------------------------------------------------------------


def pick_iter_by_step_ratio(
    rows: list[CandidateRow],
    *,
    ratio_threshold: float = 0.5,
) -> int | None:
    """Find the smallest iteration count at the plateau of the step sequence.

    The step sequence is `step_max__V` over the candidates. Quadratic Newton
    convergence gives geometrically-decreasing steps; once the ratio
    `step_n / step_{n-1}` crosses `ratio_threshold` the solver has hit its
    numerical floor and the last candidate with meaningful progress is `n-1`.

    Args:
        rows: Sweep results ordered by ascending iteration count; the leading
            row carries no step.
        ratio_threshold: Plateau threshold; a smaller value declares the
            plateau sooner, a larger one demands a steeper drop to keep
            iterating.

    Returns:
        The picked candidate's `iter_count`, or `None` when the plateau was
        not reached within the candidate range.
    """
    if len(rows) < 3:
        return None

    # First check: is the entire observed sequence already on the plateau?
    # Happens when the solver converges within the very first candidate (e.g.
    # `n_inner = 1` for a tightly-conditioned chip) — every observable step
    # is already fp-floor noise of similar magnitude.
    valid_steps = [r.step_max__V for r in rows if r.step_max__V is not None]
    if len(valid_steps) >= 2:
        max_step = max(valid_steps)
        min_step = min(valid_steps)
        if max_step > 0.0 and min_step / max_step > ratio_threshold:
            # Steps vary by less than (1/ratio_threshold)× across the sweep —
            # nothing changes after the first candidate.
            return rows[0].iter_count

    for i in range(2, len(rows)):
        prev_step = rows[i - 1].step_max__V
        curr_step = rows[i].step_max__V
        if prev_step is None or curr_step is None:
            continue
        if prev_step <= 0.0:
            return rows[i - 1].iter_count
        ratio = curr_step / prev_step
        if ratio > ratio_threshold:
            return rows[i - 1].iter_count
    return None


# ---------------------------------------------------------------------------
# Relative residual guard
# ---------------------------------------------------------------------------


# Per-residual-class denominator selector: which workload scale this residual
# is measured against. The names are the field keys in CandidateRow.residual_max.
_RESIDUAL_DENOMINATOR_KIND: dict[str, str] = {
    "cell__uA": "current",
    "f_bl_kcl__uA": "current",
    "f_sl_kcl__uA": "current",
    "f_bl_clamp__V": "voltage",
    "f_sl_clamp__V": "voltage",
}


def check_residual_relative_guard(
    row: CandidateRow,
    scale: WorkloadScale,
    *,
    reltol: float = 1e-2,
) -> tuple[bool, dict[str, float]]:
    """Verify residuals are tiny relative to workload-derived signal scales.

    A residual passes the guard iff `residual / scale < reltol`. Both the
    residuals and the scale come from the same workload sweep, so the
    threshold is chip-independent. The guard is a sanity check on a plateau
    pick: it flags a candidate that converged in step yet failed to meet a
    meaningful residual budget. A residual class the scale table does not
    cover is skipped.

    Args:
        row: The candidate's residual stats.
        scale: Workload-derived signal scales.
        reltol: Relative tolerance, as a fraction of the signal scale.

    Returns:
        `(passed, ratios)` — `passed` is `False` iff any residual exceeds
        the guard; `ratios` maps each residual key to its `residual / scale`.
    """
    ratios: dict[str, float] = {}
    passed = True
    for key, residual in row.residual_max.items():
        kind = _RESIDUAL_DENOMINATOR_KIND.get(key)
        if kind == "current":
            denom = max(scale.i_cell_typ__uA, 1e-30)
        elif kind == "voltage":
            denom = max(scale.v_node_typ__V, 1e-30)
        else:
            # Unknown residual class — skip (caller should not include unknowns).
            continue
        ratio = residual / denom
        ratios[key] = ratio
        if ratio >= reltol:
            passed = False
    return passed, ratios


# ---------------------------------------------------------------------------
# Combined pick
# ---------------------------------------------------------------------------


def pick_with_plateau_and_guard(
    rows: list[CandidateRow],
    scale: WorkloadScale,
    *,
    ratio_threshold: float = 0.5,
    reltol: float = 1e-2,
) -> PickResult:
    """Run plateau detection then verify the pick passes the residual guard."""
    n_pick = pick_iter_by_step_ratio(rows, ratio_threshold=ratio_threshold)
    if n_pick is None:
        return PickResult(
            iter_count=None,
            reason=("plateau not reached within the candidate range — increase n_max or widen ratio_threshold"),
            residual_guard_ratios={},
        )
    row_pick = next(r for r in rows if r.iter_count == n_pick)
    passed, ratios = check_residual_relative_guard(row_pick, scale, reltol=reltol)
    if not passed:
        worst = max(ratios.items(), key=lambda kv: kv[1])
        return PickResult(
            iter_count=None,
            reason=(
                f"plateau at iter={n_pick} but residual guard FAILED: "
                f"{worst[0]} ratio={worst[1]:.3e} >= reltol={reltol:.1e} "
                "— candidate did not satisfy the residual guard"
            ),
            residual_guard_ratios=ratios,
        )
    return PickResult(
        iter_count=n_pick,
        reason=f"step-ratio plateau at iter={n_pick}; residual guard passed",
        residual_guard_ratios=ratios,
    )
