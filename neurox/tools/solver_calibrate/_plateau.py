"""Shared step-ratio plateau detection + workload-relative residual guard.

The framework here is **chip-parameter-free** by design:

  * Convergence criterion (primary) is the ratio of consecutive solver step
    sizes ``|u_n − u_{n-1}|``. When the ratio crosses ``ratio_threshold``
    (default 0.5) the solver has stopped making meaningful progress —
    further iterations only oscillate within fp round-off.

  * Residual guard (sanity) is purely relative — ``|F(u_n)|.max <
    reltol × signal_scale`` where ``signal_scale`` is derived from the
    workload itself (``max |I_cell|`` for current residuals,
    ``max |V_node|`` for voltage residuals). Never compared to an
    absolute target tied to a specific chip's electrical envelope.

This file owns the **pure-Python pick logic + result containers**. The
sample-streaming aggregation (which builds the per-candidate rows) lives
in :mod:`_common`; it is solver-family-specific because the unknowns and
residual fields differ across :class:`Solver1T1R` and :class:`OpAmpTIA`.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateRow:
    """Per-candidate aggregate stats over the full workload.

    Attributes:
        iter_count: Logical iteration count of this candidate (e.g.
            ``n_newton`` for TIA, ``n_outer`` for the
            current axis being swept in nested).
        step_max__V: ``max |u_n − u_{n-1}|`` across all 5 (or fewer)
            unknown classes and all (batch, col, row). The primary
            convergence indicator. ``None`` for the first candidate
            (no previous iterate to subtract from).
        step_per_class__V: Per-unknown-class breakdown of
            ``step_max__V`` (debugging). Keys depend on solver family.
        residual_max: Mapping ``residual_name → max(|residual|)`` over
            the workload. Units are baked into the name (e.g.
            ``"cell__uA"``).
    """

    iter_count: int
    step_max__V: float | None
    step_per_class__V: dict[str, float]
    residual_max: dict[str, float]


@dataclass(frozen=True)
class WorkloadScale:
    """Signal scales derived from the workload itself.

    Used as denominators for the relative residual guard so the guard
    threshold (e.g. 1e-3) carries the same chip-independent meaning
    across solvers and workloads.

    Attributes:
        i_cell_typ__uA: ``max |I_cell|`` over the full workload. Used
            as denominator for cell + wire current residuals.
        v_node_typ__V: ``max |V_BL_node|`` over the full workload.
            Used as denominator for clamp voltage residuals. For TIA
            standalone calibration this is the TIA's own ``V_clamp``
            scale instead.
    """

    i_cell_typ__uA: float
    v_node_typ__V: float


@dataclass(frozen=True)
class PickResult:
    """Outcome of running plateau detection + residual guard on a sweep."""

    iter_count: int | None
    """The smallest candidate at the plateau; ``None`` if not reached."""
    reason: str
    """Human-readable explanation of the pick or the rejection."""
    residual_guard_ratios: dict[str, float]
    """Per-residual-class ratio ``residual / scale`` at the picked candidate."""


# ---------------------------------------------------------------------------
# Plateau detection
# ---------------------------------------------------------------------------


def pick_iter_by_step_ratio(
    rows: list[CandidateRow],
    *,
    ratio_threshold: float = 0.5,
) -> int | None:
    """Find the smallest iter_count at the plateau of the step sequence.

    The step sequence is ``step_max__V`` over the candidates. Quadratic
    Newton convergence gives geometrically-decreasing steps; when the
    ratio ``step_n / step_{n-1}`` crosses ``ratio_threshold`` (default
    0.5 — "step from n-1 to n did not shrink to less than half of the
    previous step"), the solver has hit its numerical floor and the last
    candidate with meaningful progress is ``n-1``.

    Returns the ``iter_count`` of that candidate, or ``None`` if the
    plateau was not reached within the candidate range.

    Args:
        rows: Sweep results ordered by ascending iter_count. The first
            row's ``step_max__V`` is ``None`` (no predecessor).
        ratio_threshold: Plateau threshold; smaller values are more
            permissive (declare plateau sooner), larger values are
            stricter (require steeper drop to keep iterating).
    """
    if len(rows) < 3:
        return None

    # First check: is the entire observed sequence already on the plateau?
    # Happens when the solver converges within the very first candidate (e.g.
    # ``n_inner = 1`` for a tightly-conditioned chip) — every observable step
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
    "wire_bl__uA": "current",
    "wire_sl__uA": "current",
    "clamp_bl__V": "voltage",
    "clamp_sl__V": "voltage",
}


def check_residual_relative_guard(
    row: CandidateRow,
    scale: WorkloadScale,
    *,
    reltol: float = 1e-2,
) -> tuple[bool, dict[str, float]]:
    """Verify residuals are tiny relative to workload-derived signal scales.

    A residual passes the guard iff
    ``residual / scale < reltol``. Both the residuals and the scale come
    from the same workload sweep, so the threshold (default ``1e-2``
    = 1%) is chip-independent.

    The guard is a sanity check that rejects a plateau pick if the
    residuals there are still large relative to the workload's signal
    scale. The primary convergence indicator is the step-ratio plateau,
    not residuals; this guard only flags candidates that converged in
    step but failed to meet a meaningful residual budget.

    Args:
        row: Candidate's residual stats.
        scale: Workload-derived signal scales.
        reltol: Relative tolerance. Default ``1e-2`` = 1% sits safely
            above the fp32 accumulated-round-off floor (~ ε_fp32 ·
            sqrt(N_ops) · signal_scale ≈ 0.8% for our chip's wire
            ladder) while still flagging genuine divergence (a 1%
            residual ratio means wire KCL is off by 1% of cell
            current, which is clearly broken). fp64 workloads have
            8+ orders of headroom under this threshold.

    Returns:
        ``(passed, ratios)``: ``passed`` is ``False`` iff any residual
        exceeds the guard; ``ratios`` maps each residual key to its
        ``residual / scale`` value.
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
