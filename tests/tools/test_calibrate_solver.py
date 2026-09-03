"""Unit tests for the host-agnostic solver calibration plumbing.

Synthetic-data tests only — no end-to-end macro sweep (that is exercised by
the real calibration runs). Covers the engine sub-phase unroll mirror, the
config-space solver-table patch, the record aggregation laws over directly
constructed records, and the shipped run-TOML schema.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from neurox.primitive.xbar.cell import XbarCell1t1rDcop, XbarCell1t1rDetailRecord
from neurox.primitive.xbar.solver import ColBlColSlDcop, ColBlColSlRecord
from neurox.tools.calibrate_solver._common import (
    aggregate_residual_trajectory,
    build_residual_trajectories,
    current_residual_max,
    load_macro_config_dict,
    resolve_macro_files,
    resolve_solver_table,
    solver_residual_max,
    step_delta_over_streams,
    unroll_sub_phase,
)
from neurox.tools.calibrate_solver.col_bl_col_sl import CalibrateSolverColBlColSlConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHIPPED_RUN_TOML = _REPO_ROOT / "validations/xue2020jssc/tools/calibrate_solver.toml"


# ---------------------------------------------------------------------------
# (a) plane-unroll law
# ---------------------------------------------------------------------------


class TestUnrollSubPhaseLaw:
    def _planes_active_mask(self, *, input_num: int, active_inputs: int) -> torch.Tensor:
        """Return the per-plane boolean active mask."""
        x = torch.ones((1, input_num), dtype=torch.long)
        out = unroll_sub_phase(x, input_num=input_num, active_inputs=active_inputs, inst_shape=())
        # Shape: [1, P, input_num] -> [P, input_num]
        return out[0].bool()

    @pytest.mark.parametrize(("input_num", "active_inputs"), [(12, 4), (12, 3), (10, 4), (16, 5)])
    def test_planes_partition_inputs(self, input_num: int, active_inputs: int) -> None:
        """Every plane has at most active_inputs live inputs and together covers all."""
        mask = self._planes_active_mask(input_num=input_num, active_inputs=active_inputs)
        n_planes = mask.shape[0]
        assert n_planes == -(-input_num // active_inputs)
        # Shape: [P, input_num] -> [P]
        inputs_per_plane = mask.sum(dim=-1)
        assert int(inputs_per_plane.max().item()) <= active_inputs
        # Shape: [P, input_num] -> [input_num]
        planes_per_input = mask.long().sum(dim=0)
        assert int(planes_per_input.max().item()) == 1
        assert bool((planes_per_input == 1).all().item())

    def test_full_active_inputs_is_single_dense_plane(self) -> None:
        """active_inputs == input_num collapses to exactly one all-active plane."""
        mask = self._planes_active_mask(input_num=16, active_inputs=16)
        assert mask.shape[0] == 1
        assert bool(mask.all().item())

    def test_inst_span_slots_inserted(self) -> None:
        """The unroll inserts the P axis + inst-span size-1 slots left of input."""
        x = torch.ones((2, 8), dtype=torch.long)
        out = unroll_sub_phase(x, input_num=8, active_inputs=4, inst_shape=(1,))
        # Shape: [..., P, *inst_shape=1, input_num]
        assert tuple(out.shape) == (2, 2, 1, 8)


# ---------------------------------------------------------------------------
# (b) dict-patch law
# ---------------------------------------------------------------------------


def _macro_dict() -> dict:
    """A minimal macro-config-shaped dict with a nested-solver table."""
    return {
        "col_num": 8,
        "row_num": 8,
        "array_config": {
            "leakage_per_inst__uW": 1.0,
            "solver_config": {
                "_neurox_class": "ColBlColSlSolverConfig",
                "n_outer": 20,
                "n_inner": 4,
            },
        },
    }


class TestResolveSolverTableLaw:
    def test_patch_changes_only_target_key(self) -> None:
        """Patching the dotted solver table leaves everything else unchanged."""
        base = _macro_dict()
        patched = copy.deepcopy(base)
        table = resolve_solver_table(patched, "array_config.solver_config")
        table.update({"n_outer": 30, "n_inner": 6})
        # The located table saw the overrides.
        assert patched["array_config"]["solver_config"]["n_outer"] == 30
        assert patched["array_config"]["solver_config"]["n_inner"] == 6
        # Everything outside the solver table is byte-for-byte the base.
        expected = copy.deepcopy(base)
        expected["array_config"]["solver_config"]["n_outer"] = 30
        expected["array_config"]["solver_config"]["n_inner"] = 6
        assert patched == expected
        # The caller's base dict was not mutated.
        assert base == _macro_dict()

    def test_missing_segment_raises(self) -> None:
        with pytest.raises(KeyError, match="segment 'nope' not found"):
            resolve_solver_table(_macro_dict(), "array_config.nope")

    def test_non_table_leaf_raises(self) -> None:
        with pytest.raises(TypeError, match="not a table"):
            resolve_solver_table(_macro_dict(), "array_config.leakage_per_inst__uW")

    def test_wrong_discriminator_raises(self) -> None:
        d = _macro_dict()
        d["array_config"]["solver_config"]["_neurox_class"] = "SomeOtherSolverConfig"
        with pytest.raises(TypeError, match="not ColBlColSlSolverConfig"):
            resolve_solver_table(d, "array_config.solver_config")

    def test_no_discriminator_requires_swept_keys(self) -> None:
        d = _macro_dict()
        table = d["array_config"]["solver_config"]
        del table["_neurox_class"]
        del table["n_inner"]
        with pytest.raises(ValueError, match="does not look like a nested-solver config"):
            resolve_solver_table(d, "array_config.solver_config")

    def test_no_discriminator_with_swept_keys_ok(self) -> None:
        d = _macro_dict()
        del d["array_config"]["solver_config"]["_neurox_class"]
        table = resolve_solver_table(d, "array_config.solver_config")
        assert table["n_outer"] == 20


# ---------------------------------------------------------------------------
# (c) record alignment / aggregation law (synthetic records, no macro)
# ---------------------------------------------------------------------------


_GRID = (2, 3)
_COL = (2,)


def _terminal(*, outer: int = 2, v_bl_node__V: float, v_x__V: float) -> ColBlColSlRecord:
    """The record one synthetic solve closes with: the converged DCOP."""
    ones = torch.ones(_GRID)
    cell = XbarCell1t1rDcop(
        i__uA=ones * 7.0,
        di_dvbl__uS=ones,
        di_dvsl__uS=-ones,
        v_x__V=ones * v_x__V,
    )
    dcop: ColBlColSlDcop = ColBlColSlDcop(
        i_bl_driver__uA=torch.ones(_COL),
        i_sl_driver__uA=torch.ones(_COL),
        v_bl_node__V=ones * v_bl_node__V,
        v_sl_node__V=ones * 0.0,
        cell=cell,
        v_bl_clamp__V=torch.ones(_COL) * 0.3,
        v_sl_drive__V=torch.zeros(_COL),
    )
    return ColBlColSlRecord(
        outer=outer,
        inner=0,
        f_bl_clamp__V=None,
        f_sl_clamp__V=None,
        f_bl_kcl__uA=None,
        f_sl_kcl__uA=None,
        dcop=dcop,
    )


def _clamp_event(*, outer: int, clamp_bl: float) -> ColBlColSlRecord:
    """One outer clamp event carrying its own clamp residuals."""
    return ColBlColSlRecord(
        outer=outer,
        inner=0,
        f_bl_clamp__V=torch.ones(_COL) * clamp_bl,
        f_sl_clamp__V=torch.zeros(_COL),
        f_bl_kcl__uA=None,
        f_sl_kcl__uA=None,
        dcop=None,
    )


def _inner_step(*, outer: int, inner: int, wire_bl: float) -> ColBlColSlRecord:
    """One inner Newton step carrying its own wire residuals."""
    ones = torch.ones(_GRID)
    return ColBlColSlRecord(
        outer=outer,
        inner=inner,
        f_bl_clamp__V=None,
        f_sl_clamp__V=None,
        f_bl_kcl__uA=ones * wire_bl,
        f_sl_kcl__uA=ones * 0.0,
        dcop=None,
    )


def _cell_step(residual__uA: float) -> XbarCell1t1rDetailRecord:
    return XbarCell1t1rDetailRecord(cell__uA=torch.ones(_GRID) * residual__uA)


def _solve(*, clamp_bl: tuple[float, ...], wire_bl: tuple[float, ...]) -> list[ColBlColSlRecord]:
    """One whole synthetic solve: its trajectory then its terminal record.

    The two residual sequences are laid out so the LAST entry of each is the
    iterate the solve stopped on, which is the only one the guard reads.
    """
    trajectory: list[ColBlColSlRecord] = []
    for outer, (clamp, wire) in enumerate(zip(clamp_bl, wire_bl, strict=True)):
        trajectory.append(_clamp_event(outer=outer, clamp_bl=clamp))
        trajectory.append(_inner_step(outer=outer, inner=1, wire_bl=wire))
    trajectory.append(_terminal(outer=len(clamp_bl), v_bl_node__V=0.0, v_x__V=0.0))
    return trajectory


class TestRecordAggregationLaw:
    def test_full_solver_and_cell_trajectories_are_retained_per_solve(self) -> None:
        solver_records = (
            *_solve(clamp_bl=(1.0, 0.1), wire_bl=(2.0, 0.2)),
            *_solve(clamp_bl=(3.0, 0.3), wire_bl=(4.0, 0.4)),
        )
        # n_outer=2, n_inner=1: seed + two iterative evaluations + terminal.
        cell_records = tuple(_cell_step(value) for value in (9.0, 8.0, 7.0, 0.07, 6.0, 5.0, 4.0, 0.04))

        trajectories = build_residual_trajectories(
            solver_records,
            cell_records,
            n_outer=2,
            n_inner=1,
        )

        assert len(trajectories) == 2
        assert trajectories[0].solver_records == tuple(solver_records[:5])
        assert trajectories[0].cell_records == cell_records[:4]
        assert trajectories[1].solver_records == tuple(solver_records[5:])
        assert trajectories[1].cell_records == cell_records[4:]

        solver_trace, cell_trace = aggregate_residual_trajectory(
            trajectories,
            n_outer=2,
            n_inner=1,
        )
        assert [(point.outer, point.inner) for point in solver_trace] == [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert solver_trace[0].f_bl_clamp__V == pytest.approx(3.0)
        assert solver_trace[1].f_bl_kcl__uA == pytest.approx(4.0)
        assert [point.cell__uA for point in cell_trace] == pytest.approx([9.0, 8.0, 7.0, 0.07])
        assert [point.stage for point in cell_trace] == ["seed", "iteration", "iteration", "terminal"]

    def test_current_residual_uses_each_solve_stopping_point(self) -> None:
        solver_records = tuple(_solve(clamp_bl=(9.0, 0.02), wire_bl=(8.0, 0.5)))
        cell_records = tuple(_cell_step(value) for value in (7.0, 6.0, 5.0, 0.03))
        trajectories = build_residual_trajectories(
            solver_records,
            cell_records,
            n_outer=2,
            n_inner=1,
        )

        residual = current_residual_max(trajectories)

        assert residual["cell__uA"] == pytest.approx(0.03)
        assert residual["f_bl_kcl__uA"] == pytest.approx(0.5)
        assert residual["f_bl_clamp__V"] == pytest.approx(0.02)

    def test_cell_trajectory_must_align_with_complete_solves(self) -> None:
        solver_records = tuple(_solve(clamp_bl=(0.1,), wire_bl=(0.2,)))
        with pytest.raises(ValueError, match="expected 3 per solve"):
            build_residual_trajectories(
                solver_records,
                tuple(_cell_step(value) for value in (1.0, 0.1)),
                n_outer=1,
                n_inner=1,
            )

    def test_cell_without_residual_channel_keeps_solver_trajectory(self) -> None:
        solver_records = tuple(_solve(clamp_bl=(0.1,), wire_bl=(0.2,)))
        trajectories = build_residual_trajectories(
            solver_records,
            (),
            n_outer=1,
            n_inner=1,
        )

        assert len(trajectories) == 1
        assert trajectories[0].solver_records == solver_records
        assert trajectories[0].cell_records == ()
        assert "cell__uA" not in current_residual_max(trajectories)

    def test_step_delta_is_max_abs_field_difference(self) -> None:
        prev = [
            _terminal(v_bl_node__V=0.0, v_x__V=0.0),
            _terminal(v_bl_node__V=1.0, v_x__V=1.0),
        ]
        curr = [
            _terminal(v_bl_node__V=0.5, v_x__V=0.2),
            _terminal(v_bl_node__V=1.0, v_x__V=3.0),
        ]
        step = step_delta_over_streams(prev, curr)
        # v_bl_node__V: max(|0.5-0|, |1-1|) = 0.5
        assert step["v_bl_node__V"] == pytest.approx(0.5)
        # v_x__V: max(|0.2-0|, |3-1|) = 2.0
        assert step["v_x__V"] == pytest.approx(2.0)
        # v_sl_node__V / clamps unchanged -> 0
        assert step["v_sl_node__V"] == pytest.approx(0.0)

    def test_identical_streams_give_zero_step(self) -> None:
        stream = [_terminal(v_bl_node__V=1.0, v_x__V=1.0)]
        step = step_delta_over_streams(stream, [copy.copy(stream[0])])
        assert max(step.values()) == 0.0

    def test_misaligned_streams_raise(self) -> None:
        one = [_terminal(v_bl_node__V=1.0, v_x__V=1.0)]
        with pytest.raises(ValueError, match="misaligned"):
            step_delta_over_streams(one, one + one)

    def test_step_delta_refuses_an_iteration_record(self) -> None:
        """A step delta compares converged iterates, so it demands a DCOP."""
        stream = [_inner_step(outer=0, inner=1, wire_bl=0.0)]
        with pytest.raises(ValueError, match="terminal record"):
            step_delta_over_streams(stream, stream)

    def test_residual_max_is_per_field_max_abs_over_solves(self) -> None:
        """Each solve contributes the residuals driving its final updates."""
        records = [
            *_solve(clamp_bl=(0.1,), wire_bl=(2.0,)),
            *_solve(clamp_bl=(0.05,), wire_bl=(5.0,)),
        ]
        residual = solver_residual_max(records)
        assert residual["f_bl_kcl__uA"] == pytest.approx(5.0)
        assert residual["f_bl_clamp__V"] == pytest.approx(0.1)
        assert residual["f_sl_kcl__uA"] == pytest.approx(0.0)

    def test_only_the_last_iterate_of_a_solve_is_read(self) -> None:
        """DESCENT LAW: the residuals a solve passed through never enter the max.

        A converging solve's early iterates are large by construction, so
        counting them would report divergence for every well-behaved solve.
        """
        records = _solve(clamp_bl=(9.0, 8.0, 0.02), wire_bl=(7.0, 6.0, 0.5))
        residual = solver_residual_max(records)
        assert residual["f_bl_kcl__uA"] == pytest.approx(0.5)
        assert residual["f_bl_clamp__V"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# (d) shipped run-TOML schema
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _SHIPPED_RUN_TOML.exists(),
    reason="Shipped calibrate_solver run-config is a Batch-2 artifact "
    "(validations/xue2020jssc/tools/calibrate_solver.toml) not yet created; "
    "the isub_iadc_1t1r scheme that previously shipped it was deleted in this redesign.",
)
class TestShippedRunConfig:
    def test_parses_and_validates(self) -> None:
        cfg = CalibrateSolverColBlColSlConfig.from_file(_SHIPPED_RUN_TOML)
        assert cfg.workload.batch_w >= 1
        assert cfg.workload.active_inputs >= 1
        assert cfg.macro.solver_section

    def test_solver_section_resolves_in_macro_config(self) -> None:
        cfg = CalibrateSolverColBlColSlConfig.from_file(_SHIPPED_RUN_TOML)
        config_paths, _policy_path = resolve_macro_files(cfg.macro, base=_SHIPPED_RUN_TOML)
        macro_dict = load_macro_config_dict(config_paths, config_section=cfg.macro.config_section)
        table = resolve_solver_table(macro_dict, cfg.macro.solver_section)
        assert {"n_outer", "n_inner"} <= set(table)

    def test_active_inputs_matches_max_active_num(self) -> None:
        """active_inputs is the production-faithful max_active_num."""
        cfg = CalibrateSolverColBlColSlConfig.from_file(_SHIPPED_RUN_TOML)
        config_paths, _policy_path = resolve_macro_files(cfg.macro, base=_SHIPPED_RUN_TOML)
        macro_dict = load_macro_config_dict(config_paths, config_section=cfg.macro.config_section)
        assert cfg.workload.active_inputs == macro_dict["max_active_num"]
