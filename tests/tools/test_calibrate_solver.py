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

from neurox.primitive.xbar.cell import XbarCell1t1rDcop
from neurox.primitive.xbar.solver import SolverDcop, SolverRecord
from neurox.tools.calibrate_solver._common import (
    load_macro_config_dict,
    resolve_macro_files,
    resolve_solver_table,
    solver_residual_max,
    step_delta_over_streams,
    unroll_sub_phase,
)
from neurox.tools.calibrate_solver.nested import CalibrateSolverNestedConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHIPPED_RUN_TOML = _REPO_ROOT / "validations/xue2020jssc/tools/calibrate_solver.toml"


# ---------------------------------------------------------------------------
# (a) plane-unroll law
# ---------------------------------------------------------------------------


class TestUnrollSubPhaseLaw:
    def _planes_active_mask(self, *, row_num: int, active_rows: int) -> torch.Tensor:
        """Return the per-plane boolean active mask."""
        x = torch.ones((1, row_num), dtype=torch.long)
        out = unroll_sub_phase(x, row_num=row_num, active_rows=active_rows, inst_rank=0)
        # Ones where active, 0 where WL off.
        # Shape: [1, P, row_num] -> [P, row_num]
        return out[0].bool()

    @pytest.mark.parametrize(("row_num", "active_rows"), [(12, 4), (12, 3), (10, 4), (16, 5)])
    def test_planes_partition_rows(self, row_num: int, active_rows: int) -> None:
        """Every plane has <= active_rows live rows, disjoint, union covers all."""
        mask = self._planes_active_mask(row_num=row_num, active_rows=active_rows)
        n_planes = mask.shape[0]
        assert n_planes == -(-row_num // active_rows)
        # Each plane drives at most active_rows rows.
        # Shape: [P, row_num] -> [P]
        rows_per_plane = mask.sum(dim=-1)
        assert int(rows_per_plane.max().item()) <= active_rows
        # Pairwise disjoint: at most one plane owns each row.
        # Shape: [P, row_num] -> [row_num]
        planes_per_row = mask.long().sum(dim=0)
        assert int(planes_per_row.max().item()) == 1
        # Union covers every row exactly once.
        assert bool((planes_per_row == 1).all().item())

    def test_full_active_rows_is_single_dense_plane(self) -> None:
        """active_rows == row_num collapses to exactly one all-active plane."""
        mask = self._planes_active_mask(row_num=16, active_rows=16)
        assert mask.shape[0] == 1
        assert bool(mask.all().item())

    def test_inst_span_slots_inserted(self) -> None:
        """The unroll inserts the P axis + inst-span size-1 slots left of row."""
        x = torch.ones((2, 8), dtype=torch.long)
        out = unroll_sub_phase(x, row_num=8, active_rows=4, inst_rank=1)
        # Shape: [..., P, *inst_shape=1, row_num]
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
                "_neurox_class": "NestedParallelRailSolverConfig",
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
        with pytest.raises(TypeError, match="not NestedParallelRailSolverConfig"):
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


def _record(*, v_bl_node: float, v_x: float, wire_bl: float, clamp_bl: float) -> SolverRecord:
    """One synthetic SolverRecord with scalar-broadcast tensor fields."""
    ones = torch.ones((2, 3))
    cell = XbarCell1t1rDcop(
        i__uA=ones * 7.0,
        di_dvbl__uS=ones,
        di_dvsl__uS=-ones,
        v_x__V=ones * v_x,
    )
    dcop: SolverDcop = SolverDcop(
        i_bl_driver=torch.ones((2,)),
        i_sl_driver=torch.ones((2,)),
        v_bl_node=ones * v_bl_node,
        v_sl_node=ones * 0.0,
        cell=cell,
        v_bl_clamp=torch.ones((2,)) * 0.3,
        v_sl_drive=torch.zeros((2,)),
    )
    return SolverRecord(
        emitter="synthetic",
        dcop=dcop,
        wire_bl__uA=ones * wire_bl,
        wire_sl__uA=ones * 0.0,
        clamp_bl__V=torch.ones((2,)) * clamp_bl,
        clamp_sl__V=torch.zeros((2,)),
    )


class TestRecordAggregationLaw:
    def test_step_delta_is_max_abs_field_difference(self) -> None:
        prev = [
            _record(v_bl_node=0.0, v_x=0.0, wire_bl=0.0, clamp_bl=0.0),
            _record(v_bl_node=1.0, v_x=1.0, wire_bl=0.0, clamp_bl=0.0),
        ]
        curr = [
            _record(v_bl_node=0.5, v_x=0.2, wire_bl=0.0, clamp_bl=0.0),
            _record(v_bl_node=1.0, v_x=3.0, wire_bl=0.0, clamp_bl=0.0),
        ]
        step = step_delta_over_streams(prev, curr)
        # v_bl_node: max(|0.5-0|, |1-1|) = 0.5
        assert step["v_bl_node"] == pytest.approx(0.5)
        # v_x: max(|0.2-0|, |3-1|) = 2.0
        assert step["v_x"] == pytest.approx(2.0)
        # v_sl_node / clamps unchanged -> 0
        assert step["v_sl_node"] == pytest.approx(0.0)

    def test_identical_streams_give_zero_step(self) -> None:
        stream = [_record(v_bl_node=1.0, v_x=1.0, wire_bl=0.0, clamp_bl=0.0)]
        step = step_delta_over_streams(stream, [copy.copy(stream[0])])
        assert max(step.values()) == 0.0

    def test_misaligned_streams_raise(self) -> None:
        one = [_record(v_bl_node=1.0, v_x=1.0, wire_bl=0.0, clamp_bl=0.0)]
        with pytest.raises(ValueError, match="misaligned"):
            step_delta_over_streams(one, one + one)

    def test_residual_max_is_per_field_max_abs(self) -> None:
        records = [
            _record(v_bl_node=0.0, v_x=0.0, wire_bl=2.0, clamp_bl=0.1),
            _record(v_bl_node=0.0, v_x=0.0, wire_bl=5.0, clamp_bl=0.05),
        ]
        residual = solver_residual_max(records)
        assert residual["wire_bl__uA"] == pytest.approx(5.0)
        assert residual["clamp_bl__V"] == pytest.approx(0.1)
        assert residual["wire_sl__uA"] == pytest.approx(0.0)


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
        cfg = CalibrateSolverNestedConfig.from_file(_SHIPPED_RUN_TOML)
        # inst_shape is bound to [batch_w] and active_rows is in range.
        assert list(cfg.workload.inst_shape) == [cfg.workload.batch_w]
        assert cfg.workload.active_rows >= 1
        assert cfg.macro.solver_section

    def test_solver_section_resolves_in_macro_config(self) -> None:
        cfg = CalibrateSolverNestedConfig.from_file(_SHIPPED_RUN_TOML)
        config_paths, _policy_path = resolve_macro_files(cfg.macro, base=_SHIPPED_RUN_TOML)
        macro_dict = load_macro_config_dict(config_paths, config_section=cfg.macro.config_section)
        table = resolve_solver_table(macro_dict, cfg.macro.solver_section)
        assert {"n_outer", "n_inner"} <= set(table)

    def test_active_rows_matches_max_active_num(self) -> None:
        """active_rows is the production-faithful max_active_num (= active_row_num)."""
        cfg = CalibrateSolverNestedConfig.from_file(_SHIPPED_RUN_TOML)
        config_paths, _policy_path = resolve_macro_files(cfg.macro, base=_SHIPPED_RUN_TOML)
        macro_dict = load_macro_config_dict(config_paths, config_section=cfg.macro.config_section)
        assert cfg.workload.active_rows == macro_dict["max_active_num"]
