"""Tests for the side-channel profiler and ``ProfileMixin`` mixin.

Phase A contract:

* physical modules log dynamic energy + latency through ``_log_dynamic``;
* fabrication records ``_inst_count`` on every ``ProfileMixin``;
* ``NeuroxProfiler.analyze_static`` sums area + leakage power from those;
* the profiler's ``summary`` derives leakage *energy* centrally from
  static leakage power and total runtime latency.
"""

from __future__ import annotations

import pytest
import torch

from neurox.config import DEFAULT_1T1R_MACRO_TOML
from neurox.digital import Accumulator, AccumulatorConfig, Requantizer, RequantizerConfig, ShiftAdder, ShiftAdderConfig
from neurox.common.mixin import ProfileMixin
from neurox.common.profiler import NeuroxProfiler
from example.common.macro_factory import build_macro_factory


N_LOGICAL = 16
K_LOGICAL = 32
_DEFAULT_W_LOGICAL_SHAPE = (N_LOGICAL, K_LOGICAL)


def _build_ideal_macro(name: str = "fc1.macro"):
    factory = build_macro_factory(DEFAULT_1T1R_MACRO_TOML, xbar="ideal")
    return factory(name=name, w_logical_shape=_DEFAULT_W_LOGICAL_SHAPE)


def _program_small(macro) -> None:
    weight = torch.randint(-3, 4, _DEFAULT_W_LOGICAL_SHAPE, dtype=torch.int32)
    macro.program(weight)
    macro.fabricate()


# ---------------------------------------------------------------------------
# Mixin contract
# ---------------------------------------------------------------------------


class TestProfileMixin:
    """``ProfileMixin`` builds a complete instance even without nn.Module."""

    def test_name_qualified_module_type(self) -> None:
        cfg = AccumulatorConfig(
            bit_width=32,
            energy_per_op__fJ=0.0,
            latency_per_op__ns=0.0,
            leakage_per_inst__uW=0.5,
            area_per_inst__um2=10.0,
        )
        acc = Accumulator(cfg=cfg, name="fc1.macro.col_accumulator", inst_shape=(1,))
        assert acc.qualified_name == "fc1.macro.col_accumulator"
        assert acc.module_type == "Accumulator"

    def test_log_static_int(self) -> None:
        cfg = AccumulatorConfig(
            bit_width=32,
            energy_per_op__fJ=0.0,
            latency_per_op__ns=0.0,
            leakage_per_inst__uW=1.0,
            area_per_inst__um2=60.0,
        )
        acc = Accumulator(cfg=cfg, name="m.acc", inst_shape=(8,))
        assert acc.inst_count == 8
        assert acc.inst_area__um2 == pytest.approx(60.0 * 8)
        assert acc.inst_leakage__uW == pytest.approx(1.0 * 8)

    def test_log_static_shape(self) -> None:
        cfg = ShiftAdderConfig(
            bit_width=32,
            energy_per_op__fJ=0.0,
            latency_per_op__ns=0.0,
            leakage_per_inst__uW=1.5,
            area_per_inst__um2=75.0,
        )
        sa = ShiftAdder(cfg=cfg, name="m.sa", inst_shape=(2, 3, 4))  # 24 instances
        assert sa.inst_count == 24
        assert sa.inst_area__um2 == pytest.approx(75.0 * 24)

    def test_log_dynamic_no_op_outside_context(self) -> None:
        cfg = AccumulatorConfig(
            bit_width=32,
            energy_per_op__fJ=5.0,
            latency_per_op__ns=0.5,
            leakage_per_inst__uW=0.0,
            area_per_inst__um2=0.0,
        )
        acc = Accumulator(cfg=cfg, name="m.acc", inst_shape=(1,))
        # No profiler bound -> call is a silent no-op.
        acc._log_dynamic(10.0, 0.5)
        assert NeuroxProfiler.get_current() is None


# ---------------------------------------------------------------------------
# Static aggregation via macro cascade
# ---------------------------------------------------------------------------


class TestStaticAggregation:
    """Constructed macro reports area + leakage via ``analyze_static``."""

    def test_ideal_macro_static_sums_known_components(self) -> None:
        macro = _build_ideal_macro()
        _program_small(macro)

        # Expected contributors for the ideal-tile macro (no SAR ADC):
        # - col_accumulator: 60 um2 area, 1.0 uW leakage
        # - sw_shift_adder + sa_shift_adder: 75 um2 each, 1.5 uW each
        # - requantizer: 0 (zero PPA in the macro TOML)
        # - IdealXbar inherits PPA from the physical twin (500 um2, 5.0 uW)
        # - each inst_count is 1 for tile-fitting weight
        static = NeuroxProfiler.analyze_static(macro)
        assert static.area__um2 == pytest.approx(60.0 + 75.0 + 75.0 + 500.0)
        assert static.leakage_power__uW == pytest.approx(1.0 + 1.5 + 1.5 + 5.0)
        assert static.latency__ns == 0.0  # no runtime events yet

    def test_hierarchical_names(self) -> None:
        macro = _build_ideal_macro(name="model.fc1.macro")
        _program_small(macro)

        names = {m.qualified_name for m in macro.modules() if isinstance(m, ProfileMixin)}
        assert "model.fc1.macro.xbar" in names
        assert "model.fc1.macro.col_accumulator" in names
        assert "model.fc1.macro.sw_shift_adder" in names
        assert "model.fc1.macro.sa_shift_adder" in names
        assert "model.fc1.macro.requantizer" in names

    def test_collect_static_records(self) -> None:
        macro = _build_ideal_macro(name="m")
        _program_small(macro)
        records = NeuroxProfiler.collect_static(macro)
        by_name = {r.qualified_name: r for r in records}
        # Accumulator area = 60 * 1
        assert by_name["m.col_accumulator"].area__um2 == pytest.approx(60.0)
        # ShiftAdder leakage = 1.5 * 1
        assert by_name["m.sw_shift_adder"].leakage_power__uW == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Dynamic side channel
# ---------------------------------------------------------------------------


class TestDynamicEvents:
    """Digital modules emit one event per ``operate`` call."""

    def test_accumulator_emits_event(self) -> None:
        cfg = AccumulatorConfig(
            bit_width=32,
            energy_per_op__fJ=4.0,
            latency_per_op__ns=0.5,
            leakage_per_inst__uW=0.0,
            area_per_inst__um2=0.0,
        )
        acc = Accumulator(cfg=cfg, name="acc", inst_shape=(1,))

        x = torch.zeros(2, 3, dtype=torch.int32)
        with NeuroxProfiler() as p:
            acc.operate(x, dim=-1)
        assert len(p.events) == 1
        evt = p.events[0]
        assert evt.qualified_name == "acc"
        assert evt.module_type == "Accumulator"
        # energy = sum over output element count (which is 2 after dim=-1 reduction)
        assert evt.dynamic_energy__fJ == pytest.approx(4.0 * 2)
        assert evt.latency__ns == pytest.approx(0.5)

    def test_shift_adder_emits_event(self) -> None:
        cfg = ShiftAdderConfig(
            bit_width=32,
            energy_per_op__fJ=2.0,
            latency_per_op__ns=0.25,
            leakage_per_inst__uW=0.0,
            area_per_inst__um2=0.0,
        )
        sa = ShiftAdder(cfg=cfg, name="sa", inst_shape=(1,))
        x = torch.zeros(4, dtype=torch.int32)
        with NeuroxProfiler() as p:
            sa.operate(x, scale=2, dim=-1, init_val=None)
        assert len(p.events) == 1
        # output is scalar after reduction, energy = 2.0 * 1
        assert p.events[0].dynamic_energy__fJ == pytest.approx(2.0)

    def test_requantizer_emits_event(self) -> None:
        cfg = RequantizerConfig(
            bit_width=32,
            energy_per_op__fJ=1.0,
            latency_per_op__ns=0.5,
            leakage_per_inst__uW=0.0,
            area_per_inst__um2=0.0,
        )
        req = Requantizer(cfg=cfg, name="req", inst_shape=(1,))
        x = torch.zeros(2, 3, dtype=torch.int32)
        m = torch.ones(1, dtype=torch.int32)
        s = torch.zeros(1, dtype=torch.int32)
        with NeuroxProfiler() as p:
            req.operate(x, m, s, None)
        assert len(p.events) == 1
        assert p.events[0].dynamic_energy__fJ == pytest.approx(1.0 * 2 * 3)

    def test_no_events_outside_context(self) -> None:
        cfg = AccumulatorConfig(
            bit_width=32,
            energy_per_op__fJ=4.0,
            latency_per_op__ns=0.5,
            leakage_per_inst__uW=0.0,
            area_per_inst__um2=0.0,
        )
        acc = Accumulator(cfg=cfg, name="acc", inst_shape=(1,))
        x = torch.zeros(2, 3, dtype=torch.int32)
        # No active profiler — must not raise and must produce nothing.
        acc.operate(x, dim=-1)
        assert NeuroxProfiler.get_current() is None


# ---------------------------------------------------------------------------
# Centralised leakage-energy derivation
# ---------------------------------------------------------------------------


class TestLeakageEnergyCentralized:
    """Leakage energy is derived once in ``summary`` from runtime latency."""

    def test_summary_derives_leakage_energy(self) -> None:
        cfg = AccumulatorConfig(
            bit_width=32,
            energy_per_op__fJ=4.0,
            latency_per_op__ns=2.0,
            leakage_per_inst__uW=3.0,
            area_per_inst__um2=0.0,
        )
        acc = Accumulator(cfg=cfg, name="acc", inst_shape=(5,))  # 5 instances, total leakage = 15 uW

        x = torch.zeros(8, dtype=torch.int32)
        with NeuroxProfiler() as p:
            acc.operate(x, dim=-1)  # one event with latency 2.0 ns
            acc.operate(x, dim=-1)  # second event, total latency 4.0 ns

        # analyze_static on a single ProfileMixin (wrap in dummy parent)
        from neurox.common.profiler import StaticMetrics
        static = StaticMetrics(
            area__um2=acc._inst_area__um2,
            leakage_power__uW=acc._inst_leakage__uW,
            latency__ns=p.total_latency__ns,
        )
        out = p.summary(static=static)
        # The summary line for leakage_energy = 15 uW * 4.0 ns = 60 fJ
        assert "leakage_energy_total_fJ: 60.0000" in out

    def test_static_metrics_derived_property(self) -> None:
        from neurox.common.profiler import StaticMetrics
        s = StaticMetrics(area__um2=100.0, leakage_power__uW=2.5, latency__ns=4.0)
        assert s.leakage_energy__fJ == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Composite ownership rule
# ---------------------------------------------------------------------------


class TestCompositeOwnership:
    """Composite modules (Xbar, ReadOut) thread names but never double-log."""

    def test_macro_walk_no_duplicate_inst_area(self) -> None:
        """No two ProfileMixin contributions share the same qualified_name."""
        macro = _build_ideal_macro(name="m")
        _program_small(macro)

        names = [m.qualified_name for m in macro.modules() if isinstance(m, ProfileMixin)]
        assert len(names) == len(set(names)), f"duplicates in {names}"
