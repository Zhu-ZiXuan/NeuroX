"""Tests for the top-level ``neurox`` package surface (Phase D).

The three lifecycle stages — preparation, evaluator construction,
execution + profiling — must all be reachable via ``import neurox``
without any subpackage import.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import neurox


class TestStage1Preparation:
    """Stage 1 names: training-side conversion + checkpoint export."""

    def test_quantspec_visible(self) -> None:
        assert hasattr(neurox, "QuantSpec")

    def test_hat_helpers_visible(self) -> None:
        assert callable(neurox.replace_for_hat)
        assert callable(neurox.freeze_hat_observers)
        assert callable(neurox.fold_batchnorm)
        assert callable(neurox.extract_neurox_state)


class TestStage2EvaluatorConstruction:
    """Stage 2 names: build_evaluator + staged building blocks + policy."""

    def test_one_shot_wrapper(self) -> None:
        assert callable(neurox.build_evaluator)

    def test_staged_functions(self) -> None:
        assert callable(neurox.replace_model)
        assert callable(neurox.load_neurox_state)
        assert callable(neurox.bind_output_calibration)
        assert callable(neurox.fabricate_model)

    def test_policy_surface(self) -> None:
        assert neurox.ReplacementPolicy is not None
        assert neurox.ReplacementRule is not None
        assert neurox.ReplacementContext is not None
        assert callable(neurox.default_policy)
        assert callable(neurox.name_excluded_match)
        assert callable(neurox.by_attr_match)
        assert callable(neurox.heterogeneous_macro_policy)

    def test_state_diagnostic_types(self) -> None:
        assert issubclass(neurox.NeuroxStateError, ValueError)
        assert neurox.StateBindingReport is not None
        assert neurox.StructuralReport is not None


class TestStage3ExecutionAndProfiling:
    """Stage 3 names: NeuroxProfiler + side-channel types."""

    def test_profiler_visible(self) -> None:
        assert neurox.NeuroxProfiler is not None
        assert neurox.ProfilerReport is not None
        assert neurox.StaticMetrics is not None
        assert neurox.RuntimeEvent is not None
        assert neurox.StaticRecord is not None
        assert neurox.ProfileMixin is not None


class TestRoundtripImportOnly:
    """Build + run + profile using only ``import neurox`` (no submodule reach-in)."""

    def test_end_to_end_with_fake_macro(self) -> None:
        from neurox.macro.ideal import IdealMacro  # only this single non-public reach-in

        class Tiny(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(4, 3)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        def factory(*, name: str = "", w_logical_shape: tuple[int, ...] = (1, 1)) -> IdealMacro:
            del name
            return IdealMacro(
                x_value_range=(-7, 7),
                w_value_range=(-7, 7),
                w_logical_shape=w_logical_shape,
            )

        # Stage 2 staged path through the top-level surface.
        model = Tiny()
        neurox.replace_model(model, factory, verbose=False)
        ckpt = {"schema": "neurox_flat", "state_dict": model.state_dict()}
        neurox.load_neurox_state(model, ckpt, strict=True)
        neurox.bind_output_calibration(model)
        neurox.program_model(model)
        neurox.fabricate_model(model)

        # Stage 3 execution + profiling.
        x = torch.zeros(2, 4)
        with neurox.NeuroxProfiler() as profiler:
            y = model(x)
        assert y.shape == (2, 3)
        static = neurox.NeuroxProfiler.analyze_static(model)
        # IdealMacro carries no PPA-bearing children; static aggregation
        # is well-defined but yields zero.
        assert static.area__um2 >= 0.0
        # ``profiler.summary`` runs without errors.
        out = profiler.summary(static=static)
        assert "dynamic_energy_total_fJ" in out


