"""Tests for the rule-driven replacement policy.

Phase C contract:

* :class:`ReplacementPolicy` is an ordered ruleset, first-match-wins;
* :func:`default_policy` builds the canonical Linear/Conv2d sweep ruleset;
* Replaced subtrees are opaque (the walker does not descend into them);
* Helpers (``name_excluded_match``, ``by_attr_match``,
  ``heterogeneous_macro_policy``) compose into real policies that handle
  partial replacement, exclusions, and per-prefix macro routing;
* ``unmatched={"keep_float", "warn", "error"}`` dispatches correctly.
"""

from __future__ import annotations

import warnings

import pytest
import torch.nn as nn

from neurox.macro.ideal import IdealMacro
from neurox.operator import QuantConv2d, QuantLinear
from neurox.replace import (
    ReplacementContext,
    ReplacementPolicy,
    ReplacementRule,
    by_attr_match,
    default_policy,
    heterogeneous_macro_policy,
    name_excluded_match,
    replace_model,
)


def _fake_factory(*, name: str = "", w_logical_shape: tuple[int, ...] = (1, 1)) -> IdealMacro:
    del name
    return IdealMacro(
        x_value_range=(-7, 7),
        w_value_range=(-7, 7),
        w_logical_shape=w_logical_shape,
    )


def _model_with_three_layers() -> nn.Module:
    """Three Linear layers at root: fc1 (in=4, out=8), fc2 (in=8, out=16), lm_head (in=16, out=2)."""

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 16)
            self.lm_head = nn.Linear(16, 2)

        def forward(self, x):  # noqa: ANN001
            return self.lm_head(self.fc2(self.fc1(x)))

    return Net()


class TestDefaultPolicy:
    """``default_policy`` builds the canonical Linear/Conv2d sweep ruleset."""

    def test_replaces_all_linears(self) -> None:
        model = _model_with_three_layers()
        policy = default_policy(_fake_factory)
        report = policy.apply(model)
        assert len(report.replaced) == 3
        assert all(isinstance(m, QuantLinear) for m in (model.fc1, model.fc2, model.lm_head))

    def test_replace_model_uses_default_when_none(self) -> None:
        a = _model_with_three_layers()
        b = _model_with_three_layers()
        replace_model(a, _fake_factory, verbose=False)
        replace_model(b, _fake_factory, policy=default_policy(_fake_factory), verbose=False)
        # Same structural outcome: every child has the same class name.
        a_kids = [type(c).__name__ for _, c in a.named_modules() if _ != ""]
        b_kids = [type(c).__name__ for _, c in b.named_modules() if _ != ""]
        assert a_kids == b_kids


class TestPartialReplacement:
    """Excluding specific layer names keeps them in float."""

    def test_exclude_lm_head(self) -> None:
        model = _model_with_three_layers()
        excluded = name_excluded_match(["lm_head"])
        rule = ReplacementRule(
            name="linear-except-head",
            match=lambda mod, ctx: isinstance(mod, nn.Linear)
            and not isinstance(mod, QuantLinear)
            and excluded(mod, ctx),
            build=lambda mod, ctx: QuantLinear.from_torch(
                mod, _fake_factory(name=ctx.qualified_name), ctx.qualified_name
            ),
        )
        policy = ReplacementPolicy([rule])
        policy.apply(model)
        assert isinstance(model.fc1, QuantLinear)
        assert isinstance(model.fc2, QuantLinear)
        assert isinstance(model.lm_head, nn.Linear)
        assert not isinstance(model.lm_head, QuantLinear)


class TestByAttrPredicate:
    """``by_attr_match`` selects on module attributes."""

    def test_only_large_out_features(self) -> None:
        model = _model_with_three_layers()  # fc1: out=8, fc2: out=16, lm_head: out=2
        large = by_attr_match("out_features", lambda v: v >= 16)
        rule = ReplacementRule(
            name="linear-large-only",
            match=lambda mod, ctx: isinstance(mod, nn.Linear)
            and not isinstance(mod, QuantLinear)
            and large(mod, ctx),
            build=lambda mod, ctx: QuantLinear.from_torch(
                mod, _fake_factory(name=ctx.qualified_name), ctx.qualified_name
            ),
        )
        policy = ReplacementPolicy([rule])
        policy.apply(model)
        assert not isinstance(model.fc1, QuantLinear)  # out=8 < 16
        assert isinstance(model.fc2, QuantLinear)
        assert not isinstance(model.lm_head, QuantLinear)


class TestHeterogeneousMacros:
    """Different prefixes route to different macro factories."""

    def test_per_prefix_routing(self) -> None:
        model = _model_with_three_layers()
        # Two distinct factories; assert each layer's macro came from the right one
        fc_macros: list[IdealMacro] = []
        head_macros: list[IdealMacro] = []

        def fc_factory(*, name: str = "", w_logical_shape: tuple[int, ...] = (1, 1)) -> IdealMacro:
            del name
            m = IdealMacro(
                x_value_range=(-7, 7),
                w_value_range=(-7, 7),
                w_logical_shape=w_logical_shape,
            )
            fc_macros.append(m)
            return m

        def head_factory(*, name: str = "", w_logical_shape: tuple[int, ...] = (1, 1)) -> IdealMacro:
            del name
            m = IdealMacro(
                x_value_range=(-1, 1),
                w_value_range=(-1, 1),
                w_logical_shape=w_logical_shape,
            )
            head_macros.append(m)
            return m

        policy = heterogeneous_macro_policy(
            {
                "lm_head": head_factory,
                "": fc_factory,  # catch-all
            }
        )
        policy.apply(model)
        # fc1 + fc2 -> fc_factory, lm_head -> head_factory
        assert len(fc_macros) == 2
        assert len(head_macros) == 1
        assert model.lm_head.macro is head_macros[0]
        # Sanity: macros' w_bits differ
        assert head_macros[0].w_value_range != fc_macros[0].w_value_range


class TestUnmatchedHandling:
    """Three modes for eligible-but-unmatched float modules."""

    def test_keep_float_silent(self) -> None:
        model = _model_with_three_layers()
        # Empty rule list -> nothing matches.
        policy = ReplacementPolicy(rules=[], unmatched="keep_float")
        report = policy.apply(model)
        assert report.replaced == []
        assert sorted(report.unmatched) == ["fc1", "fc2", "lm_head"]

    def test_warn_emits_warning(self) -> None:
        model = _model_with_three_layers()
        policy = ReplacementPolicy(rules=[], unmatched="warn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            policy.apply(model)
        assert any("unmatched" in str(w.message) for w in caught)

    def test_error_raises(self) -> None:
        model = _model_with_three_layers()
        policy = ReplacementPolicy(rules=[], unmatched="error")
        with pytest.raises(ValueError, match="unmatched"):
            policy.apply(model)


class TestOpaqueReplacement:
    """The walker does not descend into replaced subtrees."""

    def test_no_recursion_into_quantlinear(self) -> None:
        # If the walker recursed into a replaced subtree it would try to
        # match the macro's internals — which contain no nn.Linear /
        # nn.Conv2d, so the test mostly guards against double-replacement
        # and infinite recursion.
        model = _model_with_three_layers()
        report = default_policy(_fake_factory).apply(model)
        assert len(report.replaced) == 3
        # Second pass: every layer is already QuantLinear, so default
        # rules should NOT re-match.
        report2 = default_policy(_fake_factory).apply(model)
        assert report2.replaced == []


class TestPriorityOrdering:
    """Higher-priority rules evaluate first."""

    def test_higher_priority_wins(self) -> None:
        model = _model_with_three_layers()
        calls: list[str] = []

        def lo_build(mod, ctx):
            calls.append(f"lo:{ctx.qualified_name}")
            return QuantLinear.from_torch(
                mod, _fake_factory(name=ctx.qualified_name), ctx.qualified_name
            )

        def hi_build(mod, ctx):
            calls.append(f"hi:{ctx.qualified_name}")
            return QuantLinear.from_torch(
                mod, _fake_factory(name=ctx.qualified_name), ctx.qualified_name
            )

        rule_lo = ReplacementRule(
            name="lo",
            match=lambda mod, ctx: isinstance(mod, nn.Linear) and not isinstance(mod, QuantLinear),
            build=lo_build,
            priority=0,
        )
        rule_hi = ReplacementRule(
            name="hi",
            match=lambda mod, ctx: isinstance(mod, nn.Linear) and not isinstance(mod, QuantLinear),
            build=hi_build,
            priority=10,
        )
        policy = ReplacementPolicy([rule_lo, rule_hi])
        policy.apply(model)
        assert all(c.startswith("hi:") for c in calls)
