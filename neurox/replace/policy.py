"""Rule-driven replacement policy."""

from __future__ import annotations

import fnmatch
import sys
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import IO, Any, Literal

import torch.nn as nn

from neurox.macro.base import NeuroxMacroQuantMatMul
from neurox.operator import NeuroxOperator, QuantConv2d, QuantLinear

# ---------------------------------------------------------------------------
# Rule + Context dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplacementContext:
    """Context object handed to :class:`ReplacementRule` ``match`` and ``build``.

    Attributes:
        qualified_name: Dotted path of the candidate module within the
            root model (e.g. ``encoder.layers.0.attn.q_proj``).
        parent: The :class:`nn.Module` owning the candidate as an
            attribute.  Useful for shape introspection or attribute
            replacement that needs more than the candidate itself.
        depth: Recursion depth from the root model (root children have
            ``depth == 0``).
        metadata: Free-form per-walk dictionary that rules can read or
            mutate to carry side state (e.g. an index counter).
    """

    qualified_name: str
    parent: nn.Module
    depth: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplacementRule:
    """One rule in a :class:`ReplacementPolicy`.

    Attributes:
        name: Human-readable rule identifier surfaced in reports.
        match: Predicate ``(module, context) -> bool``.  Receives the
            candidate child module and a :class:`ReplacementContext`.
        build: Constructor ``(module, context) -> nn.Module`` returning
            the replacement module.  Only invoked when ``match`` is true.
        priority: Higher values evaluate first when sorted by priority.
            Defaults to ``0``; rules with equal priority preserve
            declaration order.
    """

    name: str
    match: Callable[[nn.Module, ReplacementContext], bool]
    build: Callable[[nn.Module, ReplacementContext], nn.Module]
    priority: int = 0


@dataclass(frozen=True)
class StructuralReport:
    """Outcome of a :meth:`ReplacementPolicy.apply` pass.

    Attributes:
        replaced: ``[(qualified_name, rule_name)]`` for every successful match.
        unmatched: Qualified names of float modules that were eligible
            (``nn.Linear`` / ``nn.Conv2d``) but no rule matched.
    """

    replaced: list[tuple[str, str]] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Policy driver
# ---------------------------------------------------------------------------


class ReplacementPolicy:
    """Ordered ruleset + unmatched-module handling.

    Rules evaluate in priority-descending (then declaration) order;
    first match wins, ``build`` runs, the result is bound on the parent.
    Replaced subtrees are opaque (not recursed into).

    ``unmatched`` controls behaviour when an eligible ``nn.Linear`` /
    ``nn.Conv2d`` is not matched by any rule: ``"keep_float"`` leaves it
    in place, ``"warn"`` emits a stderr warning, ``"error"`` raises.
    """

    def __init__(
        self,
        rules: Sequence[ReplacementRule],
        *,
        unmatched: Literal["keep_float", "warn", "error"] = "keep_float",
    ) -> None:
        if unmatched not in ("keep_float", "warn", "error"):
            raise ValueError(f"unmatched must be 'keep_float', 'warn', or 'error'; got {unmatched!r}")
        # Stable sort: priority desc, declaration order preserved within.
        self.rules = tuple(sorted(rules, key=lambda r: -r.priority))
        self.unmatched = unmatched

    def apply(
        self,
        model: nn.Module,
        *,
        verbose: bool = False,
        report_file: IO[str] | None = None,
    ) -> StructuralReport:
        """Walk ``model`` in place and replace every matching module.

        Args:
            model: Float model to transform.
            verbose: When ``True``, print a brief replacement summary to
                ``report_file``.
            report_file: Output stream (default ``sys.stderr``).

        Returns:
            :class:`StructuralReport` with per-rule replacement count
            and the list of eligible-but-unmatched modules.
        """
        report = StructuralReport()
        stream = report_file or sys.stderr

        def is_eligible_float(mod: nn.Module) -> bool:
            return isinstance(mod, (nn.Linear, nn.Conv2d)) and not isinstance(mod, NeuroxOperator)

        def recurse(parent: nn.Module, prefix: str, depth: int) -> None:
            # Snapshot children so in-place ``setattr`` does not break iteration.
            for name, child in list(parent.named_children()):
                qualified = f"{prefix}.{name}" if prefix else name
                ctx = ReplacementContext(
                    qualified_name=qualified,
                    parent=parent,
                    depth=depth,
                    metadata={},
                )
                rule = self._first_match(child, ctx)
                if rule is not None:
                    new_module = rule.build(child, ctx)
                    setattr(parent, name, new_module)
                    report.replaced.append((qualified, rule.name))
                    # Replaced subtree is opaque — do not descend.
                    continue
                if is_eligible_float(child):
                    report.unmatched.append(qualified)
                # Recurse into the (possibly un-replaced) subtree.
                recurse(child, qualified, depth + 1)

        recurse(model, "", 0)

        if self.unmatched != "keep_float" and report.unmatched:
            msg = (
                "policy: "
                + ("ERROR" if self.unmatched == "error" else "WARNING")
                + f" {len(report.unmatched)} eligible float module(s) unmatched: "
                + ", ".join(report.unmatched[:10])
                + ("..." if len(report.unmatched) > 10 else "")
            )
            if self.unmatched == "error":
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)

        if verbose:
            stream.write(
                f"replacement policy: {len(report.replaced)} replaced, {len(report.unmatched)} eligible-but-unmatched\n"
            )
            for q, rule_name in report.replaced:
                stream.write(f"  + {q}  ({rule_name})\n")
            for q in report.unmatched:
                stream.write(f"  - {q}  (unmatched float)\n")
            stream.flush()

        return report

    def _first_match(self, module: nn.Module, ctx: ReplacementContext) -> ReplacementRule | None:
        for rule in self.rules:
            if rule.match(module, ctx):
                return rule
        return None


# ---------------------------------------------------------------------------
# Default policy + helpers
# ---------------------------------------------------------------------------


def default_policy(
    macro_factory: Callable[..., NeuroxMacroQuantMatMul],
) -> ReplacementPolicy:
    """Default ``Linear -> QuantLinear``, ``Conv2d -> QuantConv2d`` policy.

    Args:
        macro_factory: Per-layer macro factory called as
            ``macro_factory(name=qualified_name)``.
    """
    return ReplacementPolicy(
        rules=[
            ReplacementRule(
                name="linear-to-quant",
                match=lambda mod, ctx: isinstance(mod, nn.Linear) and not isinstance(mod, QuantLinear),
                build=lambda mod, ctx: QuantLinear.from_torch(
                    mod, macro_factory(name=ctx.qualified_name), ctx.qualified_name
                ),
            ),
            ReplacementRule(
                name="conv2d-to-quant",
                match=lambda mod, ctx: isinstance(mod, nn.Conv2d) and not isinstance(mod, QuantConv2d),
                build=lambda mod, ctx: QuantConv2d.from_torch(
                    mod, macro_factory(name=ctx.qualified_name), ctx.qualified_name
                ),
            ),
        ],
        unmatched="keep_float",
    )


def name_excluded_match(
    patterns: str | Iterable[str],
) -> Callable[[nn.Module, ReplacementContext], bool]:
    """``match`` predicate rejecting qualified names that hit any glob pattern."""
    pat_list = [patterns] if isinstance(patterns, str) else list(patterns)

    def predicate(mod: nn.Module, ctx: ReplacementContext) -> bool:
        del mod
        return not any(fnmatch.fnmatch(ctx.qualified_name, p) for p in pat_list)

    return predicate


def by_attr_match(
    attr: str,
    test: Callable[[Any], bool],
) -> Callable[[nn.Module, ReplacementContext], bool]:
    """Build a ``match`` predicate based on a module attribute.

    Example: ``by_attr_match("out_features", lambda v: v >= 512)``
    matches any module whose ``out_features`` is at least 512.
    Returns ``False`` when the attribute is absent.
    """

    def predicate(mod: nn.Module, ctx: ReplacementContext) -> bool:
        del ctx
        if not hasattr(mod, attr):
            return False
        return bool(test(getattr(mod, attr)))

    return predicate


def heterogeneous_macro_policy(
    assignments: Mapping[str, Callable[..., NeuroxMacroQuantMatMul]],
    *,
    unmatched: Literal["keep_float", "warn", "error"] = "keep_float",
) -> ReplacementPolicy:
    """Map qualified-name prefixes to distinct macro factories.

    Args:
        assignments: ``{prefix: macro_factory}``; ``""`` is the catch-all
            default. Longer prefixes match first.
        unmatched: Behavior for modules not covered by any prefix.
    """
    rules: list[ReplacementRule] = []
    sorted_items = sorted(assignments.items(), key=lambda kv: -len(kv[0]))  # longest prefix first
    for prefix, factory in sorted_items:

        def make_match(p: str) -> Callable[[nn.Module, ReplacementContext], bool]:
            def matcher(mod: nn.Module, ctx: ReplacementContext) -> bool:
                if not isinstance(mod, (nn.Linear, nn.Conv2d)) or isinstance(mod, NeuroxOperator):
                    return False
                if p == "":
                    return True
                return ctx.qualified_name == p or ctx.qualified_name.startswith(p + ".")

            return matcher

        def make_build(
            f: Callable[..., NeuroxMacroQuantMatMul],
        ) -> Callable[[nn.Module, ReplacementContext], nn.Module]:
            def builder(mod: nn.Module, ctx: ReplacementContext) -> nn.Module:
                macro = f(name=ctx.qualified_name)
                if isinstance(mod, nn.Linear):
                    return QuantLinear.from_torch(mod, macro, ctx.qualified_name)
                if isinstance(mod, nn.Conv2d):
                    return QuantConv2d.from_torch(mod, macro, ctx.qualified_name)
                raise TypeError(f"unsupported eligible module: {type(mod).__name__}")

            return builder

        rules.append(
            ReplacementRule(
                name=f"prefix={prefix!r}",
                match=make_match(prefix),
                build=make_build(factory),
            )
        )
    return ReplacementPolicy(rules, unmatched=unmatched)
