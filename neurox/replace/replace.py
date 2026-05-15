"""Model conversion API for NeuroX crossbar inference.

The replacement subsystem is **staged**: ``build_evaluator`` is a thin
wrapper around four public building blocks that users can also call
individually for staged or selective conversion flows.

Stages
------
1. :func:`replace_model` — structurally swap matched float modules for
   NeuroX crossbar operators using a :class:`ReplacementPolicy` (default:
   every ``nn.Linear`` → :class:`QuantLinear`, every ``nn.Conv2d`` →
   :class:`QuantConv2d`).  No state loading, no fabrication.
2. :func:`load_neurox_state` — load a NeuroX-flat checkpoint into the
   replaced model.  Strict by default: missing required NeuroX operator
   buffers raise :class:`NeuroxStateError`.
3. :func:`bind_output_calibration` — fold each macro's
   ``output_rescale_factor`` into the freshly-loaded
   ``(rescale_multiplier, rescale_rshift, bias_int)`` so one
   checkpoint serves any macro backend.
4. :func:`fabricate_model` — call ``fabricate()`` on every
   :class:`NeuroxOperator` so each macro programs its physical
   device state from the integer weights.

:func:`build_evaluator` orchestrates the four stages in order and
preserves the one-call user surface.

Example::

    import neurox
    model = torchvision_like_float_model()
    model = neurox.build_evaluator(model, "checkpoint.pth", xbar1t1r_macro_factory)
    model.eval()
    logits = model(images)
"""

import sys
import warnings
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import IO, TYPE_CHECKING

import torch
import torch.nn as nn

from neurox.macro.base import NeuroxMacroQuantMatMul
from neurox.operator import (
    NeuroxOperator,
    QuantConv2d,
    QuantLinear,
)
from neurox.operator.qat_util import derive_multiplier_and_shift_tensor

from .state import NeuroxStateError, StateBindingReport

if TYPE_CHECKING:
    from .policy import ReplacementPolicy

_NEUROX_FLAT_SCHEMA = "neurox_flat"

# Per-layer keys every ``QuantLinear`` / ``QuantConv2d`` requires.
# Order is irrelevant; the names match each operator's registered buffers.
_REQUIRED_OP_STATE_KEYS: tuple[str, ...] = (
    "weight_int",
    "bias_int",
    "rescale_multiplier",
    "rescale_rshift",
    "output_zero_point",
    "output_qmin",
    "output_qmax",
    "input_scale",
    "input_zero_point",
    "input_qmin",
    "input_qmax",
    "output_scale",
)


def count_unreplaced_ops(model: nn.Module) -> dict[str, int]:
    """Tabulate every leaf ``nn.Module`` *not* covered by NeuroX operators.

    Walks the module tree but treats every ``NeuroxOperator`` subtree as
    opaque — analog macro internals (RRAM, NMOS, ADC, etc.) are
    deliberately *not* counted, since they belong to ops we did replace.
    Everything that survives a replace pass and isn't inside a
    ``Quant*`` operator shows up here: typically activation functions
    (ReLU, GELU), normalisers (LayerNorm), pooling (MaxPool2d),
    embeddings, dropouts, etc.  These modules still run in float and
    consume no analog energy.

    Args:
        model: Replaced or unreplaced model.

    Returns:
        ``{type_name: count}`` sorted by descending count.
    """
    counts: Counter[str] = Counter()

    def walk(module: nn.Module) -> None:
        children = list(module.children())
        if not children:
            counts[type(module).__name__] += 1
            return
        for child in children:
            # Treat every replaced operator as opaque — its internals
            # are macro primitives (RRAM/NMOS/ADC), not unreplaced ops.
            if isinstance(child, NeuroxOperator):
                continue
            walk(child)

    walk(model)
    return dict(counts.most_common())


def _format_table(counts: dict[str, int], header: str) -> str:
    """Render a 2-column ``type | count`` table for the report."""
    if not counts:
        return f"{header}: (none)\n"
    width = max(len(name) for name in counts)
    lines = [header, "-" * (width + 12)]
    lines.extend(f"  {name:<{width}}   {n:>5}" for name, n in counts.items())
    lines.append("-" * (width + 12))
    lines.append(f"  {'total':<{width}}   {sum(counts.values()):>5}")
    return "\n".join(lines) + "\n"


def report_replacement(
    model: nn.Module,
    *,
    before_str: str | None = None,
    file: IO[str] | None = None,
) -> None:
    """Print a before/after replacement report to ``file`` (default stdout).

    Args:
        model: The model *after* ``_replace`` has run.
        before_str: ``str(model)`` captured before the replace pass; pass
            ``None`` to skip the "before" block.
        file: Output stream (default ``sys.stdout``).  The full ``str(model)``
            is verbose for big graphs — pipe to a log file when noisy.
    """
    out = file if file is not None else sys.stdout
    if before_str is not None:
        out.write("=" * 72 + "\n")
        out.write("MODEL STRUCTURE  ──  before replacement\n")
        out.write("=" * 72 + "\n")
        out.write(before_str + "\n\n")
    out.write("=" * 72 + "\n")
    out.write("MODEL STRUCTURE  ──  after replacement\n")
    out.write("=" * 72 + "\n")
    out.write(str(model) + "\n\n")
    out.write("=" * 72 + "\n")
    out.write("REPLACEMENT SUMMARY\n")
    out.write("=" * 72 + "\n")
    counts = count_xbar_layers(model)
    replaced = {
        original: counts[neurox]
        for original, neurox in (("Linear", "NeuroxLinear"), ("Conv2d", "NeuroxConv2d"))
        if counts[neurox] > 0
    }
    out.write(_format_table(replaced, "Replaced operators (now run on NeuroX macros)"))
    out.write("\n")
    out.write(
        _format_table(
            count_unreplaced_ops(model),
            "Unreplaced leaf ops (still run in float — no analog energy)",
        )
    )
    out.write("=" * 72 + "\n\n")
    out.flush()


def _replace(
    model: nn.Module,
    macro_factory: Callable[..., NeuroxMacroQuantMatMul],
) -> nn.Module:
    """Recursively swap ``nn.Linear`` / ``nn.Conv2d`` for crossbar operators.

    Args:
        model: PyTorch model to transform.
        macro_factory: Callable that returns a fresh macro instance per layer.
            Accepts ``name=<qualified module name>`` so the macro and
            every nested physical module receive a hierarchical
            profiler identity (e.g. ``fc1.macro.xbar.core.tia``).

    Returns:
        The same model, mutated in place.
    """

    def recursive_replace(module: nn.Module, prefix: str) -> None:
        for name, child in list(module.named_children()):
            qualified = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not isinstance(child, QuantLinear):
                setattr(module, name, QuantLinear.from_torch(child, macro_factory(name=qualified), qualified))
            elif isinstance(child, nn.Conv2d) and not isinstance(child, QuantConv2d):
                setattr(module, name, QuantConv2d.from_torch(child, macro_factory(name=qualified), qualified))
            else:
                recursive_replace(child, qualified)

    recursive_replace(model, "")
    return model


def fabricate_model(model: nn.Module) -> None:
    """Fabricate every crossbar operator's physical state in ``model``.

    Programs every :class:`NeuroxOperator`'s macro from its loaded
    integer weights.  Called automatically by :func:`build_evaluator`
    after :func:`load_neurox_state` and :func:`bind_output_calibration`.
    """
    for module in model.modules():
        if isinstance(module, NeuroxOperator):
            module.fabricate()


def fabricate(model: nn.Module) -> None:
    """Deprecated alias for :func:`fabricate_model`."""
    warnings.warn(
        "neurox.replace.fabricate is deprecated; use neurox.replace.fabricate_model instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    fabricate_model(model)


def replace_model(
    model: nn.Module,
    macro_factory: Callable[..., NeuroxMacroQuantMatMul],
    *,
    policy: "ReplacementPolicy | None" = None,
    verbose: bool = True,
    report_file: IO[str] | None = None,
) -> nn.Module:
    """Structurally replace matched float modules with NeuroX operators.

    When ``policy`` is ``None`` the default rule set replaces every
    ``nn.Linear`` with :class:`QuantLinear` and every ``nn.Conv2d``
    with :class:`QuantConv2d`, bound to fresh macros from
    ``macro_factory(name=qualified_module_name)``.  Provide a
    :class:`~neurox.replace.policy.ReplacementPolicy` for partial
    replacement, heterogeneous macros, or predicate-based filtering.

    Args:
        model: Float model to transform in place.
        macro_factory: Callable returning a fresh macro per replaced
            layer.  Receives ``name=`` so each macro's physical leaves
            carry a hierarchical profiler identity.
        policy: Optional :class:`ReplacementPolicy`.  Default rules
            preserve the legacy ``Linear`` / ``Conv2d`` behavior.
        verbose: When ``True``, print a before/after structural report.
        report_file: Output stream for the report (default ``sys.stdout``).

    Returns:
        The same model object, mutated in place.
    """
    before_str = str(model) if verbose else None
    if policy is None:
        _replace(model, macro_factory)
    else:
        policy.apply(model, verbose=False)
    if verbose:
        report_replacement(model, before_str=before_str, file=report_file)
    return model


def load_neurox_state(
    model: nn.Module,
    checkpoint: str | Path | dict[str, object],
    *,
    strict: bool = True,
) -> StateBindingReport:
    """Load a NeuroX-flat checkpoint into a replaced model.

    Strict by default: every :class:`QuantLinear` / :class:`QuantConv2d`
    must receive its full 12-buffer slice (``weight_int``, ``bias_int``,
    rescale + zero-point + scale + qmin/qmax).  Missing keys raise
    :class:`NeuroxStateError`; unexpected keys are passed through
    ``load_state_dict(strict=False)`` so float-side buffers (BN
    parameters, biases on float layers, …) keep loading.

    Args:
        model: Model already processed by :func:`replace_model`.
        checkpoint: Path to a ``neurox_flat`` checkpoint or a pre-loaded
            ``{"schema": "neurox_flat", "state_dict": {...}, ...}`` dict.
        strict: When ``True``, raise :class:`NeuroxStateError` if any
            replaced operator is missing required buffers.  When
            ``False`` (relaxed), the missing keys land in the returned
            report so callers can decide.

    Returns:
        :class:`StateBindingReport` with the per-call diagnostic.

    Raises:
        ValueError: When the checkpoint schema is absent or unknown.
        NeuroxStateError: In strict mode when required keys are missing.
    """
    ckpt = _load_checkpoint(checkpoint)
    schema = ckpt.get("schema")
    if schema != _NEUROX_FLAT_SCHEMA:
        raise ValueError(
            f"unsupported checkpoint schema: {schema!r}; expected {_NEUROX_FLAT_SCHEMA!r}. "
            "Use example.common.pt2e.pt2e_to_neurox_state or neurox.extract_neurox_state to produce one."
        )
    state_dict = ckpt.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint['state_dict'] missing or not a dict")

    # Build the required-key set from the replaced model and compare.
    required: set[str] = set()
    loaded: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, (QuantLinear, QuantConv2d)):
            loaded.append(name)
            for suffix in _REQUIRED_OP_STATE_KEYS:
                required.add(f"{name}.{suffix}")
    present = required & state_dict.keys()
    missing = sorted(required - state_dict.keys())
    # Float-only buffers (BN, dropout, etc.) typically live outside the
    # required set; treat any extra key that targets a non-NeuroX module
    # as "unexpected" only when the model has no matching parameter.
    model_keys = set(model.state_dict().keys())
    unexpected = sorted(state_dict.keys() - model_keys)

    if strict and missing:
        raise NeuroxStateError(
            f"checkpoint is missing {len(missing)} required NeuroX operator buffers; first few: {missing[:5]}",
            missing=missing,
            unexpected=unexpected,
        )

    model.load_state_dict(state_dict, strict=False)
    return StateBindingReport(
        loaded=loaded,
        missing=missing,
        unexpected=unexpected,
    )


def bind_output_calibration(model: nn.Module) -> None:
    """Fold each macro's ``output_rescale_factor`` into the loaded buffers.

    The checkpoint is extracted with ``rescale_factor=1.0`` (macro-
    agnostic).  After :func:`load_neurox_state`, this function
    re-derives ``(rescale_multiplier, rescale_rshift, bias_int)`` by
    multiplying the original ``(sx * sw / sy)`` scale by the target
    macro's ``rf``.  One checkpoint works with any macro backend.

    Operators whose macro has ``rf == 1.0`` (e.g. :class:`IdealMacro` or
    a tile whose codes already cover the ideal state range) are
    skipped — no correction needed.
    """
    for module in model.modules():
        if not isinstance(module, (QuantLinear, QuantConv2d)):
            continue
        rf = module.macro.output_rescale_factor
        if rf == 1.0:
            continue

        sx = module.input_scale.to(torch.float64)
        sy = module.output_scale.to(torch.float64)

        # The old (mult, rshift) encoded ``(sx*sw/sy)`` with ``rf=1``;
        # ``old_scale = mult / 2^rshift``; new scale = ``old_scale * rf``.
        old_mult = module.rescale_multiplier.to(torch.float64)
        old_rshift = module.rescale_rshift.to(torch.float64)
        old_scale = old_mult / (2.0**old_rshift)
        new_scale = old_scale * rf

        new_mult, new_rsh = derive_multiplier_and_shift_tensor(new_scale.to(torch.float32))
        module.rescale_multiplier.copy_(new_mult)
        module.rescale_rshift.copy_(new_rsh)

        # Bias was folded with ``rf=1``; rescale by ``/rf`` and re-round.
        int32_info = torch.iinfo(torch.int32)
        module.bias_int.copy_(
            torch.round(module.bias_int.to(torch.float64) / rf)
            .clamp(min=int32_info.min, max=int32_info.max)
            .to(torch.int32)
        )

        # Quiet the unused references the linter sees in some
        # configurations — ``sx`` / ``sy`` are part of the documented
        # derivation even though the implementation above recovers the
        # combined scale directly from the stored ``(mult, rshift)``.
        del sx, sy


def build_evaluator(
    model: nn.Module,
    checkpoint: str | Path | dict[str, object],
    macro_factory: Callable[..., NeuroxMacroQuantMatMul],
    *,
    policy: "ReplacementPolicy | None" = None,
    verbose: bool = True,
    report_file: IO[str] | None = None,
) -> nn.Module:
    """Build a crossbar-backed inference model in one call.

    Thin orchestrator over the four staged functions:

    1. :func:`replace_model`
    2. :func:`load_neurox_state` (strict)
    3. :func:`bind_output_calibration`
    4. :func:`fabricate_model`

    Args:
        model: The original float model (same architecture used during QAT).
        checkpoint: Path to a ``neurox_flat`` checkpoint file, or a
            pre-loaded checkpoint dict with ``"schema": "neurox_flat"`` and
            a ``"state_dict"`` entry.
        macro_factory: Per-layer macro factory (accepts ``name=``).
        policy: Optional replacement policy; default replaces every
            ``nn.Linear`` / ``nn.Conv2d``.
        verbose: Print before/after model structure.
        report_file: Stream for the verbose report; defaults to ``sys.stdout``.

    Returns:
        The (same) model with operators replaced, state loaded,
        calibration bound, and physical state fabricated.

    Raises:
        ValueError: If the checkpoint schema is unknown or missing.
        NeuroxStateError: If required operator buffers are missing.
    """
    replace_model(model, macro_factory, policy=policy, verbose=verbose, report_file=report_file)
    load_neurox_state(model, checkpoint, strict=True)
    bind_output_calibration(model)
    fabricate_model(model)
    return model


def _load_checkpoint(checkpoint: str | Path | dict[str, object]) -> dict[str, object]:
    """Coerce ``checkpoint`` to a dict, loading from disk if a path is given."""
    if isinstance(checkpoint, (str, Path)):
        # weights_only=False because the payload is a plain dict containing
        # tensor values and Python primitives (schema string, qat_config).
        return torch.load(checkpoint, weights_only=False, map_location="cpu")
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"expected path or dict checkpoint, got {type(checkpoint).__name__}")


def count_xbar_layers(model: nn.Module) -> dict[str, int]:
    """Count standard and NeuroX layers in ``model``.

    Useful for verifying a replace pass covered the whole network.
    """
    counts = {"Linear": 0, "Conv2d": 0, "NeuroxLinear": 0, "NeuroxConv2d": 0}

    for module in model.modules():
        if isinstance(module, QuantLinear):
            counts["NeuroxLinear"] += 1
        elif isinstance(module, nn.Linear):
            counts["Linear"] += 1
        elif isinstance(module, QuantConv2d):
            counts["NeuroxConv2d"] += 1
        elif isinstance(module, nn.Conv2d):
            counts["Conv2d"] += 1

    return counts
