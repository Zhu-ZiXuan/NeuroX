"""BERT-small quantization model helpers — in-place `nn.Linear` swap.

Two helpers walk the HuggingFace `BertForSequenceClassification` module tree
and replace every `nn.Linear` (Q/K/V/output attention projections, FFN
intermediate/output, pooler, classifier — 26 in total for BERT-small) with
either `QATLinear` for training or `QuantLinear` for inference. LayerNorm,
GELU, embeddings and attention softmax stay in float, outside the macro's
contract.
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from neurox.architecture.unit.cim import CimUnit, CimUnitConfig, CimUnitPolicy

from .quant import QATLinear, QuantLinear

MacroFactory = Callable[..., CimUnit[CimUnitConfig, CimUnitPolicy]]
ModePicker = Callable[[str], int]


def _replace_linear(model: nn.Module, builder: Callable[[str, nn.Linear], nn.Module]) -> int:
    """Walk the module tree, replace every `nn.Linear` via `builder`, count the swaps."""
    n_replaced = 0
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and not isinstance(child, (QATLinear, QuantLinear)):
                qualified = f"{parent_name}.{child_name}" if parent_name else child_name
                setattr(parent, child_name, builder(qualified, child))
                n_replaced += 1
    return n_replaced


def to_qat(model: nn.Module) -> int:
    """In-place: swap every `nn.Linear` in `model` for `QATLinear`.

    The QAT layer's `weight` / `bias` Parameter objects are re-pointed at the
    original tensors, so the float values are shared rather than copied.

    Returns:
        Number of layers replaced.
    """

    def build(qualified: str, original: nn.Linear) -> nn.Module:
        del qualified  # name not needed for QAT layer state
        qat = QATLinear(original.in_features, original.out_features, bias=original.bias is not None)
        qat.weight = original.weight
        if original.bias is not None:
            qat.bias = original.bias
        return qat.to(original.weight.device)

    return _replace_linear(model, build)


def to_quant(
    model: nn.Module,
    layer_state: dict[str, dict],
    macro_factory: MacroFactory,
    mode_picker: ModePicker | int = 0,
) -> int:
    """In-place: swap every `nn.Linear` for `QuantLinear` bound to a macro.

    Args:
        model: Module tree whose linears are replaced.
        layer_state: Exported QAT state keyed by qualified layer name.
        macro_factory: Builds one unit per layer from its logical weight shape.
        mode_picker: Quantization-mode index, or a callable
            `qualified_name → mode` when the layers need different windows.

    Returns:
        Number of layers replaced.

    Raises:
        KeyError: A replaced layer has no entry in `layer_state`.
    """
    pick = mode_picker if callable(mode_picker) else (lambda _name: mode_picker)

    def build(qualified: str, original: nn.Linear) -> nn.Module:
        if qualified not in layer_state:
            raise KeyError(f"layer_state has no entry for {qualified!r}")
        macro = macro_factory(
            w_logical_shape=(original.out_features, original.in_features),
        )
        return QuantLinear.from_state(
            macro=macro,
            state=layer_state[qualified],
            quantization_mode=pick(qualified),
        ).to(original.weight.device)

    return _replace_linear(model, build)
