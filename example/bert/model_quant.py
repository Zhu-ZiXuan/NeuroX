"""BERT-small quantization model helpers — in-place ``nn.Linear`` swap.

Two helpers walk the HuggingFace ``BertForSequenceClassification`` module
tree and replace every ``nn.Linear`` (Q/K/V/output attention projections,
FFN intermediate/output, pooler, classifier — 26 in total for BERT-small)
with either :class:`QATLinear` (training) or :class:`QuantLinear`
(inference). LayerNorm, GELU, embeddings, attention softmax stay in
float — those are not in the macro's contract.
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from neurox.architecture.unit import LinearUnit

from .quant import QATLinear, QuantLinear

MacroFactory = Callable[..., LinearUnit]
ModePicker = Callable[[str], int]


def _replace_linear(model: nn.Module, builder: Callable[[str, nn.Linear], nn.Module]) -> int:
    """Walk module tree; replace every ``nn.Linear`` via ``builder``. Returns count."""
    n_replaced = 0
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and not isinstance(child, (QATLinear, QuantLinear)):
                qualified = f"{parent_name}.{child_name}" if parent_name else child_name
                setattr(parent, child_name, builder(qualified, child))
                n_replaced += 1
    return n_replaced


def to_qat(model: nn.Module) -> int:
    """In-place: swap every ``nn.Linear`` in ``model`` for :class:`QATLinear`.

    Float weights are carried over by re-pointing the QAT layer's
    ``weight`` / ``bias`` Parameter objects at the original tensors.
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
    """In-place: swap every ``nn.Linear`` for :class:`QuantLinear` bound to a macro.

    ``mode_picker``: int constant or callable ``qualified_name → mode``.
    Default 0 routes every layer through quantization mode 0. BERT-small
    linears fully fill 64-row xbar tiles so a single mode is reasonable;
    refine per layer by passing a custom picker if eval shows over-rescale.
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
