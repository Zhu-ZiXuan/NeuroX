"""Hand-built crossbar LeNet-5 — uses local ``quant.py`` only.

Two model variants, one shared architecture:

- ``QATLeNet5``: training-time. Every ``nn.Conv2d`` / ``nn.Linear`` is a
  local ``QATConv2d`` / ``QATLinear`` (fake-quant + observers, no macro).
  Suitable for ``train_quant.py`` standard QAT.
- ``QuantLeNet5``: inference-time. Every weight layer is a local
  ``QuantConv2d`` / ``QuantLinear`` bound to a macro. Per-layer ADC mode
  is hard-wired in :func:`_LAYER_MODE` below — picked manually so each
  layer's expected ``max|dot|`` lands within the chip's analog range
  (see ``log/lenet_rescale_audit.log`` for the underlying numbers).

The training → inference handoff is a flat ``{layer_name: layer_state}``
dict produced by ``quant.export_qat_state`` and consumed by
``QuantLeNet5.from_qat_state``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn
from torch import Tensor

from neurox.macro import NeuroxMacroQuantMatMul

from .quant import QATConv2d, QATLinear, QuantConv2d, QuantLinear

MacroFactory = Callable[..., NeuroxMacroQuantMatMul]

# Per-layer ADC operating-mode pick. Indices match the chip preset's
# ``v_refs__V`` list (currently [0.8, 0.4, 0.2]); add 0.1 / 0.05 entries
# and bump fc3 to index 4 once #276 lands. Picked to match each layer's
# expected max|dot| under ternary weights (see log/lenet_rescale_audit.log):
# conv1 / fc1: max|dot|≈16–17 → mode 1 (v_ref=0.4) gives r_ADC≈1.7
# conv2:       max|dot|≈24    → mode 0 (v_ref=0.8) gives r_ADC≈3.5
# fc2:         max|dot|≈10    → mode 2 (v_ref=0.2)
# fc3:         max|dot|≈2     → mode 2 today (still over-rescaled;
#                              waiting on smaller v_ref mode).
_LAYER_MODE: dict[str, int] = {
    # Picked for the post-solver-fix mid1 chip: v_diff p99 ≈ 0.025 V across
    # the synthetic uniform workload (see log/adc_stat/), so the smallest
    # v_ref mode (mode 4, v_ref = 0.05 V → LSB ≈ 0.4 mV) is the one that
    # gives non-trivial bits for typical activations. The larger modes
    # (0.1, 0.2, 0.4, 0.8 V) give 1–2 LSB resolution and round signal
    # back to noise. ideal_xbar TOMLs ignore the mode and use their own
    # synthetic full-range ADC.
    "conv1": 4,
    "conv2": 4,
    "fc1": 4,
    "fc2": 4,
    "fc3": 4,
}


# ---------------------------------------------------------------------------
# Training-time architecture
# ---------------------------------------------------------------------------


class QATLeNet5(nn.Module):
    """LeNet-5 with every weight layer replaced by its QAT (fake-quant) variant."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = QATConv2d(1, 6, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = QATConv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = QATLinear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = QATLinear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = QATLinear(84, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x


# ---------------------------------------------------------------------------
# Inference-time architecture
# ---------------------------------------------------------------------------


def _conv_macro(
    factory: MacroFactory, name: str, out_channels: int, in_channels: int, kernel_size: int
) -> NeuroxMacroQuantMatMul:
    """Build a macro shaped for this conv layer's unfolded matmul."""
    return factory(name=name, w_logical_shape=(out_channels, in_channels * kernel_size * kernel_size))


def _linear_macro(factory: MacroFactory, name: str, out_features: int, in_features: int) -> NeuroxMacroQuantMatMul:
    return factory(name=name, w_logical_shape=(out_features, in_features))


class QuantLeNet5(nn.Module):
    """LeNet-5 with every weight layer replaced by its macro-backed Quant variant.

    Built from a flat state dict (produced by :class:`QATLeNet5` via
    ``quant.export_qat_state``) plus a macro factory. ``adc_mode`` per layer
    comes from :data:`_LAYER_MODE`.
    """

    def __init__(
        self,
        macro_factory: MacroFactory,
        layer_state: dict[str, dict],
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.conv1 = QuantConv2d.from_state(
            macro=_conv_macro(macro_factory, "conv1", 6, 1, 5),
            state=layer_state["conv1"],
            adc_mode=_LAYER_MODE["conv1"],
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = QuantConv2d.from_state(
            macro=_conv_macro(macro_factory, "conv2", 16, 6, 5),
            state=layer_state["conv2"],
            adc_mode=_LAYER_MODE["conv2"],
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = QuantLinear.from_state(
            macro=_linear_macro(macro_factory, "fc1", 120, 16 * 5 * 5),
            state=layer_state["fc1"],
            adc_mode=_LAYER_MODE["fc1"],
        )
        self.relu3 = nn.ReLU()
        self.fc2 = QuantLinear.from_state(
            macro=_linear_macro(macro_factory, "fc2", 84, 120),
            state=layer_state["fc2"],
            adc_mode=_LAYER_MODE["fc2"],
        )
        self.relu4 = nn.ReLU()
        self.fc3 = QuantLinear.from_state(
            macro=_linear_macro(macro_factory, "fc3", num_classes, 84),
            state=layer_state["fc3"],
            adc_mode=_LAYER_MODE["fc3"],
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x
