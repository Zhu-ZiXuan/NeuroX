"""Hand-built crossbar LeNet-5: direct construction (no ``replace_*``).

Mirrors :class:`example.lenet.model.LeNet5` layer-for-layer, but every
``nn.Linear`` / ``nn.Conv2d`` is replaced at construction with its
crossbar-backed counterpart bound to a macro built by the supplied
factory. Activations / pooling / flatten stay in float.

Usage:
    factory = build_macro_factory(config_path, ideal_xbar=False)
    model = QuantLeNet5(factory)
    flat = torch.load(checkpoint)["state_dict"]
    model.load_state_dict(flat, strict=False)
    neurox.fabricate_model(model)
    neurox.program_model(model)
    model = model.to(device).eval()
"""

from collections.abc import Callable

import torch.nn as nn
from torch import Tensor

from neurox.macro.base import NeuroxMacroQuantMatMul
from neurox.operator import QuantConv2d, QuantLinear, w_logical_shape_for

MacroFactory = Callable[..., NeuroxMacroQuantMatMul]


def _conv(macro_factory: MacroFactory, name: str, conv: nn.Conv2d) -> QuantConv2d:
    macro = macro_factory(name=name, w_logical_shape=w_logical_shape_for(conv))
    return QuantConv2d.from_torch(conv, macro, name)


def _linear(macro_factory: MacroFactory, name: str, linear: nn.Linear) -> QuantLinear:
    macro = macro_factory(name=name, w_logical_shape=w_logical_shape_for(linear))
    return QuantLinear.from_torch(linear, macro, name)


class QuantLeNet5(nn.Module):
    """Crossbar-backed LeNet-5; constructed directly, not via ``replace_*``.

    Layer stack matches :class:`LeNet5`:
        conv1 (1 -> 6, 5x5, pad 2) -> relu -> maxpool 2x2
        conv2 (6 -> 16, 5x5)       -> relu -> maxpool 2x2
        flatten
        fc1 (400 -> 120) -> relu
        fc2 (120 -> 84)  -> relu
        fc3 (84  -> num_classes)

    The macro factory is consulted once per weight layer at ``__init__``
    using the qualified attribute name (``conv1``, ``fc3``, ...).
    """

    def __init__(self, macro_factory: MacroFactory, num_classes: int = 10) -> None:
        super().__init__()

        self.conv1 = _conv(macro_factory, "conv1", nn.Conv2d(1, 6, kernel_size=5, padding=2))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = _conv(macro_factory, "conv2", nn.Conv2d(6, 16, kernel_size=5))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()

        self.fc1 = _linear(macro_factory, "fc1", nn.Linear(16 * 5 * 5, 120))
        self.relu3 = nn.ReLU()
        self.fc2 = _linear(macro_factory, "fc2", nn.Linear(120, 84))
        self.relu4 = nn.ReLU()
        self.fc3 = _linear(macro_factory, "fc3", nn.Linear(84, num_classes))

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x
