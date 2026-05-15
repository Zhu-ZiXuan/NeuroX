"""LeNet-5 architecture for MNIST.

Plain ``nn.Module`` with no quantization-specific scaffolding — the
PT2E quantization flow (``torchao.quantization.pt2e``) operates on the
exported FX graph, so ``QuantStub`` / ``DeQuantStub`` and explicit
``fuse_modules_qat`` calls aren't needed; the quantizer pattern-matches
``Conv -> ReLU`` / ``Linear -> ReLU`` directly on the graph.

Input spec:
    Shape: ``[N, 1, 28, 28]`` (standard MNIST).  The first conv uses
    ``padding=2`` to recover the classic LeNet-5 32x32 input feature map
    after the initial padding.
"""

import torch.nn as nn
from torch import Tensor


class LeNet5(nn.Module):
    """Classic LeNet-5 for MNIST.

    Layer stack:
        conv1 (1 -> 6, 5x5, pad 2) -> relu -> maxpool 2x2
        conv2 (6 -> 16, 5x5)       -> relu -> maxpool 2x2
        flatten
        fc1 (400 -> 120) -> relu
        fc2 (120 -> 84)  -> relu
        fc3 (84  -> num_classes)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x
