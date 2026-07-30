"""LeNet-5 architecture for MNIST.

Plain ``nn.Module``; the PT2E quantization flow
(``torchao.quantization.pt2e``) operates on the exported FX graph.

Input spec:
    Standard MNIST; the first conv uses ``padding=2`` to recover the classic
    LeNet-5 32×32 input feature map.
"""

import torch.nn as nn
from torch import Tensor


class LeNet5(nn.Module):
    """Classic LeNet-5 for MNIST.

    Layer stack:
        conv1 (1 -> 6, 5×5, pad 2) -> relu -> maxpool 2×2
        conv2 (6 -> 16, 5×5)       -> relu -> maxpool 2×2
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
