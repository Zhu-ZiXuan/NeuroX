"""MNIST data loading for the LeNet example."""

from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def mnist_transform() -> transforms.Compose:
    """Standard MNIST eval/train transform: ToTensor + per-channel normalize."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def create_mnist_dataloader(
    dataset_dir: Path,
    batch_size: int,
    device: torch.device,
    *,
    split: str = "val",
    shuffle: bool = False,
    download: bool = True,
    indices: Sequence[int] | None = None,
) -> DataLoader:
    """Create a MNIST dataloader rooted at `dataset_dir`.

    Args:
        dataset_dir: Root directory holding `MNIST/raw/`, or the directory the
            corpus is downloaded into.
        batch_size: Samples per batch; the trailing batch may be short.
        device: Runtime device, consulted only to decide `pin_memory`.
        split: `"train"` selects the training split, anything else the test
            split, which serves as validation here.
        shuffle: Reshuffles the split on every epoch.
        download: Fetches the corpus on first use when it is absent.
        indices: Subset indices for sharded evaluation.

    Returns:
        DataLoader yielding `(images, targets)` pairs.
    """
    dataset = datasets.MNIST(
        root=str(dataset_dir),
        train=split == "train",
        download=download,
        transform=mnist_transform(),
    )
    if indices is not None:
        dataset = Subset(dataset, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
