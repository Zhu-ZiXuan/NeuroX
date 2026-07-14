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
    """Create a MNIST dataloader rooted at ``dataset_dir``.

    Args:
        dataset_dir: Root directory under which ``MNIST/raw/`` lives (or
            will be downloaded to when ``download=True``).
        device: Runtime device (used only for ``pin_memory``).
        split: ``"train"`` for the training split; anything else selects
            the test split used as validation.
        download: If ``True`` and the dataset is not found under
            ``dataset_dir``, torchvision downloads it on first use.
        indices: Optional subset indices for sharded evaluation.

    Returns:
        DataLoader over the requested MNIST split.
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
