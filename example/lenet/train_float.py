"""Floating-point training of LeNet-5 on MNIST.

Trains from random init with SGD + momentum + cosine LR schedule, keeps
the best-validation checkpoint, and writes its ``state_dict`` to
``--checkpoint``.  The resulting file is the pretrained starting point
consumed by ``example/lenet/qat.py``.
"""

# ruff: noqa: T201

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from example.lenet.data import create_mnist_dataloader
from example.lenet.model import LeNet5


def _validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return top-1 accuracy of ``model`` on ``loader``."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            correct += model(images).argmax(1).eq(targets).sum().item()
            total += targets.size(0)
    return correct / total if total > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LeNet-5 on MNIST (float)")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="MNIST root directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Output float state_dict path")
    parser.add_argument("--device", type=str, default="cuda:1", help="Torch device")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = LeNet5().to(device)

    train_loader = create_mnist_dataloader(args.dataset_dir, args.batch_size, device, split="train", shuffle=True)
    val_loader = create_mnist_dataloader(args.dataset_dir, args.batch_size, device, split="val")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            output = model(images)
            loss = criterion(output, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += output.argmax(1).eq(targets).sum().item()
            total += targets.size(0)

        scheduler.step()
        train_acc = correct / total
        val_acc = _validate(model, val_loader, device)

        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            marker = " *best*"

        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"loss={total_loss / len(train_loader):.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
            f"lr={scheduler.get_last_lr()[0]:.5f}{marker}"
        )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    final_state = best_state if best_state is not None else model.state_dict()
    torch.save(final_state, args.checkpoint)
    print(f"Saved best float model (val_acc={best_acc:.4f}) to {args.checkpoint}")


if __name__ == "__main__":
    main()
