"""LeNet pure-QAT training. No macro in the forward.

Stages:
1. Load the pretrained float LeNet5 state_dict.
2. Build :class:`QATLeNet5` (every weight layer becomes a fake-quant variant
   with observers attached). Copy float weights over.
3. Calibration: forward N batches in train mode without gradients to settle
   the input / weight / output observers, then freeze them so subsequent
   training steps don't move the qparams.
4. Fine-tune with optional KD from a frozen float teacher. Forward is
   plain ``F.linear`` / ``F.conv2d`` under fake-quant; backward is STE.
5. Save the per-layer flat state dict (weight_int + scales + zero-points,
   bias kept as float) under ``--checkpoint``. The schema is consumed by
   :class:`QuantLeNet5` at deployment.
"""

# ruff: noqa: T201

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from example.lenet.data import create_mnist_dataloader
from example.lenet.model_float import LeNet5
from example.lenet.model_quant import QATLeNet5
from example.lenet.quant import (
    X_QMAX,
    X_QMIN,
    W_QMAX,
    Y_QMAX,
    Y_QMIN,
    export_qat_state,
    freeze_observers,
)

QAT_SCHEMA = "lenet_qat_v1"


def _validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Top-1 accuracy of ``model`` on ``loader`` (eval mode)."""
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


def _copy_float_weights(qat_model: QATLeNet5, float_state: dict[str, torch.Tensor]) -> None:
    """Load float LeNet5 weights into the QAT model (matching parameter names)."""
    # QATConv2d / QATLinear are subclasses of nn.Conv2d / nn.Linear so the
    # parameter names ('weight', 'bias') align. Observer buffers are added
    # automatically and don't appear in the float state.
    missing, unexpected = qat_model.load_state_dict(float_state, strict=False)
    obs_buffers = {n for n in missing if "observer" in n}
    real_missing = [n for n in missing if n not in obs_buffers]
    if real_missing:
        raise RuntimeError(f"float checkpoint is missing weight layers: {real_missing}")
    if unexpected:
        raise RuntimeError(f"float checkpoint has stray keys: {unexpected}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LeNet-5 pure-QAT training (no macro in forward)")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="MNIST root directory")
    parser.add_argument(
        "--float-checkpoint", type=Path, required=True, help="Pretrained float state_dict from train_float.py"
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Output QAT checkpoint path")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=64,
        help="Forward-only train-mode batches before training so observers settle",
    )
    parser.add_argument("--kd-alpha", type=float, default=0.15, help="Hard-label CE weight; (1-α) → KL(student||teacher)")
    parser.add_argument("--kd-temperature", type=float, default=4.0)
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- 1-2. Load float weights into QAT student ---
    float_state = torch.load(args.float_checkpoint, map_location="cpu", weights_only=True)
    model = QATLeNet5()
    _copy_float_weights(model, float_state)
    model = model.to(device)
    print(f"Loaded float checkpoint: {args.float_checkpoint}")
    print(f"QAT grid: x[{X_QMIN},{X_QMAX}]  w[±{W_QMAX}]  y[{Y_QMIN},{Y_QMAX}]")

    teacher: nn.Module | None = None
    if args.kd_alpha < 1.0:
        teacher = LeNet5()
        teacher.load_state_dict(float_state)
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"KD: alpha_ce={args.kd_alpha}  T={args.kd_temperature}")

    train_loader = create_mnist_dataloader(args.dataset_dir, args.batch_size, device, split="train", shuffle=True)
    val_loader = create_mnist_dataloader(args.dataset_dir, args.batch_size, device, split="val")

    # --- 3. Calibration ---
    if args.calibration_batches > 0:
        print(f"Calibrating {args.calibration_batches} batches (no-grad train-mode)...")
        model.train()
        with torch.no_grad():
            for i, (images, _) in enumerate(train_loader):
                if i >= args.calibration_batches:
                    break
                model(images.to(device, non_blocking=True))
    n_frozen = freeze_observers(model)
    print(f"Froze {n_frozen} observers")

    # --- 4. Fine-tune ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
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
            ce = criterion(output, targets)
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                T = args.kd_temperature
                kd = F.kl_div(
                    F.log_softmax(output / T, dim=-1),
                    F.softmax(teacher_logits / T, dim=-1),
                    reduction="batchmean",
                ) * (T * T)
                loss = args.kd_alpha * ce + (1.0 - args.kd_alpha) * kd
            else:
                loss = ce
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
            f"lr={scheduler.get_last_lr()[0]:.6f}{marker}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best QAT val_acc: {best_acc:.4f}")

    # --- 5. Save per-layer flat state ---
    layer_state = export_qat_state(model)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": QAT_SCHEMA,
            "grid": {"x_qmin": X_QMIN, "x_qmax": X_QMAX, "w_qmax": W_QMAX, "y_qmin": Y_QMIN, "y_qmax": Y_QMAX},
            "layers": layer_state,
        },
        args.checkpoint,
    )
    print(f"Saved QAT checkpoint ({len(layer_state)} layers) → {args.checkpoint}")


if __name__ == "__main__":
    main()
