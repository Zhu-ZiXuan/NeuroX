"""Hardware-aware QAT for LeNet-5 on MNIST.

End-to-end HAT pipeline (no pt2e):

1. Load the float ``state_dict`` produced by ``train.py``.
2. ``fold_batchnorm`` (no-op for LeNet, but on by default for future
   models that have BN).
3. ``replace_for_hat`` swaps every supported ``nn.Linear`` /
   ``nn.Conv2d`` for its HAT counterpart, bound to a fresh
   xbar macro instance per layer backed by the lossless
   :class:`IdealXbar` tile derived from the physical chip.  Other
   modules (ReLU, MaxPool, Flatten, ...) stay in float.
4. Fine-tune with the ideal macro in the forward pass — same ADC
   quant grid as the deployed hardware, no noise in the backward,
   fast enough to keep training practical.  Backward uses the
   float-reference gradient via STE so training is stable.
5. ``extract_neurox_state`` → save a NeuroX-flat checkpoint readable by
   ``neurox.build_evaluator``.

CLI knobs for hardware targeting: ``--config`` selects the chip
TOML, ``--xbar`` selects ``physical`` vs ``ideal``.  HAT defaults to
``--xbar ideal`` for speed; evaluation defaults to ``--xbar physical``
to mirror the deployed chip.
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

from example.common import build_macro_factory, derive_quant_spec
from example.lenet.data import create_mnist_dataloader
from example.lenet.model import LeNet5
from neurox import replace as neurox
from neurox.config import DEFAULT_1T1R_TOML


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Hardware-aware QAT for LeNet-5 on MNIST")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="MNIST root directory")
    parser.add_argument(
        "--float-checkpoint",
        type=Path,
        required=True,
        help="Pretrained float state_dict produced by example.lenet.train",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Output NeuroX-flat checkpoint path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for HAT fine-tuning")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_1T1R_TOML,
        help="Chip TOML (bundled default covers the reference 1T1R tile).",
    )
    parser.add_argument(
        "--xbar",
        choices=("physical", "ideal"),
        default="ideal",
        help=(
            "Tile implementation used as the HAT calculator.  ``ideal`` "
            "(recommended) is the lossless reference derived from the "
            "physical tile — same ADC grid, no noise in the backward, "
            "fast.  ``physical`` runs the full 1T1R circuit solver with "
            "noise for noise-aware training (much slower)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=4,
        help="Forward-only batches before the first optimizer step so observers settle",
    )
    # Knowledge distillation: a frozen float teacher (same architecture,
    # same init weights) supervises the quantised student.  With a 26-
    # total-layer 4-bit output grid the STE loss alone stalls around
    # 88 %; adding logit-KL from the float teacher pushes MNIST LeNet
    # back close to the 99 % float baseline.  ``--kd-alpha=1.0`` falls
    # back to pure hard-label CE (KD disabled).
    parser.add_argument(
        "--kd-alpha", type=float, default=0.3, help="Weight on hard-label CE; (1-alpha) goes to logit KL"
    )
    parser.add_argument("--kd-temperature", type=float, default=4.0, help="Softening temperature for logit KD")
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- 1. Load float weights ---
    float_state = torch.load(args.float_checkpoint, map_location="cpu", weights_only=True)
    model = LeNet5()
    model.load_state_dict(float_state)
    print(f"Loaded float checkpoint: {args.float_checkpoint}")

    # Frozen float teacher — same architecture + same init weights.
    teacher: nn.Module | None = None
    if args.kd_alpha < 1.0:
        teacher = LeNet5()
        teacher.load_state_dict(float_state)
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"KD: alpha_ce={args.kd_alpha}  T={args.kd_temperature}")

    # --- 2. BN fold (no-op for LeNet) ---
    neurox.fold_batchnorm(model)

    # --- 3. Replace supported ops with HAT counterparts ---
    # The two CLI knobs (``--config`` and ``--xbar``) fully specify the
    # hardware target: the TOML provides the quant grid, and the
    # chosen xbar kind selects between the physical 1T1R solver and
    # the lossless reference.  Default is ``ideal`` for speed.
    spec = derive_quant_spec(args.config)
    macro_factory = build_macro_factory(args.config, xbar=args.xbar)
    model = neurox.replace_for_hat(model, macro_factory, spec)
    model = model.to(device)
    print(
        f"HAT grid: x[{spec.x_qmin},{spec.x_qmax}]  w[±{spec.w_qmax}]  "
        f"y[{spec.y_qmin},{spec.y_qmax}]  xbar={args.xbar}  (from {args.config.name})"
    )
    from neurox.operator import HATConv2d, HATLinear

    n_lin = sum(1 for m in model.modules() if isinstance(m, HATLinear))
    n_conv = sum(1 for m in model.modules() if isinstance(m, HATConv2d))
    print(f"HAT layers: {n_lin} Linear, {n_conv} Conv2d")

    train_loader = create_mnist_dataloader(args.dataset_dir, args.batch_size, device, split="train", shuffle=True)
    val_loader = create_mnist_dataloader(args.dataset_dir, args.batch_size, device, split="val")

    # --- 4. Calibration: train-mode forwards to settle observers ---
    if args.calibration_batches > 0:
        print(f"Calibrating {args.calibration_batches} batches (train-mode no-grad to settle observers)...")
        model.train()
        with torch.no_grad():
            for i, (images, _) in enumerate(train_loader):
                if i >= args.calibration_batches:
                    break
                model(images.to(device, non_blocking=True))
    # Lock observer stats once calibration is done.  Letting the EMA
    # keep updating during training at 16-level output grids feeds
    # STE-noisy output ranges back into ``(s_y, zp_y)`` and
    # destabilises the integer requantize — the cascade observed
    # initially was loss-spike at epoch 2.
    n_frozen = neurox.freeze_hat_observers(model)
    print(f"Froze {n_frozen} HAT observers")

    # --- 5. Fine-tune with macro-in-the-loop ---
    # Adam (over SGD+momentum): HAT gradients through the 16-level
    # output grid are quantisation-noisy, and momentum accumulates that
    # noise into stepwise weight drift that spikes the loss after a
    # few epochs.  Adam's per-parameter second-moment scaling tolerates
    # the STE noise without blowing up.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
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
    print(f"Best HAT-eval val_acc: {best_acc:.4f}")

    # --- 6. Extract NeuroX-flat state and save ---
    flat = neurox.extract_neurox_state(model)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "neurox_flat",
            "state_dict": flat,
            "qat_config": {
                "x_qmin": spec.x_qmin,
                "x_qmax": spec.x_qmax,
                "w_qmax": spec.w_qmax,
                "y_qmin": spec.y_qmin,
                "y_qmax": spec.y_qmax,
                "xbar_used_during_hat": args.xbar,
                "hardware_config": str(args.config),
            },
        },
        args.checkpoint,
    )
    print(f"Saved NeuroX-flat HAT checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
