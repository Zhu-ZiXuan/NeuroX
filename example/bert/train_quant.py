"""BERT-small pure-QAT training on SST-2. No macro in the forward.

1. Load float BertForSequenceClassification + fine-tuned state.
2. In-place swap every nn.Linear for QATLinear (observers attached).
3. Calibrate observers on N batches, freeze.
4. Fine-tune with optional KD from frozen float teacher (CE + KL on logits).
5. Save per-layer flat state dict consumed by ``evaluate.py``.
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

from example.bert.data import create_sst2_dataloader
from example.bert.model_float import create_bert_small
from example.bert.model_quant import to_qat
from example.bert.quant import (
    W_QMAX,
    X_QMAX,
    X_QMIN,
    Y_QMAX,
    Y_QMIN,
    export_qat_state,
    freeze_observers,
)

QAT_SCHEMA = "bert_qat_v1"


def _validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attn, ttids, labels in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)
            ttids = ttids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(input_ids=input_ids, attention_mask=attn, token_type_ids=ttids).logits
            correct += logits.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def _kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    s_log = F.log_softmax(student_logits / temperature, dim=-1)
    t_p = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s_log, t_p, reduction="batchmean") * (temperature * temperature)


def main() -> None:
    parser = argparse.ArgumentParser(description="BERT-small pure-QAT (no macro in forward) on SST-2")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="HF cache dir")
    parser.add_argument("--float-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--calibration-batches", type=int, default=16)
    parser.add_argument("--kd-alpha", type=float, default=0.3)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    float_state = torch.load(args.float_checkpoint, map_location="cpu", weights_only=True)

    teacher = create_bert_small(num_labels=2, cache_dir=str(args.dataset_dir))
    teacher.load_state_dict(float_state)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = create_bert_small(num_labels=2, cache_dir=str(args.dataset_dir))
    student.load_state_dict(float_state)
    student = student.to(device)
    n_replaced = to_qat(student)
    print(f"Loaded float checkpoint; QAT-replaced {n_replaced} Linear layers")
    print(f"QAT grid: x[{X_QMIN},{X_QMAX}]  w[±{W_QMAX}]  y[{Y_QMIN},{Y_QMAX}]")
    print(f"KD: alpha_ce={args.kd_alpha}  T={args.kd_temperature}")

    train_loader = create_sst2_dataloader(
        args.dataset_dir, args.batch_size, device, split="train", shuffle=True, max_length=args.max_length
    )
    val_loader = create_sst2_dataloader(
        args.dataset_dir, args.batch_size, device, split="validation", max_length=args.max_length
    )

    if args.calibration_batches > 0:
        print(f"Calibrating {args.calibration_batches} batches (no-grad train-mode)...")
        student.train()
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= args.calibration_batches:
                    break
                ids, attn, tt, _ = batch
                student(
                    input_ids=ids.to(device, non_blocking=True),
                    attention_mask=attn.to(device, non_blocking=True),
                    token_type_ids=tt.to(device, non_blocking=True),
                )
    n_frozen = freeze_observers(student)
    print(f"Froze {n_frozen} observers")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(args.epochs):
        student.train()
        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0
        correct = 0
        total = 0
        for input_ids, attn, ttids, labels in train_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)
            ttids = ttids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                teacher_logits = teacher(input_ids=input_ids, attention_mask=attn, token_type_ids=ttids).logits
            student_logits = student(input_ids=input_ids, attention_mask=attn, token_type_ids=ttids).logits

            ce = criterion(student_logits, labels)
            kd = _kd_loss(student_logits, teacher_logits, args.kd_temperature)
            loss = args.kd_alpha * ce + (1.0 - args.kd_alpha) * kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_ce += ce.item()
            total_kd += kd.item()
            correct += student_logits.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc = _validate(student, val_loader, device)
        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(student.state_dict())
            marker = " *best*"
        nb = len(train_loader)
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"loss={total_loss / nb:.4f} (ce={total_ce / nb:.4f}, kd={total_kd / nb:.4f}), "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, lr={scheduler.get_last_lr()[0]:.2e}{marker}"
        )

    if best_state is not None:
        student.load_state_dict(best_state)
    print(f"Best QAT val_acc: {best_acc:.4f}")

    layer_state = export_qat_state(student)
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
