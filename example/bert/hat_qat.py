"""Hardware-aware QAT for BERT-small on SST-2 with knowledge distillation.

Replaces every ``nn.Linear`` in the BERT model — Q/K/V/output
attention projections, FFN intermediate / output, pooler, and the
classification head — with ``HATLinear`` bound to a fresh
``XbarMacro`` instance per layer backed by the lossless
:class:`IdealXbar` tile derived from the physical chip.  Embeddings,
LayerNorm, GELU, and the attention softmax stay in float (NeuroX
doesn't touch them).

The HAT forward runs the ideal macro inside each replaced linear —
same ADC quant grid as the deployed hardware, no noise in the
backward — and backward uses STE through the float-reference path.
The physical ``xbar1t1r`` macro is reserved for post-HAT evaluation.

Knowledge distillation
----------------------
A frozen float teacher (the original fine-tuned BERT-small) supervises
the quantized student during HAT training.  The total loss is

    L = α · CE(student, labels)
      + (1 - α) · T² · KL(softmax(student/T) || softmax(teacher/T))
      + β · MSE(student_pooled, teacher_pooled)

* The CE term keeps the student grounded to the ground-truth labels.
* The KD term transfers the teacher's full class-probability distribution.
* The pooled-output MSE term provides intermediate-representation
  supervision — the teacher and student share architecture so the
  ``[CLS]`` pooled embeddings are directly comparable.  This is the
  most effective single ingredient for quantized-BERT distillation.

The combined loss gives a much richer training signal than CE alone
and helps the student recover accuracy lost to the coarse per-tile
ADC quantization.
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
from example.bert.model import create_bert_small
from example.common import build_macro_factory, derive_quant_spec
from neurox import replace as neurox
from neurox.config import DEFAULT_1T1R_TOML


def _validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Top-1 accuracy of ``model`` on ``loader`` (eval mode)."""
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
    """KL divergence between teacher and student soft-label distributions.

    Standard Hinton distillation: temperature ``T`` softens both
    distributions before the KL.  The ``T**2`` factor preserves the
    gradient magnitude relative to the unsoftened CE loss.
    """
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temperature * temperature)


def _forward_pooled_and_logits(
    model: nn.Module,
    input_ids: torch.Tensor,
    attn: torch.Tensor,
    ttids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single forward pass that yields both ``[CLS]`` pooled output and logits.

    ``BertForSequenceClassification`` normally hides the pooled output
    behind ``model.bert(...)`` and ``model(...)`` returns only logits.
    For hidden-state distillation we need both, so we drive the
    encoder + classifier explicitly to avoid two HAT forwards.
    """
    bert_outputs = model.bert(
        input_ids=input_ids,
        attention_mask=attn,
        token_type_ids=ttids,
    )
    pooled = bert_outputs.pooler_output
    logits = model.classifier(model.dropout(pooled))
    return pooled, logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Hardware-aware QAT + distillation for BERT-small on SST-2")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="HuggingFace cache directory")
    parser.add_argument(
        "--float-checkpoint", type=Path, required=True, help="Pretrained float state_dict from train.py"
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
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Cap on training batches per epoch (0 = use the full train split). Useful for speed probes.",
    )
    # Optimization
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=16,
        help="Forward-only batches before the first optimizer step so observers settle",
    )
    # Distillation
    parser.add_argument(
        "--kd-alpha", type=float, default=0.3, help="Weight on hard-label CE; (1-alpha) goes to logit KL"
    )
    parser.add_argument("--kd-temperature", type=float, default=2.0, help="Softening temperature for logit KD")
    parser.add_argument(
        "--kd-hidden-weight", type=float, default=1.0, help="Weight on pooled-output MSE; set 0 to disable"
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # Chip config drives both the operator quantisation grid and the
    # macro-factory wiring, so the two CLI knobs (``--config`` and
    # ``--xbar``) are sufficient to fully specify the hardware target.
    spec = derive_quant_spec(args.config)
    macro_factory = build_macro_factory(args.config, xbar=args.xbar)

    # --- 1. Load fine-tuned float weights ---
    float_state = torch.load(args.float_checkpoint, map_location="cpu", weights_only=True)

    # Teacher: frozen float BERT-small (eval mode, no grad).
    teacher = create_bert_small(num_labels=2, cache_dir=str(args.dataset_dir))
    teacher.load_state_dict(float_state)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Student: same architecture, initialised from the same weights,
    # then HAT-replaced.
    student = create_bert_small(num_labels=2, cache_dir=str(args.dataset_dir))
    student.load_state_dict(float_state)
    print(f"Loaded float checkpoint: {args.float_checkpoint}")

    # --- 2. BN fold (no-op for BERT) ---
    neurox.fold_batchnorm(student)

    # --- 3. Replace nn.Linear modules with HATLinear ---
    student = neurox.replace_for_hat(student, macro_factory, spec)
    student = student.to(device)
    from neurox.operator import HATConv2d, HATLinear

    n_lin = sum(1 for m in student.modules() if isinstance(m, HATLinear))
    n_conv = sum(1 for m in student.modules() if isinstance(m, HATConv2d))
    print(
        f"HAT grid: x[{spec.x_qmin},{spec.x_qmax}]  w[±{spec.w_qmax}]  "
        f"y[{spec.y_qmin},{spec.y_qmax}]  xbar={args.xbar}  (from {args.config.name})"
    )
    print(f"HAT layers: {n_lin} Linear, {n_conv} Conv2d")
    print(f"KD: alpha_ce={args.kd_alpha}  T={args.kd_temperature}  hidden_weight={args.kd_hidden_weight}")

    train_loader = create_sst2_dataloader(
        args.dataset_dir,
        args.batch_size,
        device,
        split="train",
        shuffle=True,
        max_length=args.max_length,
    )
    val_loader = create_sst2_dataloader(
        args.dataset_dir,
        args.batch_size,
        device,
        split="validation",
        max_length=args.max_length,
    )

    # --- 4. Calibration: train-mode forwards to settle observers ---
    if args.calibration_batches > 0:
        print(f"Calibrating {args.calibration_batches} batches (no-grad to settle observers)...")
        student.train()
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= args.calibration_batches:
                    break
                input_ids, attn, ttids, _ = batch
                student(
                    input_ids=input_ids.to(device, non_blocking=True),
                    attention_mask=attn.to(device, non_blocking=True),
                    token_type_ids=ttids.to(device, non_blocking=True),
                )
    # Lock observer stats: EMA updates during training chase the
    # STE-noisy output ranges and shift ``(s_y, zp_y)`` between steps,
    # destabilising the integer requantize.  BERT also benefits even
    # though LayerNorm softens the cascade.
    n_frozen = neurox.freeze_hat_observers(student)
    print(f"Froze {n_frozen} HAT observers")

    # --- 5. Fine-tune with macro-in-the-loop + distillation ---
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(args.epochs):
        student.train()
        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0
        total_hidden = 0.0
        correct = 0
        total = 0
        for step, (input_ids, attn, ttids, labels) in enumerate(train_loader):
            if args.max_train_batches and step >= args.max_train_batches:
                break
            input_ids = input_ids.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)
            ttids = ttids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Teacher forward (frozen, no grad) — single pass.
            with torch.no_grad():
                teacher_pooled, teacher_logits = _forward_pooled_and_logits(
                    teacher,
                    input_ids,
                    attn,
                    ttids,
                )

            # Student forward (HAT through macro) — single pass.
            student_pooled, student_logits = _forward_pooled_and_logits(
                student,
                input_ids,
                attn,
                ttids,
            )

            # --- loss components ---
            ce = criterion(student_logits, labels)
            kd = _kd_loss(student_logits, teacher_logits, args.kd_temperature)
            if args.kd_hidden_weight > 0:
                hidden = F.mse_loss(student_pooled, teacher_pooled)
            else:
                hidden = torch.zeros((), device=device)

            loss = args.kd_alpha * ce + (1.0 - args.kd_alpha) * kd + args.kd_hidden_weight * hidden

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_ce += ce.item()
            total_kd += kd.item()
            total_hidden += hidden.item()
            correct += student_logits.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc = _validate(student, val_loader, device)

        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(student.state_dict())
            marker = " *best*"

        n_batches = len(train_loader)
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"loss={total_loss / n_batches:.4f} "
            f"(ce={total_ce / n_batches:.4f}, kd={total_kd / n_batches:.4f}, "
            f"hidden={total_hidden / n_batches:.4f}), "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
            f"lr={scheduler.get_last_lr()[0]:.2e}{marker}"
        )

    if best_state is not None:
        student.load_state_dict(best_state)
    print(f"Best HAT-eval val_acc: {best_acc:.4f}")

    # --- 6. Extract NeuroX-flat state and save ---
    flat = neurox.extract_neurox_state(student)
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
                "hardware_config": str(args.config),
                "xbar_used_during_hat": args.xbar,
                "kd_alpha": args.kd_alpha,
                "kd_temperature": args.kd_temperature,
                "kd_hidden_weight": args.kd_hidden_weight,
            },
        },
        args.checkpoint,
    )
    print(f"Saved NeuroX-flat HAT checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
