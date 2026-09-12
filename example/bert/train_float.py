"""Floating-point fine-tuning of BERT-small on SST-2.

Loads the HuggingFace-pretrained BERT-small encoder plus a randomly
initialised classification head, fine-tunes on the SST-2 training split with
AdamW and cosine decay, keeps the best-validation checkpoint, and writes its
`state_dict` to `--checkpoint`.
"""

# ruff: noqa: T201

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from example.bert.data import create_sst2_dataloader
from example.bert.model_float import create_bert_small


def _validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Top-1 accuracy of `model` on `loader`, in eval mode."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BERT-small on SST-2 (float)")
    parser.add_argument(
        "--dataset-dir", type=Path, required=True, help="HuggingFace cache directory for SST-2 + tokenizer"
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Output float state_dict path")
    parser.add_argument("--device", type=str, required=True, help="Torch device")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-length", type=int, default=128, help="Max token sequence length")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = create_bert_small(num_labels=2, cache_dir=str(args.dataset_dir)).to(device)

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for input_ids, attn, ttids, labels in train_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)
            ttids = ttids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(input_ids=input_ids, attention_mask=attn, token_type_ids=ttids).logits
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            correct += logits.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

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
            f"lr={scheduler.get_last_lr()[0]:.2e}{marker}"
        )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    final_state = best_state if best_state is not None else model.state_dict()
    torch.save(final_state, args.checkpoint)
    print(f"Saved best float model (val_acc={best_acc:.4f}) to {args.checkpoint}")


if __name__ == "__main__":
    main()
