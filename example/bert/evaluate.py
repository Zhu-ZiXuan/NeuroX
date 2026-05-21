"""Evaluate a NeuroX-flat BERT-small checkpoint on SST-2.

Loads a HAT-trained NeuroX-flat checkpoint, builds an evaluator with
the chosen crossbar macro backend, and runs the SST-2 validation
split.  BERT's dataloader yields 4-tuples ``(input_ids,
attention_mask, token_type_ids, labels)`` and the model's forward
returns a HuggingFace output object — so this script implements its
own evaluation loop instead of reusing
``example.common.procedures.run_evaluate`` (which is image-only).
"""

# ruff: noqa: T201

import argparse
import time
from pathlib import Path

import torch

from example.bert.data import create_sst2_dataloader
from example.bert.model import create_bert_small
from example.common import build_macro_factory
from neurox import replace as neurox
from neurox.config import DEFAULT_1T1R_MACRO_TOML
from neurox.replace import NeuroxProfiler, count_xbar_layers


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a NeuroX-flat BERT-small checkpoint on SST-2")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="HuggingFace cache directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="NeuroX-flat checkpoint path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device for inference")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_1T1R_MACRO_TOML,
        help="Chip TOML (defaults to the bundled 1T1R reference).",
    )
    parser.add_argument(
        "--xbar",
        choices=("physical", "ideal"),
        default="physical",
        help=(
            "Tile implementation.  ``physical`` runs the full 1T1R "
            "circuit solver with noise (deployment-accurate); "
            "``ideal`` swaps in the lossless reference derived from "
            "the physical tile (noise-free, same ADC grid)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap on samples processed (useful for quick smoke tests).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # Build a fresh float skeleton so the architecture matches the
    # checkpoint (HuggingFace pretrained encoder + 2-class classifier);
    # build_evaluator then swaps every nn.Linear for QuantLinear and
    # loads the int weights / rescale buffers from the checkpoint.
    float_model = create_bert_small(num_labels=2, cache_dir=str(args.dataset_dir))
    macro_factory = build_macro_factory(args.config, xbar=args.xbar)
    model = neurox.build_evaluator(float_model, args.checkpoint, macro_factory)
    model = model.to(device).eval()
    print(f"Layer summary: {count_xbar_layers(model)}")

    loader = create_sst2_dataloader(
        args.dataset_dir,
        args.batch_size,
        device,
        split="validation",
        max_length=args.max_length,
    )

    correct = 0
    total = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    with torch.inference_mode(), NeuroxProfiler() as profiler:
        for input_ids, attn, ttids, labels in loader:
            if args.max_samples is not None and total >= args.max_samples:
                break
            input_ids = input_ids.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)
            ttids = ttids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(input_ids=input_ids, attention_mask=attn, token_type_ids=ttids).logits
            correct += logits.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

    elapsed = time.time() - t0
    acc = correct / total if total else 0.0
    static = NeuroxProfiler.analyze_static(model)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0

    print(f"xbar:               {args.xbar}  (from {args.config.name})")
    print(f"samples:            {total}")
    print(f"top1_accuracy:      {acc:.4f}")
    print(f"wall_time_s:        {elapsed:.2f}")
    print(f"time_per_sample_s:  {elapsed / max(total, 1):.4f}")
    print(f"peak_gpu_memory_MB: {peak_mb:.1f}")
    print(profiler.summary(static=static))


if __name__ == "__main__":
    main()
