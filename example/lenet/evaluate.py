"""Evaluate a LeNet QAT checkpoint by running through macros.

Loads a flat per-layer QAT checkpoint, builds :class:`QuantLeNet5` with
the chosen macro flavour (``ideal_xbar_macro.toml`` /
``macro_with_ideal_xbar.toml`` / ``macro_with_physical_xbar.toml``), and
runs MNIST val. Each layer's per-layer ADC mode is hard-wired in
``model_quant._LAYER_MODE`` — edit that mapping to retarget modes.
"""

# ruff: noqa: T201

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from example.lenet.data import create_mnist_dataloader
from example.lenet.macro_factory import build_macro_factory
from example.lenet.model_quant import QuantLeNet5
from example.lenet.train_quant import QAT_SCHEMA
from neurox.common.profiler import NeuroxProfiler

CONFIG_DIR = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a LeNet QAT checkpoint")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="MNIST root directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="QAT checkpoint produced by train_quant.py")
    parser.add_argument(
        "--macro-config",
        required=True,
        help=f"Macro TOML filename under {CONFIG_DIR.name}/ (e.g. macro_with_physical_xbar.toml)",
    )
    parser.add_argument(
        "--xbar",
        choices=("physical", "ideal"),
        default="physical",
        help="Tile implementation. 'ideal' swaps physical for lossless twin; only meaningful "
        "when the chosen TOML carries a physical xbar.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None, help="Cap on samples processed")
    parser.add_argument(
        "--solve-chunk-size-x",
        type=int,
        default=0,
        help="CircuitCore1T1RPolicy.solve_chunk_size_x; 0 = no x-batch chunking (default).",
    )
    parser.add_argument(
        "--solve-chunk-size-inst",
        type=int,
        default=0,
        help="CircuitCore1T1RPolicy.solve_chunk_size_inst; 0 = no inst chunking (default).",
    )
    args = parser.parse_args()

    config_path = CONFIG_DIR / args.macro_config
    if not config_path.is_file():
        raise SystemExit(f"--macro-config: file not found: {config_path}")

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if ckpt.get("schema") != QAT_SCHEMA:
        raise SystemExit(f"unsupported schema {ckpt.get('schema')!r}; expected {QAT_SCHEMA!r}")

    macro_factory = build_macro_factory(
        config_path,
        ideal_xbar=(args.xbar == "ideal"),
        solve_chunk_size_x=args.solve_chunk_size_x,
        solve_chunk_size_inst=args.solve_chunk_size_inst,
    )
    model = QuantLeNet5(macro_factory, ckpt["layers"]).to(device).eval()

    loader: DataLoader = create_mnist_dataloader(args.dataset_dir, args.batch_size, device, split="val")
    correct = 0
    total = 0
    t0 = time.time()
    with NeuroxProfiler() as profiler, torch.no_grad():
        for images, targets in loader:
            if args.max_samples is not None and total >= args.max_samples:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            pred = model(images).argmax(1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
    elapsed = time.time() - t0
    acc = correct / total if total else 0.0
    static = NeuroxProfiler.analyze_static(model)
    leakage_energy__fJ = static.leakage_power__uW * profiler.total_latency__ns
    print(f"macro:                    {args.macro_config} (xbar={args.xbar})")
    print(f"samples:                  {total}")
    print(f"top1_accuracy:            {acc:.4f}")
    print(f"wall_time_s:              {elapsed:.2f}")
    print(f"time_per_sample_s:        {elapsed / max(total, 1):.4f}")
    print(f"area_total_um2:           {static.area__um2:.4f}")
    print(f"leakage_power_total_uW:   {static.leakage_power__uW:.4f}")
    print(f"dynamic_energy_total_fJ:  {profiler.total_dynamic_energy__fJ:.4f}")
    print(f"modeled_latency_total_ns: {profiler.total_latency__ns:.4f}")
    print(f"leakage_energy_total_fJ:  {leakage_energy__fJ:.4f}")
    by_type = profiler.energy_by_type
    if by_type:
        print("dynamic_energy_by_type_fJ:")
        for k in sorted(by_type, key=lambda n: -by_type[n]):
            v = by_type[k]
            if v > 0:
                print(f"  {k:<28s} {v:.4f}")


if __name__ == "__main__":
    main()
