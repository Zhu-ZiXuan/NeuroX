"""Evaluate a BERT-small QAT checkpoint by running through macros on SST-2 val."""

# ruff: noqa: T201

import argparse
import statistics
import time
from pathlib import Path

import torch
import torch._dynamo
import torch.nn as nn
from torch.utils.data import DataLoader

from example.bert.data import create_sst2_dataloader
from example.bert.macro_factory import build_macro_factory
from example.bert.model_float import create_bert_small
from example.bert.model_quant import to_quant
from example.bert.quant import QuantLinear
from example.bert.train_quant import QAT_SCHEMA
from neurox.architecture.unit.cim.base import CimUnit
from neurox.architecture.unit.cim.engine.base import CimEngine
from neurox.common.profiler import NeuroxProfiler, neurox_roots
from neurox.primitive.macro.cim.base import CimMacro

CONFIG_DIR = Path(__file__).parent


def latency_per_token__ns(model: nn.Module) -> float:
    """Modelled duration of one token's pass through every macro-backed layer [ns].

    Each layer times the unit it drives at the operating point it resolved,
    over one input vector of the contraction width its programmed weight fixes.
    A linear operator lowers to a single output plane, so the sample and token
    axes a layer receives are parallel and never enter the duration; one token
    therefore costs one call per layer.
    """
    total__ns = 0.0
    for layer in model.modules():
        if not isinstance(layer, QuantLinear):
            continue
        for root in neurox_roots(layer):
            # Narrowing by type is a temporary stand-in for a proper latency
            # interface: only these three families declare `latency__ns`, and
            # no interface spans them yet.
            if not isinstance(root, (CimUnit, CimEngine, CimMacro)):
                continue
            # Shape: [K], the fan-in one output channel contracts over.
            total__ns += root.latency__ns((layer.in_features,), adc_bits=layer.adc_bits)
    return total__ns


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a BERT-small QAT checkpoint on SST-2")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        default="macro.toml",
        help=f"Circuit config TOML under {CONFIG_DIR.name}/ (default: macro.toml). "
        "macro_ideal.toml is a standalone ideal reference for flow bring-up only "
        "(not a production result); for a faithful ideal twin use --cim_macro ideal on the physical config.",
    )
    parser.add_argument(
        "--policy",
        default="macro.policy.toml",
        help=f"Nonideality policy TOML under {CONFIG_DIR.name}/ (default: macro.policy.toml). "
        "Use macro_ideal.policy.toml for the ideal path.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--cim_macro",
        choices=("physical", "ideal"),
        default="physical",
        help="'ideal' swaps the configured tile for its to_ideal() twin, the faithful reference "
        "of a physical macro; 'physical' runs the macro as configured.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config_path = CONFIG_DIR / args.config
    if not config_path.is_file():
        raise SystemExit(f"--config: file not found: {config_path}")
    policy_path = CONFIG_DIR / args.policy
    if not policy_path.is_file():
        raise SystemExit(f"--policy: file not found: {policy_path}")

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if ckpt.get("schema") != QAT_SCHEMA:
        raise SystemExit(f"unsupported schema {ckpt.get('schema')!r}; expected {QAT_SCHEMA!r}")

    macro_factory = build_macro_factory(
        config_path,
        policy_path,
        ideal_macro=(args.cim_macro == "ideal"),
    )
    model = create_bert_small(num_labels=2, cache_dir=str(args.dataset_dir))
    model = model.to(device)
    # The shipped ideal config declares a single conversion window: mode 0.
    n_replaced = to_quant(model, ckpt["layers"], macro_factory, mode_picker=0)
    print(
        f"Quant-replaced {n_replaced} Linear layers; config={args.config} policy={args.policy} (cim_macro={args.cim_macro})"
    )
    model.eval()

    loader: DataLoader = create_sst2_dataloader(
        args.dataset_dir, args.batch_size, device, split="validation", max_length=args.max_length
    )
    correct = 0
    total = 0
    batch_times: list[float] = []
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    dynamic_energy__fJ = 0.0
    energy_by_type__fJ: dict[str, float] = {}
    token_num = 0
    t0 = time.time()
    with torch.no_grad():
        for input_ids, attn, ttids, labels in loader:
            if args.max_samples is not None and total >= args.max_samples:
                break
            input_ids = input_ids.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)
            ttids = ttids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if cuda:
                torch.cuda.synchronize(device)
            tb = time.time()
            # A unit operation is one token, so the caller leading is [B, T].
            with NeuroxProfiler(leading_rank=2) as profiler:
                logits = model(input_ids=input_ids, attention_mask=attn, token_type_ids=ttids).logits
            if cuda:
                torch.cuda.synchronize(device)
            batch_times.append(time.time() - tb)
            correct += logits.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
            token_num += input_ids.shape[0] * input_ids.shape[1]
            dynamic_energy__fJ += profiler.total_dynamic_energy__fJ
            for name, e__fJ in profiler.energy_by_type.items():
                energy_by_type__fJ[name] = energy_by_type__fJ.get(name, 0.0) + e__fJ
    elapsed = time.time() - t0
    acc = correct / total if total else 0.0
    static = NeuroxProfiler.analyze_static(model)
    # Leakage power and the access time are two independent figures. Static
    # energy is leakage times the duty-cycle period a deployment holds the macro
    # for, which is a property of that deployment rather than of the access time
    # below, so the two are reported separately.
    latency__ns = latency_per_token__ns(model)
    print(f"samples:                  {total}")
    print(f"top1_accuracy:            {acc:.4f}")
    print(f"wall_time_s:              {elapsed:.2f}")
    print(f"time_per_sample_s:        {elapsed / max(total, 1):.4f}")
    if batch_times:
        print(f"cold_batch_s:             {batch_times[0]:.2f}   (first batch; includes torch.compile)")
        if len(batch_times) > 1:
            print(f"warm_batch_s:             {statistics.median(batch_times[1:]):.4f}   (median of later batches)")
    if cuda:
        print(f"peak_gpu_mem_gib:         {torch.cuda.max_memory_allocated(device) / (1024**3):.3f}")
    print(f"dynamo_unique_graphs:     {torch._dynamo.utils.counters['stats'].get('unique_graphs', 0)}")
    print(f"area_total_um2:           {static.area__um2:.4f}")
    print(f"leakage_power_total_uW:   {static.leakage_power__uW:.4f}")
    print(f"tokens:                   {token_num}")
    print(f"dynamic_energy_total_fJ:  {dynamic_energy__fJ:.4f}")
    print(f"modeled_latency_per_token_ns: {latency__ns:.4f}")
    if energy_by_type__fJ:
        print("dynamic_energy_by_type_fJ:")
        for k in sorted(energy_by_type__fJ, key=lambda n: -energy_by_type__fJ[n]):
            v = energy_by_type__fJ[k]
            if v > 0:
                print(f"  {k:<28s} {v:.4f}")


if __name__ == "__main__":
    main()
