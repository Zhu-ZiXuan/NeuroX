"""Evaluate a LeNet QAT checkpoint by running through macros.

Loads a flat per-layer QAT checkpoint, builds `QuantLeNet5` with the chosen
macro flavour (`ideal_xbar_macro.toml` / `macro_with_ideal_xbar.toml` /
`macro_with_physical_xbar.toml`), and runs MNIST val. Each layer's quantization
mode comes from `model_quant._LAYER_MODE`.
"""

# ruff: noqa: T201

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from example.lenet.data import create_mnist_dataloader
from example.lenet.macro_factory import build_macro_factory
from example.lenet.model_quant import QuantLeNet5
from example.lenet.quant import QuantConv2d, QuantLinear
from example.lenet.train_quant import QAT_SCHEMA
from neurox import Reporter, stamp_names
from neurox.architecture.unit.cim import CimUnit
from neurox.architecture.unit.cim.engine import CimEngine
from neurox.common import ModuleBase, Profiler, neurox_roots
from neurox.primitive.macro.cim import CimMacro

CONFIG_DIR = Path(__file__).parent

# The two operator interfaces a NeuroX root exposes (`linear.py`, `conv2d.py`);
# a root has exactly one of these, never both.
_ROOT_ENTRY_METHODS = ("linear", "conv2d")


def _wrap_entry(root: ModuleBase, name: str, shapes: dict[ModuleBase, tuple[int, ...]]) -> Callable[[], None]:
    """Shadow `root`'s bound `name` method with a shape-capturing wrapper.

    The wrapper lives in the instance dict, in front of the class method.

    Returns:
        A callback that drops the instance attribute, so lookups reach the
        class method again and nothing holds `root` past the capture.
    """
    original: Callable[..., Tensor] = getattr(root, name)

    def wrapped(input: Tensor, *, quantization_mode: int, adc_bits: int | None) -> Tensor:
        shapes[root] = tuple(input.shape)
        return original(input, quantization_mode=quantization_mode, adc_bits=adc_bits)

    setattr(root, name, wrapped)
    return lambda: delattr(root, name)


def capture_root_input_shapes(model: nn.Module) -> tuple[dict[ModuleBase, tuple[int, ...]], list[Callable[[], None]]]:
    """Shadow every NeuroX root's entry point to capture the shape it receives.

    `QuantLeNet5` is a plain `nn.Module`, so its roots — each layer's `.macro` —
    are buried and each sees a different shape; `neurox_roots` discovers them
    without hard-coding layer geometry in the script.

    A root's real entry point is `linear()` or `conv2d()` (`LinearUnit` /
    `Conv2dUnit`), called straight rather than through `__call__`, so no
    `forward()` ever runs to hook; the capture shadows that entry point in the
    same pre-call spirit.

    Only the shape is recorded, an input that cannot be computed. Latency
    follows from shape plus config and is derived rather than captured.

    Returns:
        The per-root shape dict, populated once the model's forward runs,
        and the restore callbacks that undo the shadowing.
    """
    shapes: dict[ModuleBase, tuple[int, ...]] = {}
    restores: list[Callable[[], None]] = []
    for root in neurox_roots(model):
        for name in _ROOT_ENTRY_METHODS:
            if hasattr(root, name):
                restores.append(_wrap_entry(root, name, shapes))
                break
    return shapes, restores


def latency_per_sample__ns(model: nn.Module, root_shapes: dict[ModuleBase, tuple[int, ...]]) -> float:
    """Modelled duration of one sample's pass through every macro-backed layer [ns].

    Each layer's root times the call it actually received, using the captured
    input shape — the one runtime extent (`M`, a convolution unit's
    output-position count) that depends on the input resolution and so cannot
    be read from config alone. `M` is a time axis: the engine unrolls one
    macro-access schedule per output position, so a conv layer's duration
    scales with its output map. Only the sample batch stays outside the figure,
    which is one sample's pass by definition.

    Raises:
        RuntimeError: A macro-backed layer has no captured input shape.
    """
    total__ns = 0.0
    for layer in model.modules():
        if not isinstance(layer, (QuantConv2d, QuantLinear)):
            continue
        (root,) = neurox_roots(layer)
        # Narrowing by type is a temporary stand-in for a proper latency
        # interface: only these three families declare `latency__ns`, and no
        # interface spans them yet.
        if not isinstance(root, (CimUnit, CimEngine, CimMacro)):
            continue
        shape = root_shapes.get(root)
        if shape is None:
            raise RuntimeError(f"no captured input shape for {type(root).__name__}; forward never ran")
        total__ns += root.latency__ns(shape, adc_bits=layer.adc_bits)
    return total__ns


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a LeNet QAT checkpoint")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="MNIST root directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="QAT checkpoint produced by train_quant.py")
    parser.add_argument(
        "--config",
        required=True,
        help=f"Circuit config TOML filename under {CONFIG_DIR.name}/ (e.g. macro_with_physical_xbar.toml)",
    )
    parser.add_argument(
        "--policy",
        required=True,
        help=f"Nonideality policy TOML filename under {CONFIG_DIR.name}/ (e.g. macro_with_physical_xbar.policy.toml)",
    )
    parser.add_argument(
        "--cim_macro",
        choices=("physical", "ideal"),
        default="physical",
        help="Tile implementation. 'ideal' swaps the configured tile for its to_ideal() twin, "
        "the faithful reference of a physical macro; 'physical' runs the macro as configured.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None, help="Cap on samples processed")
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
    model = QuantLeNet5(macro_factory, ckpt["layers"]).to(device).eval()
    # A module never knows its own name: the assembled tree hands it one, and a
    # record carries that name. Bind the reporter before the run, so a missing
    # or stale stamp is caught here rather than at the first reported row.
    stamp_names(model)
    reporter = Reporter(model)

    loader: DataLoader = create_mnist_dataloader(args.dataset_dir, args.batch_size, device, split="val")
    correct = 0
    total = 0
    dynamic_energy__fJ = 0.0
    energy_by_name__fJ: dict[str, float] = {}
    root_shapes, restore_root_entries = capture_root_input_shapes(model)
    t0 = time.time()
    with torch.no_grad():
        for images, targets in loader:
            if args.max_samples is not None and total >= args.max_samples:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            # A unit operation is one image, so the caller leading is [B]. The
            # records stay where they were recorded: the reporter reduces a
            # whole book in one transfer, so parking them on the host first
            # would cost one sync per record instead.
            with Profiler(leading_rank=1) as profiler:
                logits = model(images)
            correct += logits.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
            batch_by_name__fJ = reporter.by_name(profiler)
            dynamic_energy__fJ += sum(batch_by_name__fJ.values())
            for name, e__fJ in batch_by_name__fJ.items():
                energy_by_name__fJ[name] = energy_by_name__fJ.get(name, 0.0) + e__fJ
    for restore in restore_root_entries:
        restore()
    elapsed = time.time() - t0
    acc = correct / total if total else 0.0
    static = reporter.static
    # Leakage power and the access time are two independent figures. Static
    # energy is leakage times the duty-cycle period a deployment holds the macro
    # for, which is a property of that deployment rather than of the access time
    # below, so the two are reported separately.
    latency__ns = latency_per_sample__ns(model, root_shapes)
    print(f"config:                   {args.config} (cim_macro={args.cim_macro})")
    print(f"policy:                   {args.policy}")
    print(f"samples:                  {total}")
    print(f"top1_accuracy:            {acc:.4f}")
    print(f"wall_time_s:              {elapsed:.2f}")
    print(f"time_per_sample_s:        {elapsed / max(total, 1):.4f}")
    print(f"area_total_um2:           {static.area__um2:.4f}")
    print(f"leakage_power_total_uW:   {static.leakage__uW:.4f}")
    print(f"dynamic_energy_total_fJ:  {dynamic_energy__fJ:.4f}")
    print(f"modeled_latency_per_sample_ns: {latency__ns:.4f}")
    if energy_by_name__fJ:
        print("dynamic_energy_by_name_fJ:")
        width = max(len(name) for name in energy_by_name__fJ)
        for name in sorted(energy_by_name__fJ, key=lambda n: -energy_by_name__fJ[n]):
            e__fJ = energy_by_name__fJ[name]
            if e__fJ > 0:
                print(f"  {name:<{width}s} {e__fJ:.4f}")


if __name__ == "__main__":
    main()
