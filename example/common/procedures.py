"""Shared evaluation procedure for NeuroX examples.

Each example's ``evaluate.py`` calls ``run_evaluate`` with its own model
+ transform + dataloader builder.  The loop itself, the profiler
wrapping, and the summary print are model-agnostic and live here.
"""

# ruff: noqa: T201

import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import neurox
from neurox.macro.base import NeuroxMacroQuantMatMul


def run_evaluate(
    float_model: nn.Module,
    checkpoint: Path,
    macro_factory: Callable[[], NeuroxMacroQuantMatMul],
    loader_factory: Callable[[torch.device], DataLoader],
    *,
    device: torch.device,
    macro_name: str,
    max_samples: int | None = None,
) -> float:
    """Replace float ops → build evaluator → run inference → print report.

    Args:
        float_model: Fresh (unloaded) float model matching the QAT architecture.
        checkpoint: Path to a ``neurox_flat`` checkpoint.
        macro_factory: Per-layer macro factory (``fake`` / ``xbar_ideal`` / ...).
        loader_factory: Callable that returns a val-split ``DataLoader`` given a device.
        device: Inference device.
        macro_name: Label printed in the report.
        max_samples: Optional cap on the number of samples processed.

    Returns:
        Top-1 accuracy as a float in ``[0, 1]``.
    """
    # NB: ``neurox`` is imported as the root package; every public
    # entry point (``build_evaluator``, ``NeuroxProfiler``, etc.) is a
    # top-level attribute per Phase D's API consolidation.
    from neurox.replace import count_xbar_layers  # diagnostic only

    model = neurox.build_evaluator(float_model, checkpoint, macro_factory)
    model = model.to(device).eval()
    print(f"Layer summary: {count_xbar_layers(model)}")
    loader = loader_factory(device)

    correct = 0
    total = 0
    t0 = time.time()
    with torch.inference_mode(), neurox.NeuroxProfiler() as profiler:
        for images, targets in loader:
            if max_samples is not None and total >= max_samples:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            pred = model(images).argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)

    elapsed = time.time() - t0
    acc = correct / total if total else 0.0
    static = neurox.NeuroxProfiler.analyze_static(model)

    print(f"macro:              {macro_name}")
    print(f"samples:            {total}")
    print(f"top1_accuracy:      {acc:.4f}")
    print(f"wall_time_s:        {elapsed:.2f}")
    print(f"time_per_sample_s:  {elapsed / max(total, 1):.4f}")
    print(profiler.summary(static=static))
    return acc
