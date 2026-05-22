"""Shared evaluation procedure for NeuroX examples.

Each example's ``evaluate.py`` constructs its model (via either the
automated ``build_evaluator`` path or a hand-built quantised model class)
and hands it to ``run_evaluate``. The loop itself, the profiler wrapping,
and the summary print are model-agnostic and live here.
"""

# ruff: noqa: T201

import time
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import neurox
from neurox.replace import count_xbar_layers


def run_evaluate(
    model: nn.Module,
    loader_factory: Callable[[torch.device], DataLoader],
    *,
    device: torch.device,
    macro_name: str,
    max_samples: int | None = None,
) -> float:
    """Run inference on a pre-built model and print a report.

    Args:
        model: Ready-to-run model already placed on ``device`` and switched
            to ``eval()`` (the caller owns construction — automated
            ``build_evaluator`` or a hand-built quantised class).
        loader_factory: Callable that returns a val-split ``DataLoader`` given a device.
        device: Inference device.
        macro_name: Label printed in the report.
        max_samples: Optional cap on the number of samples processed.

    Returns:
        Top-1 accuracy as a float in ``[0, 1]``.
    """
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
