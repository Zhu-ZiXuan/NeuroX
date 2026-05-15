"""Evaluate a NeuroX-flat LeNet checkpoint on MNIST.

Thin wrapper over ``example.common.run_evaluate``; this file owns only
the model / dataset binding.  See ``example/common/`` for the shared
macro factories and the generic evaluation loop.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from example.common import build_macro_factory, run_evaluate
from example.lenet.data import create_mnist_dataloader
from example.lenet.model import LeNet5
from neurox.config import DEFAULT_1T1R_TOML


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a NeuroX-flat LeNet checkpoint on MNIST")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="MNIST root directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="NeuroX-flat checkpoint path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device for inference")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_1T1R_TOML,
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Cap on samples processed (useful with slow macros)"
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    def _loader(dev: torch.device) -> DataLoader:
        return create_mnist_dataloader(args.dataset_dir, args.batch_size, dev, split="val")

    run_evaluate(
        float_model=LeNet5(),
        checkpoint=args.checkpoint,
        macro_factory=build_macro_factory(args.config, xbar=args.xbar),
        loader_factory=_loader,
        device=device,
        macro_name=f"{args.xbar}@{args.config.name}",
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
