"""Evaluate a NeuroX-flat LeNet checkpoint on MNIST.

Selects the macro configuration via ``--macro-config <file>`` (path
resolved against this directory). The ``--xbar`` flag is accepted only
when the chosen config carries a physical xbar that has an ideal twin.
``--build`` picks between the automated graph rewrite (``replace``) and
the hand-built crossbar model (``direct``).
"""

# ruff: noqa: T201

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import neurox
from example.common import build_macro_factory, read_macro_config, run_evaluate, supports_xbar_override
from example.lenet.data import create_mnist_dataloader
from example.lenet.model import LeNet5
from example.lenet.model_quant import QuantLeNet5

CONFIG_DIR = Path(__file__).parent


def _build_model_replace(
    checkpoint: Path,
    macro_factory,  # noqa: ANN001
    device: torch.device,
) -> torch.nn.Module:
    """Automated graph-rewrite path."""
    model = neurox.build_evaluator(LeNet5(), checkpoint, macro_factory)
    return model.to(device).eval()


def _build_model_direct(
    checkpoint: Path,
    macro_factory,  # noqa: ANN001
    device: torch.device,
) -> torch.nn.Module:
    """Hand-built ``QuantLeNet5`` path."""
    model = QuantLeNet5(macro_factory)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ckpt.get("schema") != "neurox_flat":
        raise ValueError(f"expected schema 'neurox_flat', got {ckpt.get('schema')!r}")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    neurox.fabricate_model(model)
    neurox.program_model(model)
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a NeuroX-flat LeNet checkpoint on MNIST")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="MNIST root directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="NeuroX-flat checkpoint path")
    parser.add_argument(
        "--macro-config",
        required=True,
        help=f"Macro config filename under {CONFIG_DIR.name}/ (e.g. macro_with_physical_xbar.toml)",
    )
    parser.add_argument(
        "--build",
        choices=("replace", "direct"),
        required=True,
        help="Model construction: 'replace' rewrites the float LeNet5 graph; "
        "'direct' instantiates QuantLeNet5 from scratch.",
    )
    parser.add_argument(
        "--xbar",
        choices=("physical", "ideal"),
        default=None,
        help="Swap a physical xbar for its ideal twin at runtime. "
        "Valid only when --macro-config carries a physical xbar; defaults to 'physical' there.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device for inference")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Cap on samples processed (useful with slow macros)"
    )
    args = parser.parse_args()

    config_path = CONFIG_DIR / args.macro_config
    if not config_path.is_file():
        raise SystemExit(f"--macro-config: file not found: {config_path}")

    cfg = read_macro_config(config_path)
    can_override = supports_xbar_override(cfg)
    if args.xbar is not None and not can_override:
        raise SystemExit(
            f"--xbar is valid only when --macro-config carries a physical xbar; "
            f"{args.macro_config} does not (cfg={type(cfg).__name__})."
        )
    ideal_xbar = can_override and args.xbar == "ideal"

    device = torch.device(args.device)
    macro_factory = build_macro_factory(config_path, ideal_xbar=ideal_xbar)

    if args.build == "replace":
        model = _build_model_replace(args.checkpoint, macro_factory, device)
    else:
        model = _build_model_direct(args.checkpoint, macro_factory, device)

    def _loader(dev: torch.device) -> DataLoader:
        return create_mnist_dataloader(args.dataset_dir, args.batch_size, dev, split="val")

    flavour = f"ideal-twin@{args.macro_config}" if ideal_xbar else args.macro_config
    run_evaluate(
        model,
        _loader,
        device=device,
        macro_name=f"{args.build}/{flavour}",
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
