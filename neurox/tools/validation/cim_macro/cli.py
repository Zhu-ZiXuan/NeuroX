"""Command-line and logging conventions for CIM-macro validation."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from neurox.tools.cli import add_runtime_args, positive_int
from neurox.tools.run import tool_run


@dataclass(frozen=True)
class ValidationArgs:
    """Normalized arguments shared by CIM-macro validation campaigns."""

    n_w: int
    n_x: int
    repeat: int
    seed: int
    device: torch.device
    output_dir: Path
    log_level: str


def get_cli_args(
    *,
    description: str | None,
    campaign: str,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Define and read the command-line arguments shared by campaigns."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--n-w", type=positive_int, default=64, help="Weight draws per round.")
    parser.add_argument("--n-x", type=positive_int, default=256, help="Inputs per weight draw.")
    parser.add_argument("--repeat", type=positive_int, default=8, help="Independent workload rounds.")
    parser.add_argument("--seed", type=int, default=0)
    add_runtime_args(parser, output_dir=Path("log/validation") / campaign)
    return parser.parse_args(argv)


def parse_cli_args(cli_args: argparse.Namespace) -> ValidationArgs:
    """Normalize raw command-line values for a validation campaign."""
    return ValidationArgs(
        n_w=cli_args.n_w,
        n_x=cli_args.n_x,
        repeat=cli_args.repeat,
        seed=cli_args.seed,
        device=torch.device(cli_args.device),
        output_dir=cli_args.output_dir,
        log_level=cli_args.log_level,
    )


@contextmanager
def validation_run(args: ValidationArgs, *, campaign: str) -> Iterator[ValidationArgs]:
    """Provide campaign arguments whose output directory belongs to one logged run."""
    with tool_run(
        name=campaign, output_dir=args.output_dir, log_level=args.log_level, parameters=asdict(args)
    ) as output:
        yield replace(args, output_dir=output.directory)
