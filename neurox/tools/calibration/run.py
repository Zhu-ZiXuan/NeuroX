"""Configuration loading within the shared offline run lifecycle."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager

from neurox.common.serialize_mixin import SerializeMixin
from neurox.tools.output import RunOutput
from neurox.tools.run import tool_run

__all__ = ["calibration_run"]


@contextmanager
def calibration_run[ConfigT: SerializeMixin](
    config_type: type[ConfigT],
    args: argparse.Namespace,
    *,
    name: str,
) -> Iterator[tuple[ConfigT, RunOutput]]:
    """Load a task's config and record its values inside one logged run.

    `config.json` records parsed values; relative input paths retain their
    meaning against the original config path recorded in `run.json`.
    """
    parameters = {**vars(args), "config": str(args.config.resolve())}
    with tool_run(name=name, output_dir=args.output_dir, log_level=args.log_level, parameters=parameters) as output:
        config = config_type.from_file(args.config)
        output.write_json("config.json", config.to_dict())
        yield config, output
