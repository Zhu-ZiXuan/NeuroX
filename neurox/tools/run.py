"""Single-run lifecycle shared by calibration and validation commands."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

from .logging import run_logging
from .output import RunOutput

__all__ = ["tool_run"]

logger = logging.getLogger(__name__)


@contextmanager
def tool_run(
    *,
    name: str,
    output_dir: Path,
    log_level: str,
    parameters: Mapping[str, object],
) -> Iterator[RunOutput]:
    """Allocate artifacts, record invocation metadata, and bracket one command.

    `run.json` records execution status and elapsed time; completion means the
    command returned normally, while scientific acceptance remains a task
    result. Failures retain completed artifacts and propagate after logging.

    The yielded `RunOutput` names a new timestamped child of `output_dir`.
    Parameters are serialized for provenance, with unsupported JSON values
    represented as strings. The context installs console and file logging for
    its lifetime and restores logging afterwards. An exception is recorded and
    re-raised; callers decide which completed artifacts are scientifically
    usable.

    Args:
        name: Nonempty single-component task name for the run directory.
        output_dir: Parent directory in which a timestamped run directory is
            created.
        log_level: Logging level name for the console and run log.
        parameters: Invocation metadata to record in the run artifact.

    Yields:
        The artifact namespace for the newly allocated run directory.
    """
    output = RunOutput.create(output_dir, name=name)
    with run_logging(output.path("run.log"), level=log_level):
        started = perf_counter()
        metadata: dict[str, object] = {
            "task": name,
            "parameters": dict(parameters),
            "working_directory": str(Path.cwd()),
            "status": "running",
        }
        output.write_json("run.json", metadata)
        logger.info("run directory: %s", output.directory)
        logger.info("parameters: %s", dict(parameters))
        try:
            yield output
        except BaseException as error:
            metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
            logger.exception("run failed")
            raise
        else:
            metadata["status"] = "completed"
        finally:
            elapsed = perf_counter() - started
            metadata["elapsed_seconds"] = elapsed
            output.write_json("run.json", metadata)
            logger.info("elapsed wall time: %.3f s", elapsed)
