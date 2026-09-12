"""Run directories and atomic artifact writes for offline tools."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

__all__ = ["RunOutput", "atomic_output"]


@contextmanager
def atomic_output(path: Path) -> Iterator[Path]:
    """Yield a temporary sibling and replace `path` only after a successful write.

    A failed write preserves the previous artifact and removes the temporary
    file. The destination's parent directories are created as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class RunOutput:
    """Artifact namespace for one offline run; names are relative to its directory."""

    directory: Path

    @classmethod
    def create(cls, parent: Path, *, name: str) -> RunOutput:
        """Allocate a timestamped directory, resolving concurrent name collisions."""
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("run name must be one nonempty path component")
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        for index in range(1000):
            suffix = "" if index == 0 else f"_{index + 1}"
            directory = parent / f"{name}_{stamp}{suffix}"
            try:
                directory.mkdir()
            except FileExistsError:
                continue
            return cls(directory)
        raise FileExistsError(f"could not allocate a run directory under {parent}")

    def path(self, name: str) -> Path:
        """Prepare the parent of a run-relative artifact and return its path."""
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact name must remain inside the run directory")
        path = self.directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_text(self, name: str, text: str) -> Path:
        """Atomically write UTF-8 text under this run."""
        path = self.path(name)
        with atomic_output(path) as temporary:
            temporary.write_text(text, encoding="utf-8")
        return path

    def write_json(self, name: str, values: Mapping[str, object]) -> Path:
        """Record metadata, representing path and device objects as strings."""
        return self.write_text(name, json.dumps(values, indent=2, default=str) + "\n")

    def plot_dir(self, override: Path | None = None) -> Path:
        """Create the selected figures directory, defaulting to this run."""
        directory = override if override is not None else self.directory / "figures"
        directory.mkdir(parents=True, exist_ok=True)
        return directory
