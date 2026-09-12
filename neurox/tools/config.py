"""Path resolution for offline run configurations."""

from __future__ import annotations

from pathlib import Path

__all__ = ["resolve_relative_path"]


def resolve_relative_path(path: Path | str, base: Path) -> Path:
    """Resolve a TOML-supplied path against `base`'s directory.

    An absolute path is returned unchanged.

    Args:
        path: Raw value from the TOML.
        base: The TOML file's own path; a relative `path` resolves against
            `base.parent`.
    """
    p = Path(path) if not isinstance(path, Path) else path
    if p.is_absolute():
        return p
    return base.parent / p
