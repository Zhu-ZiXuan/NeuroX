"""Static check for broken intra-repo Markdown links.

Scope: relative links from any ``.md`` file under ``docs/``, the
top-level ``README.md``, and every ``.md`` under ``example/``.
External URLs (``http://``, ``https://``, ``mailto:``, ``ftp://``,
``tel:``) and pure anchor fragments (``#section``) are skipped. Each
target is resolved against the linking file's directory and asserted
to exist on disk (file or directory).

Failure mode: print every broken link with source file, line number,
and resolved target so a single CI run lists all stale references.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
EXAMPLE_ROOT = REPO_ROOT / "example"

_LINK_PATTERN = re.compile(r"(?<!!)\[(?:[^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_SKIP_SCHEMES = ("http://", "https://", "mailto:", "ftp://", "tel:")


def _iter_links(md_path: Path) -> list[tuple[int, str]]:
    """Yield ``(line_number, target)`` for every Markdown link in ``md_path``."""
    out: list[tuple[int, str]] = []
    for ln, line in enumerate(md_path.read_text().splitlines(), start=1):
        out.extend((ln, match.group(1)) for match in _LINK_PATTERN.finditer(line))
    return out


def _is_skippable(target: str) -> bool:
    if not target:
        return True
    if target.startswith("#"):
        return True
    return any(target.startswith(scheme) for scheme in _SKIP_SCHEMES)


def _resolve_target(md_path: Path, target: str) -> Path:
    """Resolve ``target`` against ``md_path``'s directory, stripping anchors."""
    target_no_anchor = target.split("#", 1)[0]
    if not target_no_anchor:
        return md_path
    candidate = (md_path.parent / target_no_anchor).resolve()
    return candidate


def _iter_target_files() -> list[Path]:
    """Every ``.md`` file in scope, deduplicated and sorted."""
    out: set[Path] = set()
    out.update(DOCS_ROOT.rglob("*.md"))
    out.update(EXAMPLE_ROOT.rglob("*.md"))
    top_readme = REPO_ROOT / "README.md"
    if top_readme.is_file():
        out.add(top_readme)
    return sorted(out)


def test_no_broken_relative_markdown_links() -> None:
    broken: list[str] = []
    for md_path in _iter_target_files():
        for ln, target in _iter_links(md_path):
            if _is_skippable(target):
                continue
            resolved = _resolve_target(md_path, target)
            if not resolved.exists():
                broken.append(f"{md_path.relative_to(REPO_ROOT)}:{ln}  →  {target}  (resolved: {resolved})")
    if broken:
        pytest.fail("Broken Markdown links:\n  " + "\n  ".join(broken))
