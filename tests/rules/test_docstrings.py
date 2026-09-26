"""Docstrings keep exact module See Also entries."""

from __future__ import annotations

import ast
import re
from itertools import pairwise
from pathlib import Path
from typing import NamedTuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = (REPO_ROOT / "neurox", REPO_ROOT / "validations")

_MANAGED_SECTIONS = frozenset({"Args", "Returns", "Yields", "Raises", "See Also", "Examples", "Attributes"})
_HEADER_PATTERN = re.compile(r"(?P<name>[A-Za-z]+(?: [A-Za-z]+)*):")
_DOC_PATH_PATTERN = re.compile(r"docs/[\w/.-]+\.md")
_DOC_ENTRY_PATTERN = re.compile(r" {4}(?P<path>docs/[\w/.-]+\.md)")


class Docstring(NamedTuple):
    path: Path
    lineno: int
    kind: str
    owner: str
    text: str

    def where(self) -> str:
        return f"{self.path.relative_to(REPO_ROOT)}:{self.lineno} ({self.kind} {self.owner})"


def _iter_python_files() -> list[Path]:
    out: list[Path] = []
    for root in SCAN_ROOTS:
        out.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    return sorted(out)


def _attribute_docstrings(path: Path, body: list[ast.stmt], scope: str) -> list[Docstring]:
    out: list[Docstring] = []
    for previous, statement in pairwise(body):
        if not isinstance(previous, ast.Assign | ast.AnnAssign):
            continue
        if not isinstance(statement, ast.Expr):
            continue
        literal = statement.value
        if not (isinstance(literal, ast.Constant) and isinstance(literal.value, str)):
            continue
        target = previous.target if isinstance(previous, ast.AnnAssign) else previous.targets[0]
        name = target.id if isinstance(target, ast.Name) else "<attribute>"
        owner = f"{scope}.{name}".lstrip(".")
        out.append(Docstring(path, literal.lineno, "attribute", owner, literal.value))
    return out


def _collect(path: Path, node: ast.AST, scope: str, out: list[Docstring]) -> None:
    if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
        return
    text = ast.get_docstring(node, clean=False)
    if text is not None:
        kind = {ast.Module: "module", ast.ClassDef: "class"}.get(type(node), "function")
        out.append(Docstring(path, node.body[0].lineno, kind, scope or "<module>", text))
    out.extend(_attribute_docstrings(path, node.body, scope))
    for child in node.body:
        if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            _collect(path, child, f"{scope}.{child.name}".lstrip("."), out)


def _iter_docstrings() -> list[Docstring]:
    out: list[Docstring] = []
    for path in _iter_python_files():
        _collect(path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path)), "", out)
    return out


def _normalized_lines(text: str) -> list[str]:
    lines = text.expandtabs().splitlines()
    if not lines:
        return []
    indents = [len(line) - len(line.lstrip()) for line in lines[1:] if line.strip()]
    margin = min(indents, default=0)
    return [lines[0].strip(), *(line[margin:].rstrip() for line in lines[1:])]


def _section_headers(docstring: Docstring) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for offset, line in enumerate(_normalized_lines(docstring.text)):
        match = _HEADER_PATTERN.fullmatch(line)
        if match is not None and match.group("name") in _MANAGED_SECTIONS:
            out.append((offset, docstring.lineno + offset, match.group("name")))
    return out


@pytest.fixture(scope="module")
def docstrings() -> list[Docstring]:
    found = _iter_docstrings()
    assert found, f"No docstrings found under {SCAN_ROOTS}; check the scan roots and collector."
    return found


def test_section_reader_distinguishes_managed_headers_from_prose_and_nested_labels() -> None:
    docstring = Docstring(
        Path("probe.py"),
        1,
        "function",
        "probe",
        """Summary.

        Run:
            python probe.py

        Args:
            value: Tensor value.
                Shape: `[item]`.

        Examples:
            >>> probe()
        """,
    )
    assert [name for _, _, name in _section_headers(docstring)] == ["Args", "Examples"]


def test_see_also_contains_only_bare_existing_paths_in_module_docstrings(docstrings: list[Docstring]) -> None:
    misplaced: list[str] = []
    duplicate: list[str] = []
    badly_indented_headers: list[str] = []
    empty: list[str] = []
    malformed: list[str] = []
    dangling: list[str] = []
    pointers_outside_section: list[str] = []
    for docstring in docstrings:
        lines = _normalized_lines(docstring.text)
        headers = _section_headers(docstring)
        see_also_indices = [index for index, (_, _, name) in enumerate(headers) if name == "See Also"]
        if see_also_indices and docstring.kind != "module":
            misplaced.append(docstring.where())
        if len(see_also_indices) > 1:
            duplicate.append(docstring.where())

        badly_indented_headers.extend(
            f"{docstring.where()} line {docstring.lineno + offset}: {line}"
            for offset, line in enumerate(lines)
            if line.strip() == "See Also:" and line != "See Also:"
        )

        see_also_body_lines: set[int] = set()
        for index in see_also_indices:
            offset = headers[index][0]
            end = headers[index + 1][0] if index + 1 < len(headers) else len(lines)
            body_offsets = list(range(offset + 1, end))
            see_also_body_lines.update(docstring.lineno + body_offset for body_offset in body_offsets)

            content_started = False
            content_ended = False
            entry_found = False
            for body_offset in body_offsets:
                line_text = lines[body_offset]
                line = docstring.lineno + body_offset
                if not line_text:
                    if content_started:
                        content_ended = True
                    else:
                        malformed.append(f"{docstring.where()} line {line}: blank line before the first entry")
                    continue

                if content_ended:
                    malformed.append(f"{docstring.where()} line {line}: entry follows a blank line")
                content_started = True
                entry_found = True
                match = _DOC_ENTRY_PATTERN.fullmatch(line_text)
                if match is None:
                    malformed.append(f"{docstring.where()} line {line}: {line_text}")
                    continue
                path = match.group("path")
                if not (REPO_ROOT / path).is_file():
                    dangling.append(f"{docstring.where()} line {line}: {path}")

            if not entry_found:
                empty.append(f"{docstring.where()} line {headers[index][1]}")

        for offset, line_text in enumerate(lines):
            line = docstring.lineno + offset
            pointers_outside_section.extend(
                f"{docstring.where()} line {line}: {line_text.strip()}"
                for _ in _DOC_PATH_PATTERN.findall(line_text)
                if line not in see_also_body_lines
            )

    if misplaced or duplicate or badly_indented_headers or empty or malformed or dangling or pointers_outside_section:
        groups = []
        if misplaced:
            groups.append("`See Also:` outside a module docstring:\n  " + "\n  ".join(misplaced))
        if duplicate:
            groups.append("Repeated `See Also:` sections:\n  " + "\n  ".join(duplicate))
        if badly_indented_headers:
            groups.append("`See Also:` headers that are not top-level:\n  " + "\n  ".join(badly_indented_headers))
        if empty:
            groups.append("`See Also:` sections with no entry:\n  " + "\n  ".join(empty))
        if malformed:
            groups.append("Entries that are not contiguous four-space-indented paths:\n  " + "\n  ".join(malformed))
        if dangling:
            groups.append("Paths that do not exist:\n  " + "\n  ".join(dangling))
        if pointers_outside_section:
            groups.append("Document paths outside a `See Also:` entry:\n  " + "\n  ".join(pointers_outside_section))
        pytest.fail(
            "Rule: a module docstring has at most one top-level `See Also:` section. Its non-empty body "
            "is a contiguous list of four-space-indented, bare, existing `docs/...md` paths; no document "
            "pointer appears elsewhere in a docstring.\n"
            "Fix: move a necessary pointer into that section, use one path per line with exact indentation, "
            "and update or delete stale paths.\n" + "\n".join(groups)
        )
