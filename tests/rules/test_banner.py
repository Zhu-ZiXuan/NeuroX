"""Class banners use `===`, function banners use `---`, and modules use neither.

A banner occupies its own line, with a blank line below and normally one above. When a function banner
follows that function's docstring, no blank line intervenes. A padded two-sided candidate uses exactly
three matching markers; one-sided or incompletely padded comments remain prose. Unnamed runs are refused.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path
from typing import NamedTuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = (REPO_ROOT / "neurox", REPO_ROOT / "validations")

_SCOPE_FOR_SYMBOL = {"=": "class", "-": "function"}
_BANNER_CANDIDATE = re.compile(r"^# (?P<open>-{2,}|={2,}) (?P<name>.*?) (?P<close>-{2,}|={2,})$")
_LEGAL_BANNER = re.compile(r"^# (?P<mark>---|===) (?P<name>\S(?:.*\S)?) (?P=mark)$")
_UNNAMED_RUN = re.compile(r"^#\s*[-=]{2,}\s*$")


class BannerSite(NamedTuple):
    path: Path
    lineno: int
    scope: str
    text: str
    own_line: bool
    blank_before: bool
    blank_after: bool
    after_docstring: bool

    def where(self) -> str:
        return f"{self.path.relative_to(REPO_ROOT)}:{self.lineno}  ({self.scope} scope)  {self.text}"

    def symbol(self) -> str:
        match = _LEGAL_BANNER.fullmatch(self.text)
        assert match is not None
        return match.group("mark")[0]


def _iter_python_files() -> list[Path]:
    out: list[Path] = []
    for root in SCAN_ROOTS:
        out.extend(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(out)


def _definition_spans(source: str, tree: ast.Module) -> list[tuple[int, int, str]]:
    lines = source.splitlines()
    out: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        end = node.end_lineno or node.lineno
        while end < len(lines):
            text = lines[end]
            if not text.lstrip().startswith("#") or len(text) - len(text.lstrip()) <= node.col_offset:
                break
            end += 1
        out.append((node.lineno, end, "class" if isinstance(node, ast.ClassDef) else "function"))
    return out


def _scope_of(lineno: int, spans: list[tuple[int, int, str]]) -> str:
    enclosing = [span for span in spans if span[0] <= lineno <= span[1]]
    if not enclosing:
        return "module"
    return min(enclosing, key=lambda span: span[1] - span[0])[2]


def _docstring_end_lines(tree: ast.Module) -> set[int]:
    ends: set[int] = set()
    owners = ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
    for node in ast.walk(tree):
        if not isinstance(node, owners) or not node.body:
            continue
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            assert first.end_lineno is not None
            ends.add(first.end_lineno)
    return ends


def _previous_nonblank_line(lines: list[str], lineno: int) -> int | None:
    for index in range(lineno - 2, -1, -1):
        if lines[index].strip():
            return index + 1
    return None


def _comment_tokens(source: str) -> list[tuple[int, str, bool]]:
    out: list[tuple[int, str, bool]] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            own_line = not token.line[: token.start[1]].strip()
            out.append((token.start[0], token.string, own_line))
    return out


def _classify(text: str) -> str:
    if _LEGAL_BANNER.fullmatch(text):
        return "banner"
    if _BANNER_CANDIDATE.fullmatch(text):
        return "malformed"
    if _UNNAMED_RUN.fullmatch(text):
        return "divider"
    return "prose"


def _collect() -> tuple[list[BannerSite], int]:
    sites: list[BannerSite] = []
    visited = 0
    for path in _iter_python_files():
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        tree = ast.parse(source, filename=str(path))
        spans = _definition_spans(source, tree)
        docstring_ends = _docstring_end_lines(tree)
        comments = _comment_tokens(source)
        visited += len(comments)
        for lineno, text, own_line in comments:
            if _classify(text) == "prose":
                continue
            sites.append(
                BannerSite(
                    path=path,
                    lineno=lineno,
                    scope=_scope_of(lineno, spans),
                    text=text,
                    own_line=own_line,
                    blank_before=lineno > 1 and not lines[lineno - 2].strip(),
                    blank_after=lineno < len(lines) and not lines[lineno].strip(),
                    after_docstring=_previous_nonblank_line(lines, lineno) in docstring_ends,
                )
            )
    return sites, visited


@pytest.fixture(scope="module")
def sites() -> list[BannerSite]:
    return _collect()[0]


def test_the_grammar_reads_the_ruled_forms() -> None:
    witnesses = {
        "# === Nominal buffers ===": "banner",
        "# --- Reference bank ---": "banner",
        "# --- 2: condense the cell network onto wire nodes ---": "banner",
        "# == State ==": "malformed",
        "# ==== State ====": "malformed",
        "# == State ===": "malformed",
        "# --- State --": "malformed",
        "# == State --": "malformed",
        "# ---  ---": "malformed",
        "# ---------------------------": "divider",
        "# ====": "divider",
        "# --- State": "prose",
        "# State ---": "prose",
        "# ---State---": "prose",
        "#--- State ---": "prose",
        "# --device auto picks CUDA when available": "prose",
        "# a --> b": "prose",
        "# x == y": "prose",
        "# Result containers": "prose",
        "#!/usr/bin/env python3": "prose",
        "# -*- coding: utf-8 -*-": "prose",
    }
    assert {text: _classify(text) for text in witnesses} == witnesses


def test_the_scan_reads_comments_not_string_contents() -> None:
    source = (
        '"""Doc.\n\n# --- not a banner ---\n"""\n\n'
        'TEXT = "# === not a banner ==="\n'
        "value = 1  # === inline banner ===\n"
        "# === own-line banner ===\n"
    )
    assert _comment_tokens(source) == [
        (7, "# === inline banner ===", False),
        (8, "# === own-line banner ===", True),
    ]


def test_the_scan_reaches_the_tree(sites: list[BannerSite]) -> None:
    assert _iter_python_files(), f"No `.py` file found under {SCAN_ROOTS}; the scan roots are stale."

    _, visited = _collect()
    assert visited, "No comment tokenized; the comment reader is broken."
    banners = [site for site in sites if _classify(site.text) == "banner"]
    assert banners, "No banner found in the tree; the exact banner reader is broken."

    symbols = {banner.symbol() for banner in banners}
    assert symbols == set(_SCOPE_FOR_SYMBOL), (
        f"Only {sorted(symbols)} banners were read, so the scope check below covers one symbol at most."
    )
    scopes = {banner.scope for banner in banners}
    assert set(_SCOPE_FOR_SYMBOL.values()) <= scopes, (
        f"Banners resolved to {sorted(scopes)} only; the scope resolver misses a definition body."
    )


def test_every_banner_candidate_uses_the_exact_grammar(sites: list[BannerSite]) -> None:
    malformed = [site.where() for site in sites if _classify(site.text) == "malformed"]
    dividers = [site.where() for site in sites if _classify(site.text) == "divider"]
    if malformed or dividers:
        sections: list[str] = []
        if malformed:
            sections.append("Two-sided candidates that are not exact banners:\n  " + "\n  ".join(malformed))
        if dividers:
            sections.append("Unnamed separator runs:\n  " + "\n  ".join(dividers))
        pytest.fail(
            "Rule: a two-sided banner candidate is exactly `# === text ===` or `# --- text ---`; "
            "each side has three matching markers and text is non-empty. A symbol-only divider is "
            "never written. One-sided or incompletely padded comments remain prose.\n\n"
            + "\n\n".join(sections)
            + "\n\nFix: use the exact banner form when a named structural boundary helps, use a plain comment "
            "for prose, or delete an unnamed divider."
        )


def test_each_banner_occupies_an_isolated_line(sites: list[BannerSite]) -> None:
    offenders: list[str] = []
    for site in sites:
        if _classify(site.text) != "banner":
            continue
        faults: list[str] = []
        if not site.own_line:
            faults.append("shares a line with code")
        else:
            if site.scope == "function" and site.after_docstring and site.blank_before:
                faults.append("has a blank line after the preceding docstring")
            elif not (site.scope == "function" and site.after_docstring) and not site.blank_before:
                faults.append("has no blank line immediately above")
            if not site.blank_after:
                faults.append("has no blank line immediately below")
        if faults:
            offenders.append(f"{site.where()}  -- {', '.join(faults)}")
    if offenders:
        pytest.fail(
            "Rule: a banner occupies its own line with a blank line immediately below. It also has a "
            "blank line above unless a function banner follows that function's docstring, in which "
            "case no blank line intervenes.\n"
            "Fix: move an inline banner onto its own line, keep the lower blank line, and use the "
            "correct upper boundary for its position.\n  " + "\n  ".join(offenders)
        )


def test_each_banner_stands_in_the_scope_its_symbol_opens(sites: list[BannerSite]) -> None:
    banners = [site for site in sites if _classify(site.text) == "banner"]
    offenders = [
        f"{banner.where()}  -- `{banner.symbol()}` opens a {_SCOPE_FOR_SYMBOL[banner.symbol()]}-scope group"
        for banner in banners
        if banner.scope != _SCOPE_FOR_SYMBOL[banner.symbol()]
    ]
    if offenders:
        pytest.fail(
            "Rule: `===` marks class structure, `---` marks a function or method procedure, and module "
            "scope admits neither.\n"
            "Fix: use `===` in a class body, `---` in a function or method body, or a plain comment at "
            "module scope.\n  " + "\n  ".join(offenders)
        )
