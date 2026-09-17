"""Banner markers, scope, and blank-line contracts."""

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

_SCOPE_FOR_SYMBOL = {"#": "module", "=": "class", "-": "function"}
_BANNER_CANDIDATE = re.compile(r"^# (?P<open>\#{2,}|-{2,}|={2,}) (?P<name>.*?) (?P<close>\#{2,}|-{2,}|={2,})$")
_LEGAL_BANNER = re.compile(r"^# (?P<mark>\#\#\#|---|===) (?P<name>\S(?:.*\S)?) (?P=mark)$")
_UNNAMED_RUN = re.compile(r"^#\s*[#=-]{2,}\s*$")
_NUMBERED_TITLE = re.compile(r"^\d+(?:\.\d+)*(?:[.:]|\s|$)")


class BannerSite(NamedTuple):
    path: Path
    lineno: int
    scope: str
    text: str
    own_line: bool
    blank_before: int
    blank_after: int
    preceding_boundary: str | None

    def where(self) -> str:
        return f"{self.path.relative_to(REPO_ROOT)}:{self.lineno}  ({self.scope} scope)  {self.text}"

    def symbol(self) -> str:
        match = _LEGAL_BANNER.fullmatch(self.text)
        assert match is not None
        return match.group("mark")[0]

    def required_blank_lines(self) -> tuple[int, int]:
        after = 2 if self.scope == "module" else 1
        before = after
        if self.scope == "module" and self.preceding_boundary == "import":
            before = 1
        elif (self.scope == "class" and self.preceding_boundary == "class_header") or (
            self.scope == "function" and self.preceding_boundary == "function_docstring"
        ):
            before = 0
        return before, after


def _iter_python_files() -> list[Path]:
    out: list[Path] = []
    for root in SCAN_ROOTS:
        out.extend(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(out)


def _definition_spans(lines: list[str], tree: ast.Module) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        end = node.end_lineno or node.lineno
        while end < len(lines):
            text = lines[end]
            if not text.strip():
                end += 1
                continue
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


def _class_header_end_lines(tokens: list[tokenize.TokenInfo], tree: ast.Module) -> set[int]:
    starts = {(node.lineno, node.col_offset) for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    ends: set[int] = set()
    for index, token in enumerate(tokens):
        if token.start not in starts:
            continue
        depth = 0
        for header_token in tokens[index:]:
            if header_token.type != tokenize.OP:
                continue
            if header_token.string in "([{":
                depth += 1
            elif header_token.string in ")]}":
                depth -= 1
            elif header_token.string == ":" and depth == 0:
                ends.add(header_token.end[0])
                break
    return ends


def _comment_tokens(tokens: list[tokenize.TokenInfo]) -> list[tuple[int, str, bool]]:
    out: list[tuple[int, str, bool]] = []
    for token in tokens:
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


def _scan_source(path: Path, source: str) -> list[BannerSite]:
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    spans = _definition_spans(lines, tree)
    boundaries = dict.fromkeys(_class_header_end_lines(tokens, tree), "class_header")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and ast.get_docstring(node) is not None:
            boundaries[node.body[0].end_lineno] = "function_docstring"
    for node in tree.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            boundaries[node.end_lineno] = "import"

    sites = []
    comments = _comment_tokens(tokens)
    for lineno, text, own_line in comments:
        if _classify(text) == "prose":
            continue
        index = lineno - 1
        before = after = 0
        while index - before > 0 and not lines[index - before - 1].strip():
            before += 1
        while index + after + 1 < len(lines) and not lines[index + after + 1].strip():
            after += 1
        sites.append(
            BannerSite(
                path=path,
                lineno=lineno,
                scope=_scope_of(lineno, spans),
                text=text,
                own_line=own_line,
                blank_before=before,
                blank_after=after,
                preceding_boundary=boundaries.get(index - before),
            )
        )
    return sites


@pytest.fixture(scope="module")
def sites() -> list[BannerSite]:
    found = []
    for path in _iter_python_files():
        found.extend(_scan_source(path, path.read_text(encoding="utf-8")))
    assert found, f"No banners found under {SCAN_ROOTS}; check the scan roots and collector."
    return found


def test_the_grammar_reads_the_ruled_forms() -> None:
    witnesses = {
        "# ### Access-node solver ###": "banner",
        "# === Nominal buffers ===": "banner",
        "# --- Reference bank ---": "banner",
        "# --- 2: condense the cell network onto wire nodes ---": "banner",
        "# == State ==": "malformed",
        "# ==== State ====": "malformed",
        "# == State ===": "malformed",
        "# --- State --": "malformed",
        "# == State --": "malformed",
        "# ---  ---": "malformed",
        "# ## Solver ##": "malformed",
        "# #### Solver ####": "malformed",
        "# ### Solver ##": "malformed",
        "# ### Solver ===": "malformed",
        "# ###  ###": "malformed",
        "# ---------------------------": "divider",
        "# ====": "divider",
        "# ###": "divider",
        "# --- State": "prose",
        "# State ---": "prose",
        "# ---State---": "prose",
        "#--- State ---": "prose",
        "# --device selection belongs to the caller": "prose",
        "# a --> b": "prose",
        "# x == y": "prose",
        "# Result containers": "prose",
        "# ### Solver": "prose",
        "# ###Solver###": "prose",
        "#!/usr/bin/env python3": "prose",
        "# -*- coding: utf-8 -*-": "prose",
    }
    assert {text: _classify(text) for text in witnesses} == witnesses


def test_scope_reader_distinguishes_module_class_and_function_groups() -> None:
    source = (
        "# ### First component ###\n"
        "\n"
        "class Example:\n"
        "    # === Fields ===\n"
        "\n"
        "    value: int\n"
        "\n"
        "    def run(self):\n"
        "        # --- Work ---\n"
        "\n"
        "        pass\n"
        "\n"
        "        # ### Wrong scope ###\n"
        "\n"
        "# ### Second component ###\n"
        "\n"
        "def run():\n"
        "    pass\n"
    )
    scanned = _scan_source(REPO_ROOT / "example.py", source)
    assert [site.scope for site in scanned] == [
        "module",
        "class",
        "function",
        "function",
        "module",
    ]


@pytest.mark.parametrize(
    ("prefix", "banner", "suffix", "expected_before", "expected_after"),
    [
        ("VALUE = 1\n", "# ### Component ###\n", "class Example:\n    pass\n", 2, 2),
        ("import ast\n", "# ### Component ###\n", "class Example:\n    pass\n", 1, 2),
        ("from ast import (\n    AST,\n)\n", "# ### Component ###\n", "class Example:\n    pass\n", 1, 2),
        ("class Example:\n", "    # === Fields ===\n", "    value: int\n", 0, 1),
        ("class Example(\n    object,\n):\n", "    # === Fields ===\n", "    value: int\n", 0, 1),
        ('class Example:\n    """Doc."""\n', "    # === Fields ===\n", "    value: int\n", 1, 1),
        ("class Example:\n    value: int\n", "    # === Fields ===\n", "    other: int\n", 1, 1),
        ('def run():\n    """Doc."""\n', "    # --- Work ---\n", "    pass\n", 0, 1),
        ('async def run():\n    """Doc.\n\n    Detail.\n    """\n', "    # --- Work ---\n", "    pass\n", 0, 1),
        ("def run():\n    value = 1\n", "    # --- Work ---\n", "    pass\n", 1, 1),
    ],
)
def test_banner_spacing_accepts_only_the_required_counts(
    prefix: str, banner: str, suffix: str, expected_before: int, expected_after: int
) -> None:
    for before in range(4):
        for after in range(4):
            source = prefix + "\n" * before + banner + "\n" * after + suffix
            scanned = _scan_source(REPO_ROOT / "example.py", source)
            (site,) = scanned
            assert (site.blank_before, site.blank_after) == (before, after), source
            assert site.required_blank_lines() == (expected_before, expected_after), source
            if (before, after) == (expected_before, expected_after):
                test_each_banner_has_the_required_blank_lines(scanned)
            else:
                with pytest.raises(AssertionError, match="blank lines"):
                    test_each_banner_has_the_required_blank_lines(scanned)


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
            "Rule: a two-sided banner candidate is exactly `# ### text ###`, `# === text ===`, or `# --- text ---`; "
            "each side has three matching markers and text is non-empty. A symbol-only divider is "
            "never written. One-sided or incompletely padded comments remain prose.\n\n"
            + "\n\n".join(sections)
            + "\n\nFix: use the exact banner form when a named structural boundary helps, use a plain comment "
            "for prose, or delete an unnamed divider."
        )


def test_each_banner_occupies_its_own_line(sites: list[BannerSite]) -> None:
    offenders = [site.where() for site in sites if _classify(site.text) == "banner" and not site.own_line]
    assert not offenders, (
        "Rule: every banner occupies its own line.\n"
        "Fix: move an inline banner onto its own line.\n  " + "\n  ".join(offenders)
    )


def test_each_banner_has_the_required_blank_lines(sites: list[BannerSite]) -> None:
    offenders = []
    for site in sites:
        if _classify(site.text) != "banner" or not site.own_line:
            continue
        before, after = site.required_blank_lines()
        if (site.blank_before, site.blank_after) != (before, after):
            offenders.append(
                f"{site.where()} -- {site.blank_before} blank lines before, {site.blank_after} after; "
                f"expected {before} before, {after} after"
            )
    assert not offenders, (
        "Rule (docs/conventions/code_style.md): module banners have two blank lines on each side; "
        "class and function banners have one. Before a banner, imports require one blank line, "
        "while a class header or function docstring requires none.\n"
        "Fix: use the required count for the banner's scope and preceding boundary.\n  " + "\n  ".join(offenders)
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
            "Rule: `###` marks module structure, `===` marks class structure, and `---` marks a "
            "function or method procedure.\n"
            "Fix: use the symbol matching the scope, or a plain comment for prose.\n  " + "\n  ".join(offenders)
        )


def test_structural_banners_are_unnumbered(sites: list[BannerSite]) -> None:
    offenders = []
    for site in sites:
        match = _LEGAL_BANNER.fullmatch(site.text)
        if match is not None and site.symbol() in {"#", "="} and _NUMBERED_TITLE.match(match.group("name")):
            offenders.append(site.where())
    assert not offenders, (
        "Rule: module and class banners name structural groups without numbering; "
        "numbered steps belong in function or method bodies.\n  " + "\n  ".join(offenders)
    )
