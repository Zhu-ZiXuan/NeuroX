"""Code identifiers use one well-formed `__<unit>` suffix.

A unit expression made only from the standard atoms, products, quotients, and powers is legal
everywhere. Any other well-formed unit is legal only for the exact code binding named in
`NONSTANDARD_UNIT_GRANTS`. The check reads Python bindings, not comments, docstrings, or strings.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = ("neurox", "validations")

STANDARD_UNIT_ATOMS = (
    "V",
    "uA",
    "uS",
    "MOhm",
    "fF",
    "fC",
    "ns",
    "fJ",
    "uW",
    "K",
    "um",
)

# Each entry groups one module's stable code bindings whose well-formed expressions contain
# nonstandard atoms.
NONSTANDARD_UNIT_GRANTS: dict[str, tuple[str, ...]] = {
    # SI constants retain their own SI units; consumers rescale them at the boundary where needed.
    "neurox/primitive/physics.py": (
        "ELEM_CHARGE__fC",
        "K_BOLTZMANN__fJ_per_K",
        "EPS_0__fF_per_um",
    ),
    # PDK configuration quotes the foundry units without converting its human-facing surface.
    "neurox/primitive/device/mosfet.py": (
        "MosfetConfig.mu0__cm2_per_V_s",
        "MosfetConfig.A_vt__mV_um",
        "Mosfet._on_temperature_changed.nominal_mu__cm2_per_V_s",
    ),
}

UnitExpression = tuple[tuple[str, ...], tuple[str, ...]]
Binding = tuple[str, str, str, int]


def _parse_product(text: str) -> tuple[str, ...] | None:
    factors = tuple(text.split("_"))
    if not factors:
        return None
    if any(
        not factor
        or factor == "per"
        or factor[0] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        or not factor.isascii()
        or not factor.isalnum()
        for factor in factors
    ):
        return None
    return factors


def _parse_unit_expression(suffix: str) -> UnitExpression | None:
    if suffix.startswith("per_"):
        denominator = _parse_product(suffix.removeprefix("per_"))
        return ((), denominator) if denominator is not None else None

    parts = suffix.split("_per_")
    if len(parts) > 2:
        return None
    numerator = _parse_product(parts[0])
    if numerator is None:
        return None
    if len(parts) == 1:
        return numerator, ()
    denominator = _parse_product(parts[1])
    return (numerator, denominator) if denominator is not None else None


def _read_unit(name: str) -> tuple[str, UnitExpression] | None:
    cuts = [i for i in range(len(name) - 1) if name[i : i + 2] == "__"]
    cuts = [i for i in cuts if i != 0 and i != len(name) - 2]
    if not cuts:
        return None
    if len(cuts) != 1:
        raise ValueError("a unit-bearing identifier has exactly one interior `__` separator")

    suffix = name[cuts[0] + 2 :]
    expression = _parse_unit_expression(suffix)
    if expression is None:
        raise ValueError("the unit suffix is not a product, quotient, or power of unit factors")
    return suffix, expression


def _is_standard_factor(factor: str) -> bool:
    if factor in STANDARD_UNIT_ATOMS:
        return True
    for atom in sorted(STANDARD_UNIT_ATOMS, key=len, reverse=True):
        exponent = factor.removeprefix(atom) if factor.startswith(atom) else ""
        if exponent and exponent.isascii() and exponent.isdecimal() and not exponent.startswith("0"):
            return int(exponent) >= 2
    return False


def _is_standard_expression(expression: UnitExpression) -> bool:
    numerator, denominator = expression
    return all(_is_standard_factor(factor) for factor in (*numerator, *denominator))


class _BindingVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope: list[str] = []
        self.bindings: list[tuple[str, str, int]] = []

    def _add(self, name: str, line: int) -> None:
        qualname = ".".join((*self.scope, name))
        self.bindings.append((qualname, name, line))

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self._add(node.id, node.lineno)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Store):
            self._add(node.attr, node.lineno)
        self.visit(node.value)

    def visit_arg(self, node: ast.arg) -> None:
        self._add(node.arg, node.lineno)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._add(node.name, node.lineno)
        self.scope.append(node.name)
        self.visit(node.args)
        for statement in node.body:
            self.visit(statement)
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add(node.name, node.lineno)
        self.scope.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.scope.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.scope.append("<lambda>")
        self.visit(node.args)
        self.visit(node.body)
        self.scope.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add(alias.asname or alias.name.split(".", maxsplit=1)[0], node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self._add(alias.asname or alias.name, node.lineno)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self._add(node.name, node.lineno)
        for statement in node.body:
            self.visit(statement)


def _iter_bindings() -> list[Binding]:
    out: list[Binding] = []
    for root in SCAN_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            visitor = _BindingVisitor()
            visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
            relpath = str(path.relative_to(REPO_ROOT))
            seen: set[str] = set()
            for qualname, name, line in visitor.bindings:
                if qualname not in seen:
                    out.append((relpath, qualname, name, line))
                    seen.add(qualname)
    return sorted(out)


def test_the_unit_parser_reads_the_name_grammar() -> None:
    unsuffixed = ("__init__", "__all__", "__attr", "row_num")
    valid = {
        "t_wl__ns": ("ns", (("ns",), ())),
        "_sigma_beta__uA_per_V2": ("uA_per_V2", (("uA",), ("V2",))),
        "inv_scale__per_V": ("per_V", ((), ("V",))),
        "cap_density__fF_per_um2": ("fF_per_um2", (("fF",), ("um2",))),
        "constant__J_per_K": ("J_per_K", (("J",), ("K",))),
    }
    malformed = ("x__2", "a___b", "a__b__uA", "x__per_", "x__uA_per_V_per_s")

    assert all(_read_unit(name) is None for name in unsuffixed)
    assert {name: _read_unit(name) for name in valid} == valid
    for name in malformed:
        with pytest.raises(ValueError, match="unit"):
            _read_unit(name)


def test_code_unit_suffixes_are_standard_or_granted() -> None:
    grants = {(relpath, qualname) for relpath, qualnames in NONSTANDARD_UNIT_GRANTS.items() for qualname in qualnames}
    malformed: list[str] = []
    ungranted: list[str] = []
    for relpath, qualname, name, line in _iter_bindings():
        try:
            unit = _read_unit(name)
        except ValueError as error:
            malformed.append(f"{relpath}:{line}  `{qualname}`  -- {error}")
            continue
        if unit is None:
            continue
        suffix, expression = unit
        if not _is_standard_expression(expression) and (relpath, qualname) not in grants:
            ungranted.append(f"{relpath}:{line}  `{qualname}`  suffix `{suffix}`")

    if malformed or ungranted:
        sections: list[str] = []
        if malformed:
            sections.append("Malformed unit-bearing identifiers:\n  " + "\n  ".join(malformed))
        if ungranted:
            sections.append("Well-formed nonstandard units without an exact grant:\n  " + "\n  ".join(ungranted))
        pytest.fail(
            "Rule: a code binding with an interior `__` has one well-formed unit expression. An "
            "expression composed only of standard unit atoms, products, quotients, and powers is "
            "legal everywhere; any other unit requires its qualified binding in that module's "
            "NONSTANDARD_UNIT_GRANTS group.\n\n"
            + "\n\n".join(sections)
            + "\n\nFix: correct malformed `__<unit>` syntax; otherwise use standard unit atoms or add one "
            "reviewed grant for the precise binding whose external or industrial unit must remain."
        )


def test_nonstandard_unit_grants_are_exact_and_live() -> None:
    bindings = {(relpath, qualname): (name, line) for relpath, qualname, name, line in _iter_bindings()}
    faults: list[str] = []
    for relpath, qualnames in NONSTANDARD_UNIT_GRANTS.items():
        if not qualnames:
            faults.append(f"{relpath}  -- empty grant group")
        for qualname in qualnames:
            if qualnames.count(qualname) > 1:
                faults.append(f"{relpath}  `{qualname}`  -- duplicate grant")
            target = bindings.get((relpath, qualname))
            if target is None:
                faults.append(f"{relpath}  `{qualname}`  -- no such code binding")
                continue
            name, _ = target
            try:
                unit = _read_unit(name)
            except ValueError as error:
                faults.append(f"{relpath}  `{qualname}`  -- {error}")
                continue
            if unit is None:
                faults.append(f"{relpath}  `{qualname}`  -- the binding has no unit suffix")
                continue
            suffix, expression = unit
            if _is_standard_expression(expression):
                faults.append(f"{relpath}  `{qualname}`  suffix `{suffix}`  -- already a standard expression")

    if faults:
        pytest.fail(
            "Rule: every nonstandard-unit grant names one unique, live code binding whose suffix is "
            "well formed but cannot be composed solely from standard unit atoms.\n  "
            + "\n  ".join(sorted(dict.fromkeys(faults)))
            + "\n\nFix: remove a duplicate or dead grant, correct its stable qualified binding, or delete a "
            "grant made unnecessary by the standard unit-expression grammar."
        )
