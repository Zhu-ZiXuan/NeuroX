"""Imports follow package ownership, public surfaces, privacy, and dependency direction."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBRARY_ROOT = REPO_ROOT / "neurox"
IMPORT_SCAN_ROOTS = (LIBRARY_ROOT, REPO_ROOT / "validations")
CLI_ROOT = LIBRARY_ROOT / "tools"
PACKAGE = "neurox"

_ROOT_LIFT_EXCEPTIONS = {
    "common": frozenset({"Profiler", "Reporter", "check_unique_neurox_bindings", "fabricate", "stamp_names"})
}

_ALLOWED_TOP_LEVEL_DEPENDENCIES: dict[str, frozenset[str]] = {
    "common": frozenset({"common"}),
    "primitive": frozenset({"common", "primitive"}),
    "architecture": frozenset({"architecture", "common", "primitive"}),
    "works": frozenset({"architecture", "common", "primitive", "works"}),
    "tools": frozenset({"architecture", "common", "primitive", "tools", "works"}),
}


def _is_cli_path(path: Path) -> bool:
    return path == CLI_ROOT or CLI_ROOT in path.parents


def _iter_source_files() -> list[Path]:
    return sorted(path for root in IMPORT_SCAN_ROOTS for path in root.rglob("*.py"))


def _iter_library_files() -> list[Path]:
    return sorted(LIBRARY_ROOT.rglob("*.py"))


def _iter_package_faces() -> list[Path]:
    return sorted(path for path in LIBRARY_ROOT.rglob("__init__.py") if not _is_cli_path(path))


def _package_parts(path: Path) -> tuple[str, ...]:
    return path.parent.relative_to(REPO_ROOT).parts


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)


def _dotted_path(name: str) -> Path:
    return REPO_ROOT.joinpath(*name.split("."))


def _module_file(name: str) -> Path | None:
    path = _dotted_path(name).with_suffix(".py")
    return path if path.is_file() else None


def _package_face(name: str) -> Path | None:
    path = _dotted_path(name) / "__init__.py"
    return path if path.is_file() else None


def _declared_exports(package_face: Path) -> list[str] | None:
    tree = ast.parse(package_face.read_text(encoding="utf-8"), filename=str(package_face))
    declarations: list[ast.Assign | ast.AnnAssign] = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
        )
        or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__")
    ]

    if len(declarations) != 1:
        return None
    declaration = declarations[0]
    if (
        not isinstance(declaration, ast.Assign)
        or len(declaration.targets) != 1
        or not isinstance(declaration.targets[0], ast.Name)
        or not isinstance(declaration.value, ast.List)
        or not all(isinstance(item, ast.Constant) and isinstance(item.value, str) for item in declaration.value.elts)
    ):
        return None
    return [
        item.value for item in declaration.value.elts if isinstance(item, ast.Constant) and isinstance(item.value, str)
    ]


def _relative_target(importer: Path, name: str) -> tuple[Path | None, Path | None]:
    stem = importer.parent / name
    module = stem.with_suffix(".py")
    package = stem / "__init__.py"
    return (module if module.is_file() else None, package if package.is_file() else None)


def _is_strict_ancestor(candidate: tuple[str, ...], descendant: tuple[str, ...]) -> bool:
    return len(candidate) < len(descendant) and descendant[: len(candidate)] == candidate


def _package_relationship(importer: tuple[str, ...], target: tuple[str, ...]) -> str:
    if target == importer:
        return "same package"
    if len(target) == len(importer) + 1 and target[:-1] == importer:
        return "direct child package"
    if _is_strict_ancestor(target, importer):
        return "strict ancestor package"
    return "other package"


def _resolve_from(node: ast.ImportFrom, package: tuple[str, ...]) -> str | None:
    if node.level == 0:
        return node.module
    kept = len(package) - (node.level - 1)
    if kept <= 0:
        return None
    base = ".".join(package[:kept])
    return f"{base}.{node.module}" if node.module else base


def _iter_imports(path: Path) -> list[tuple[int, str, list[str]]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = _package_parts(path)
    imports: list[tuple[int, str, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.append((node.lineno, ast.unparse(node), [alias.name for alias in node.names]))
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_from(node, package)
            candidates = [] if resolved is None else [resolved, *(f"{resolved}.{alias.name}" for alias in node.names)]
            imports.append((node.lineno, ast.unparse(node), candidates))
    return imports


def _private_modules() -> dict[str, Path]:
    return {
        _module_name(path): path
        for path in _iter_library_files()
        if path.stem.startswith("_") and not path.stem.startswith("__")
    }


def _private_targets(candidates: list[str], private_modules: dict[str, Path]) -> set[str]:
    targets: set[str] = set()
    for candidate in candidates:
        parts = candidate.split(".")
        for end in range(1, len(parts) + 1):
            target = ".".join(parts[:end])
            if target in private_modules:
                targets.add(target)
    return targets


def _top_level_roles(candidates: list[str]) -> set[str]:
    roles: set[str] = set()
    for candidate in candidates:
        parts = candidate.split(".")
        if len(parts) >= 2 and parts[0] == PACKAGE and parts[1] in _ALLOWED_TOP_LEVEL_DEPENDENCIES:
            roles.add(parts[1])
    return roles


def test_package_surfaces_are_static_and_explicit() -> None:
    malformed: list[str] = []
    duplicates: list[str] = []
    private_names: list[str] = []
    private_sources: list[str] = []
    private_modules = _private_modules()

    for path in _iter_package_faces():
        exports = _declared_exports(path)
        if exports is None:
            malformed.append(str(path.relative_to(REPO_ROOT)))
        else:
            repeated = sorted({name for name in exports if exports.count(name) > 1})
            if repeated:
                duplicates.append(f"{path.relative_to(REPO_ROOT)} ({', '.join(repeated)})")
            hidden = sorted(name for name in exports if name.startswith("_"))
            if hidden:
                private_names.append(f"{path.relative_to(REPO_ROOT)} ({', '.join(hidden)})")

        for lineno, source, candidates in _iter_imports(path):
            private_sources.extend(
                f"{path.relative_to(REPO_ROOT)}:{lineno} {source} (source: {target})"
                for target in sorted(_private_targets(candidates, private_modules))
            )

    if malformed or duplicates or private_names or private_sources:
        details: list[str] = []
        if malformed:
            details.append("Missing or noncanonical `__all__` declarations:\n  " + "\n  ".join(malformed))
        if duplicates:
            details.append("Repeated exports:\n  " + "\n  ".join(duplicates))
        if private_names:
            details.append("Private names declared public:\n  " + "\n  ".join(private_names))
        if private_sources:
            details.append("Package faces importing private files:\n  " + "\n  ".join(private_sources))
        pytest.fail(
            "Rule: every importable library package declares its public face once as an unannotated "
            "literal string list. Its entries are unique and public, and no public face draws from a "
            "leading-underscore file. Command-line-only packages under `neurox/tools` are exempt.\n"
            "Fix: write one `__all__ = [...]`; move a public definition to a non-underscore file before "
            "exporting it.\n" + "\n".join(details)
        )


def test_parent_faces_do_not_lift_child_package_members() -> None:
    offenders: list[str] = []
    for path in _iter_package_faces():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in (item for item in ast.walk(tree) if isinstance(item, ast.ImportFrom)):
            if node.level != 1 or node.module is None:
                continue
            child = node.module.split(".", 1)[0]
            if not (path.parent / child / "__init__.py").is_file():
                continue
            lifted = []
            for alias in node.names:
                allowed = (
                    path == LIBRARY_ROOT / "__init__.py"
                    and node.module in _ROOT_LIFT_EXCEPTIONS
                    and alias.name in _ROOT_LIFT_EXCEPTIONS[node.module]
                    and alias.asname is None
                )
                if not allowed:
                    lifted.append(alias.name)
            if lifted:
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno} {ast.unparse(node)} "
                    f"(lifted: {', '.join(sorted(lifted))})"
                )
    if offenders:
        pytest.fail(
            "Rule: a package face may expose a direct child package object, but does not copy that "
            "child package's members into the parent namespace. The root's named user entry points are "
            "the only exceptions.\nFix: keep `from . import child_package` and let callers descend through "
            "the child package's own face.\nOffenders:\n  " + "\n  ".join(offenders)
        )


def test_relative_imports_reach_only_direct_files_or_child_package_faces() -> None:
    invalid_targets: list[str] = []
    unexported: list[str] = []
    for path in _iter_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in (item for item in ast.walk(tree) if isinstance(item, ast.ImportFrom)):
            if node.level != 1:
                continue
            if node.module is None:
                unresolved = [
                    alias.name
                    for alias in node.names
                    if alias.name != "*" and not any(_relative_target(path, alias.name))
                ]
                if unresolved:
                    invalid_targets.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno} {ast.unparse(node)} "
                        f"(not direct children: {', '.join(unresolved)})"
                    )
                continue
            if "." in node.module:
                invalid_targets.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} {ast.unparse(node)}")
                continue
            module, package = _relative_target(path, node.module)
            if module is None and package is None:
                invalid_targets.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} {ast.unparse(node)}")
                continue
            if package is not None:
                exports = _declared_exports(package)
                if exports is None:
                    continue
                missing = sorted(alias.name for alias in node.names if alias.name != "*" and alias.name not in exports)
                if missing:
                    unexported.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno} {ast.unparse(node)} "
                        f"(not in {package.parent.relative_to(REPO_ROOT)}/__all__: {', '.join(missing)})"
                    )
    if invalid_targets or unexported:
        details: list[str] = []
        if invalid_targets:
            details.append("Targets beyond a direct child:\n  " + "\n  ".join(invalid_targets))
        if unexported:
            details.append("Names outside the child package face:\n  " + "\n  ".join(unexported))
        pytest.fail(
            "Rule: a single-dot relative import reaches only a sibling file or a direct child package; "
            "a child package is used through names in its `__all__`. Ruff separately forbids parent "
            "relative imports.\nFix: import a sibling file as `from .module import Name`, or a direct child "
            "package as `from .subpackage import PublicName`.\n" + "\n".join(details)
        )


def test_absolute_imports_match_ownership_and_package_surfaces() -> None:
    wrong_form: list[str] = []
    concrete_crossings: list[str] = []
    unexported: list[str] = []

    for path in _iter_source_files():
        importer_package = _package_parts(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module != PACKAGE and not node.module.startswith(f"{PACKAGE}."):
                    continue
                module = _module_file(node.module)
                if module is not None:
                    owner = tuple(node.module.split("."))[:-1]
                    relationship = _package_relationship(importer_package, owner)
                    if relationship == "strict ancestor package":
                        continue
                    offender = f"{path.relative_to(REPO_ROOT)}:{node.lineno} {ast.unparse(node)}"
                    if relationship == "same package":
                        wrong_form.append(f"{offender} (sibling file requires a relative import)")
                    else:
                        concrete_crossings.append(f"{offender} ({relationship})")
                    continue

                package = _package_face(node.module)
                if package is None:
                    continue
                relationship = _package_relationship(importer_package, tuple(node.module.split(".")))
                offender = f"{path.relative_to(REPO_ROOT)}:{node.lineno} {ast.unparse(node)}"
                if relationship != "other package":
                    wrong_form.append(f"{offender} ({relationship})")
                    continue
                exports = _declared_exports(package)
                if exports is None:
                    continue
                missing = sorted(alias.name for alias in node.names if alias.name != "*" and alias.name not in exports)
                if missing:
                    unexported.append(
                        f"{offender} (not in {package.parent.relative_to(REPO_ROOT)}/__all__: {', '.join(missing)})"
                    )

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name != PACKAGE and not alias.name.startswith(f"{PACKAGE}."):
                        continue
                    module = _module_file(alias.name)
                    target_package = tuple(alias.name.split("."))
                    if module is not None:
                        owner = target_package[:-1]
                        relationship = _package_relationship(importer_package, owner)
                        if relationship == "strict ancestor package":
                            continue
                        offender = f"{path.relative_to(REPO_ROOT)}:{node.lineno} {ast.unparse(node)}"
                        if relationship == "same package":
                            wrong_form.append(f"{offender} (sibling file requires a relative import)")
                        else:
                            concrete_crossings.append(f"{offender} ({relationship})")
                        continue
                    if _package_face(alias.name) is None:
                        continue
                    relationship = _package_relationship(importer_package, target_package)
                    if relationship != "other package":
                        wrong_form.append(
                            f"{path.relative_to(REPO_ROOT)}:{node.lineno} {ast.unparse(node)} ({relationship})"
                        )

    if wrong_form or concrete_crossings or unexported:
        details: list[str] = []
        if wrong_form:
            details.append("Imports using the wrong relative/absolute form:\n  " + "\n  ".join(wrong_form))
        if concrete_crossings:
            details.append("Concrete files crossed outside their ownership case:\n  " + "\n  ".join(concrete_crossings))
        if unexported:
            details.append("Imports bypassing a package's `__all__` face:\n  " + "\n  ".join(unexported))
        pytest.fail(
            "Rule: sibling files and direct child packages use single-dot relative imports; a descendant "
            "uses a strict ancestor's directly owned file by absolute path; every other NeuroX dependency "
            "uses the target package's absolute `__all__` face. The same rule applies to library, tool, "
            "and validation callers.\nFix: choose the import form from the ownership relationship, and "
            "export cross-package names from their owning package.\n" + "\n".join(details)
        )


def test_private_modules_stay_inside_their_package_subtree() -> None:
    private_modules = _private_modules()
    breaches: set[tuple[Path, int, str, Path]] = set()
    for importer in _iter_source_files():
        for lineno, _, candidates in _iter_imports(importer):
            for target in _private_targets(candidates, private_modules):
                private_module = private_modules[target]
                if not importer.is_relative_to(private_module.parent):
                    breaches.add((importer, lineno, target, private_module.parent))

    if breaches:
        offenders = [
            f"{importer.relative_to(REPO_ROOT)}:{lineno} imports {target} (private to {owner.relative_to(REPO_ROOT)}/)"
            for importer, lineno, target, owner in sorted(breaches, key=lambda item: (str(item[0]), *item[1:3]))
        ]
        pytest.fail(
            "Rule: a leading-underscore module is private to its containing package subtree and cannot "
            "supply a public package face.\nFix: keep the helper where its only users are; if it owns a "
            "public API, give the file a public non-underscore name.\nOffenders:\n  " + "\n  ".join(offenders)
        )


def test_top_level_dependencies_follow_declared_roles() -> None:
    source_roles = {
        path.relative_to(LIBRARY_ROOT).parts[0] for path in _iter_library_files() if path.parent != LIBRARY_ROOT
    }
    unruled = sorted(source_roles - _ALLOWED_TOP_LEVEL_DEPENDENCIES.keys())
    if unruled:
        pytest.fail(
            "Rule: every top-level Python source directory has an explicit dependency role.\n"
            f"Unruled: {', '.join(unruled)}\n"
            "Fix: add its allowed top-level dependencies to `_ALLOWED_TOP_LEVEL_DEPENDENCIES`."
        )

    violations: list[str] = []
    for path in _iter_library_files():
        relative = path.relative_to(LIBRARY_ROOT)
        if len(relative.parts) < 2:
            continue
        importer_role = relative.parts[0]
        allowed = _ALLOWED_TOP_LEVEL_DEPENDENCIES.get(importer_role)
        if allowed is None:
            continue
        for lineno, source, candidates in _iter_imports(path):
            violations.extend(
                (f"{path.relative_to(REPO_ROOT)}:{lineno} [{importer_role}] {source} -> [{imported_role}]")
                for imported_role in sorted(_top_level_roles(candidates) - allowed)
            )

    if violations:
        pytest.fail(
            "Rule: top-level roles follow their declared dependency directions. `works` contains concrete "
            "design extensions and `tools` contains terminal consumers; neither role is an abstraction "
            "rank above the core library.\nOffenders:\n  " + "\n  ".join(violations)
        )
