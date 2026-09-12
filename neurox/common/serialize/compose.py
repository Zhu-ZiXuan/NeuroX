"""Resolve, merge, and select configuration mappings."""

from __future__ import annotations

from pathlib import Path

from .file import dict_from_file
from .keys import (
    CLASS_DISCRIMINATOR,
    USE_DIRECTIVE,
    USE_PRESET_DIRECTIVE,
)
from .value import ConfigDict, ConfigValue


def _deep_fill_defaults(override: ConfigDict, default: ConfigDict, strict_type: bool) -> ConfigDict:
    """Fill missing keys in the override mapping from the default one, recursively."""
    merged = override.copy()
    for k, d_v in default.items():
        if k not in merged:
            merged[k] = d_v
            continue
        o_v = merged[k]
        if isinstance(o_v, dict) and isinstance(d_v, dict):
            merged[k] = _deep_fill_defaults(o_v, d_v, strict_type=strict_type)
        elif strict_type and isinstance(o_v, dict) != isinstance(d_v, dict):
            raise ValueError(
                f"Cannot merge key '{k}': type mismatch. override={type(o_v).__name__}, default={type(d_v).__name__}",
            )
    return merged


def merge_dicts(*dicts: ConfigDict, strict_type: bool = True) -> ConfigDict:
    """Deep-merge dictionaries in descending priority order.

    Args:
        dicts: Earlier dictionaries take precedence over later ones.
        strict_type: Reject a dictionary/non-dictionary conflict instead of
            resolving it by priority.

    Returns:
        New merged dictionary; inputs are unchanged.

    Raises:
        ValueError: A dictionary/non-dictionary conflict occurs in strict mode.
    """
    if not dicts:
        return {}
    if len(dicts) == 1:
        return dicts[0].copy()

    merged = dicts[0].copy()
    for d in dicts[1:]:
        merged = _deep_fill_defaults(merged, d, strict_type=strict_type)
    return merged


def _lookup_section(root: ConfigDict, section: str) -> ConfigValue:
    """Look up one section of a mapping; a dotted name descends nested tables.

    An exact top-level key wins; otherwise the name is split on `.` and walked
    table by table, so `"a.b.c"` reaches the `[a.b.c]` TOML table.

    Raises:
        TypeError: A path segment is reached inside a non-table.
        KeyError: A path segment is absent.
    """
    if section in root:
        return root[section]
    node: ConfigValue = root
    for part in section.split("."):
        if not isinstance(node, dict):
            raise TypeError(f"section {section!r}: segment {part!r} sits inside a {type(node).__name__}, not a table")
        if part not in node:
            raise KeyError(f"section {section!r} not found: segment {part!r} missing (available keys: {sorted(node)})")
        node = node[part]
    return node


def _resolve_fragment_path(rel: str, base_dir: Path) -> Path:
    """Resolve a `_neurox_use` relative path to an existing file.

    A suffix-free path tries `.toml`, then `.yaml` / `.yml`.
    """
    candidate = base_dir / rel
    if candidate.exists():
        return candidate
    if candidate.suffix == "":
        for suffix in (".toml", ".yaml", ".yml"):
            with_suffix = candidate.with_suffix(suffix)
            if with_suffix.exists():
                return with_suffix
    raise FileNotFoundError(f"{USE_DIRECTIVE} fragment {rel!r} not found relative to {base_dir}")


def _parse_use_ref(ref: ConfigValue, base_dir: Path) -> tuple[Path, str]:
    """Parse `"<rel_path>:<section>"` into `(absolute_path, section_name)`."""
    if not isinstance(ref, str):
        raise TypeError(f"{USE_DIRECTIVE} must be a string, got {type(ref).__name__}")
    if ":" not in ref:
        raise ValueError(f"{USE_DIRECTIVE} reference {ref!r} missing ':' (expected '<path>:<section>')")
    rel, section = ref.split(":", 1)
    if not rel:
        raise ValueError(f"{USE_DIRECTIVE} reference {ref!r} has an empty path")
    if not section:
        raise ValueError(f"{USE_DIRECTIVE} reference {ref!r} has an empty section")
    return _resolve_fragment_path(rel, base_dir), section


def _presets_root() -> Path:
    """Return the installed `neurox/presets/` path."""
    import importlib.resources

    return Path(str(importlib.resources.files("neurox") / "presets"))


def _validate_preset_ref_path(rel: str) -> None:
    """Reject preset paths that are absolute or try to escape the presets root."""
    if not rel:
        raise ValueError(f"{USE_PRESET_DIRECTIVE} path is empty")
    if rel.startswith(("./", "/", "\\")):
        raise ValueError(f"{USE_PRESET_DIRECTIVE} path must not start with './' or be absolute: {rel!r}")
    if any(part == ".." for part in rel.replace("\\", "/").split("/")):
        raise ValueError(f"{USE_PRESET_DIRECTIVE} path must not contain '..' segments: {rel!r}")


def _resolve_preset_fragment_path(rel: str) -> Path:
    """Resolve a preset-relative path to an existing file under `neurox/presets/`."""
    root = _presets_root()
    candidate = root / rel
    if candidate.is_file():
        return candidate
    if candidate.suffix == "":
        for suffix in (".toml", ".yaml", ".yml"):
            with_suffix = candidate.with_suffix(suffix)
            if with_suffix.is_file():
                return with_suffix
    raise FileNotFoundError(f"{USE_PRESET_DIRECTIVE} fragment {rel!r} not found under {root}")


def parse_preset_ref(ref: str) -> tuple[Path, str]:
    """Parse a preset `"<rel_path>:<section>"` anchored at `neurox/presets/`."""
    if not isinstance(ref, str):
        raise TypeError(f"{USE_PRESET_DIRECTIVE} must be a string, got {type(ref).__name__}")
    if ":" not in ref:
        raise ValueError(f"{USE_PRESET_DIRECTIVE} reference {ref!r} missing ':' (expected '<path>:<section>')")
    rel, section = ref.split(":", 1)
    if not rel:
        raise ValueError(f"{USE_PRESET_DIRECTIVE} reference {ref!r} has an empty path")
    if not section:
        raise ValueError(f"{USE_PRESET_DIRECTIVE} reference {ref!r} has an empty section")
    _validate_preset_ref_path(rel)
    return _resolve_preset_fragment_path(rel), section


def _resolve_directive_branch(
    value: ConfigDict,
    *,
    directive: str,
    path: Path,
    section: str,
    base_dir_for_fragment: Path,
    base_dir_for_inline: Path,
    in_preset_for_fragment: bool,
    in_preset_for_inline: bool,
    cache: dict[Path, ConfigDict],
    in_progress: frozenset[tuple[Path, str]],
) -> ConfigDict:
    """Resolve and merge one referenced configuration fragment."""
    key = (path, section)
    if key in in_progress:
        trail = " -> ".join(f"{p.name}:{s}" for p, s in in_progress)
        raise ValueError(f"{directive} cycle detected: {trail} -> {path.name}:{section}")
    if path not in cache:
        cache[path] = dict_from_file(path)
    root = cache[path]
    try:
        target = _lookup_section(root, section)
    except KeyError as exc:
        raise KeyError(f"{directive} target in {path}: {exc.args[0]}") from None
    if not isinstance(target, dict):
        raise TypeError(f"{directive} target {value[directive]!r} must be a table, got {type(target).__name__}")
    resolved_fragment = _resolve_uses_in_dict(
        target,
        base_dir_for_fragment,
        cache=cache,
        in_progress=in_progress | {key},
        in_preset=in_preset_for_fragment,
    )
    inline = {k: v for k, v in value.items() if k != directive}
    resolved_inline = _resolve_uses_in_dict(
        inline,
        base_dir_for_inline,
        cache=cache,
        in_progress=in_progress,
        in_preset=in_preset_for_inline,
    )
    return merge_dicts(resolved_inline, resolved_fragment, strict_type=True)


def _resolve_uses_in_dict(
    value: ConfigDict,
    base_dir: Path,
    *,
    cache: dict[Path, ConfigDict],
    in_progress: frozenset[tuple[Path, str]],
    in_preset: bool = False,
) -> ConfigDict:
    """Recursively resolve use directives in one configuration mapping."""
    has_use = USE_DIRECTIVE in value
    has_preset = USE_PRESET_DIRECTIVE in value
    if has_use and has_preset:
        raise ValueError(f"{USE_DIRECTIVE!r} and {USE_PRESET_DIRECTIVE!r} are mutually exclusive in the same table")
    if (has_use or has_preset) and CLASS_DISCRIMINATOR in value:
        directive = USE_DIRECTIVE if has_use else USE_PRESET_DIRECTIVE
        raise ValueError(
            f"{directive!r} table may not also declare {CLASS_DISCRIMINATOR!r}; "
            f"the referenced fragment/preset is the sole class authority "
            f"(table keys: {sorted(value)})"
        )
    if in_preset and has_use:
        raise ValueError(f"{USE_DIRECTIVE!r} is forbidden inside neurox/presets/; use {USE_PRESET_DIRECTIVE!r} instead")
    if has_preset:
        ref = value[USE_PRESET_DIRECTIVE]
        if not isinstance(ref, str):
            raise TypeError(f"{USE_PRESET_DIRECTIVE} must be a string, got {type(ref).__name__}")
        path, section = parse_preset_ref(ref)
        return _resolve_directive_branch(
            value,
            directive=USE_PRESET_DIRECTIVE,
            path=path,
            section=section,
            base_dir_for_fragment=_presets_root(),
            base_dir_for_inline=base_dir,
            in_preset_for_fragment=True,
            in_preset_for_inline=in_preset,
            cache=cache,
            in_progress=in_progress,
        )
    if has_use:
        path, section = _parse_use_ref(value[USE_DIRECTIVE], base_dir)
        return _resolve_directive_branch(
            value,
            directive=USE_DIRECTIVE,
            path=path,
            section=section,
            base_dir_for_fragment=path.parent,
            base_dir_for_inline=base_dir,
            in_preset_for_fragment=in_preset,
            in_preset_for_inline=in_preset,
            cache=cache,
            in_progress=in_progress,
        )
    return {
        key: _resolve_uses_in_value(item, base_dir, cache=cache, in_progress=in_progress, in_preset=in_preset)
        for key, item in value.items()
    }


def _resolve_uses_in_value(
    value: ConfigValue,
    base_dir: Path,
    *,
    cache: dict[Path, ConfigDict],
    in_progress: frozenset[tuple[Path, str]],
    in_preset: bool = False,
) -> ConfigValue:
    """Recursively resolve use directives in one configuration value."""
    if isinstance(value, dict):
        return _resolve_uses_in_dict(
            value,
            base_dir,
            cache=cache,
            in_progress=in_progress,
            in_preset=in_preset,
        )
    if isinstance(value, list):
        return [
            _resolve_uses_in_value(x, base_dir, cache=cache, in_progress=in_progress, in_preset=in_preset)
            for x in value
        ]
    return value


def resolve_uses(data: ConfigDict, base_dir: Path) -> ConfigDict:
    """Expand every `_neurox_use` / `_neurox_use_preset` directive in a mapping.

    `_neurox_use = "<rel_path>:<section>"` resolves the path relative to
    `base_dir` (the directory of the file containing the directive) and pulls
    the named section from that file (a dotted section name descends nested
    tables); inline keys override the fragment. `_neurox_use_preset` follows the
    same merge semantics but resolves paths from `neurox/presets/` and forbids
    `_neurox_use` inside the preset subtree.

    Returns:
        New dict with every directive expanded.

    Raises:
        ValueError: A malformed reference, a resolution cycle, a preset path
            that is not forward-relative, or a `_neurox_use` reached inside a
            preset subtree.
        FileNotFoundError: A referenced fragment file does not exist.
        KeyError: The referenced section is absent from the target file.
        TypeError: A directive value, or the section it names, is not the
            expected type.
    """
    return _resolve_uses_in_dict(data, base_dir, cache={}, in_progress=frozenset())


def _pluck_section(data: ConfigDict, section: str | None) -> ConfigDict:
    if section is None:
        return data
    sub = _lookup_section(data, section)
    if not isinstance(sub, dict):
        raise TypeError(f"Section '{section}' must be a table, got {type(sub).__name__}")
    return sub


def load_config_dict(
    *files: Path,
    section: str | None = None,
    encoding: str | None = "utf-8",
    strict_type: bool = True,
) -> ConfigDict:
    """Load, resolve, merge, and pluck one or more config files into a plain dict.

    Each file is parsed, its `_neurox_use` / `_neurox_use_preset` directives are
    expanded (relative to that file's own directory), then `section` is plucked
    (if given). The per-file results are merged in descending priority (first
    wins).

    Args:
        files: Config file paths, ordered by descending priority.
        section: Optional table name to extract from each file; a dotted
            name descends nested tables.
        encoding: YAML text encoding (ignored for TOML).
        strict_type: Reject a dict / non-dict conflict during the merge.

    Returns:
        The merged dict, ready for coercion into a dataclass.

    Raises:
        ValueError: No files were given (at least one is required).
    """
    if not files:
        raise ValueError("At least one config file must be provided")
    raw = [
        _pluck_section(
            resolve_uses(dict_from_file(f, encoding=encoding), base_dir=f.parent),
            section,
        )
        for f in files
    ]
    return merge_dicts(*raw, strict_type=strict_type)
