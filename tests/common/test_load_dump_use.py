"""Tests for the ``_neurox_use`` / ``_neurox_use_preset`` directives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import pytest

from neurox.common.serialize import compose, dict_from_file
from neurox.common.serialize.build import dataclass_from_dict
from neurox.common.serialize.compose import load_config_dict, resolve_uses

T = TypeVar("T")


@dataclass(frozen=True)
class _Inner:
    a: float
    b: float


@dataclass(frozen=True)
class _Outer:
    name: str
    inner: _Inner


# --- fixtures ---


def _write(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def dataclass_from_file(
    cls: type[T],
    *files: Path,
    section: str | None = None,
) -> T:
    """Load-resolve-merge-coerce a config file into ``cls`` (white-box test helper)."""
    return dataclass_from_dict(cls, load_config_dict(*files, section=section))


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    return tmp_path


# --- basic resolution ---


def test_basic_reference_expands_fragment(config_dir: Path) -> None:
    _write(config_dir / "frag.toml", "[piece]\na = 1.0\nb = 2.0\n")
    _write(config_dir / "main.toml", '[outer]\nname = "x"\n[outer.inner]\n_neurox_use = "frag:piece"\n')
    obj = dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")
    assert obj == _Outer(name="x", inner=_Inner(a=1.0, b=2.0))


def test_inline_keys_override_fragment(config_dir: Path) -> None:
    _write(config_dir / "frag.toml", "[piece]\na = 1.0\nb = 2.0\n")
    _write(
        config_dir / "main.toml",
        '[outer]\nname = "x"\n[outer.inner]\n_neurox_use = "frag:piece"\nb = 9.5\n',
    )
    obj = dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")
    assert obj == _Outer(name="x", inner=_Inner(a=1.0, b=9.5))


def test_same_fragment_broadcast_to_two_slots(config_dir: Path) -> None:
    @dataclass(frozen=True)
    class TwoInners:
        left: _Inner
        right: _Inner

    _write(config_dir / "frag.toml", "[piece]\na = 3.0\nb = 4.0\n")
    _write(
        config_dir / "main.toml",
        '[outer.left]\n_neurox_use = "frag:piece"\n[outer.right]\n_neurox_use = "frag:piece"\n',
    )
    obj = dataclass_from_file(TwoInners, config_dir / "main.toml", section="outer")
    assert obj == TwoInners(left=_Inner(a=3.0, b=4.0), right=_Inner(a=3.0, b=4.0))


def test_suffixless_path_tries_toml_then_yaml(config_dir: Path) -> None:
    _write(config_dir / "frag.toml", "[piece]\na = 5.0\nb = 6.0\n")
    _write(config_dir / "main.toml", '[outer]\nname = "y"\n[outer.inner]\n_neurox_use = "frag:piece"\n')
    obj = dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")
    assert obj == _Outer(name="y", inner=_Inner(a=5.0, b=6.0))


def test_explicit_suffix_also_works(config_dir: Path) -> None:
    _write(config_dir / "frag.toml", "[piece]\na = 7.0\nb = 8.0\n")
    _write(config_dir / "main.toml", '[outer]\nname = "z"\n[outer.inner]\n_neurox_use = "frag.toml:piece"\n')
    obj = dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")
    assert obj == _Outer(name="z", inner=_Inner(a=7.0, b=8.0))


# --- recursion ---


def test_nested_use_inside_fragment_resolves(config_dir: Path) -> None:
    _write(config_dir / "leaf.toml", "[atom]\na = 10.0\nb = 20.0\n")
    _write(config_dir / "mid.toml", '[piece]\n_neurox_use = "leaf:atom"\n')
    _write(config_dir / "main.toml", '[outer]\nname = "n"\n[outer.inner]\n_neurox_use = "mid:piece"\n')
    obj = dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")
    assert obj == _Outer(name="n", inner=_Inner(a=10.0, b=20.0))


def test_mid_layer_can_override_leaf(config_dir: Path) -> None:
    _write(config_dir / "leaf.toml", "[atom]\na = 10.0\nb = 20.0\n")
    _write(config_dir / "mid.toml", '[piece]\n_neurox_use = "leaf:atom"\nb = 99.0\n')
    _write(config_dir / "main.toml", '[outer]\nname = "n"\n[outer.inner]\n_neurox_use = "mid:piece"\n')
    obj = dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")
    assert obj == _Outer(name="n", inner=_Inner(a=10.0, b=99.0))


def test_main_inline_overrides_chain(config_dir: Path) -> None:
    _write(config_dir / "leaf.toml", "[atom]\na = 10.0\nb = 20.0\n")
    _write(config_dir / "mid.toml", '[piece]\n_neurox_use = "leaf:atom"\nb = 99.0\n')
    _write(
        config_dir / "main.toml",
        '[outer]\nname = "n"\n[outer.inner]\n_neurox_use = "mid:piece"\nb = 1.0\n',
    )
    obj = dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")
    assert obj == _Outer(name="n", inner=_Inner(a=10.0, b=1.0))


# --- errors ---


def test_cycle_is_rejected(config_dir: Path) -> None:
    _write(config_dir / "a.toml", '[piece]\n_neurox_use = "b:piece"\n')
    _write(config_dir / "b.toml", '[piece]\n_neurox_use = "a:piece"\n')
    _write(config_dir / "main.toml", '[outer]\nname = "c"\n[outer.inner]\n_neurox_use = "a:piece"\n')
    with pytest.raises(ValueError, match="_neurox_use cycle detected"):
        dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")


def test_missing_section_raises(config_dir: Path) -> None:
    _write(config_dir / "frag.toml", "[piece]\na = 1.0\nb = 2.0\n")
    _write(config_dir / "main.toml", '[outer]\nname = "x"\n[outer.inner]\n_neurox_use = "frag:nonexistent"\n')
    with pytest.raises(KeyError, match="nonexistent"):
        dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")


def test_missing_file_raises(config_dir: Path) -> None:
    _write(config_dir / "main.toml", '[outer]\nname = "x"\n[outer.inner]\n_neurox_use = "no_such:piece"\n')
    with pytest.raises(FileNotFoundError, match="no_such"):
        dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")


def test_malformed_use_string_raises(config_dir: Path) -> None:
    _write(config_dir / "main.toml", '[outer]\nname = "x"\n[outer.inner]\n_neurox_use = "missing_separator"\n')
    with pytest.raises(ValueError, match="missing ':'"):
        dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")


def test_use_value_must_be_string(config_dir: Path) -> None:
    _write(config_dir / "main.toml", '[outer]\nname = "x"\n[outer.inner]\n_neurox_use = 42\n')
    with pytest.raises(TypeError, match="_neurox_use must be a string"):
        dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")


# --- pure dict-level resolver ---


def test_resolve_uses_pure_dict_form(config_dir: Path) -> None:
    _write(config_dir / "frag.toml", "[piece]\na = 1.0\nb = 2.0\n")
    raw = {"outer": {"inner": {"_neurox_use": "frag:piece", "b": 5.0}}}
    expanded = resolve_uses(raw, base_dir=config_dir)
    assert expanded == {"outer": {"inner": {"a": 1.0, "b": 5.0}}}


def test_resolve_uses_preserves_no_use_data(config_dir: Path) -> None:
    raw = {"only": {"a": 1, "b": 2}, "list_ish": [1, 2, 3]}
    assert resolve_uses(raw, base_dir=config_dir) == raw


def test_dict_from_file_keeps_raw_use(config_dir: Path) -> None:
    _write(config_dir / "frag.toml", "[piece]\na = 1.0\nb = 2.0\n")
    _write(config_dir / "main.toml", '[outer.inner]\n_neurox_use = "frag:piece"\n')
    raw = dict_from_file(config_dir / "main.toml")
    assert raw == {"outer": {"inner": {"_neurox_use": "frag:piece"}}}


# --- _neurox_use_preset --------------------------------------------------------


@pytest.fixture
def presets_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Override ``_presets_root()`` to a hermetic tmp directory for the test."""
    root = tmp_path / "presets"
    root.mkdir()
    monkeypatch.setattr(compose, "_presets_root", lambda: root)
    return root


def test_preset_basic_resolution(config_dir: Path, presets_root: Path) -> None:
    _write(presets_root / "frag.toml", "[piece]\na = 1.0\nb = 2.0\n")
    _write(config_dir / "main.toml", '[outer]\nname = "x"\n[outer.inner]\n_neurox_use_preset = "frag:piece"\n')
    obj = dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")
    assert obj == _Outer(name="x", inner=_Inner(a=1.0, b=2.0))


def test_preset_resolves_from_subdirectory(config_dir: Path, presets_root: Path) -> None:
    (presets_root / "process").mkdir()
    _write(presets_root / "process" / "rram.toml", "[rram_x]\na = 11.0\nb = 22.0\n")
    _write(
        config_dir / "main.toml",
        '[outer]\nname = "p"\n[outer.inner]\n_neurox_use_preset = "process/rram:rram_x"\n',
    )
    obj = dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")
    assert obj == _Outer(name="p", inner=_Inner(a=11.0, b=22.0))


def test_preset_chains_via_preset_only(config_dir: Path, presets_root: Path) -> None:
    _write(presets_root / "leaf.toml", "[atom]\na = 7.0\nb = 8.0\n")
    _write(presets_root / "mid.toml", '[piece]\n_neurox_use_preset = "leaf:atom"\n')
    _write(
        config_dir / "main.toml",
        '[outer]\nname = "c"\n[outer.inner]\n_neurox_use_preset = "mid:piece"\n',
    )
    obj = dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")
    assert obj == _Outer(name="c", inner=_Inner(a=7.0, b=8.0))


def test_preset_inline_overrides_fragment(config_dir: Path, presets_root: Path) -> None:
    _write(presets_root / "frag.toml", "[piece]\na = 1.0\nb = 2.0\n")
    _write(
        config_dir / "main.toml",
        '[outer]\nname = "x"\n[outer.inner]\n_neurox_use_preset = "frag:piece"\nb = 99.0\n',
    )
    obj = dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")
    assert obj == _Outer(name="x", inner=_Inner(a=1.0, b=99.0))


def test_preset_forbids_neurox_use_inside_subtree(config_dir: Path, presets_root: Path) -> None:
    _write(presets_root / "leaf.toml", "[atom]\na = 1.0\nb = 2.0\n")
    # Preset tries to chain via the user-side directive — forbidden.
    _write(presets_root / "bad.toml", '[piece]\n_neurox_use = "leaf:atom"\n')
    _write(
        config_dir / "main.toml",
        '[outer]\nname = "n"\n[outer.inner]\n_neurox_use_preset = "bad:piece"\n',
    )
    with pytest.raises(ValueError, match="forbidden inside neurox/presets/"):
        dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")


def test_preset_mutual_exclusion_with_use(config_dir: Path, presets_root: Path) -> None:
    _write(presets_root / "p.toml", "[piece]\na = 1.0\nb = 2.0\n")
    _write(config_dir / "u.toml", "[piece]\na = 3.0\nb = 4.0\n")
    _write(
        config_dir / "main.toml",
        '[outer]\nname = "m"\n[outer.inner]\n_neurox_use = "u:piece"\n_neurox_use_preset = "p:piece"\n',
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")


@pytest.mark.parametrize(
    "bad_path",
    ["./frag:piece", "../frag:piece", "/abs/frag:piece", "frag/../other:piece"],
)
def test_preset_rejects_forbidden_path_forms(config_dir: Path, presets_root: Path, bad_path: str) -> None:
    _write(presets_root / "frag.toml", "[piece]\na = 1.0\nb = 2.0\n")
    _write(config_dir / "main.toml", f'[outer]\nname = "x"\n[outer.inner]\n_neurox_use_preset = "{bad_path}"\n')
    with pytest.raises(ValueError, match=r"(must not (start with|contain)|absolute)"):
        dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")


def test_preset_cycle_is_rejected(config_dir: Path, presets_root: Path) -> None:
    _write(presets_root / "a.toml", '[piece]\n_neurox_use_preset = "b:piece"\n')
    _write(presets_root / "b.toml", '[piece]\n_neurox_use_preset = "a:piece"\n')
    _write(
        config_dir / "main.toml",
        '[outer]\nname = "y"\n[outer.inner]\n_neurox_use_preset = "a:piece"\n',
    )
    with pytest.raises(ValueError, match="_neurox_use_preset cycle detected"):
        dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")


def test_preset_missing_file_raises(config_dir: Path, presets_root: Path) -> None:
    _write(
        config_dir / "main.toml",
        '[outer]\nname = "x"\n[outer.inner]\n_neurox_use_preset = "no_such:piece"\n',
    )
    with pytest.raises(FileNotFoundError, match="no_such"):
        dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")


def test_preset_missing_section_raises(config_dir: Path, presets_root: Path) -> None:
    _write(presets_root / "frag.toml", "[piece]\na = 1.0\nb = 2.0\n")
    _write(
        config_dir / "main.toml",
        '[outer]\nname = "x"\n[outer.inner]\n_neurox_use_preset = "frag:nonexistent"\n',
    )
    with pytest.raises(KeyError, match="nonexistent"):
        dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")


def test_preset_value_must_be_string(config_dir: Path, presets_root: Path) -> None:
    _write(config_dir / "main.toml", '[outer]\nname = "x"\n[outer.inner]\n_neurox_use_preset = 42\n')
    with pytest.raises(TypeError, match="_neurox_use_preset must be a string"):
        dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")


def test_preset_malformed_string_raises(config_dir: Path, presets_root: Path) -> None:
    _write(
        config_dir / "main.toml",
        '[outer]\nname = "x"\n[outer.inner]\n_neurox_use_preset = "missing_separator"\n',
    )
    with pytest.raises(ValueError, match="missing ':'"):
        dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")


# --- strictness ---


@dataclass(frozen=True)
class _Pair:
    """Fixed-length tuple field for the strictness suite."""

    rng: tuple[int, int]


def test_unknown_key_rejected(config_dir: Path) -> None:
    """Typos must error, not silently fall back to defaults."""
    _write(
        config_dir / "main.toml",
        '[outer]\nname = "x"\nmispelled = "y"\n[outer.inner]\na = 1.0\nb = 2.0\n',
    )
    with pytest.raises(TypeError, match=r"unknown key.*mispelled"):
        dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")


def test_unknown_key_rejected_in_nested(config_dir: Path) -> None:
    _write(
        config_dir / "main.toml",
        '[outer]\nname = "x"\n[outer.inner]\na = 1.0\nb = 2.0\nrogue = 3.0\n',
    )
    with pytest.raises(TypeError, match=r"unknown key.*rogue"):
        dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")


def test_tuple_length_strict_too_few(config_dir: Path) -> None:
    """Fixed-length tuples must error on element-count mismatch."""
    _write(
        config_dir / "main.toml",
        "[outer]\nrng = [1]\n",
    )
    with pytest.raises(ValueError, match=r"tuple length mismatch"):
        dataclass_from_file(_Pair, config_dir / "main.toml", section="outer")


def test_tuple_length_strict_too_many(config_dir: Path) -> None:
    _write(
        config_dir / "main.toml",
        "[outer]\nrng = [1, 2, 3]\n",
    )
    with pytest.raises(ValueError, match=r"tuple length mismatch"):
        dataclass_from_file(_Pair, config_dir / "main.toml", section="outer")
