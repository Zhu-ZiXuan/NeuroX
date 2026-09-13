"""Tests for the `_neurox_use` / `_neurox_use_preset` directives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from neurox.common.serialize import compose, dataclass_from_dict, dict_from_file, load_config_dict, resolve_uses


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


def dataclass_from_file[T](
    cls: type[T],
    *files: Path,
    section: str | None = None,
) -> T:
    """Load-resolve-merge-coerce a config file into `cls`."""
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
    """Override `_presets_root()` to a hermetic tmp directory."""
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


def test_preset_cycle_is_rejected(config_dir: Path, presets_root: Path) -> None:
    _write(presets_root / "a.toml", '[piece]\n_neurox_use_preset = "b:piece"\n')
    _write(presets_root / "b.toml", '[piece]\n_neurox_use_preset = "a:piece"\n')
    _write(
        config_dir / "main.toml",
        '[outer]\nname = "y"\n[outer.inner]\n_neurox_use_preset = "a:piece"\n',
    )
    with pytest.raises(ValueError, match="_neurox_use_preset cycle detected"):
        dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")


# --- strictness ---


def test_unknown_key_rejected_in_nested(config_dir: Path) -> None:
    _write(
        config_dir / "main.toml",
        '[outer]\nname = "x"\n[outer.inner]\na = 1.0\nb = 2.0\nrogue = 3.0\n',
    )
    with pytest.raises(TypeError, match=r"unknown key.*rogue"):
        dataclass_from_file(_Outer, config_dir / "main.toml", section="outer")
