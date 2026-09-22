"""Config references preserve branch independence, override precedence, and source boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from neurox.common.serialize import compose, load_config_dict, resolve_uses


def _write(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_inline_keys_override_fragment(tmp_path: Path) -> None:
    _write(tmp_path / "frag.toml", "[piece]\na = 1.0\nb = 2.0\n")
    _write(
        tmp_path / "main.toml",
        '[outer]\nname = "x"\n[outer.inner]\n_neurox_use = "frag:piece"\nb = 9.5\n',
    )
    assert load_config_dict(tmp_path / "main.toml", section="outer") == {"name": "x", "inner": {"a": 1.0, "b": 9.5}}


def test_same_fragment_resolves_in_two_slots_with_optional_file_suffix(tmp_path: Path) -> None:
    _write(tmp_path / "frag.toml", "[piece]\na = 3.0\nb = 4.0\n")
    _write(
        tmp_path / "main.toml",
        '[outer.left]\n_neurox_use = "frag:piece"\n[outer.right]\n_neurox_use = "frag.toml:piece"\n',
    )
    assert load_config_dict(tmp_path / "main.toml", section="outer") == {
        "left": {"a": 3.0, "b": 4.0},
        "right": {"a": 3.0, "b": 4.0},
    }


def test_nested_reference_override_precedence(tmp_path: Path) -> None:
    _write(tmp_path / "leaf.toml", "[atom]\na = 10.0\nb = 20.0\n")
    _write(tmp_path / "mid.toml", '[piece]\n_neurox_use = "leaf:atom"\nb = 99.0\n')
    _write(tmp_path / "main.toml", '[outer]\nname = "n"\n[outer.inner]\n_neurox_use = "mid:piece"\n')
    assert load_config_dict(tmp_path / "main.toml", section="outer") == {"name": "n", "inner": {"a": 10.0, "b": 99.0}}

    _write(
        tmp_path / "main.toml",
        '[outer]\nname = "n"\n[outer.inner]\n_neurox_use = "mid:piece"\nb = 1.0\n',
    )
    assert load_config_dict(tmp_path / "main.toml", section="outer") == {"name": "n", "inner": {"a": 10.0, "b": 1.0}}


def test_resolve_uses_preserves_unreferenced_branches(tmp_path: Path) -> None:
    _write(tmp_path / "frag.toml", "[piece]\na = 1.0\nb = 2.0\n")
    raw = {"outer": {"inner": {"_neurox_use": "frag:piece", "b": 5.0}}, "untouched": {"values": [1, 2, 3]}}
    expanded = resolve_uses(raw, base_dir=tmp_path)
    assert expanded == {"outer": {"inner": {"a": 1.0, "b": 5.0}}, "untouched": {"values": [1, 2, 3]}}


@pytest.fixture
def presets_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "presets"
    root.mkdir()
    monkeypatch.setattr(compose, "_presets_root", lambda: root)
    return root


def test_inline_keys_override_a_preset_from_a_subdirectory(tmp_path: Path, presets_root: Path) -> None:
    (presets_root / "process").mkdir()
    _write(presets_root / "process" / "rram.toml", "[rram_x]\na = 11.0\nb = 22.0\n")
    _write(
        tmp_path / "main.toml",
        '[outer]\nname = "p"\n[outer.inner]\n_neurox_use_preset = "process/rram:rram_x"\nb = 99.0\n',
    )
    assert load_config_dict(tmp_path / "main.toml", section="outer") == {"name": "p", "inner": {"a": 11.0, "b": 99.0}}


def test_preset_chains_via_preset_only(tmp_path: Path, presets_root: Path) -> None:
    _write(presets_root / "leaf.toml", "[atom]\na = 7.0\nb = 8.0\n")
    _write(presets_root / "mid.toml", '[piece]\n_neurox_use_preset = "leaf:atom"\n')
    _write(
        tmp_path / "main.toml",
        '[outer]\nname = "c"\n[outer.inner]\n_neurox_use_preset = "mid:piece"\n',
    )
    assert load_config_dict(tmp_path / "main.toml", section="outer") == {"name": "c", "inner": {"a": 7.0, "b": 8.0}}


def test_preset_forbids_neurox_use_inside_subtree(tmp_path: Path, presets_root: Path) -> None:
    _write(presets_root / "leaf.toml", "[atom]\na = 1.0\nb = 2.0\n")
    _write(presets_root / "bad.toml", '[piece]\n_neurox_use = "leaf:atom"\n')
    _write(
        tmp_path / "main.toml",
        '[outer]\nname = "n"\n[outer.inner]\n_neurox_use_preset = "bad:piece"\n',
    )
    with pytest.raises(ValueError, match="forbidden inside neurox/presets/"):
        load_config_dict(tmp_path / "main.toml", section="outer")


@pytest.mark.parametrize("preset", [False, True])
def test_reference_cycles_are_rejected(tmp_path: Path, presets_root: Path, preset: bool) -> None:
    root = presets_root if preset else tmp_path
    directive = "_neurox_use_preset" if preset else "_neurox_use"
    _write(root / "a.toml", f'[piece]\n{directive} = "b:piece"\n')
    _write(root / "b.toml", f'[piece]\n{directive} = "a:piece"\n')
    _write(
        tmp_path / "main.toml",
        f'[outer]\nname = "y"\n[outer.inner]\n{directive} = "a:piece"\n',
    )
    with pytest.raises(ValueError, match=f"{directive} cycle detected"):
        load_config_dict(tmp_path / "main.toml", section="outer")
