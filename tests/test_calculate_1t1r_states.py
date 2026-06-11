"""Tests for neurox.tools.calculate_1t1r_states."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from neurox.common import dataclass_from_file
from neurox.device import NMOSConfig, RRAMConfig
from neurox.tools.calculate_1t1r_states import (
    _CellBias,
    calculate_state_map,
    main,
)

_PRESET_DIR = Path(__file__).parent.parent / "neurox" / "presets" / "process"


def _load_configs() -> tuple[RRAMConfig, NMOSConfig]:
    rram = dataclass_from_file(RRAMConfig, _PRESET_DIR / "rram.toml", section="default")
    nmos = dataclass_from_file(NMOSConfig, _PRESET_DIR / "mos.toml", section="nmos_28_rvt")
    return rram, nmos


def _bias() -> _CellBias:
    return _CellBias(v_wl__V=1.2, v_bl__V=0.2, v_sl__V=0.0)


def _kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "g_max__uS": 100.0,
        "n_states": 4,
        "access_nmos_W__um": 0.150,
        "access_nmos_L__um": 0.028,
        "temperature_K": 300.0,
    }
    base.update(overrides)
    return base


def _write_config(
    tmp_path: Path,
    *,
    v_bl__V: float = 0.2,
    v_wl__V: float = 1.2,
    v_sl__V: float = 0.0,
    n_states: int = 4,
    g_max__uS: float = 100.0,
    access_nmos_W__um: float = 0.150,
    access_nmos_L__um: float = 0.028,
    temperature__K: float = 300.0,
) -> Path:
    """Write a calculate_1t1r_states run TOML into ``tmp_path``."""
    cfg_path = tmp_path / "run.toml"
    cfg_path.write_text(
        f"""[rram]
_neurox_use_preset = "process/rram:default"

[nmos]
_neurox_use_preset = "process/mos:nmos_28_rvt"

[bias]
v_wl__V = {v_wl__V}
v_bl__V = {v_bl__V}
v_sl__V = {v_sl__V}
temperature__K = {temperature__K}

[design]
g_max__uS = {g_max__uS}
n_states = {n_states}
access_nmos_W__um = {access_nmos_W__um}
access_nmos_L__um = {access_nmos_L__um}
"""
    )
    return cfg_path


def test_linear_rram_monotonic_state_map() -> None:
    """With ohmic RRAM (alpha=0), state-to-g map is strictly increasing and endpoint-snapped."""
    rram_config, nmos_config = _load_configs()
    rram_config = replace(rram_config, nonlinearity_alpha=0.0)

    result = calculate_state_map(rram_config=rram_config, nmos_config=nmos_config, bias=_bias(), **_kwargs())

    assert len(result.state_to_g_map__uS) == 4
    assert result.state_to_g_map__uS[0] == rram_config.g_min__uS
    assert result.state_to_g_map__uS[-1] == 100.0
    for k in range(1, 4):
        assert result.state_to_g_map__uS[k] > result.state_to_g_map__uS[k - 1]


def test_target_current_accuracy() -> None:
    """Solved currents recreate the linear target ladder within tolerance."""
    rram_config, nmos_config = _load_configs()
    result = calculate_state_map(rram_config=rram_config, nmos_config=nmos_config, bias=_bias(), **_kwargs(n_states=6))

    for k in range(6):
        err = abs(result.solved_currents__uA[k] - result.target_currents__uA[k])
        assert err <= 1e-9, f"state {k}: err {err:.3e}"


def test_n_states_two_short_circuit() -> None:
    """n_states=2 returns just the two endpoints, no interior solver work."""
    rram_config, nmos_config = _load_configs()
    result = calculate_state_map(rram_config=rram_config, nmos_config=nmos_config, bias=_bias(), **_kwargs(n_states=2))

    assert result.state_to_g_map__uS == [rram_config.g_min__uS, 100.0]


def test_cli_smoke(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Invoke main() and inspect logger output for the paste-ready lines."""
    caplog.set_level(logging.INFO, logger="neurox.tools.calculate_1t1r_states")
    cfg = _write_config(tmp_path)
    main(["--config", str(cfg)])
    assert "rram_g_max__uS = 100.0" in caplog.text
    assert "state_to_g_map__uS = [" in caplog.text


def test_cli_rejects_v_bl_le_v_sl(tmp_path: Path) -> None:
    """V_BL <= V_SL fast-fails after config load."""
    cfg = _write_config(tmp_path, v_bl__V=0.0)
    with pytest.raises(SystemExit):
        main(["--config", str(cfg)])


def test_cli_requires_config() -> None:
    """--config is required (no default)."""
    with pytest.raises(SystemExit):
        main([])
