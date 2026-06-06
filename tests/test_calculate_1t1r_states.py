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


def _cli_args(**overrides: str) -> list[str]:
    base: dict[str, str] = {
        "--rram-config": str(_PRESET_DIR / "rram.toml"),
        "--rram-section": "default",
        "--nmos-config": str(_PRESET_DIR / "mos.toml"),
        "--nmos-section": "nmos_28_rvt",
        "--v-wl-V": "1.2",
        "--v-bl-V": "0.2",
        "--v-sl-V": "0.0",
        "--g-max-uS": "100.0",
        "--n-states": "4",
        "--access-nmos-W-um": "0.150",
        "--access-nmos-L-um": "0.028",
        "--temperature-K": "300.0",
    }
    base.update(overrides)
    return [token for flag, value in base.items() for token in (flag, value)]


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


def test_cli_smoke(caplog: pytest.LogCaptureFixture) -> None:
    """Invoke main() and inspect logger output for the paste-ready lines."""
    caplog.set_level(logging.INFO, logger="neurox.tools.calculate_1t1r_states")
    main(_cli_args())
    assert "rram_g_max__uS = 100.0" in caplog.text
    assert "state_to_g_map__uS = [" in caplog.text


def test_cli_rejects_v_bl_le_v_sl() -> None:
    """V_BL <= V_SL fast-fails at CLI parse."""
    with pytest.raises(SystemExit):
        main(_cli_args(**{"--v-bl-V": "0.0"}))


def test_cli_requires_temperature() -> None:
    """--temperature-K is required (no default)."""
    args = _cli_args()
    # Drop the --temperature-K flag and its value to verify argparse rejects the call.
    idx = args.index("--temperature-K")
    args.pop(idx)  # flag
    args.pop(idx)  # value
    with pytest.raises(SystemExit):
        main(args)
