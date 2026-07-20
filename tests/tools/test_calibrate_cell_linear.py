"""Linear-fragment extraction sanity for :mod:`neurox.tools.calibrate_cell`.

Pulls a real Detail cell fragment through the tool-run config path, extracts
the divider linearization (chord conductance + BL-side drop fraction) at a
nominal operating point, and checks that the emitted fragment deserializes
into a buildable :class:`XbarCell1t1rLinearConfig` that reproduces the
Detail branch current and access node at that operating point.
"""

import tomllib
from pathlib import Path

import pytest
import torch

from neurox.primitive.xbar.cell import (
    XbarCell,
    XbarCell1t1rDetailConfig,
    XbarCell1t1rLinear,
    XbarCell1t1rLinearConfig,
    XbarCell1t1rLinearPolicy,
)
from neurox.tools.calibrate_cell._1t1r import (
    CalibrateCell1t1rConfig,
    _build_cell,
    extract_linear_cell_config,
    linear_fragment_text,
    n_newton_fragment_text,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ISUB_DEFAULT = _REPO_ROOT / "neurox" / "works" / "macro" / "cim" / "isub_iadc_1t1r" / "params" / "default.toml"

_V_BL_OP__V = 0.3
_V_SL_OP__V = 0.0
_V_WL_OFF__V = 0.0
_V_WL_ON__V = 0.9

_RUN_TOML = f"""\
[cell_config]
_neurox_use = "{_ISUB_DEFAULT}:cim_macro.array_config.cell_config"

[grid]
v_terminal_min__V = 0.0
v_terminal_max__V = 0.4
n_terminal = 4
v_wl_off__V = {_V_WL_OFF__V}
v_wl_on__V = {_V_WL_ON__V}
v_bl_op__V = {_V_BL_OP__V}
v_sl_op__V = {_V_SL_OP__V}

[sweep]
candidates = [1, 2, 3]
ratio_threshold = 0.5
reltol = 1e-2
margin = 1

[runtime]
dtype = "float64"
"""


@pytest.fixture(scope="module")
def detail_config(tmp_path_factory: pytest.TempPathFactory) -> XbarCell1t1rDetailConfig:
    """Detail cell fragment pulled through the tool-run config path."""
    run_toml = tmp_path_factory.mktemp("calibrate_cell") / "run.toml"
    run_toml.write_text(_RUN_TOML)
    cfg = CalibrateCell1t1rConfig.from_file(run_toml)
    assert isinstance(cfg.cell_config, XbarCell1t1rDetailConfig)
    return cfg.cell_config


@pytest.fixture(scope="module")
def linear_config(detail_config: XbarCell1t1rDetailConfig) -> XbarCell1t1rLinearConfig:
    """Divider linearization of the Detail cell at the nominal OP."""
    return extract_linear_cell_config(
        detail_config,
        v_bl_op__V=_V_BL_OP__V,
        v_sl_op__V=_V_SL_OP__V,
        v_wl_off__V=_V_WL_OFF__V,
        v_wl_on__V=_V_WL_ON__V,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )


def test_fragment_fields_copied_from_detail(
    detail_config: XbarCell1t1rDetailConfig,
    linear_config: XbarCell1t1rLinearConfig,
) -> None:
    n_states = len(detail_config.state_to_g_map__uS)
    assert linear_config.c_bl__fF == detail_config.c_bl__fF
    assert linear_config.c_x__fF == detail_config.c_x__fF
    assert linear_config.c_sl__fF == detail_config.c_sl__fF
    assert linear_config.c_wl__fF == detail_config.c_wl__fF
    assert linear_config.v_wl_on_threshold__V == pytest.approx((_V_WL_OFF__V + _V_WL_ON__V) / 2.0)
    for table in (linear_config.g_cell_off_table__uS, linear_config.g_cell_on_table__uS):
        assert len(table) == n_states
        assert all(entry >= 0.0 for entry in table)
    for table in (linear_config.vx_ratio_off_table, linear_config.vx_ratio_on_table):
        assert len(table) == n_states
        assert all(0.0 <= entry <= 1.0 for entry in table)


def test_emitted_fragment_deserializes_and_builds(
    linear_config: XbarCell1t1rLinearConfig,
    tmp_path: Path,
) -> None:
    path = tmp_path / "cell_linear.toml"
    path.write_text(linear_fragment_text(linear_config, v_bl_op__V=_V_BL_OP__V, v_sl_op__V=_V_SL_OP__V))

    loaded = XbarCell1t1rLinearConfig.from_file(path, section="cell_config")
    assert isinstance(loaded, XbarCell1t1rLinearConfig)
    assert loaded == linear_config

    cell = XbarCell.from_config(
        config=loaded,
        policy=XbarCell1t1rLinearPolicy(),
        inst_shape=(len(loaded.g_cell_off_table__uS),),
        dtype=torch.float64,
        T__K=300.0,
    )
    assert isinstance(cell, XbarCell1t1rLinear)


def test_n_newton_fragment_parses(tmp_path: Path) -> None:
    path = tmp_path / "cell_detail_n_newton.toml"
    path.write_text(n_newton_fragment_text(7))
    data = tomllib.loads(path.read_text())
    assert data["cell_config"]["n_newton"] == 7


def test_divider_reproduces_detail_at_op(
    detail_config: XbarCell1t1rDetailConfig,
    linear_config: XbarCell1t1rLinearConfig,
) -> None:
    """Chord conductance and drop fraction match the Detail solve at the OP.

    ``I = g_cell * span`` and ``V_X = v_bl_op - vx_ratio * span`` reproduce
    the Detail branch at both WL levels — the full-span denominators keep
    the cut-off (WL-off) level as well-conditioned as the conducting one.
    """
    cell = _build_cell(
        detail_config,
        n_newton=detail_config.n_newton,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    span__V = _V_BL_OP__V - _V_SL_OP__V
    v_bl = torch.full((1, 1), _V_BL_OP__V, dtype=torch.float64)
    v_sl = torch.full((1, 1), _V_SL_OP__V, dtype=torch.float64)
    levels = (
        (_V_WL_OFF__V, linear_config.g_cell_off_table__uS, linear_config.vx_ratio_off_table),
        (_V_WL_ON__V, linear_config.g_cell_on_table__uS, linear_config.vx_ratio_on_table),
    )
    for s in range(len(detail_config.state_to_g_map__uS)):
        cell.program(torch.full((1,), s, dtype=torch.long))
        for v_wl__V, g_table, vx_table in levels:
            v_wl = torch.full((1, 1), v_wl__V, dtype=torch.float64)
            snap = cell.snapshot(control=v_wl, shape=(1, 1), multi_coords=None, t_elapsed=0.0)
            dcop = cell.solve_dc(v_bl, v_sl, snap)

            g_cell__uS = g_table[s]
            vx_ratio = vx_table[s]
            assert g_cell__uS * span__V == pytest.approx(float(dcop.i__uA), rel=1e-9)
            assert _V_BL_OP__V - vx_ratio * span__V == pytest.approx(float(dcop.v_x__V), rel=1e-9, abs=1e-12)
