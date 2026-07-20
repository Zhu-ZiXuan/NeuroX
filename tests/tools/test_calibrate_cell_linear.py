"""Linear-fragment extraction sanity for :mod:`neurox.tools.calibrate_cell`.

Pulls a real Detail cell fragment through the tool-run config path, extracts
the secant linearization at a nominal operating point, and checks that the
emitted fragment deserializes into a buildable
:class:`XbarCell1t1rLinearConfig` whose series conductance reproduces the
Detail branch current at that operating point.
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
    """Secant linearization of the Detail cell at the nominal OP."""
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
    assert len(linear_config.g_bl_table__uS) == n_states
    assert len(linear_config.g_sl_table__uS) == n_states
    for table in (linear_config.g_bl_table__uS, linear_config.g_sl_table__uS):
        for row in table:
            assert len(row) == 2
            assert all(entry > 0.0 for entry in row)


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
        inst_shape=(len(loaded.g_bl_table__uS),),
        dtype=torch.float64,
        T__K=300.0,
    )
    assert isinstance(cell, XbarCell1t1rLinear)


def test_n_newton_fragment_parses(tmp_path: Path) -> None:
    path = tmp_path / "cell_detail_n_newton.toml"
    path.write_text(n_newton_fragment_text(7))
    data = tomllib.loads(path.read_text())
    assert data["cell_config"]["n_newton"] == 7


def test_series_g_reproduces_detail_current_at_op(
    detail_config: XbarCell1t1rDetailConfig,
    linear_config: XbarCell1t1rLinearConfig,
) -> None:
    """Series secant conductance at WL-on matches the Detail solve at the OP."""
    cell = _build_cell(
        detail_config,
        n_newton=detail_config.n_newton,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    v_bl = torch.full((1, 1), _V_BL_OP__V, dtype=torch.float64)
    v_sl = torch.full((1, 1), _V_SL_OP__V, dtype=torch.float64)
    v_wl = torch.full((1, 1), _V_WL_ON__V, dtype=torch.float64)
    for s in range(len(detail_config.state_to_g_map__uS)):
        cell.program(torch.full((1,), s, dtype=torch.long))
        snap = cell.snapshot(control=v_wl, shape=(1, 1), multi_coords=None, t_elapsed=0.0)
        i_detail__uA = float(cell.solve_dc(v_bl, v_sl, snap).i__uA)

        g_bl_on__uS = linear_config.g_bl_table__uS[s][1]
        g_sl_on__uS = linear_config.g_sl_table__uS[s][1]
        g_series__uS = g_bl_on__uS * g_sl_on__uS / (g_bl_on__uS + g_sl_on__uS)
        i_linear__uA = g_series__uS * (_V_BL_OP__V - _V_SL_OP__V)

        assert i_linear__uA == pytest.approx(i_detail__uA, rel=1e-3)
