"""Closed-form and registry checks for `XbarCell1t1rLinear`.

Covers registry dispatch from the config type, the WL-switched
division-free branch math against hand-built tables, table validation
bounds, and the empty-policy deserialization path.
"""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from neurox.primitive.device import MosfetPolicy, RramPolicy
from neurox.primitive.xbar.cell import (
    XbarCell1t1r,
    XbarCell1t1rDetailPolicy,
    XbarCell1t1rLinear,
    XbarCell1t1rLinearConfig,
    XbarCell1t1rLinearPolicy,
    XbarCell1t1rPolicy,
)

_G_CELL_OFF_TABLE__uS = (1e-4, 2e-4)
_G_CELL_ON_TABLE__uS = (5.0, 95.0)
_VX_RATIO_OFF_TABLE = (2e-5, 1e-5)
_VX_RATIO_ON_TABLE = (0.98, 0.94)
_V_WL_ON_THRESHOLD__V = 0.45


def _hand_built_config() -> XbarCell1t1rLinearConfig:
    return XbarCell1t1rLinearConfig(
        g_cell_off_table__uS=_G_CELL_OFF_TABLE__uS,
        g_cell_on_table__uS=_G_CELL_ON_TABLE__uS,
        vx_ratio_off_table=_VX_RATIO_OFF_TABLE,
        vx_ratio_on_table=_VX_RATIO_ON_TABLE,
        v_wl_on_threshold__V=_V_WL_ON_THRESHOLD__V,
    )


def _build_cell(inst_shape: tuple[int, ...]) -> XbarCell1t1rLinear:
    cell = XbarCell1t1r.from_config(
        config=_hand_built_config(),
        policy=XbarCell1t1rLinearPolicy(),
        inst_shape=inst_shape,
        dtype=torch.float64,
        T__K=300.0,
    )
    assert isinstance(cell, XbarCell1t1rLinear)
    cell.eval()
    cell.fabricate()
    return cell


def test_registry_dispatch_yields_linear_leaf() -> None:
    cell = XbarCell1t1r.from_config(
        config=_hand_built_config(),
        policy=XbarCell1t1rLinearPolicy(),
        inst_shape=(2, 2),
        dtype=torch.float64,
        T__K=300.0,
    )
    assert type(cell) is XbarCell1t1rLinear


def test_solve_dc_matches_table_conductance() -> None:
    cell = _build_cell((2, 2))
    w_state = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    cell.program(w_state)

    # A word line runs along a row and is shared by every column, so the array
    # gates every cell of a row at the same voltage on the per-cell grid.
    v_wl__V = torch.tensor([0.0, 0.9], dtype=torch.float64).expand(2, 2)
    snap = cell.snapshot(control=v_wl__V, shape=(2, 2), t_elapsed=0.0)

    g_cell_off = torch.tensor(_G_CELL_OFF_TABLE__uS, dtype=torch.float64)[w_state]
    g_cell_on = torch.tensor(_G_CELL_ON_TABLE__uS, dtype=torch.float64)[w_state]
    on = v_wl__V > _V_WL_ON_THRESHOLD__V
    g_cell__uS = torch.where(on, g_cell_on, g_cell_off)

    v_bl__V = torch.full((2, 2), 0.3, dtype=torch.float64)
    v_sl__V = torch.full((2, 2), 0.05, dtype=torch.float64)
    dcop = cell.solve_dc(v_bl__V, v_sl__V, snap)

    torch.testing.assert_close(dcop.i__uA, g_cell__uS * (v_bl__V - v_sl__V))
    torch.testing.assert_close(dcop.di_dvbl__uS, g_cell__uS)
    torch.testing.assert_close(dcop.di_dvsl__uS, -g_cell__uS)


def test_wl_threshold_switches_off_at_and_below() -> None:
    cell = _build_cell((1, 1))
    cell.program(torch.tensor([[1]], dtype=torch.long))
    v_bl__V = torch.full((1, 1), 0.3, dtype=torch.float64)
    v_sl__V = torch.zeros((1, 1), dtype=torch.float64)

    i_levels = []
    for v_wl__V in (_V_WL_ON_THRESHOLD__V, _V_WL_ON_THRESHOLD__V + 0.01):
        v_wl_grid__V = torch.full((1, 1), v_wl__V, dtype=torch.float64)
        snap = cell.snapshot(control=v_wl_grid__V, shape=(1, 1), t_elapsed=0.0)
        i_levels.append(float(cell.solve_dc(v_bl__V, v_sl__V, snap).i__uA))
    i_at_threshold, i_above = i_levels

    g_off = _G_CELL_OFF_TABLE__uS[1]
    assert i_at_threshold == pytest.approx(g_off * 0.3)
    assert i_above > i_at_threshold * 1e3


def test_solve_dc_vx_multiplication_form() -> None:
    cell = _build_cell((1, 1))
    cell.program(torch.tensor([[0]], dtype=torch.long))
    v_bl__V = torch.full((1, 1), 0.3, dtype=torch.float64)
    v_sl__V = torch.zeros((1, 1), dtype=torch.float64)
    v_wl__V = torch.full((1, 1), 0.9, dtype=torch.float64)
    snap = cell.snapshot(control=v_wl__V, shape=(1, 1), t_elapsed=0.0)

    dcop = cell.solve_dc(v_bl__V, v_sl__V, snap)

    g_cell_on = _G_CELL_ON_TABLE__uS[0]
    vx_ratio_on = _VX_RATIO_ON_TABLE[0]
    assert float(dcop.i__uA) == pytest.approx(g_cell_on * 0.3)
    assert float(dcop.v_x__V) == pytest.approx(0.3 - vx_ratio_on * 0.3)


def test_table_validation_bounds() -> None:
    base = _hand_built_config()
    # Boundary values are legal: zero chord conductance (cut-off leakage)
    # and drop fractions at 0 / 1.
    replace(base, g_cell_off_table__uS=(0.0, 2e-4)).validate()
    replace(base, vx_ratio_off_table=(0.0, 0.5), vx_ratio_on_table=(1.0, 0.25)).validate()
    with pytest.raises(ValueError, match=r"g_cell_off_table__uS\[0\] \(-1\.0\) >= 0"):
        replace(base, g_cell_off_table__uS=(-1.0, 2e-4)).validate()
    with pytest.raises(ValueError, match=r"g_cell_on_table__uS\[0\] \(inf\) >= 0"):
        replace(base, g_cell_on_table__uS=(float("inf"), 95.0)).validate()
    with pytest.raises(ValueError, match=r"vx_ratio_on_table\[0\] \(1\.5\) in closed interval \[0\.0, 1\.0\]"):
        replace(base, vx_ratio_on_table=(1.5, 0.94)).validate()
    with pytest.raises(ValueError, match=r"vx_ratio_off_table\[0\] \(-0\.1\) in closed interval \[0\.0, 1\.0\]"):
        replace(base, vx_ratio_off_table=(-0.1, 1e-5)).validate()
    with pytest.raises(ValueError, match=r"vx_ratio_off_table\[0\] \(nan\) in closed interval \[0\.0, 1\.0\]"):
        replace(base, vx_ratio_off_table=(float("nan"), 1e-5)).validate()
    with pytest.raises(ValueError, match=r"require: len\(vx_ratio_on_table\) \(1\) == len\(g_cell_off_table__uS\)"):
        replace(base, vx_ratio_on_table=(0.98,)).validate()
    with pytest.raises(ValueError, match=r"require: len\(g_cell_off_table__uS\) \(0\) >= 1"):
        replace(
            base,
            g_cell_off_table__uS=(),
            g_cell_on_table__uS=(),
            vx_ratio_off_table=(),
            vx_ratio_on_table=(),
        ).validate()


def test_empty_policy_deserializes(tmp_path: Path) -> None:
    policy_toml = tmp_path / "policy.toml"
    policy_toml.write_text('[policy]\n_neurox_class = "XbarCell1t1rLinearPolicy"\n')
    policy = XbarCell1t1rPolicy.from_file(policy_toml, section="policy")
    assert isinstance(policy, XbarCell1t1rLinearPolicy)


def test_wrong_policy_type_raises() -> None:
    detail_policy = XbarCell1t1rDetailPolicy(
        rram_policy=RramPolicy(
            prog_gamma=False,
            drift=False,
            stuck_at=False,
            read_telegraph=False,
            read_thermal=False,
        ),
        nmos_policy=MosfetPolicy(A_vt_mismatch=False, A_beta_mismatch=False),
    )
    with pytest.raises(
        TypeError,
        match=r"no XbarCell1t1r module registered for config XbarCell1t1rLinearConfig "
        r"and policy XbarCell1t1rDetailPolicy",
    ):
        XbarCell1t1r.from_config(
            config=_hand_built_config(),
            policy=detail_policy,
            inst_shape=(1, 1),
            dtype=torch.float64,
            T__K=300.0,
        )
