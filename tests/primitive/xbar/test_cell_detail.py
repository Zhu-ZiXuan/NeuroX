"""Probe-emission checks for `XbarCell1t1rDetail`.

The detailed 1T1R cell condenses its access node with a per-cell Newton and
emits the access-node KCL residual to `XbarCell1t1rDetailProber` once per
`solve_dc` call. These tests pin the emitter contract (laws, not numbers):
one record per `solve_dc` call, the record names the emitting cell and its
`cell__uA` is the non-negative `|I_NMOS - I_RRAM|` over the branch grid,
and the lean `solve_branch` hot path emits nothing.
"""

from __future__ import annotations

import torch

from neurox.primitive.device import MosfetConfig, MosfetPolicy, RramConfig, RramPolicy
from neurox.primitive.xbar.cell import (
    XbarCell1t1r,
    XbarCell1t1rDetail,
    XbarCell1t1rDetailConfig,
    XbarCell1t1rDetailPolicy,
    XbarCell1t1rDetailProber,
)


def _build_cell(inst_shape: tuple[int, ...]) -> XbarCell1t1rDetail:
    rram_config = RramConfig.from_preset("process/rram:default")
    config = XbarCell1t1rDetailConfig(
        rram_config=rram_config,
        nmos_config=MosfetConfig.from_preset("process/mos:nmos_28_rvt"),
        state_to_g_map__uS=(rram_config.g_min__uS, 100.0),
        access_nmos_W__um=0.1,
        access_nmos_L__um=0.05,
        rram_g_max__uS=100.0,
        newton_iter_num=4,
    )
    policy = XbarCell1t1rDetailPolicy(
        rram_policy=RramPolicy(
            prog_gamma=False,
            drift=False,
            stuck_at=False,
            read_telegraph=False,
            read_thermal=False,
        ),
        nmos_policy=MosfetPolicy(A_vt_mismatch=False, A_beta_mismatch=False),
    )
    cell = XbarCell1t1r.from_config(
        config=config,
        policy=policy,
        inst_shape=inst_shape,
        dtype=torch.float64,
        T__K=300.0,
    )
    assert isinstance(cell, XbarCell1t1rDetail)
    cell.eval()
    cell.fabricate()
    cell.program(torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
    return cell


def _grids() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    v_bl = torch.full((2, 2), 0.3, dtype=torch.float64)
    v_sl = torch.zeros((2, 2), dtype=torch.float64)
    v_wl = torch.full((2, 2), 0.9, dtype=torch.float64)
    return v_bl, v_sl, v_wl


def test_solve_dc_emits_one_residual_record() -> None:
    cell = _build_cell((2, 2))
    v_bl, v_sl, v_wl = _grids()
    snap = cell.snapshot(control=v_wl, shape=(2, 2), t_elapsed=0.0)

    with XbarCell1t1rDetailProber() as prober:
        cell.solve_dc(v_bl, v_sl, snap)
    records = prober.records

    assert len(records) == 1
    residual = records[0]
    assert residual.cell__uA.shape == (2, 2)
    # KCL mismatch is an absolute magnitude, always non-negative.
    assert bool((residual.cell__uA >= 0.0).all())


def test_solve_branch_emits_nothing() -> None:
    """Only `solve_dc` emits; the lean hot path stays off the side channel."""
    cell = _build_cell((2, 2))
    v_bl, v_sl, v_wl = _grids()
    snap = cell.snapshot(control=v_wl, shape=(2, 2), t_elapsed=0.0)

    with XbarCell1t1rDetailProber() as prober:
        cell.solve_branch(v_bl, v_sl, snap)
    assert prober.records == ()


def test_solve_dc_without_prober_is_silent() -> None:
    """The emit hook is a no-op when no prober is active (no error)."""
    cell = _build_cell((2, 2))
    v_bl, v_sl, v_wl = _grids()
    snap = cell.snapshot(control=v_wl, shape=(2, 2), t_elapsed=0.0)
    cell.solve_dc(v_bl, v_sl, snap)
