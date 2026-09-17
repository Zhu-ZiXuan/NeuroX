"""Detailed-cell current continuity and trace selection."""

from __future__ import annotations

import torch

from neurox.primitive.device.mosfet import MosfetConfig, MosfetPolicy
from neurox.primitive.device.rram import RramConfig, RramPolicy
from neurox.primitive.nonideality import StateDependentGammaConfig, StuckAtFaultConfig, TelegraphConfig
from neurox.primitive.xbar.cell.x1t1r_detail import (
    XbarCell1t1rDetail,
    XbarCell1t1rDetailConfig,
    XbarCell1t1rDetailPolicy,
    XbarCell1t1rDetailSnap,
)

_CPU = torch.device("cpu")


def _rram_config() -> RramConfig:
    return RramConfig(
        g_min__uS=10.0,
        nonlinearity_alpha=0.5,
        drift_decay_rate=0.03,
        drift_t0=1.0,
        read_thermal__uS=0.002,
        prog_gamma=StateDependentGammaConfig(
            k_slope=0.0,
            k_intercept=100.0,
            theta=1.0,
            min_val=10.0,
            max_val=100.0,
        ),
        read_telegraph=TelegraphConfig(
            amplitude_mean=0.005,
            amplitude_std=0.001,
            p_high_state=0.01,
        ),
        stuck_at=StuckAtFaultConfig(p_at_min=0.001, p_at_max=0.001),
    )


def _mosfet_config() -> MosfetConfig:
    return MosfetConfig(
        T_nom__K=300.0,
        c_ox__fF_per_um2=31.4,
        mu0__cm2_per_V_s=200.0,
        ute=1.5,
        vth0__V=0.4,
        kt1__V=-0.002,
        n_factor=1.25,
        A_vt__mV_um=3.0,
        A_beta_relative__um=0.002,
    )


def _rram_policy() -> RramPolicy:
    return RramPolicy(
        prog_gamma=False,
        drift=False,
        stuck_at=False,
        read_telegraph=False,
        read_thermal=False,
    )


def _mosfet_policy() -> MosfetPolicy:
    return MosfetPolicy(A_vt_mismatch=False, A_beta_mismatch=False)


def _build_cell(
    *,
    inst_shape: tuple[int, ...] = (2, 2),
    dtype: torch.dtype = torch.float64,
    device: torch.device = _CPU,
) -> XbarCell1t1rDetail:
    config = XbarCell1t1rDetailConfig(
        rram_config=_rram_config(),
        nmos_config=_mosfet_config(),
        state_to_g_map__uS=(10.0, 100.0),
        access_nmos_W__um=0.1,
        access_nmos_L__um=0.05,
        rram_g_max__uS=100.0,
    )
    cell = XbarCell1t1rDetail(
        config=config,
        policy=XbarCell1t1rDetailPolicy(
            rram_policy=_rram_policy(),
            nmos_policy=_mosfet_policy(),
        ),
        inst_shape=inst_shape,
        dtype=dtype,
    )
    cell.to(device)
    cell.eval()
    cell.fabricate()
    program = torch.arange(cell.inst_count, dtype=torch.long, device=device).remainder(2).reshape(inst_shape)
    cell.program(program)
    return cell


def _snap(
    cell: XbarCell1t1rDetail,
    *,
    v_wl__V: float,
    shape: tuple[int, ...] = (2, 2),
) -> XbarCell1t1rDetailSnap:
    control = cell.rram.snapshot(shape=shape).g__uS.new_full(shape, v_wl__V)
    return cell.snapshot(control=control, shape=shape)


def test_trace_selection_preserves_the_current_balanced_solution() -> None:
    cell = _build_cell()
    snap = _snap(cell, v_wl__V=0.9)
    v_bl__V = torch.full((2, 2), 0.3, dtype=torch.float64)
    v_sl__V = torch.zeros((2, 2), dtype=torch.float64)
    trace_mask = torch.tensor([[True, False], [False, True]])

    plain_dcop = cell.solve_dc(v_bl__V, v_sl__V, snap)
    traced_dcop, trace = cell.solve_dc_trace(v_bl__V, v_sl__V, snap, trace_mask=trace_mask)
    valid = ~trace.residual__uA.isnan()
    assert not valid[~trace_mask].any()
    assert valid[trace_mask, 0].all()
    assert not ((~valid[..., :-1]) & valid[..., 1:]).any()
    assert torch.equal(trace.threshold__uA.isnan(), ~valid)
    assert torch.equal(trace.dv_x_abs__V.isnan(), ~valid)

    torch.testing.assert_close(plain_dcop.i__uA, traced_dcop.i__uA)
    torch.testing.assert_close(plain_dcop.v_x__V, traced_dcop.v_x__V)
    # Independently check current continuity at the solved access node,
    # without reproducing the Newton update or its stopping criterion.
    nmos = cell.nmos.solve_dc(snap.v_wl__V, plain_dcop.v_x__V, v_sl__V, snap.nmos_snap)
    rram = cell.rram.solve_dc(v_bl__V - plain_dcop.v_x__V, snap.rram_snap)
    torch.testing.assert_close(nmos.ids__uA, rram.i__uA, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(plain_dcop.i__uA, rram.i__uA)
    assert valid[..., 0].sum() == trace_mask.sum()
    assert (valid.sum(dim=(0, 1)) <= trace_mask.sum()).all()
