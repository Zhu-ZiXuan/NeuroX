"""MOSFET current polarity, monotonicity, and terminal derivatives."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from neurox.primitive.device.mosfet import Mosfet, MosfetConfig, MosfetPolicy, Nmos, Pmos


def _make(cls: type[Mosfet], *, vth0__V: float, inst_shape: tuple[int, ...]) -> Mosfet:
    dev = cls(
        config=MosfetConfig(
            mu0__cm2_per_V_s=200.0,
            c_ox__fF_per_um2=31.4,
            vth0__V=vth0__V,
            n_factor=1.25,
            T_nom__K=300.0,
            ute=1.5,
            kt1__V=-0.002,
            A_vt__mV_um=0.0,
            A_beta_relative__um=0.0,
        ),
        policy=MosfetPolicy(A_vt_mismatch=False, A_beta_mismatch=False),
        inst_shape=inst_shape,
        dtype=torch.float64,
        W__um=1.0,
        L__um=1.0,
    )
    dev.fabricate()
    return dev


def test_nmos_enhancement_conducts_and_partial_signs() -> None:
    k = 3
    dev = _make(Nmos, vth0__V=0.4, inst_shape=(k,))
    snap = dev.snapshot(shape=(k,))
    vg = torch.linspace(0.5, 1.0, k, dtype=torch.float64)
    vd = torch.full((k,), 0.6, dtype=torch.float64)
    vs = torch.full((k,), 0.1, dtype=torch.float64)
    dc = dev.solve_dc(vg__V=vg, vd__V=vd, vs__V=vs, snap=snap)
    assert torch.all(dc.ids__uA > 0.0)
    assert torch.all(dc.ids__uA[1:] - dc.ids__uA[:-1] > 0.0)
    assert torch.all(dc.did_dvd__uS >= 0.0)
    assert torch.all(dc.did_dvs__uS <= 0.0)


def test_pmos_enhancement_conducts_negative() -> None:
    k = 3
    vdd = 0.9
    dev = _make(Pmos, vth0__V=-0.4, inst_shape=(k,))
    snap = dev.snapshot(shape=(k,))
    vg = torch.linspace(0.0, 0.5, k, dtype=torch.float64)
    vs = torch.full((k,), vdd, dtype=torch.float64)
    vd = torch.zeros(k, dtype=torch.float64)
    dc = dev.solve_dc(vg__V=vg, vd__V=vd, vs__V=vs, snap=snap)
    assert torch.all(dc.ids__uA < 0.0)
    assert torch.all(dc.ids__uA[1:] - dc.ids__uA[:-1] > 0.0)
    assert torch.all(dc.did_dvd__uS >= 0.0)
    assert torch.all(dc.did_dvs__uS <= 0.0)


def test_depletion_nmos_conducts_at_zero_gate() -> None:
    dev = _make(Nmos, vth0__V=-0.4, inst_shape=(1,))
    snap = dev.snapshot(shape=(1,))
    dc = dev.solve_dc(
        vg__V=torch.zeros(1, dtype=torch.float64),
        vd__V=torch.full((1,), 0.5, dtype=torch.float64),
        vs__V=torch.full((1,), 0.1, dtype=torch.float64),
        snap=snap,
    )
    assert torch.all(dc.ids__uA > 0.0)


@pytest.mark.parametrize(
    ("cls", "vth0", "vg", "vd", "vs"),
    [
        (Nmos, 0.4, [0.70, 0.80, 0.90, 1.00], [0.50, 0.45, 0.55, 0.40], [0.10, 0.12, 0.08, 0.15]),
        (Pmos, -0.4, [0.20, 0.10, 0.30, 0.05], [0.10, 0.12, 0.08, 0.15], [0.90, 0.88, 0.92, 0.85]),
    ],
)
def test_partials_match_finite_difference(
    cls: type[Mosfet],
    vth0: float,
    vg: list[float],
    vd: list[float],
    vs: list[float],
) -> None:
    k = len(vg)
    dev = _make(cls, vth0__V=vth0, inst_shape=(k,))
    snap = dev.snapshot(shape=(k,))
    g = torch.tensor(vg, dtype=torch.float64)
    d = torch.tensor(vd, dtype=torch.float64)
    s = torch.tensor(vs, dtype=torch.float64)
    dc = dev.solve_dc(vg__V=g, vd__V=d, vs__V=s, snap=snap)

    h = 1e-5

    def ids(gg: Tensor, dd: Tensor, ss: Tensor) -> Tensor:
        return dev.solve_dc(vg__V=gg, vd__V=dd, vs__V=ss, snap=snap).ids__uA

    fd_g = (ids(g + h, d, s) - ids(g - h, d, s)) / (2.0 * h)
    fd_d = (ids(g, d + h, s) - ids(g, d - h, s)) / (2.0 * h)
    fd_s = (ids(g, d, s + h) - ids(g, d, s - h)) / (2.0 * h)
    assert torch.allclose(dc.did_dvg__uS, fd_g, atol=1e-4)
    assert torch.allclose(dc.did_dvd__uS, fd_d, atol=1e-4)
    assert torch.allclose(dc.did_dvs__uS, fd_s, atol=1e-4)
