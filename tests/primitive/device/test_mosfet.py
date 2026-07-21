"""Device-level tests for the polarity-parameterized MOSFET primitive.

Exercises the EKV-softplus I-V law and its three node partials for both
the :class:`Nmos` (polarity +1) and :class:`Pmos` (polarity -1)
specializations, the finite-difference consistency of those partials,
all-off determinism, and config validation. Everything runs on CPU in
``float64`` with no ``torch.compile``.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from neurox.primitive.device import Mosfet, MosfetConfig, MosfetPolicy, Nmos, Pmos

_OFF = MosfetPolicy(A_vt_mismatch=False, A_beta_mismatch=False)

_BASE_CONFIG: dict[str, float] = {
    "mu0__cm2_per_V_s": 200.0,
    "c_ox__fF_per_um2": 31.4,
    "vth0__V": 0.40,
    "n_factor": 1.25,
    "T_nom__K": 300.0,
    "ute": 1.5,
    "kt1__V": -0.002,
    "A_vt__mV_um": 0.0,
    "A_beta_relative__um": 0.0,
}


def _config(**overrides: float) -> MosfetConfig:
    """Build a :class:`MosfetConfig` from the base field set with overrides."""
    return MosfetConfig(**{**_BASE_CONFIG, **overrides})


def _make(
    cls: type[Mosfet],
    *,
    vth0__V: float,
    inst_shape: tuple[int, ...],
    policy: MosfetPolicy = _OFF,
    W__um: float = 1.0,
    L__um: float = 1.0,
    **config_overrides: float,
) -> Mosfet:
    """Construct and fabricate a concrete MOSFET sized for the tests."""
    dev = cls(
        config=_config(vth0__V=vth0__V, **config_overrides),
        policy=policy,
        inst_shape=inst_shape,
        dtype=torch.float64,
        T__K=300.0,
        W__um=W__um,
        L__um=L__um,
    )
    dev.fabricate()
    return dev


def test_nmos_enhancement_conducts_and_partial_signs() -> None:
    """Enhancement NMOS (vth0 > 0): forward bias conducts, ids rises with Vg, partial signs hold."""
    k = 3
    dev = _make(Nmos, vth0__V=0.4, inst_shape=(k,))
    snap = dev.snapshot(shape=(k,), multi_coords=None)
    vg = torch.linspace(0.5, 1.0, k, dtype=torch.float64)
    vd = torch.full((k,), 0.6, dtype=torch.float64)
    vs = torch.full((k,), 0.1, dtype=torch.float64)
    dc = dev.solve_dc(vg__V=vg, vd__V=vd, vs__V=vs, snap=snap)
    # vd > vs and Vgs >= vth -> forward (drain -> source) conduction.
    assert torch.all(dc.ids__uA > 0.0)
    # ids increases monotonically as the gate rises (gm >= 0).
    assert torch.all(dc.ids__uA[1:] - dc.ids__uA[:-1] > 0.0)
    # Node-partial sign contract.
    assert torch.all(dc.did_dvd__uS >= 0.0)
    assert torch.all(dc.did_dvs__uS <= 0.0)


def test_pmos_enhancement_conducts_negative() -> None:
    """Enhancement PMOS (vth0 < 0): source-high / drain-low with a low gate conducts; ids < 0."""
    k = 3
    v_dd = 0.9
    dev = _make(Pmos, vth0__V=-0.4, inst_shape=(k,))
    snap = dev.snapshot(shape=(k,), multi_coords=None)
    # Gate swept low -> high; source held high (v_dd), drain low (0).
    vg = torch.linspace(0.0, 0.5, k, dtype=torch.float64)
    vs = torch.full((k,), v_dd, dtype=torch.float64)
    vd = torch.zeros(k, dtype=torch.float64)
    dc = dev.solve_dc(vg__V=vg, vd__V=vd, vs__V=vs, snap=snap)
    # Real source -> drain flow gives a negative I_ds for a p-channel device.
    assert torch.all(dc.ids__uA < 0.0)
    # A lower gate is a larger source-gate overdrive -> more negative current,
    # so ids climbs monotonically toward 0 as Vg rises.
    assert torch.all(dc.ids__uA[1:] - dc.ids__uA[:-1] > 0.0)
    assert dc.ids__uA[0] < dc.ids__uA[-1]
    # The partial-sign contract is polarity-independent.
    assert torch.all(dc.did_dvd__uS >= 0.0)
    assert torch.all(dc.did_dvs__uS <= 0.0)


def test_depletion_nmos_conducts_at_zero_gate() -> None:
    """Depletion NMOS (vth0 < 0) is accepted by config and conducts at Vg = 0 with vd > vs."""
    dev = _make(Nmos, vth0__V=-0.4, inst_shape=(1,))
    snap = dev.snapshot(shape=(1,), multi_coords=None)
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
    """Central differences of ids w.r.t. each terminal match the analytic partials."""
    k = len(vg)
    dev = _make(cls, vth0__V=vth0, inst_shape=(k,))
    snap = dev.snapshot(shape=(k,), multi_coords=None)
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


def test_all_off_snapshot_deterministic() -> None:
    """With every mismatch toggle off, fabricate + snapshot are deterministic and uniform."""
    k = 2
    dev = _make(
        Nmos,
        vth0__V=0.4,
        inst_shape=(k,),
        policy=_OFF,
        A_vt__mV_um=1.0,
        A_beta_relative__um=0.1,
    )
    snap1 = dev.snapshot(shape=(k,), multi_coords=None)
    dev.fabricate()
    snap2 = dev.snapshot(shape=(k,), multi_coords=None)
    # No mismatch applied -> every cell equals the nominal and refabricate is identical.
    assert torch.allclose(snap1.beta__uA_per_V2, dev.nominal_beta__uA_per_V2.expand(k))
    assert torch.allclose(snap1.vth__V, dev.nominal_vth__V.expand(k))
    assert torch.equal(snap1.beta__uA_per_V2, snap2.beta__uA_per_V2)
    assert torch.equal(snap1.vth__V, snap2.vth__V)


def test_abstract_base_cannot_instantiate() -> None:
    """The polarity-free MOSFET base is abstract; only NMOS / PMOS construct."""
    with pytest.raises(TypeError):
        Mosfet(
            config=_config(vth0__V=0.4),
            policy=_OFF,
            inst_shape=(1,),
            dtype=torch.float64,
            T__K=300.0,
            W__um=1.0,
            L__um=1.0,
        )


def test_config_rejects_nonpositive_mobility() -> None:
    with pytest.raises(ValueError):
        _config(mu0__cm2_per_V_s=0.0)


def test_config_rejects_nonpositive_oxide_cap() -> None:
    with pytest.raises(ValueError):
        _config(c_ox__fF_per_um2=0.0)


def test_config_rejects_subunity_n_factor() -> None:
    with pytest.raises(ValueError):
        _config(n_factor=1.0)


def test_config_accepts_either_vth_sign() -> None:
    """vth0 sign is unconstrained: enhancement (+) and depletion / p-channel (-) both validate."""
    assert _config(vth0__V=0.4).vth0__V == 0.4
    assert _config(vth0__V=-0.4).vth0__V == -0.4
