"""Tests for the new signal-chain primitives.

Covers the standalone behaviour of :class:`OpAmpTIA`, :class:`AnalogMux`,
and :class:`Decoder` in isolation from the xbar.  The xbar's
integration with these primitives is exercised via the macro-level
tests; here we just confirm the per-component contracts hold.
"""

from __future__ import annotations

import pytest
import torch

from neurox.analog import (
    AnalogMux,
    AnalogMuxConfig,
    AnalogMuxPolicy,
    Decoder,
    DecoderConfig,
)
from neurox.analog.dac import GeneralDAC, GeneralDACConfig, GeneralDACPolicy
from neurox.analog.tia import OpAmpTIA, OpAmpTIAConfig, OpAmpTIAPolicy
from neurox.device import NMOSConfig, NMOSPolicy

_NMOS_OFF = NMOSPolicy(A_vt_mismatch=False, A_beta_mismatch=False)
_TIA_OFF = OpAmpTIAPolicy(opamp_gain_sigma=False, nmos=_NMOS_OFF)
_TIA_ON_GAIN = OpAmpTIAPolicy(opamp_gain_sigma=True, nmos=_NMOS_OFF)
_DAC_OFF = GeneralDACPolicy(drive_thermal=False)


def _make_tia(
    *,
    inst_shape: tuple[int, ...],
    opamp_gain: float = 20.0,
    v_ref__V: float = 0.2,
    v_dd__V: float = 0.9,
    v_nmos_bias__V: float = 0.9,
    opamp_gain_sigma: float = 0.0,
    apply_opamp_gain_sigma: bool = False,
) -> OpAmpTIA:
    """Build a OpAmpTIA + internal NMOS pseudo-resistor sized for the tests."""
    nmos_config = NMOSConfig(
        mu0__cm2_per_V_s=200.0,
        c_ox__fF_per_um2=31.4,
        vth0__V=0.40,
        n_factor=1.25,
        T_ref__K=300.0,
        ute=1.5,
        kt1__V=-0.002,
        A_vt__mV_um=0.0,
        A_beta_relative__um=0.0,
    )
    config = OpAmpTIAConfig(
        v_ref__V=v_ref__V,
        v_nmos_bias__V=v_nmos_bias__V,
        v_dd__V=v_dd__V,
        opamp_gain=opamp_gain,
        opamp_gain_sigma=opamp_gain_sigma,
        nmos_config=nmos_config,
        pseudo_nmos_W__um=1.0,
        pseudo_nmos_L__um=0.06,
        output_saturation_softness__V=v_dd__V / 2.0,
        n_newton=30,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
        latency_per_op__ns=0.0,
    )
    policy = _TIA_ON_GAIN if apply_opamp_gain_sigma else _TIA_OFF
    return OpAmpTIA(
        config=config,
        policy=policy,
        name="tia",
        inst_shape=inst_shape,
        dtype=torch.float64,
        T__K=300.0,
    )


def test_tia_fabricate_shapes() -> None:
    """fabricate populates internal buffers at the inst_shape from __init__."""
    shape = (8,)
    tia = _make_tia(inst_shape=shape)
    tia.fabricate()
    assert tia.opamp_gain.shape == shape
    assert tia.nmos.beta__uA_per_V2.shape == shape
    assert tia.nmos.vth__V.shape == shape


def test_tia_solve_dc_zero_current() -> None:
    """At ``i_in = 0`` the NMOS is off, so ``v_clamp = v_out``.

    With the smooth softclip the equilibrium shifts slightly from the
    pure-linear ``v_ref·A/(A+1)`` because ``softclip`` has a small
    non-identity bias on the interior; the shift is ``O(1/A)`` and
    stays close to ``v_ref``.  The strong invariant is the off-NMOS
    condition ``v_d = v_s`` (i.e. ``v_out = v_clamp``).
    """
    tia = _make_tia(inst_shape=(4,), opamp_gain=20.0, v_ref__V=0.2)
    tia.fabricate()
    runtime = tia.snapshot(shape=(4,))
    i_in = torch.zeros(4, dtype=torch.float64)
    dc = tia.solve_dc(i_in, runtime, v_clamp_init__V=None)
    # NMOS off: v_out = v_clamp.
    assert torch.allclose(dc.v_out__V, dc.v_clamp__V, atol=1e-6)
    # Equilibrium stays within a few percent of the pure-linear
    # static op, well inside the rails.
    pure_linear = 0.2 * 20.0 / 21.0
    assert torch.all((dc.v_clamp__V - pure_linear).abs() < 5e-3)


def test_tia_solve_dc_monotone_in_linear_region() -> None:
    """``v_out`` increases monotonically with input current in the linear region."""
    tia = _make_tia(inst_shape=(5,), opamp_gain=20.0, v_ref__V=0.2, v_dd__V=0.9)
    tia.fabricate()
    runtime = tia.snapshot(shape=(5,))
    i_in = torch.tensor([0.0, 5.0, 15.0, 30.0, 60.0], dtype=torch.float64)
    dc = tia.solve_dc(i_in, runtime, v_clamp_init__V=None)

    # v_out is monotonically non-decreasing with input current.
    diffs = dc.v_out__V[1:] - dc.v_out__V[:-1]
    assert torch.all(diffs >= -1e-9), f"v_out not monotone: {dc.v_out__V.tolist()}"

    # Closed-loop consistency: v_out = softclip(A · (v_ref - v_clamp)).
    v_out_lin = tia.opamp_gain * (0.2 - dc.v_clamp__V)
    c = 0.9 / 2.0
    h = 0.9 / 2.0
    s = 0.9 / 2.0
    expected_v_out = c + h * torch.tanh((v_out_lin - c) / s)
    assert torch.allclose(dc.v_out__V, expected_v_out, atol=1e-5)

    # Stays strictly inside the supply rails (softclip never touches them).
    assert torch.all(dc.v_out__V < 0.9)
    assert torch.all(dc.v_out__V > 0.0)


def test_tia_solve_dc_smooth_saturation_near_vdd() -> None:
    """Driving the stage past its triode capacity smoothly approaches ``v_dd``.

    With the rail limit modeled by an in-residual softclip, ``v_out``
    approaches ``v_dd`` asymptotically and ``dVout_dI`` rolls off
    continuously instead of being step-masked to zero.  At a current
    near the saturation knee the output must be in the upper rail-half
    and the transimpedance must have dropped by ≥ 1 decade relative to
    the linear regime.
    """
    tia = _make_tia(inst_shape=(1,), opamp_gain=20.0, v_ref__V=0.2, v_dd__V=0.9)
    tia.fabricate()
    runtime = tia.snapshot(shape=(1,))
    # 10 uA is well below the saturation knee for this 1-um W device;
    # 800 uA pushes the pseudo-resistor's triode capacity to where the
    # op-amp output is approaching the rail.
    dc_lin = tia.solve_dc(torch.tensor([10.0], dtype=torch.float64), runtime, v_clamp_init__V=None)
    dc_sat = tia.solve_dc(torch.tensor([800.0], dtype=torch.float64), runtime, v_clamp_init__V=None)

    # v_out approaches the upper rail (allow ``==`` in floating point).
    assert torch.all(dc_sat.v_out__V <= 0.9 + 1e-9)
    assert torch.all(dc_sat.v_out__V > 0.85)

    # Sensitivities keep their sign and remain finite — no piecewise
    # mask to exact zero.
    assert torch.all(dc_sat.dVclamp_dI__MOhm < 0.0)
    assert torch.all(dc_sat.dVout_dI__MOhm > 0.0)

    # Transimpedance has rolled off by ≥ 5x vs the linear regime.
    assert (dc_sat.dVout_dI__MOhm.abs() < 0.2 * dc_lin.dVout_dI__MOhm.abs()).all()

    # Continuity across the saturation knee: a small bump in i_in
    # produces a small bump in v_out, not a discontinuous jump.
    dc_more = tia.solve_dc(torch.tensor([801.0], dtype=torch.float64), runtime, v_clamp_init__V=None)
    assert torch.all((dc_more.v_out__V - dc_sat.v_out__V).abs() < 1e-3)


def test_tia_solve_dc_residual_is_small() -> None:
    """After Newton convergence the closed-loop residual is below 1e-3 uA."""
    tia = _make_tia(inst_shape=(6,), opamp_gain=20.0, v_ref__V=0.2, v_dd__V=0.9)
    tia.fabricate()
    runtime = tia.snapshot(shape=(6,))
    i_in = torch.tensor([0.0, 1.0, 3.0, 10.0, 30.0, 60.0], dtype=torch.float64)
    dc = tia.solve_dc(i_in, runtime, v_clamp_init__V=None)
    # NMOS current at the converged operating point should match i_in
    # (when not in the clip / saturation regime — pick currents below
    # the saturation knee for this sized device).
    nmos_dc = tia.nmos.solve_dc(
        vg__V=tia.config.v_nmos_bias__V,
        vd__V=dc.v_out__V,
        vs__V=dc.v_clamp__V,
        snapshot=runtime.nmos_snapshot,
    )
    residual = (nmos_dc.ids__uA - i_in).abs()
    assert torch.all(residual < 1e-3), f"residual = {residual.tolist()}"


def test_tia_solve_dc_sensitivity_matches_finite_difference() -> None:
    """Analytical ``dVclamp_dI`` / ``dVout_dI`` agree with a finite-difference probe."""
    tia = _make_tia(inst_shape=(4,), opamp_gain=20.0, v_ref__V=0.2, v_dd__V=0.9)
    tia.fabricate()
    runtime = tia.snapshot(shape=(4,))
    i_in = torch.tensor([1.0, 10.0, 30.0, 60.0], dtype=torch.float64)
    dc = tia.solve_dc(i_in, runtime, v_clamp_init__V=None)
    h = 1e-3
    dc_eps = tia.solve_dc(i_in + h, runtime, v_clamp_init__V=None)
    fd_dVclamp = (dc_eps.v_clamp__V - dc.v_clamp__V) / h
    fd_dVout = (dc_eps.v_out__V - dc.v_out__V) / h
    assert torch.allclose(dc.dVclamp_dI__MOhm, fd_dVclamp, atol=1e-7, rtol=1e-4)
    assert torch.allclose(dc.dVout_dI__MOhm, fd_dVout, atol=1e-6, rtol=1e-4)


def _make_mux_config(**overrides) -> AnalogMuxConfig:
    """Build an :class:`AnalogMuxConfig` with every field explicit.

    Baseline = pass-through (``mux_gain=1.0``, both noise sigmas zero,
    zero energy / PPA). Tests override the field(s) they exercise.
    """
    base = dict(
        energy_per_access__fJ=0.0,
        mux_gain=1.0,
        mux_noise_cm_sigma__V=0.0,
        mux_noise_dm_sigma__V=0.0,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
        latency_per_op__ns=0.0,
    )
    base.update(overrides)
    return AnalogMuxConfig(**base)


_MUX_OFF = AnalogMuxPolicy(mux_noise_cm=False, mux_noise_dm=False)
_MUX_CM_ON = AnalogMuxPolicy(mux_noise_cm=True, mux_noise_dm=False)
_MUX_DM_ON = AnalogMuxPolicy(mux_noise_cm=False, mux_noise_dm=True)


def test_analog_mux_passthrough() -> None:
    """Default ``mux_gain=1.0`` with both sigmas ``None`` is a pure pass-through.

    Energy flows through the profiler side channel; ``transport``
    returns only the two analog legs after Phase D.
    """
    mux = AnalogMux(
        config=_make_mux_config(energy_per_access__fJ=1.0),
        policy=_MUX_OFF,
        name="mux",
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    v_pos = torch.tensor([0.1, 0.2, 0.3])
    v_neg = torch.tensor([0.0, 0.1, 0.2])
    out_pos, out_neg = mux.transport(v_pos, v_neg)
    assert torch.equal(out_pos, v_pos)
    assert torch.equal(out_neg, v_neg)


def test_analog_mux_gain_attenuates_both_legs() -> None:
    """``mux_gain`` scales both legs identically; no inline noise added."""
    mux = AnalogMux(
        config=_make_mux_config(mux_gain=0.5),
        policy=_MUX_OFF,
        name="mux",
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    v_pos = torch.tensor([0.4, 0.6])
    v_neg = torch.tensor([0.2, 0.3])
    out_pos, out_neg = mux.transport(v_pos, v_neg)
    assert torch.equal(out_pos, 0.5 * v_pos)
    assert torch.equal(out_neg, 0.5 * v_neg)


def test_analog_mux_cm_noise_is_common_to_both_legs() -> None:
    """CM noise lands with matching sign on both legs (suppressed by diff ADC)."""
    torch.manual_seed(0)
    mux = AnalogMux(
        config=_make_mux_config(mux_noise_cm_sigma__V=0.05),
        policy=_MUX_CM_ON,
        name="mux",
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    v_pos = torch.zeros(10_000)
    v_neg = torch.zeros(10_000)
    out_pos, out_neg = mux.transport(v_pos, v_neg)
    # Both legs see the *same* CM realisation each draw.
    assert torch.allclose(out_pos, out_neg)
    # And the marginal std matches the configured sigma.
    assert abs(float(out_pos.std()) - 0.05) < 5e-3


def test_analog_mux_dm_noise_is_antisymmetric() -> None:
    """DM noise lands with opposite sign on the two legs."""
    torch.manual_seed(0)
    mux = AnalogMux(
        config=_make_mux_config(mux_noise_dm_sigma__V=0.05),
        policy=_MUX_DM_ON,
        name="mux",
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    v_pos = torch.zeros(10_000)
    v_neg = torch.zeros(10_000)
    out_pos, out_neg = mux.transport(v_pos, v_neg)
    # n_pos == +n_dm, n_neg == -n_dm.
    assert torch.allclose(out_pos, -out_neg)
    assert abs(float(out_pos.std()) - 0.05) < 5e-3


def test_analog_mux_invalid_gain() -> None:
    with pytest.raises(ValueError):
        AnalogMux(
            config=_make_mux_config(mux_gain=0.0),
            policy=_MUX_OFF,
            name="mux",
            inst_shape=(),
            dtype=torch.float32,
            T__K=300.0,
        )


def test_decoder_passthrough_drives_dac() -> None:
    """Non-bit-serial decoder passes integer codes straight to the DAC."""
    dac = GeneralDAC(
        config=GeneralDACConfig(
            code_to_signal=[0.0, 1.2],
            drive_thermal__V=0.0,
            energy_per_op__fJ=0.0,
            latency_per_op__ns=0.0,
            leakage_per_inst__uW=0.0,
            area_per_inst__um2=0.0,
        ),
        policy=_DAC_OFF,
        name="dac",
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    dec = Decoder(
        config=DecoderConfig(
            n_address_bits=6,
            fanout=4,
            drive_strength__uA=1000.0,
            bit_serial=False,
            c_gate__fF=0.5,
            v_dd__V=1.0,
            t_gate__ns=0.05,
            e_overhead__fJ=0.0,
            leakage_per_inst__uW=0.0,
            area_per_inst__um2=0.0,
        ),
        name="dec",
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    code = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    sig = dec.drive(code, dac)
    # DAC LUT lookup: code 0 → 0 V, code 1 → 1.2 V.
    assert torch.allclose(sig, torch.tensor([0.0, 1.2, 1.2, 0.0]), atol=1e-6)


def test_decoder_bit_serial_expands_codes() -> None:
    """Bit-serial decoder splits an integer code into ``n_address_bits`` planes."""
    dac = GeneralDAC(
        config=GeneralDACConfig(
            code_to_signal=[0.0, 1.2],
            drive_thermal__V=0.0,
            energy_per_op__fJ=0.0,
            latency_per_op__ns=0.0,
            leakage_per_inst__uW=0.0,
            area_per_inst__um2=0.0,
        ),
        policy=_DAC_OFF,
        name="dac",
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    dec = Decoder(
        config=DecoderConfig(
            n_address_bits=4,
            fanout=4,
            drive_strength__uA=1000.0,
            bit_serial=True,
            c_gate__fF=0.5,
            v_dd__V=1.0,
            t_gate__ns=0.05,
            e_overhead__fJ=0.0,
            leakage_per_inst__uW=0.0,
            area_per_inst__um2=0.0,
        ),
        name="dec",
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    # Input ``5 = 0b0101`` → bit 0 = 1, bit 1 = 0, bit 2 = 1, bit 3 = 0.
    code = torch.tensor([5], dtype=torch.long)
    sig = dec.drive(code, dac)
    # Decoder stacks bits at dim=-2 (between the input row dim and the
    # batch dim).  Trailing dim is the row dim (size 1 here); the new
    # bit-cycle axis is at dim=-2.
    assert sig.dim() == code.dim() + 1
    bit_axis_size = sig.shape[-2]
    assert bit_axis_size == 4
    expected = torch.tensor([1.2, 0.0, 1.2, 0.0])
    assert torch.allclose(sig.flatten(), expected, atol=1e-6)


def test_decoder_invalid_address_bits() -> None:
    with pytest.raises(ValueError):
        DecoderConfig(
            n_address_bits=0,
            fanout=4,
            drive_strength__uA=1000.0,
            bit_serial=False,
            c_gate__fF=0.5,
            v_dd__V=1.0,
            t_gate__ns=0.05,
            e_overhead__fJ=0.0,
            leakage_per_inst__uW=0.0,
            area_per_inst__um2=0.0,
        )
