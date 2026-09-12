"""Vmux elementwise transport and accounting tests."""

import torch

from neurox import Profiler, Reporter, stamp_names
from neurox.primitive.analog import Vmux, VmuxConfig, VmuxPolicy


def test_transport_applies_gain_elementwise_and_accounts_output() -> None:
    mux = Vmux(
        config=VmuxConfig(
            mux_ratio=4,
            mux_gain=2.0,
            mux_gain_mismatch_sigma_relative=0.0,
            mux_noise_sigma__V=0.0,
            energy_per_access__fJ=3.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        policy=VmuxPolicy(
            mux_gain_mismatch=False,
            mux_noise=False,
        ),
        inst_shape=(1, 2),
        dtype=torch.float64,
        T__K=300.0,
    )
    mux.fabricate()
    stamp_names(mux)
    v__V = torch.arange(16, dtype=torch.float64).reshape(2, 4, 2).requires_grad_()

    with Profiler() as profiler:
        actual__V = mux.transport(v__V)

    assert actual__V.shape == v__V.shape
    assert not actual__V.requires_grad
    assert torch.is_grad_enabled()
    torch.testing.assert_close(actual__V, 2.0 * v__V)
    assert Reporter(mux).total_dynamic_energy__fJ(profiler) == 48.0
