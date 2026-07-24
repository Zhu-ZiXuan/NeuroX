"""Vmux transport and accounting tests."""

import pytest
import torch

from neurox.common.profiler import NeuroxProfiler
from neurox.primitive.analog import Vmux, VmuxConfig, VmuxPolicy


def test_transport_preserves_access_lane_layout_and_accounts_output() -> None:
    mux = Vmux(
        config=VmuxConfig(
            mux_ratio=4,
            mux_gain=2.0,
            mux_gain_mismatch_sigma_relative=0.0,
            mux_noise_sigma__V=0.0,
            energy_per_access__fJ=3.0,
            latency_per_op__ns=5.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        policy=VmuxPolicy(
            mux_gain_mismatch=False,
            mux_noise=False,
        ),
        inst_shape=(2,),
        dtype=torch.float64,
        T__K=300.0,
    )
    mux.fabricate()
    v__V = torch.arange(16, dtype=torch.float64).reshape(2, 4, 2)

    with NeuroxProfiler() as profiler:
        actual__V = mux.transport(v__V)

    assert actual__V.shape == v__V.shape
    torch.testing.assert_close(actual__V, 2.0 * v__V)
    assert profiler.total_dynamic_energy__fJ == 48.0
    assert profiler.total_latency__ns == 40.0


@pytest.mark.parametrize("shape", [(3, 2), (4, 3), (8,)])
def test_transport_requires_access_lane_layout(shape: tuple[int, ...]) -> None:
    mux = Vmux(
        config=VmuxConfig(
            mux_ratio=4,
            mux_gain=1.0,
            mux_gain_mismatch_sigma_relative=0.0,
            mux_noise_sigma__V=0.0,
            energy_per_access__fJ=0.0,
            latency_per_op__ns=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        policy=VmuxPolicy(mux_gain_mismatch=False, mux_noise=False),
        inst_shape=(2,),
        dtype=torch.float64,
        T__K=300.0,
    )
    mux.fabricate()

    with pytest.raises(ValueError, match="trailing axes"):
        mux.transport(torch.ones(shape))
