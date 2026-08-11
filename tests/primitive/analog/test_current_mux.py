"""Imux transports pre-scheduled single-ended currents.

The mux does not self-account rail energy or latency. Its caller owns the
access/lane layout; the mux preserves that layout and applies transport gain.
"""

from __future__ import annotations

import pytest
import torch

from neurox.common.mixin import ProfileMixin
from neurox.common.profiler import EnergyEvent, NeuroxProfiler
from neurox.primitive.analog.current_mux import (
    Imux,
    ImuxConfig,
    ImuxPolicy,
)


def _energy_total(events: list[EnergyEvent], module: ProfileMixin) -> float:
    """Sum the logged dynamic energy [fJ] of the events ``module`` emitted.

    An event payload is a per-unit-operation tensor, so each one totals to its
    own scalar before the events are summed.
    """
    return sum((float(e.dynamic_energy__fJ.sum()) for e in events if e.module is module), 0.0)


@pytest.mark.parametrize("mux_gain", [1.0, 2.0])
def test_transport_preserves_access_lane_layout(mux_gain: float) -> None:
    mux = Imux(
        config=ImuxConfig(
            mux_ratio=4,
            mux_gain=mux_gain,
        ),
        policy=ImuxPolicy(),
        inst_shape=(2,),
        dtype=torch.float64,
        T__K=300.0,
    )
    mux.eval()

    i__uA = torch.arange(16, dtype=torch.float64).reshape(2, 4, 2)
    with NeuroxProfiler() as p:
        out = mux.transport(i__uA)
    assert out.shape == i__uA.shape
    torch.testing.assert_close(out, mux_gain * i__uA)

    assert _energy_total(p.energy_events, mux) == 0.0


@pytest.mark.parametrize("shape", [(3, 2), (4, 3), (8,)])
def test_transport_requires_access_lane_layout(shape: tuple[int, ...]) -> None:
    mux = Imux(
        config=ImuxConfig(mux_ratio=4, mux_gain=1.0),
        policy=ImuxPolicy(),
        inst_shape=(2,),
        dtype=torch.float64,
        T__K=300.0,
    )
    with pytest.raises(ValueError, match="trailing axes"):
        mux.transport(torch.ones(shape))
