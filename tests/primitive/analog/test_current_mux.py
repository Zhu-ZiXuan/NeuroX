"""Imux elementwise transport tests."""

from __future__ import annotations

import pytest
import torch

from neurox import Profiler, stamp_names
from neurox.api.profiler import EnergyRecord
from neurox.common.profile_mixin import ProfileMixin
from neurox.primitive.analog import Imux, ImuxConfig, ImuxPolicy


def _energy_total(records: list[EnergyRecord], module: ProfileMixin) -> float:
    """Sum the logged dynamic energy [fJ] of the records `module` emitted.

    A record's tensor is a per-unit-operation layout, so each one totals to its
    own scalar before the records are summed.
    """
    return sum(
        (float(r.dynamic_energy__fJ.sum()) for r in records if r.qualified_name == module.qualified_name),
        0.0,
    )


@pytest.mark.parametrize("mux_gain", [1.0, 2.0])
def test_transport_applies_gain_elementwise(mux_gain: float) -> None:
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
    stamp_names(mux)

    i__uA = torch.arange(16, dtype=torch.float64).reshape(2, 4, 2)
    with Profiler() as p:
        out = mux.transport(i__uA)
    assert out.shape == i__uA.shape
    torch.testing.assert_close(out, mux_gain * i__uA)

    assert _energy_total(p.records, mux) == 0.0
