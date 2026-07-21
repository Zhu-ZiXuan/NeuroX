"""CurrentSubtractor: magnitude-and-sign current-difference primitive.

With both nonidealities off the subtractor is the exact reference transform
``i_diff = gain·|i_a - i_b|`` with ``sign = (i_a < i_b)``. Config validation
rejects a non-positive gain, a negative sigma, and negative rail / PPA
fields. The block self-bills one dynamic-energy event per ``subtract`` call
covering its three internal replica branches.
"""

from __future__ import annotations

import pytest
import torch

from neurox.common.profiler import NeuroxProfiler
from neurox.primitive.analog.current_subtractor import (
    CurrentSubtractor,
    CurrentSubtractorConfig,
    CurrentSubtractorPolicy,
)


def _make(
    *,
    gain: float = 2.0,
    v_rail__V: float = 0.8,
    area_per_inst__um2: float = 25.0,
    leakage_per_inst__uW: float = 0.2,
    inst_shape: tuple[int, ...] = (),
) -> CurrentSubtractor:
    sub = CurrentSubtractor(
        config=CurrentSubtractorConfig(
            gain=gain,
            mismatch_sigma_relative=0.1,
            offset_sigma__uA=0.5,
            v_rail__V=v_rail__V,
            area_per_inst__um2=area_per_inst__um2,
            leakage_per_inst__uW=leakage_per_inst__uW,
        ),
        policy=CurrentSubtractorPolicy(mismatch=False, offset=False),
        inst_shape=inst_shape,
        dtype=torch.float64,
        T__K=300.0,
    )
    sub.eval()
    return sub


def test_all_off_is_exact_magnitude_and_sign() -> None:
    """Both toggles off: exact ``gain·|a - b|`` magnitude and ``(a < b)`` sign."""
    gain = 2.0
    sub = _make(gain=gain)

    # One datum per relation: a > b, a == b, a < b.
    i_a__uA = torch.tensor([10.0, -4.0, 3.0], dtype=torch.float64)
    i_b__uA = torch.tensor([4.0, -4.0, 7.0], dtype=torch.float64)

    i_diff__uA, sign = sub.subtract(i_a__uA, i_b__uA, t_conduct__ns=10.0)

    torch.testing.assert_close(i_diff__uA, gain * (i_a__uA - i_b__uA).abs())
    assert torch.equal(sign, i_a__uA < i_b__uA)
    # The magnitude is unipolar; the direction travels in the sign bit.
    assert (i_diff__uA >= 0).all()


def test_validation_rejects_bad_config() -> None:
    """A non-positive gain and any negative sigma / rail / PPA field are rejected."""
    base = {
        "gain": 2.0,
        "mismatch_sigma_relative": 0.1,
        "offset_sigma__uA": 0.5,
        "v_rail__V": 0.8,
        "area_per_inst__um2": 25.0,
        "leakage_per_inst__uW": 0.2,
    }
    for override in (
        {"gain": 0.0},
        {"gain": -1.0},
        {"mismatch_sigma_relative": -1e-3},
        {"offset_sigma__uA": -1e-3},
        {"v_rail__V": -1e-3},
        {"area_per_inst__um2": -1e-3},
        {"leakage_per_inst__uW": -1e-3},
    ):
        with pytest.raises(ValueError):
            CurrentSubtractorConfig(**{**base, **override})

    # Zero sigmas denote an ideal (mismatch-free, offset-free) subtractor.
    CurrentSubtractorConfig(
        gain=1.0,
        mismatch_sigma_relative=0.0,
        offset_sigma__uA=0.0,
        v_rail__V=0.8,
        area_per_inst__um2=25.0,
        leakage_per_inst__uW=0.2,
    )


def test_subtract_self_bills_three_replica_branches() -> None:
    """``subtract`` logs one event: ``v_rail·t·(i_pos + i_neg + i_sub)`` elementwise."""
    gain = 2.0
    v_rail__V = 0.8
    t_conduct__ns = 10.0
    sub = _make(gain=gain, v_rail__V=v_rail__V, inst_shape=(2,))

    i_a__uA = torch.tensor([10.0, 3.0], dtype=torch.float64)
    i_b__uA = torch.tensor([4.0, 7.0], dtype=torch.float64)

    with NeuroxProfiler() as p:
        i_diff__uA, _ = sub.subtract(i_a__uA, i_b__uA, t_conduct__ns=t_conduct__ns)

    # All-off: the subtracted leg carries the nominal i_b (unit ratio, zero offset).
    torch.testing.assert_close(i_diff__uA, gain * (i_a__uA - i_b__uA).abs())
    expected__fJ = (v_rail__V * t_conduct__ns * (i_a__uA + i_b__uA + i_diff__uA)).sum()

    events = [e for e in p.energy_events if e.module is sub]
    assert len(events) == 1
    assert events[0].dynamic_energy__fJ == pytest.approx(expected__fJ.item())
    assert p.latency_events == []


def test_static_ppa_reports_per_instance_scaled() -> None:
    """Reporter leaf: area / leakage scale the per-instance config data by ``inst_count``."""
    sub = _make(area_per_inst__um2=25.0, leakage_per_inst__uW=0.2, inst_shape=(2,))
    assert sub.is_profile_target
    assert sub.area__um2 == pytest.approx(25.0 * 2)
    assert sub.leakage__uW == pytest.approx(0.2 * 2)
