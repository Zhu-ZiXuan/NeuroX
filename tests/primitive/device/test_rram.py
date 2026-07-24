"""RRAM device policy and drift tests."""

from dataclasses import replace

import pytest
import torch

from neurox.primitive.device import Rram, RramConfig, RramPolicy


def _policy(*, drift: bool) -> RramPolicy:
    return RramPolicy(
        prog_gamma=False,
        drift=drift,
        stuck_at=False,
        read_telegraph=False,
        read_thermal=False,
    )


def _rram(*, drift: bool) -> Rram:
    return Rram(
        config=RramConfig.from_preset("process/rram:default"),
        policy=_policy(drift=drift),
        inst_shape=(),
        dtype=torch.float64,
        T__K=300.0,
        g_max__uS=100.0,
    )


def test_rram_config_requires_positive_drift_reference_time() -> None:
    config = RramConfig.from_preset("process/rram:default")
    with pytest.raises(ValueError, match="drift_t0"):
        replace(config, drift_t0=0.0)


def test_rram_drift_is_controlled_by_policy() -> None:
    target = torch.tensor(50.0, dtype=torch.float64)
    elapsed = 4.0

    drift_off = _rram(drift=False)
    drift_off.program(target, elapsed)
    actual_off = drift_off.snapshot(shape=(), multi_coords=None).g__uS

    drift_on = _rram(drift=True)
    drift_on.program(target, elapsed)
    actual_on = drift_on.snapshot(shape=(), multi_coords=None).g__uS

    config = drift_on.config
    expected_on = target * (elapsed / config.drift_t0) ** (-config.drift_decay_rate)
    assert torch.equal(actual_off, target)
    assert torch.allclose(actual_on, expected_on)
