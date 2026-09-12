"""RRAM device policy and drift tests."""

from dataclasses import replace

import pytest
import torch

from neurox.primitive.device.rram import Rram, RramConfig, RramPolicy


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


def test_rram_config_requires_positive_minimum_conductance() -> None:
    config = RramConfig.from_preset("process/rram:default")
    with pytest.raises(ValueError, match="g_min__uS"):
        replace(config, g_min__uS=0.0)


def test_rram_requires_conductance_bounds_representable_by_dtype() -> None:
    config = RramConfig.from_preset("process/rram:default")
    config = replace(config, g_min__uS=torch.finfo(torch.float32).tiny / 2.0)

    with pytest.raises(ValueError, match="normal value representable"):
        Rram(
            config=config,
            policy=_policy(drift=False),
            inst_shape=(),
            dtype=torch.float32,
            T__K=300.0,
            g_max__uS=100.0,
        )


def test_rram_program_uses_zero_elapsed_time() -> None:
    target = torch.tensor(50.0, dtype=torch.float64)

    drift_off = _rram(drift=False)
    drift_off.program(target)
    actual_off = drift_off.snapshot(shape=()).g__uS

    drift_on = _rram(drift=True)
    drift_on.program(target)
    actual_on = drift_on.snapshot(shape=()).g__uS

    assert torch.equal(actual_off, target)
    assert torch.equal(actual_on, target)
