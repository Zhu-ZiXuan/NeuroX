"""Tests for stochastic rounding."""

import pytest
import torch

from neurox.common.quantization import stochastic_round


def test_evaluation_uses_floor_rounding() -> None:
    value = torch.tensor([3.7, 3.0, -0.3], dtype=torch.float64)
    assert torch.equal(
        stochastic_round(value, enabled=False),
        torch.tensor([3.0, 3.0, -1.0], dtype=torch.float64),
    )


def test_integer_values_remain_deterministic_when_enabled() -> None:
    value = torch.tensor([0.0, 1.0, 2.0, -1.0], dtype=torch.float64)
    for seed in range(4):
        torch.manual_seed(seed)
        assert torch.equal(stochastic_round(value, enabled=True), value)


def test_enabled_rounding_uses_the_fractional_probability() -> None:
    torch.manual_seed(0)
    value = torch.full((100_000,), 3.7, dtype=torch.float64)
    rounded = stochastic_round(value, enabled=True)
    assert set(rounded.tolist()) == {3.0, 4.0}
    assert rounded.to(torch.float64).mean().item() == pytest.approx(3.7, abs=0.01)
