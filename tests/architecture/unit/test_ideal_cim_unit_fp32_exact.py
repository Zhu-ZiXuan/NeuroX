"""IdealLinearUnit fp32-exact fast path, exercised via the public ``linear()``.

The substrate ``_matmul`` switches to fp32 when ``K * max|x| * max|w| <
2^24`` (every partial sum then accumulates exactly in IEEE fp32) and stays
on the int64 matmul otherwise. The fast path is what makes the unit
GPU-capable: CUDA has no integer-matmul kernel, so the fallback path is
CPU-by-design.
"""

from __future__ import annotations

import pytest
import torch

from neurox.architecture.unit import IdealLinearUnit, IdealLinearUnitConfig, IdealLinearUnitPolicy

_ADC_MODE = 0
_ADC_BITS = 0


def _build_unit(
    *,
    x_value_range: tuple[int, int],
    w_value_range: tuple[int, int],
    w_logical_shape: tuple[int, ...],
) -> IdealLinearUnit:
    unit = IdealLinearUnit(
        config=IdealLinearUnitConfig(
            x_value_range=x_value_range,
            w_value_range=w_value_range,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        policy=IdealLinearUnitPolicy(),
        w_logical_shape=w_logical_shape,
        dtype=torch.float32,
        T__K=300.0,
        ideal_xbar=False,
    )
    unit.eval()
    return unit


def _random_program_and_input(
    unit: IdealLinearUnit,
    *,
    batch: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    w_lo, w_hi = unit.w_value_range
    weight = torch.randint(w_lo, w_hi + 1, unit._w_logical_shape, dtype=torch.int32, generator=generator)
    x_lo, x_hi = unit.x_value_range
    k = unit._w_logical_shape[-1]
    x = torch.randint(x_lo, x_hi + 1, (batch, k), dtype=torch.int32, generator=generator)
    unit.program(weight.to(device))
    return weight, x


_REPRESENTATIVE = [
    # (x_value_range, w_value_range, w_logical_shape)
    pytest.param((0, 1), (-1, 1), (16, 256), id="spike_x_ternary_w"),
    pytest.param((-8, 7), (-127, 127), (32, 768), id="signed_x_int8_w"),
    pytest.param((0, 255), (-255, 255), (8, 64), id="wide_x_wide_w"),
]


class TestFastPathBitExactness:
    @pytest.mark.parametrize(("x_value_range", "w_value_range", "w_logical_shape"), _REPRESENTATIVE)
    def test_cpu_matches_int64_oracle(
        self,
        x_value_range: tuple[int, int],
        w_value_range: tuple[int, int],
        w_logical_shape: tuple[int, ...],
    ) -> None:
        unit = _build_unit(x_value_range=x_value_range, w_value_range=w_value_range, w_logical_shape=w_logical_shape)
        assert unit._fp32_exact is True
        weight, x = _random_program_and_input(unit, batch=7, seed=11, device=torch.device("cpu"))
        y = unit.linear(x, adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
        oracle = x.to(torch.int64) @ weight.to(torch.int64).transpose(-2, -1)
        assert y.dtype == torch.int64
        assert torch.equal(y, oracle)

    @pytest.mark.parametrize(("x_value_range", "w_value_range", "w_logical_shape"), _REPRESENTATIVE)
    def test_gpu_matches_cpu_oracle(
        self,
        device: torch.device,
        x_value_range: tuple[int, int],
        w_value_range: tuple[int, int],
        w_logical_shape: tuple[int, ...],
    ) -> None:
        unit = _build_unit(x_value_range=x_value_range, w_value_range=w_value_range, w_logical_shape=w_logical_shape)
        unit.to(device)
        weight, x = _random_program_and_input(unit, batch=7, seed=13, device=device)
        oracle = x.to(torch.int64) @ weight.to(torch.int64).transpose(-2, -1)
        y = unit.linear(x.to(device), adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
        assert y.device.type == device.type
        assert torch.equal(y.cpu(), oracle)

    def test_leading_batch_dims_broadcast(self) -> None:
        unit = _build_unit(x_value_range=(0, 1), w_value_range=(-3, 3), w_logical_shape=(4, 32))
        weight, _ = _random_program_and_input(unit, batch=1, seed=17, device=torch.device("cpu"))
        generator = torch.Generator().manual_seed(19)
        x = torch.randint(0, 2, (2, 3, 5, 32), dtype=torch.int32, generator=generator)
        y = unit.linear(x, adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
        oracle = x.to(torch.int64) @ weight.to(torch.int64).transpose(-2, -1)
        assert y.shape == (2, 3, 5, 4)
        assert torch.equal(y, oracle)


class TestFallbackTrigger:
    """Bound at or above ``2^24`` keeps the int64 matmul (CPU-by-design)."""

    def test_flag_disabled_and_exact_beyond_fp32(self) -> None:
        # Bound = 3 * 2^23 * 1 >= 2^24 -> fallback to int64.
        unit = _build_unit(x_value_range=(0, 1), w_value_range=(0, 2**23), w_logical_shape=(1, 3))
        assert unit._fp32_exact is False
        # Dot = 2^24 + 1 is NOT fp32-representable: the fast path would
        # round it; the int64 path must carry it exactly.
        assert torch.tensor(2**24 + 1, dtype=torch.float32).item() == 2**24
        unit.program(torch.tensor([[2**23, 2**23, 1]], dtype=torch.int32))
        x = torch.ones(1, 3, dtype=torch.int32)
        y = unit.linear(x, adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
        assert y.item() == 2**24 + 1
