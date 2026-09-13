"""Ideal linear outputs preserve exact integer products across operand ranges."""

from __future__ import annotations

import pytest
import torch

from neurox.architecture.unit.ideal import IdealLinearUnit, IdealLinearUnitConfig, IdealLinearUnitPolicy

_QUANTIZATION_MODE = 0
_ADC_BITS = 1


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
        ideal_macro=False,
    )
    unit.eval()
    return unit


def _random_program_and_input(
    unit: IdealLinearUnit,
    *,
    w_logical_shape: tuple[int, ...],
    batch: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    w_lo, w_hi = unit.w_value_range
    weight = torch.randint(w_lo, w_hi + 1, w_logical_shape, dtype=torch.int32, generator=generator)
    x_lo, x_hi = unit.x_value_range
    k = w_logical_shape[-1]
    x = torch.randint(x_lo, x_hi + 1, (batch, k), dtype=torch.int32, generator=generator)
    unit.program(weight.to(device))
    return weight, x


_REPRESENTATIVE = [
    # (x_value_range, w_value_range, w_logical_shape)
    pytest.param((0, 1), (-1, 1), (16, 256), id="spike_x_ternary_w"),
    pytest.param((-8, 7), (-127, 127), (32, 768), id="signed_x_int8_w"),
    pytest.param((0, 255), (-255, 255), (8, 64), id="wide_x_wide_w"),
]


class TestIntegerExactness:
    @pytest.mark.parametrize(("x_value_range", "w_value_range", "w_logical_shape"), _REPRESENTATIVE)
    def test_device_matches_int64_oracle(
        self,
        device: torch.device,
        x_value_range: tuple[int, int],
        w_value_range: tuple[int, int],
        w_logical_shape: tuple[int, ...],
    ) -> None:
        unit = _build_unit(x_value_range=x_value_range, w_value_range=w_value_range, w_logical_shape=w_logical_shape)
        unit.to(device)
        weight, x = _random_program_and_input(unit, w_logical_shape=w_logical_shape, batch=7, seed=13, device=device)
        oracle = x.long() @ weight.long().transpose(-2, -1)
        y = unit.linear(x.to(device), quantization_mode=_QUANTIZATION_MODE, adc_active_bits=_ADC_BITS)
        assert y.device.type == device.type
        assert y.dtype == torch.int64
        assert torch.equal(y.cpu(), oracle)

    def test_leading_batch_dims_broadcast(self) -> None:
        unit = _build_unit(x_value_range=(0, 1), w_value_range=(-3, 3), w_logical_shape=(4, 32))
        weight, _ = _random_program_and_input(
            unit, w_logical_shape=(4, 32), batch=1, seed=17, device=torch.device("cpu")
        )
        generator = torch.Generator().manual_seed(19)
        x = torch.randint(0, 2, (2, 3, 5, 32), dtype=torch.int32, generator=generator)
        y = unit.linear(x, quantization_mode=_QUANTIZATION_MODE, adc_active_bits=_ADC_BITS)
        oracle = x.long() @ weight.long().transpose(-2, -1)
        assert y.shape == (2, 3, 5, 4)
        assert torch.equal(y, oracle)


class TestBeyondFp32Precision:
    """Integer outputs remain exact beyond fp32 precision."""

    @pytest.mark.parametrize(
        ("weights", "activation", "expected"),
        [
            pytest.param((2**23, 2**23, 1), 1, 2**24 + 1, id="beyond-fp32-precision"),
            pytest.param((4095,) * 257, 4095, 4095 * 4095 * 257, id="beyond-int32-range"),
        ],
    )
    def test_preserves_large_integer_products(self, weights: tuple[int, ...], activation: int, expected: int) -> None:
        unit = _build_unit(
            x_value_range=(0, activation), w_value_range=(0, max(weights)), w_logical_shape=(1, len(weights))
        )
        unit.program(torch.tensor([weights], dtype=torch.int32))
        x = torch.full((1, len(weights)), activation, dtype=torch.int32)
        y = unit.linear(x, quantization_mode=_QUANTIZATION_MODE, adc_active_bits=_ADC_BITS)
        assert y.item() == expected
