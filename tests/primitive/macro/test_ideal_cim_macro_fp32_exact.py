"""IdealCimMacro fp32-exact dot fast path.

The per-phase dot computation switches to fp32 einsum when
``_max_phase_dot_abs < 2^24`` (every per-cell product and partial sum then
accumulates exactly in IEEE fp32) and stays on the int64 elementwise path
otherwise. Both paths must be bit-identical, on CPU and GPU, for the
lossless sentinel and the quantized per-phase path alike.
"""

from __future__ import annotations

import pytest
import torch

from neurox.primitive.analog.adc_common import AdcOperationPoint
from neurox.primitive.macro.cim.ideal import IdealCimMacro, IdealCimMacroConfig, IdealCimMacroPolicy


def _make_xbar(
    *,
    row_num: int,
    active_row_num: int,
    col_num: int,
    x_range: tuple[int, int],
    w_digit_count: int = 1,
    w_digit_radix: int = 2,
    w_digit_range: tuple[int, int] = (-3, 3),
    adc_max_bits: int = 0,
) -> IdealCimMacro:
    config = IdealCimMacroConfig(
        col_num=col_num,
        row_num=row_num,
        active_row_num=active_row_num,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        x_range=x_range,
        w_digit_count=w_digit_count,
        w_digit_radix=w_digit_radix,
        w_digit_range=w_digit_range,
        adc_mode_num=1,
        adc_max_bits=adc_max_bits,
    )
    xbar = IdealCimMacro(
        config=config,
        policy=IdealCimMacroPolicy(),
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    xbar.eval()
    return xbar


def _random_operands(xbar: IdealCimMacro, *, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Random in-range digits (programmed) and activations on CPU."""
    generator = torch.Generator().manual_seed(seed)
    d_lo, d_hi = xbar.config.w_digit_range
    digits = torch.randint(
        d_lo,
        d_hi + 1,
        (xbar.config.col_num, xbar.config.w_digit_count, xbar.config.row_num),
        dtype=torch.int32,
        generator=generator,
    )
    x_lo, x_hi = xbar.config.x_range
    x = torch.randint(x_lo, x_hi + 1, (batch, xbar.config.row_num), dtype=torch.int32, generator=generator)
    xbar.program(digits)
    return digits, x


def _phase_dot_oracle(xbar: IdealCimMacro, x: torch.Tensor) -> torch.Tensor:
    """CPU int64 per-phase partial dots ``[..., active_phase_num, col_num]``."""
    digits = xbar.digits.to("cpu", torch.int64)
    digit_weights = xbar.digit_weights.to("cpu", torch.int64).view(*([1] * (digits.ndim - 2)), -1, 1)
    w = (digits * digit_weights).sum(dim=-2)
    prod = x.to("cpu", torch.int64).unsqueeze(-2) * w
    phase_dot = prod.unflatten(-1, (xbar.config.active_phase_num, xbar.active_row_num)).sum(dim=-1)
    return phase_dot.transpose(-1, -2)


_REPRESENTATIVE = [
    # (x_range, w_digit_count, w_digit_radix, w_digit_range)
    pytest.param((0, 1), 1, 2, (-3, 3), id="spike_x_signed_w"),
    pytest.param((-8, 7), 2, 4, (0, 3), id="signed_x_unsigned_digits"),
    pytest.param((0, 255), 2, 16, (-15, 15), id="wide_x_wide_digits"),
]


class TestFastPathLossless:
    """``adc_bits == 0``: fp32 fast path == int64 oracle, CPU and GPU."""

    @pytest.mark.parametrize(("x_range", "w_digit_count", "w_digit_radix", "w_digit_range"), _REPRESENTATIVE)
    def test_cpu_matches_int64_oracle(
        self,
        x_range: tuple[int, int],
        w_digit_count: int,
        w_digit_radix: int,
        w_digit_range: tuple[int, int],
    ) -> None:
        xbar = _make_xbar(
            row_num=64,
            active_row_num=16,
            col_num=8,
            x_range=x_range,
            w_digit_count=w_digit_count,
            w_digit_radix=w_digit_radix,
            w_digit_range=w_digit_range,
        )
        assert xbar._fp32_exact is True
        _, x = _random_operands(xbar, batch=5, seed=101)
        y = xbar.vec_mat_mul(x, adc_operation_point=AdcOperationPoint(adc_mode=0, adc_bits=0))
        assert y.dtype == torch.int64
        assert torch.equal(y, _phase_dot_oracle(xbar, x))

    @pytest.mark.parametrize(("x_range", "w_digit_count", "w_digit_radix", "w_digit_range"), _REPRESENTATIVE)
    def test_gpu_matches_cpu_oracle(
        self,
        device: torch.device,
        x_range: tuple[int, int],
        w_digit_count: int,
        w_digit_radix: int,
        w_digit_range: tuple[int, int],
    ) -> None:
        xbar = _make_xbar(
            row_num=64,
            active_row_num=16,
            col_num=8,
            x_range=x_range,
            w_digit_count=w_digit_count,
            w_digit_radix=w_digit_radix,
            w_digit_range=w_digit_range,
        )
        _, x = _random_operands(xbar, batch=5, seed=202)
        oracle = _phase_dot_oracle(xbar, x)
        xbar.to(device)
        y = xbar.vec_mat_mul(x.to(device), adc_operation_point=AdcOperationPoint(adc_mode=0, adc_bits=0))
        assert y.device.type == device.type
        assert torch.equal(y.cpu(), oracle)


class TestFastPathQuantized:
    """``adc_bits > 0``: per-phase codes byte-identical to the int64 path."""

    def _quantized_xbar(self) -> IdealCimMacro:
        return _make_xbar(
            row_num=64,
            active_row_num=16,
            col_num=8,
            x_range=(-8, 7),
            w_digit_count=2,
            w_digit_radix=4,
            w_digit_range=(-3, 3),
            adc_max_bits=6,
        )

    def test_codes_match_forced_fallback_cpu(self) -> None:
        xbar_fast = self._quantized_xbar()
        xbar_ref = self._quantized_xbar()
        xbar_ref._fp32_exact = False  # force the int64 elementwise path
        _, x = _random_operands(xbar_fast, batch=5, seed=303)
        _random_operands(xbar_ref, batch=5, seed=303)
        op = AdcOperationPoint(adc_mode=0, adc_bits=4)
        y_fast = xbar_fast.vec_mat_mul(x, adc_operation_point=op)
        y_ref = xbar_ref.vec_mat_mul(x, adc_operation_point=op)
        assert y_fast.dtype == y_ref.dtype == torch.int16
        assert torch.equal(y_fast, y_ref)

    def test_codes_gpu_match_cpu_oracle(self, device: torch.device) -> None:
        xbar = self._quantized_xbar()
        _, x = _random_operands(xbar, batch=5, seed=404)
        op = AdcOperationPoint(adc_mode=0, adc_bits=4)
        y_cpu = xbar.vec_mat_mul(x, adc_operation_point=op)
        xbar.to(device)
        y_dev = xbar.vec_mat_mul(x.to(device), adc_operation_point=op)
        assert torch.equal(y_dev.cpu(), y_cpu)

    def test_training_jitter_rng_stream_identical_across_paths(self) -> None:
        """Identical fp32 inputs consume the RNG identically on both paths."""
        xbar_fast = self._quantized_xbar()
        xbar_ref = self._quantized_xbar()
        xbar_ref._fp32_exact = False
        xbar_fast.train()
        xbar_ref.train()
        _, x = _random_operands(xbar_fast, batch=5, seed=505)
        _random_operands(xbar_ref, batch=5, seed=505)
        op = AdcOperationPoint(adc_mode=0, adc_bits=4)
        torch.manual_seed(7)
        y_fast = xbar_fast.vec_mat_mul(x, adc_operation_point=op)
        torch.manual_seed(7)
        y_ref = xbar_ref.vec_mat_mul(x, adc_operation_point=op)
        assert torch.equal(y_fast, y_ref)


class TestFallbackTrigger:
    """Bound at or above ``2^24`` keeps the int64 path (and stays exact)."""

    def test_flag_disabled_and_exact_beyond_fp32(self) -> None:
        # _max_phase_dot_abs = 3 * 2^23 * 1 >= 2^24 -> fallback.
        xbar = _make_xbar(
            row_num=3,
            active_row_num=3,
            col_num=2,
            x_range=(0, 1),
            w_digit_range=(0, 2**23),
        )
        assert xbar._fp32_exact is False
        # Dot = 2^24 + 1 is NOT fp32-representable: the fast path would
        # round it; the int64 path must carry it exactly.
        assert torch.tensor(2**24 + 1, dtype=torch.float32).item() == 2**24
        digits = torch.tensor([[[2**23, 2**23, 1]], [[0, 0, 0]]], dtype=torch.int32)
        xbar.program(digits)
        x = torch.ones(3, dtype=torch.int32)
        y = xbar.vec_mat_mul(x, adc_operation_point=AdcOperationPoint(adc_mode=0, adc_bits=0))
        assert y.shape == (1, 2)  # [P, col]
        assert y[0, 0].item() == 2**24 + 1
        assert y[0, 1].item() == 0
