"""IdealCimMacro fp32-exact dot fast path.

The plane dot computation switches to fp32 einsum when
``_max_plane_dot_abs < 2^24`` (every per-cell product and partial sum then
accumulates exactly in IEEE fp32) and stays on the int64 elementwise path
otherwise. Both paths must be bit-identical, on CPU and GPU, for the
lossless sentinel and the quantized per-plane path alike. Planes arrive
pre-masked from the caller (at most ``max_active_rows`` live rows each);
the macro output keeps the leading order with primitive trailing
``[col_num]``.
"""

from __future__ import annotations

import pytest
import torch

from neurox.primitive.macro.cim.ideal import IdealCimMacro, IdealCimMacroConfig, IdealCimMacroPolicy


def _make_xbar(
    *,
    row_num: int,
    active_row_num: int,
    col_num: int,
    x_value_range: tuple[int, int],
    w_digit_count: int = 1,
    w_digit_radix: int = 2,
    w_digit_value_range: tuple[int, int] = (-3, 3),
    adc_max_bits: int = 0,
) -> IdealCimMacro:
    config = IdealCimMacroConfig(
        col_num=col_num,
        row_num=row_num,
        active_row_num=active_row_num,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        x_value_range=x_value_range,
        w_digit_count=w_digit_count,
        w_digit_radix=w_digit_radix,
        w_digit_value_range=w_digit_value_range,
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


def _random_operands(
    xbar: IdealCimMacro,
    *,
    batch: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random in-range digits (programmed) and activations on CPU."""
    generator = torch.Generator().manual_seed(seed)
    d_lo, d_hi = xbar.config.w_digit_value_range
    digits = torch.randint(
        d_lo,
        d_hi + 1,
        (xbar.config.col_num, xbar.config.w_digit_count, xbar.config.row_num),
        dtype=torch.int32,
        generator=generator,
    )
    x_lo, x_hi = xbar.config.x_value_range
    x = torch.randint(x_lo, x_hi + 1, (batch, xbar.config.row_num), dtype=torch.int32, generator=generator)
    xbar.program(digits.to(device))
    return digits, x


def _masked_planes(x: torch.Tensor, *, row_num: int, max_active_rows: int) -> torch.Tensor:
    """Pre-masked WL planes via the engine mask formula.

    Shape: [..., row_num] -> [..., P, row_num]; plane ``p`` keeps exactly
    rows ``[p*max_active_rows, (p+1)*max_active_rows)``, zeros elsewhere.
    """
    p_num = row_num // max_active_rows
    mask = torch.arange(row_num) // max_active_rows == torch.arange(p_num).unsqueeze(-1)
    # Shape: [..., row_num] -> [..., P, row_num]
    return torch.where(mask, x.unsqueeze(-2), x.new_zeros(()))


def _plane_dot_oracle(xbar: IdealCimMacro, planes: torch.Tensor) -> torch.Tensor:
    """CPU int64 lossless plane dots. Shape: [..., row_num] -> [..., col_num]."""
    digits = xbar._digits.to("cpu", torch.int64)
    digit_weights = xbar._digit_weights.to("cpu", torch.int64).view(*([1] * (digits.ndim - 2)), -1, 1)
    # Shape: [col_num, w_digit_count, row_num] -> [col_num, row_num]
    w = (digits * digit_weights).sum(dim=-2)
    # Shape: [..., row_num] -> [..., 1, row_num]; row contraction -> [..., col_num]
    return (planes.to("cpu", torch.int64).unsqueeze(-2) * w).sum(dim=-1)


_REPRESENTATIVE = [
    # (x_value_range, w_digit_count, w_digit_radix, w_digit_value_range)
    pytest.param((0, 1), 1, 2, (-3, 3), id="spike_x_signed_w"),
    pytest.param((-8, 7), 2, 4, (0, 3), id="signed_x_unsigned_digits"),
    pytest.param((0, 255), 2, 16, (-15, 15), id="wide_x_wide_digits"),
]


class TestFastPathLossless:
    """``adc_bits == 0``: fp32 fast path == int64 oracle, CPU and GPU."""

    @pytest.mark.parametrize(
        ("x_value_range", "w_digit_count", "w_digit_radix", "w_digit_value_range"), _REPRESENTATIVE
    )
    def test_cpu_matches_int64_oracle(
        self,
        x_value_range: tuple[int, int],
        w_digit_count: int,
        w_digit_radix: int,
        w_digit_value_range: tuple[int, int],
    ) -> None:
        xbar = _make_xbar(
            row_num=64,
            active_row_num=16,
            col_num=8,
            x_value_range=x_value_range,
            w_digit_count=w_digit_count,
            w_digit_radix=w_digit_radix,
            w_digit_value_range=w_digit_value_range,
        )
        assert xbar._fp32_exact is True
        _, x = _random_operands(xbar, batch=5, seed=101, device=torch.device("cpu"))
        planes = _masked_planes(x, row_num=64, max_active_rows=16)
        y = xbar.vec_mat_mul(planes, adc_mode=0, adc_bits=0)
        assert y.dtype == torch.int64
        assert y.shape == (5, 4, 8)  # leading [batch, P] preserved, trailing [col_num]
        assert torch.equal(y, _plane_dot_oracle(xbar, planes))

    @pytest.mark.parametrize(
        ("x_value_range", "w_digit_count", "w_digit_radix", "w_digit_value_range"), _REPRESENTATIVE
    )
    def test_gpu_matches_cpu_oracle(
        self,
        device: torch.device,
        x_value_range: tuple[int, int],
        w_digit_count: int,
        w_digit_radix: int,
        w_digit_value_range: tuple[int, int],
    ) -> None:
        xbar = _make_xbar(
            row_num=64,
            active_row_num=16,
            col_num=8,
            x_value_range=x_value_range,
            w_digit_count=w_digit_count,
            w_digit_radix=w_digit_radix,
            w_digit_value_range=w_digit_value_range,
        )
        xbar.to(device)
        _, x = _random_operands(xbar, batch=5, seed=202, device=device)
        planes = _masked_planes(x, row_num=64, max_active_rows=16)
        oracle = _plane_dot_oracle(xbar, planes)
        y = xbar.vec_mat_mul(planes.to(device), adc_mode=0, adc_bits=0)
        assert y.device.type == device.type
        assert torch.equal(y.cpu(), oracle)


class TestFastPathQuantized:
    """``adc_bits > 0``: per-plane codes byte-identical to the int64 path."""

    def _quantized_xbar(self) -> IdealCimMacro:
        return _make_xbar(
            row_num=64,
            active_row_num=16,
            col_num=8,
            x_value_range=(-8, 7),
            w_digit_count=2,
            w_digit_radix=4,
            w_digit_value_range=(-3, 3),
            adc_max_bits=6,
        )

    def test_codes_match_forced_fallback_cpu(self) -> None:
        xbar_fast = self._quantized_xbar()
        xbar_ref = self._quantized_xbar()
        xbar_ref._fp32_exact = False  # force the int64 elementwise path
        _, x = _random_operands(xbar_fast, batch=5, seed=303, device=torch.device("cpu"))
        _random_operands(xbar_ref, batch=5, seed=303, device=torch.device("cpu"))
        planes = _masked_planes(x, row_num=64, max_active_rows=16)
        adc_mode, adc_bits = 0, 4
        y_fast = xbar_fast.vec_mat_mul(planes, adc_mode=adc_mode, adc_bits=adc_bits)
        y_ref = xbar_ref.vec_mat_mul(planes, adc_mode=adc_mode, adc_bits=adc_bits)
        assert y_fast.dtype == y_ref.dtype == torch.int16
        assert torch.equal(y_fast, y_ref)

    def test_codes_gpu_match_cpu_oracle(self, device: torch.device) -> None:
        xbar = self._quantized_xbar()
        digits, x = _random_operands(xbar, batch=5, seed=404, device=torch.device("cpu"))
        planes = _masked_planes(x, row_num=64, max_active_rows=16)
        adc_mode, adc_bits = 0, 4
        y_cpu = xbar.vec_mat_mul(planes, adc_mode=adc_mode, adc_bits=adc_bits)
        xbar.to(device)
        xbar.program(digits.to(device))
        y_dev = xbar.vec_mat_mul(planes.to(device), adc_mode=adc_mode, adc_bits=adc_bits)
        assert torch.equal(y_dev.cpu(), y_cpu)

    def test_training_jitter_rng_stream_identical_across_paths(self) -> None:
        """Identical fp32 inputs consume the RNG identically on both paths."""
        xbar_fast = self._quantized_xbar()
        xbar_ref = self._quantized_xbar()
        xbar_ref._fp32_exact = False
        xbar_fast.train()
        xbar_ref.train()
        _, x = _random_operands(xbar_fast, batch=5, seed=505, device=torch.device("cpu"))
        _random_operands(xbar_ref, batch=5, seed=505, device=torch.device("cpu"))
        planes = _masked_planes(x, row_num=64, max_active_rows=16)
        adc_mode, adc_bits = 0, 4
        torch.manual_seed(7)
        y_fast = xbar_fast.vec_mat_mul(planes, adc_mode=adc_mode, adc_bits=adc_bits)
        torch.manual_seed(7)
        y_ref = xbar_ref.vec_mat_mul(planes, adc_mode=adc_mode, adc_bits=adc_bits)
        assert torch.equal(y_fast, y_ref)


class TestFallbackTrigger:
    """Bound at or above ``2^24`` keeps the int64 path (and stays exact)."""

    def test_flag_disabled_and_exact_beyond_fp32(self) -> None:
        # _max_plane_dot_abs = 3 * 2^23 * 1 >= 2^24 -> fallback.
        xbar = _make_xbar(
            row_num=3,
            active_row_num=3,
            col_num=2,
            x_value_range=(0, 1),
            w_digit_value_range=(0, 2**23),
        )
        assert xbar._fp32_exact is False
        # Dot = 2^24 + 1 is NOT fp32-representable: the fast path would
        # round it; the int64 path must carry it exactly.
        assert torch.tensor(2**24 + 1, dtype=torch.float32).item() == 2**24
        digits = torch.tensor([[[2**23, 2**23, 1]], [[0, 0, 0]]], dtype=torch.int32)
        xbar.program(digits)
        # Full-row plane is conformant here: active_row_num == row_num.
        x = torch.ones(3, dtype=torch.int32)
        y = xbar.vec_mat_mul(x, adc_mode=0, adc_bits=0)
        assert y.shape == (2,)  # trailing [col_num], no phase axis
        assert y[0].item() == 2**24 + 1
        assert y[1].item() == 0
