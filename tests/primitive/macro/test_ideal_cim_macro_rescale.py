"""IdealCimMacro quantize/rescale correctness across digit-range shapes.

The recovery rescale must clip exactly at the signed N-bit endpoints
under any legal ``w_digit_range`` — including signed-digit and offset
encodings whose extremes are not the canonical ``[0, radix^count - 1]``.
"""

from __future__ import annotations

import torch

from neurox.primitive.macro.cim.ideal import IdealCimMacro, IdealCimMacroConfig, IdealCimMacroPolicy

_CPU = torch.device("cpu")


def _make_xbar(
    *,
    w_digit_range: tuple[int, int],
    w_digit_radix: int,
    w_digit_count: int,
    x_range: tuple[int, int],
    row_num: int,
    col_num: int,
    adc_bits: int,
) -> IdealCimMacro:
    config = IdealCimMacroConfig(
        col_num=col_num,
        row_num=row_num,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        x_range=x_range,
        w_digit_count=w_digit_count,
        w_digit_radix=w_digit_radix,
        w_digit_range=w_digit_range,
        adc_mode_num=1,
        adc_max_bits=adc_bits,
    )
    xbar = IdealCimMacro(
        config=config,
        policy=IdealCimMacroPolicy(),
        name="x",
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    xbar.eval()
    xbar.fabricate()
    return xbar


def _expected_max_dot(
    *,
    w_digit_range: tuple[int, int],
    w_digit_radix: int,
    w_digit_count: int,
    x_range: tuple[int, int],
    row_num: int,
) -> int:
    d_lo, d_hi = w_digit_range
    max_digit_abs = max(abs(d_lo), abs(d_hi))
    weight_sum = sum((w_digit_radix**k for k in range(w_digit_count)), start=0)
    max_w_abs = max_digit_abs * weight_sum
    x_lo, x_hi = x_range
    return int(row_num * max_w_abs * max(abs(x_lo), abs(x_hi)))


class TestRescaleScope:
    """``_max_dot_abs`` must use the actual w_digit_range, not radix^count."""

    def test_signed_symmetric_range(self) -> None:
        """w_digit_range = [-d_max, d_max] — symmetric signed digit."""
        xbar = _make_xbar(
            w_digit_range=(-3, 3),
            w_digit_radix=4,
            w_digit_count=2,
            x_range=(0, 1),
            row_num=8,
            col_num=4,
            adc_bits=8,
        )
        expected = _expected_max_dot(w_digit_range=(-3, 3), w_digit_radix=4, w_digit_count=2, x_range=(0, 1), row_num=8)
        assert xbar._max_dot_abs == expected
        assert xbar._rescale_by_bits[8] == expected / ((1 << 7) - 1)

    def test_offset_nonneg_range(self) -> None:
        """w_digit_range = [0, d_max] — canonical non-negative."""
        xbar = _make_xbar(
            w_digit_range=(0, 3),
            w_digit_radix=4,
            w_digit_count=2,
            x_range=(0, 1),
            row_num=8,
            col_num=4,
            adc_bits=8,
        )
        expected = _expected_max_dot(w_digit_range=(0, 3), w_digit_radix=4, w_digit_count=2, x_range=(0, 1), row_num=8)
        assert xbar._max_dot_abs == expected

    def test_asymmetric_signed_range(self) -> None:
        """w_digit_range = [-1, 2] — asymmetric: max_abs picks the larger side."""
        xbar = _make_xbar(
            w_digit_range=(-1, 2),
            w_digit_radix=3,
            w_digit_count=2,
            x_range=(0, 1),
            row_num=8,
            col_num=4,
            adc_bits=8,
        )
        # ``radix^count - 1`` overclips for asymmetric signed digit ranges
        # where the larger |bound| < radix-1; the correct bound is
        # ``max(|d_lo|, |d_hi|) · sum(radix^k)``. See
        # ``test_naive_formula_overclips_signed`` below.
        expected = _expected_max_dot(w_digit_range=(-1, 2), w_digit_radix=3, w_digit_count=2, x_range=(0, 1), row_num=8)
        assert xbar._max_dot_abs == expected

    def test_naive_formula_overclips_signed(self) -> None:
        """w_digit_range = [-1, 1] with radix=4, count=2: the naive
        ``radix^count - 1 = 15`` bound overclips; correct is
        ``1·(1+4) = 5``."""
        xbar = _make_xbar(
            w_digit_range=(-1, 1),
            w_digit_radix=4,
            w_digit_count=2,
            x_range=(0, 1),
            row_num=8,
            col_num=4,
            adc_bits=8,
        )
        naive_overclip = 4**2 - 1
        correct = 1 * (1 + 4)
        assert xbar._max_dot_abs == 8 * correct
        # And the rescale derived from it must use the correct bound.
        assert xbar._rescale_by_bits[8] == 8 * correct / ((1 << 7) - 1)
        assert xbar._rescale_by_bits[8] != 8 * naive_overclip / ((1 << 7) - 1)


class TestProgramOwnership:
    """``program`` must take ownership of the weight tensor (clone + device)."""

    def test_program_does_not_alias_caller(self) -> None:
        xbar = _make_xbar(
            w_digit_range=(-1, 1),
            w_digit_radix=2,
            w_digit_count=1,
            x_range=(0, 1),
            row_num=4,
            col_num=2,
            adc_bits=8,
        )
        w = torch.zeros(xbar._w_layout_shape, dtype=torch.int32)
        xbar.program(w)
        # Caller-side mutation must not affect the stored state.
        w.fill_(1)
        assert xbar.digits.abs().sum().item() == 0
