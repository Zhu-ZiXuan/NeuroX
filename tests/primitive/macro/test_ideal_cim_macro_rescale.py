"""IdealCimMacro quantize/rescale correctness across digit-range shapes.

The recovery rescale must clip exactly at the signed N-bit endpoints
under any legal ``w_digit_value_range`` — including signed-digit and offset
encodings whose extremes are not the canonical ``[0, radix^count - 1]``.
The rescale denominator is the per-conversion bound ``_max_plane_dot_abs``:
each conversion digitizes one WL plane of at most ``active_row_num`` live
rows, so the bound scales with ``active_row_num``, not ``row_num``. The
macro treats every leading axis as anonymous batch — the caller (engine)
owns any phase axis and its accumulation.
"""

from __future__ import annotations

import torch

from neurox.primitive.macro.cim.ideal import IdealCimMacro, IdealCimMacroConfig, IdealCimMacroPolicy


def _make_xbar(
    *,
    w_digit_value_range: tuple[int, int],
    w_digit_radix: int,
    w_digit_count: int,
    x_value_range: tuple[int, int],
    row_num: int,
    col_num: int,
    adc_bits: int,
    active_row_num: int | None = None,
) -> IdealCimMacro:
    config = IdealCimMacroConfig(
        col_num=col_num,
        row_num=row_num,
        active_row_num=row_num if active_row_num is None else active_row_num,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        x_value_range=x_value_range,
        w_digit_count=w_digit_count,
        w_digit_radix=w_digit_radix,
        w_digit_value_range=w_digit_value_range,
        adc_mode_num=1,
        adc_max_bits=adc_bits,
    )
    xbar = IdealCimMacro(
        config=config,
        policy=IdealCimMacroPolicy(),
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    xbar.eval()
    xbar.fabricate()
    return xbar


def _expected_max_plane_dot(
    *,
    w_digit_value_range: tuple[int, int],
    w_digit_radix: int,
    w_digit_count: int,
    x_value_range: tuple[int, int],
    active_row_num: int,
) -> int:
    d_lo, d_hi = w_digit_value_range
    max_digit_abs = max(abs(d_lo), abs(d_hi))
    weight_sum = sum((w_digit_radix**k for k in range(w_digit_count)), start=0)
    max_w_abs = max_digit_abs * weight_sum
    x_lo, x_hi = x_value_range
    return int(active_row_num * max_w_abs * max(abs(x_lo), abs(x_hi)))


class TestRescaleScope:
    """``_max_plane_dot_abs`` must use the actual w_digit_value_range, not radix^count."""

    def test_signed_symmetric_range(self) -> None:
        """w_digit_value_range = [-d_max, d_max] — symmetric signed digit."""
        xbar = _make_xbar(
            w_digit_value_range=(-3, 3),
            w_digit_radix=4,
            w_digit_count=2,
            x_value_range=(0, 1),
            row_num=8,
            col_num=4,
            adc_bits=8,
        )
        expected = _expected_max_plane_dot(
            w_digit_value_range=(-3, 3), w_digit_radix=4, w_digit_count=2, x_value_range=(0, 1), active_row_num=8
        )
        assert xbar._max_plane_dot_abs == expected
        assert xbar._rescale_by_bits[8] == expected / ((1 << 7) - 1)

    def test_offset_nonneg_range(self) -> None:
        """w_digit_value_range = [0, d_max] — canonical non-negative."""
        xbar = _make_xbar(
            w_digit_value_range=(0, 3),
            w_digit_radix=4,
            w_digit_count=2,
            x_value_range=(0, 1),
            row_num=8,
            col_num=4,
            adc_bits=8,
        )
        expected = _expected_max_plane_dot(
            w_digit_value_range=(0, 3), w_digit_radix=4, w_digit_count=2, x_value_range=(0, 1), active_row_num=8
        )
        assert xbar._max_plane_dot_abs == expected

    def test_asymmetric_signed_range(self) -> None:
        """w_digit_value_range = [-1, 2] — asymmetric: max_abs picks the larger side."""
        xbar = _make_xbar(
            w_digit_value_range=(-1, 2),
            w_digit_radix=3,
            w_digit_count=2,
            x_value_range=(0, 1),
            row_num=8,
            col_num=4,
            adc_bits=8,
        )
        # ``radix^count - 1`` overclips for asymmetric signed digit ranges
        # where the larger |bound| < radix-1; the correct bound is
        # ``max(|d_lo|, |d_hi|) · sum(radix^k)``.
        expected = _expected_max_plane_dot(
            w_digit_value_range=(-1, 2), w_digit_radix=3, w_digit_count=2, x_value_range=(0, 1), active_row_num=8
        )
        assert xbar._max_plane_dot_abs == expected

    def test_naive_formula_overclips_signed(self) -> None:
        """w_digit_value_range = [-1, 1] with radix=4, count=2: the naive
        ``radix^count - 1 = 15`` bound overclips; correct is
        ``1·(1+4) = 5``."""
        xbar = _make_xbar(
            w_digit_value_range=(-1, 1),
            w_digit_radix=4,
            w_digit_count=2,
            x_value_range=(0, 1),
            row_num=8,
            col_num=4,
            adc_bits=8,
        )
        naive_overclip = 4**2 - 1
        correct = 1 * (1 + 4)
        assert xbar._max_plane_dot_abs == 8 * correct
        # And the rescale derived from it must use the correct bound.
        assert xbar._rescale_by_bits[8] == 8 * correct / ((1 << 7) - 1)
        assert xbar._rescale_by_bits[8] != 8 * naive_overclip / ((1 << 7) - 1)

    def test_partial_activation_uses_active_row_num(self) -> None:
        """``active_row_num < row_num``: the per-conversion bound counts the
        rows of one active plane, not the full array height."""
        xbar = _make_xbar(
            w_digit_value_range=(-3, 3),
            w_digit_radix=4,
            w_digit_count=2,
            x_value_range=(0, 1),
            row_num=8,
            col_num=4,
            adc_bits=8,
            active_row_num=2,
        )
        expected = _expected_max_plane_dot(
            w_digit_value_range=(-3, 3), w_digit_radix=4, w_digit_count=2, x_value_range=(0, 1), active_row_num=2
        )
        assert xbar._max_plane_dot_abs == expected
        assert xbar._rescale_by_bits[8] == expected / ((1 << 7) - 1)


class TestPlaneOutput:
    """``vec_mat_mul`` maps trailing ``[row_num]`` to ``[col_num]`` and keeps
    every leading axis anonymous — the caller owns any phase axis."""

    @staticmethod
    def _programmed_xbar(*, active_row_num: int | None, adc_bits: int) -> IdealCimMacro:
        torch.manual_seed(42)
        xbar = _make_xbar(
            w_digit_value_range=(-3, 3),
            w_digit_radix=2,
            w_digit_count=1,
            x_value_range=(0, 1),
            row_num=8,
            col_num=4,
            adc_bits=adc_bits,
            active_row_num=active_row_num,
        )
        w = torch.randint(-3, 4, xbar.w_layout_shape, dtype=torch.int32)
        xbar.program(w)
        return xbar

    def test_leading_axes_preserved(self) -> None:
        """A caller-supplied phase axis is opaque batch: trailing row_num -> col_num."""
        xbar = self._programmed_xbar(active_row_num=2, adc_bits=8)
        x = torch.randint(0, 2, (3, 5, 8), dtype=torch.int32)
        y = xbar.vec_mat_mul(x, adc_mode=0, adc_bits=8)
        assert y.shape == (3, 5, 4)  # [..., col_num]

    def test_single_leading_axis(self) -> None:
        """One leading batch axis is preserved; no phase axis is manufactured."""
        xbar = self._programmed_xbar(active_row_num=None, adc_bits=8)
        x = torch.randint(0, 2, (3, 8), dtype=torch.int32)
        y = xbar.vec_mat_mul(x, adc_mode=0, adc_bits=8)
        assert y.shape == (3, 4)

    def test_lossless_matches_whole_dot(self) -> None:
        """``adc_bits == 0``: the plane dot equals the lossless integer dot."""
        xbar = self._programmed_xbar(active_row_num=2, adc_bits=0)
        x = torch.randint(0, 2, (5, 8), dtype=torch.int32)
        y = xbar.vec_mat_mul(x, adc_mode=0, adc_bits=0)
        assert y.shape == (5, 4)
        w_logical = xbar._digits.to(torch.int64).squeeze(-2)  # [col_num, row_num], D=1
        expected = x.to(torch.int64) @ w_logical.transpose(-1, -2)
        assert torch.equal(y, expected)


class TestProgramOwnership:
    """``program`` must take ownership of the weight tensor (clone + device)."""

    def test_program_does_not_alias_caller(self) -> None:
        xbar = _make_xbar(
            w_digit_value_range=(-1, 1),
            w_digit_radix=2,
            w_digit_count=1,
            x_value_range=(0, 1),
            row_num=4,
            col_num=2,
            adc_bits=8,
        )
        w = torch.zeros(xbar.w_layout_shape, dtype=torch.int32)
        xbar.program(w)
        # Caller-side mutation must not affect the stored state.
        w.fill_(1)
        assert xbar._digits.abs().sum().item() == 0
