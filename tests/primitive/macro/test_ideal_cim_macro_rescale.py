"""IdealCimMacro window conversion law and its rescale factor.

``quantization_mode`` selects a canonical inclusive window ``[lower, upper]``
in MAC units holding ``W = upper - lower + 1`` targets. With ``b = adc_bits``
the step is ``lsb = W / 2^b``, the raw reading is
``code_u = clamp(floor((dot - lower) * 2^b / W), 0, 2^b - 1)`` and the
returned signed code subtracts the computed zero code (``0`` unsigned,
``2^(b-1)`` mid-zero). ``rescale_factor`` states that code in ideal-macro
codes, so it is the pure bit-width chain ``2^(B - b)`` here; the MAC-unit
``lsb`` belongs to the algorithm side. The macro treats every leading axis as
anonymous batch.
"""

from __future__ import annotations

import math

import pytest
import torch

from neurox.primitive.macro.cim.ideal import IdealCimMacro, IdealCimMacroConfig, IdealCimMacroPolicy


def _make_macro(
    *,
    quantization_input_ranges: tuple[tuple[int, int], ...],
    adc_max_bits: int,
    w_value_range: tuple[int, int] = (-3, 3),
    x_value_range: tuple[int, int] = (0, 1),
    input_num: int = 8,
    output_num: int = 4,
    max_active_num: int | None = None,
) -> IdealCimMacro:
    config = IdealCimMacroConfig(
        max_active_num=input_num if max_active_num is None else max_active_num,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        x_value_range=x_value_range,
        w_value_range=w_value_range,
        quantization_input_ranges=quantization_input_ranges,
        adc_max_bits=adc_max_bits,
    )
    xbar = IdealCimMacro(
        config=config,
        policy=IdealCimMacroPolicy(),
        input_num=input_num,
        output_num=output_num,
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    xbar.eval()
    xbar.fabricate()
    return xbar


def _dot_macro(*, quantization_input_ranges: tuple[tuple[int, int], ...], adc_max_bits: int) -> IdealCimMacro:
    """A 1x1 identity tile: the input value *is* the plane dot."""
    macro = _make_macro(
        quantization_input_ranges=quantization_input_ranges,
        adc_max_bits=adc_max_bits,
        w_value_range=(-1, 1),
        x_value_range=(-512, 512),
        input_num=1,
        output_num=1,
    )
    macro.program(torch.ones((1, 1), dtype=torch.int32))
    return macro


def _codes(macro: IdealCimMacro, dots: list[int], *, quantization_mode: int, adc_bits: int | None) -> list[int]:
    x = torch.tensor(dots, dtype=torch.int64).unsqueeze(-1)
    y = macro.vec_mat_mul(x, quantization_mode=quantization_mode, adc_bits=adc_bits)
    assert y.dtype == torch.int64
    return y.squeeze(-1).tolist()


_WINDOWS: tuple[tuple[int, int], ...] = ((-32, 31), (-128, 127), (0, 63), (-6, 5))


class TestRescaleFactor:
    """The ideal macro is the currency anchor: ``r_b = 2^(B - b)``."""

    @pytest.mark.parametrize("window", _WINDOWS)
    def test_max_bits_factor_is_the_identity(self, window: tuple[int, int]) -> None:
        macro = _make_macro(quantization_input_ranges=(window,), adc_max_bits=6)
        assert macro.rescale_factor(quantization_mode=0, adc_bits=6) == 1.0

    @pytest.mark.parametrize("window", _WINDOWS)
    @pytest.mark.parametrize("adc_bits", [1, 3, 6])
    def test_factor_is_the_bit_width_gap(self, window: tuple[int, int], adc_bits: int) -> None:
        macro = _make_macro(quantization_input_ranges=(window,), adc_max_bits=6)
        assert macro.rescale_factor(quantization_mode=0, adc_bits=adc_bits) == 2 ** (6 - adc_bits)

    @pytest.mark.parametrize("window", _WINDOWS)
    def test_dropping_one_bit_doubles_the_factor(self, window: tuple[int, int]) -> None:
        macro = _make_macro(quantization_input_ranges=(window,), adc_max_bits=6)
        for adc_bits in range(2, 7):
            coarse = macro.rescale_factor(quantization_mode=0, adc_bits=adc_bits - 1)
            fine = macro.rescale_factor(quantization_mode=0, adc_bits=adc_bits)
            assert coarse == 2.0 * fine

    def test_lossless_sentinel_is_identity(self) -> None:
        macro = _make_macro(quantization_input_ranges=((-32, 31),), adc_max_bits=6)
        assert macro.rescale_factor(quantization_mode=0, adc_bits=None) == 1.0

    def test_factor_is_window_independent(self) -> None:
        """Window width is dequantization currency, not rescale currency."""
        macro = _make_macro(quantization_input_ranges=((-32, 31), (-128, 127)), adc_max_bits=6)
        assert len(macro.quantization_input_ranges) == 2
        assert macro.rescale_factor(quantization_mode=0, adc_bits=5) == macro.rescale_factor(
            quantization_mode=1, adc_bits=5
        )


class TestOperatingPointValidation:
    """Out-of-range operating points are rejected on both entry points."""

    def _macro(self) -> IdealCimMacro:
        return _dot_macro(quantization_input_ranges=((-32, 31), (-128, 127)), adc_max_bits=6)

    @pytest.mark.parametrize("quantization_mode", [-1, 2])
    def test_rescale_rejects_unknown_mode(self, quantization_mode: int) -> None:
        with pytest.raises(ValueError, match=r"quantization_mode"):
            self._macro().rescale_factor(quantization_mode=quantization_mode, adc_bits=6)

    @pytest.mark.parametrize("adc_bits", [-1, 0, 7])
    def test_rescale_rejects_unsupported_bits(self, adc_bits: int) -> None:
        with pytest.raises(ValueError, match=r"adc_bits"):
            self._macro().rescale_factor(quantization_mode=0, adc_bits=adc_bits)

    @pytest.mark.parametrize("quantization_mode", [-1, 2])
    def test_conversion_rejects_unknown_mode(self, quantization_mode: int) -> None:
        with pytest.raises(ValueError, match=r"quantization_mode"):
            _codes(self._macro(), [0], quantization_mode=quantization_mode, adc_bits=6)

    def test_lossless_sentinel_still_validates_mode(self) -> None:
        """``adc_bits is None`` skips conversion but not the mode contract."""
        with pytest.raises(ValueError, match=r"quantization_mode"):
            _codes(self._macro(), [0], quantization_mode=2, adc_bits=None)

    @pytest.mark.parametrize("adc_bits", [-1, 0, 7])
    def test_conversion_rejects_unsupported_bits(self, adc_bits: int) -> None:
        """``0`` is an ordinary out-of-range width, not a sentinel."""
        with pytest.raises(ValueError, match=r"adc_bits"):
            _codes(self._macro(), [0], quantization_mode=0, adc_bits=adc_bits)


class TestWindowConversionLaw:
    """Signed codes follow ``clamp(floor((dot - lower) * 2^b / W), 0, 2^b - 1) - z``."""

    @pytest.mark.parametrize("window", _WINDOWS)
    @pytest.mark.parametrize("adc_bits", [1, 2, 3, 6])
    def test_zero_dot_converts_to_zero_code(self, window: tuple[int, int], adc_bits: int) -> None:
        """The canonical-shape rule exists exactly for this."""
        macro = _dot_macro(quantization_input_ranges=(window,), adc_max_bits=6)
        assert _codes(macro, [0], quantization_mode=0, adc_bits=adc_bits) == [0]

    def test_negative_dots_floor_toward_negative_infinity(self) -> None:
        """Never C-style truncation: -3 at lsb 2 is -2, not -1."""
        macro = _dot_macro(quantization_input_ranges=((-32, 31),), adc_max_bits=6)
        # W = 64 targets at 5 bits: lsb = 2.
        assert _codes(macro, [-1, -2, -3, -4, -5], quantization_mode=0, adc_bits=5) == [-1, -1, -2, -2, -3]
        assert _codes(macro, [1, 2, 3, 4, 5], quantization_mode=0, adc_bits=5) == [0, 1, 1, 2, 2]

    def test_clips_at_both_window_rails(self) -> None:
        macro = _dot_macro(quantization_input_ranges=((-32, 31),), adc_max_bits=6)
        # lsb = 2, zero code z = 16: codes live in [-16, 15].
        assert _codes(macro, [-32, -33, -100], quantization_mode=0, adc_bits=5) == [-16, -16, -16]
        # ``upper`` is inclusive: the top target reads as the top code.
        assert _codes(macro, [30, 31, 32, 100], quantization_mode=0, adc_bits=5) == [15, 15, 15, 15]
        # The last in-window bucket starts one lsb below the top target.
        assert _codes(macro, [29], quantization_mode=0, adc_bits=5) == [14]

    def test_output_range_is_mid_zero_at_every_width(self) -> None:
        macro = _dot_macro(quantization_input_ranges=((-32, 31),), adc_max_bits=6)
        for adc_bits in range(1, 7):
            codes = _codes(macro, list(range(-100, 101)), quantization_mode=0, adc_bits=adc_bits)
            assert min(codes) == -(1 << (adc_bits - 1))
            assert max(codes) == (1 << (adc_bits - 1)) - 1

    def test_unsigned_window_codes_are_nonnegative(self) -> None:
        """A window with ``lower == 0`` has zero code 0: no code is negative."""
        macro = _dot_macro(quantization_input_ranges=((0, 63),), adc_max_bits=6)
        # lsb = 4 at 4 bits; dots below the window clip to code 0.
        assert _codes(macro, [-10, 0, 3, 4, 63, 64, 200], quantization_mode=0, adc_bits=4) == [0, 0, 0, 1, 15, 15, 15]

    def test_mode_selects_the_conversion_window(self) -> None:
        """The same dot reads differently through a wider window."""
        macro = _dot_macro(quantization_input_ranges=((-32, 31), (-128, 127)), adc_max_bits=6)
        dots = [-40, -8, 0, 8, 40]
        assert _codes(macro, dots, quantization_mode=0, adc_bits=5) == [-16, -4, 0, 4, 15]
        assert _codes(macro, dots, quantization_mode=1, adc_bits=5) == [-5, -1, 0, 1, 5]

    def test_lossy_window_is_a_legal_operating_point(self) -> None:
        """``W > 2^b`` is allowed: the step is simply wider than one MAC unit."""
        macro = _dot_macro(quantization_input_ranges=((0, 223),), adc_max_bits=4)
        # W = 224 targets over 16 codes: lsb = 14.
        assert _codes(macro, [0, 13, 14, 27, 28, 223, 300], quantization_mode=0, adc_bits=4) == [
            0,
            0,
            1,
            1,
            2,
            15,
            15,
        ]

    def test_code_times_lsb_recovers_the_dot_within_one_step(self) -> None:
        """Algorithm-side dequantization: ``lsb = W / 2^b`` MAC units per code."""
        window, adc_bits = (-128, 127), 5
        macro = _dot_macro(quantization_input_ranges=(window,), adc_max_bits=6)
        lower, upper = window
        lsb = (upper - lower + 1) / (1 << adc_bits)
        dots = list(range(-100, 101, 7))
        for dot, code in zip(dots, _codes(macro, dots, quantization_mode=0, adc_bits=adc_bits), strict=True):
            assert 0.0 <= dot - code * lsb < lsb


class TestBitWidthNesting:
    """Lowering bits coarsens the one shared ladder: codes nest by a right shift."""

    @pytest.mark.parametrize("window", _WINDOWS)
    def test_unsigned_code_at_b_is_the_max_bits_code_shifted(self, window: tuple[int, int]) -> None:
        max_bits = 6
        macro = _dot_macro(quantization_input_ranges=(window,), adc_max_bits=max_bits)
        lower, _ = window
        dots = list(range(-200, 201, 3))
        full = _codes(macro, dots, quantization_mode=0, adc_bits=max_bits)
        full_zero = 0 if lower == 0 else 1 << (max_bits - 1)
        for adc_bits in range(1, max_bits + 1):
            lowered = _codes(macro, dots, quantization_mode=0, adc_bits=adc_bits)
            zero = 0 if lower == 0 else 1 << (adc_bits - 1)
            for code_full, code_low in zip(full, lowered, strict=True):
                assert code_low + zero == (code_full + full_zero) >> (max_bits - adc_bits)


class TestTrainingJitter:
    """Stochastic rounding is zero-preserving on the canonical shapes."""

    @pytest.mark.parametrize("window", _WINDOWS)
    def test_on_grid_dots_keep_their_eval_code(self, window: tuple[int, int]) -> None:
        adc_bits = 3
        macro = _dot_macro(quantization_input_ranges=(window,), adc_max_bits=6)
        lower, upper = window
        width = upper - lower + 1
        # A dot sits on a code boundary iff (dot - lower) * 2^b is a multiple
        # of W; jitter below W then cannot carry the floor across.
        step = width // math.gcd(1 << adc_bits, width)
        on_grid = [lower + k * step for k in range((upper - lower) // step + 1)]
        expected = _codes(macro, on_grid, quantization_mode=0, adc_bits=adc_bits)
        macro.train()
        for seed in range(8):
            torch.manual_seed(seed)
            assert _codes(macro, on_grid, quantization_mode=0, adc_bits=adc_bits) == expected

    @pytest.mark.parametrize("window", _WINDOWS)
    def test_zero_dot_never_jitters(self, window: tuple[int, int]) -> None:
        macro = _dot_macro(quantization_input_ranges=(window,), adc_max_bits=6)
        macro.train()
        for adc_bits in range(1, 7):
            assert _codes(macro, [0] * 64, quantization_mode=0, adc_bits=adc_bits) == [0] * 64

    def test_off_grid_dots_do_jitter(self) -> None:
        """The mechanism is live: an off-grid dot splits between two codes."""
        macro = _dot_macro(quantization_input_ranges=((-32, 31),), adc_max_bits=6)
        macro.train()
        torch.manual_seed(0)
        # lsb = 8 at 3 bits; dot 4 sits mid-bin between codes 0 and 1.
        codes = set(_codes(macro, [4] * 256, quantization_mode=0, adc_bits=3))
        assert codes == {0, 1}


class TestConfigValidation:
    """``quantization_input_ranges`` and ``adc_max_bits`` guards."""

    def test_empty_ranges_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"quantization_input_ranges"):
            _make_macro(quantization_input_ranges=(), adc_max_bits=6)

    def test_zero_max_bits_rejected(self) -> None:
        """The lossless oracle is a runtime value, not a declared width."""
        with pytest.raises(ValueError, match=r"adc_max_bits"):
            _make_macro(quantization_input_ranges=((-32, 31),), adc_max_bits=0)

    @pytest.mark.parametrize("window", [(-32, 32), (32, 32), (8, 32), (-32, -8), (-1, 3)])
    def test_non_canonical_window_rejected(self, window: tuple[int, int]) -> None:
        with pytest.raises(ValueError, match=r"quantization_input_range"):
            _make_macro(quantization_input_ranges=(window,), adc_max_bits=6)

    def test_unsigned_window_accepts_any_width(self) -> None:
        """``lower == 0`` fixes the zero code at 0, so the width is free."""
        macro = _make_macro(quantization_input_ranges=((0, 62),), adc_max_bits=6)
        assert len(macro.quantization_input_ranges) == 1

    def test_mid_zero_window_accepts_a_non_power_of_two(self) -> None:
        macro = _make_macro(quantization_input_ranges=((-7, 6),), adc_max_bits=6)
        assert macro.quantization_input_ranges == ((-7, 6),)


class TestInputCodeMap:
    """``map_quantization_input_code`` is the window's zero-point grid."""

    def test_mid_zero_window_offsets_to_unsigned(self) -> None:
        macro = _make_macro(quantization_input_ranges=((-8, 7),), adc_max_bits=4)
        mapped, code_range = macro.map_quantization_input_code(
            torch.tensor([-8, 0, 7], dtype=torch.int64), quantization_mode=0
        )
        assert mapped.tolist() == [0, 8, 15]
        assert code_range == (0, 15)

    def test_unsigned_window_is_the_identity(self) -> None:
        macro = _make_macro(quantization_input_ranges=((0, 223),), adc_max_bits=4)
        code = torch.tensor([0, 100, 223], dtype=torch.int64)
        mapped, code_range = macro.map_quantization_input_code(code, quantization_mode=0)
        assert torch.equal(mapped, code)
        assert code_range == (0, 223)

    def test_unknown_mode_rejected(self) -> None:
        macro = _make_macro(quantization_input_ranges=((0, 223),), adc_max_bits=4)
        with pytest.raises(ValueError, match=r"quantization_mode"):
            macro.map_quantization_input_code(torch.zeros(1, dtype=torch.int64), quantization_mode=1)


class TestPlaneOutput:
    """``vec_mat_mul`` keeps every leading axis anonymous — the caller owns any phase axis."""

    @staticmethod
    def _programmed_macro(*, max_active_num: int | None) -> tuple[IdealCimMacro, torch.Tensor]:
        torch.manual_seed(42)
        macro = _make_macro(
            quantization_input_ranges=((-32, 31),),
            adc_max_bits=8,
            w_value_range=(-3, 3),
            max_active_num=max_active_num,
        )
        w = torch.randint(-3, 4, (macro.input_num, macro.output_num), dtype=torch.int32)
        macro.program(w)
        return macro, w

    def test_leading_axes_preserved(self) -> None:
        macro, _ = self._programmed_macro(max_active_num=2)
        x = torch.randint(0, 2, (3, 5, 8), dtype=torch.int32)
        y = macro.vec_mat_mul(x, quantization_mode=0, adc_bits=8)
        assert y.shape == (3, 5, 4)

    def test_single_leading_axis(self) -> None:
        macro, _ = self._programmed_macro(max_active_num=None)
        x = torch.randint(0, 2, (3, 8), dtype=torch.int32)
        y = macro.vec_mat_mul(x, quantization_mode=0, adc_bits=8)
        assert y.shape == (3, 4)

    def test_lossless_matches_whole_dot(self) -> None:
        macro, w = self._programmed_macro(max_active_num=2)
        x = torch.randint(0, 2, (5, 8), dtype=torch.int32)
        y = macro.vec_mat_mul(x, quantization_mode=0, adc_bits=None)
        assert y.shape == (5, 4)
        expected = x.to(torch.int64) @ w.to(torch.int64)
        assert torch.equal(y, expected)


class TestProgramOwnership:
    """``program`` must take ownership of the weight tensor (clone + device)."""

    def test_program_does_not_alias_caller(self) -> None:
        macro = _make_macro(
            quantization_input_ranges=((-32, 31),),
            adc_max_bits=8,
            w_value_range=(-1, 1),
            input_num=4,
            output_num=2,
        )
        w = torch.zeros((macro.input_num, macro.output_num), dtype=torch.int32)
        macro.program(w)
        # Caller-side mutation must not affect the stored state.
        w.fill_(1)
        x = torch.ones(macro.input_num, dtype=torch.int32)
        out = macro.vec_mat_mul(x, quantization_mode=0, adc_bits=None)
        assert torch.equal(out, torch.zeros(macro.output_num, dtype=torch.int64))
