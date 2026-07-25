"""IdealCimMacro quantize/rescale correctness across value ranges.

The recovery rescale must clip exactly at the signed N-bit endpoints
under any legal ``w_value_range``. The rescale denominator is the maximum
magnitude of one conversion, so the bound scales with ``max_active_num``,
not ``input_num``. The macro treats every leading axis as anonymous batch.
"""

from __future__ import annotations

import torch

from neurox.primitive.macro.cim.ideal import IdealCimMacro, IdealCimMacroConfig, IdealCimMacroPolicy


def _make_macro(
    *,
    w_value_range: tuple[int, int],
    x_value_range: tuple[int, int],
    input_num: int,
    output_num: int,
    adc_bits: int,
    max_active_num: int | None = None,
) -> IdealCimMacro:
    config = IdealCimMacroConfig(
        max_active_num=input_num if max_active_num is None else max_active_num,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        x_value_range=x_value_range,
        w_value_range=w_value_range,
        adc_mode_num=1,
        adc_max_bits=adc_bits,
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


def _expected_max_plane_dot(
    *,
    w_value_range: tuple[int, int],
    x_value_range: tuple[int, int],
    max_active_num: int,
) -> int:
    w_lo, w_hi = w_value_range
    max_w_abs = max(abs(w_lo), abs(w_hi))
    x_lo, x_hi = x_value_range
    return int(max_active_num * max_w_abs * max(abs(x_lo), abs(x_hi)))


class TestRescaleScope:
    """The public rescale factor follows the logical weight and input ranges."""

    def test_signed_symmetric_range(self) -> None:
        macro = _make_macro(
            w_value_range=(-15, 15),
            x_value_range=(0, 1),
            input_num=8,
            output_num=4,
            adc_bits=8,
        )
        expected = _expected_max_plane_dot(
            w_value_range=(-15, 15),
            x_value_range=(0, 1),
            max_active_num=8,
        )
        assert macro.adc_rescale_factor(adc_mode=0, adc_bits=8) == expected / ((1 << 7) - 1)

    def test_offset_nonneg_range(self) -> None:
        macro = _make_macro(
            w_value_range=(0, 15),
            x_value_range=(0, 1),
            input_num=8,
            output_num=4,
            adc_bits=8,
        )
        expected = _expected_max_plane_dot(
            w_value_range=(0, 15),
            x_value_range=(0, 1),
            max_active_num=8,
        )
        assert macro.adc_rescale_factor(adc_mode=0, adc_bits=8) == expected / ((1 << 7) - 1)

    def test_asymmetric_signed_range(self) -> None:
        macro = _make_macro(
            w_value_range=(-4, 8),
            x_value_range=(0, 1),
            input_num=8,
            output_num=4,
            adc_bits=8,
        )
        expected = _expected_max_plane_dot(
            w_value_range=(-4, 8),
            x_value_range=(0, 1),
            max_active_num=8,
        )
        assert macro.adc_rescale_factor(adc_mode=0, adc_bits=8) == expected / ((1 << 7) - 1)

    def test_partial_activation_uses_max_active_num(self) -> None:
        macro = _make_macro(
            w_value_range=(-15, 15),
            x_value_range=(0, 1),
            input_num=8,
            output_num=4,
            adc_bits=8,
            max_active_num=2,
        )
        expected = _expected_max_plane_dot(
            w_value_range=(-15, 15),
            x_value_range=(0, 1),
            max_active_num=2,
        )
        assert macro.adc_rescale_factor(adc_mode=0, adc_bits=8) == expected / ((1 << 7) - 1)


class TestPlaneOutput:
    """``vec_mat_mul`` maps trailing ``[input_num]`` to ``[output_num]`` and keeps
    every leading axis anonymous — the caller owns any phase axis."""

    @staticmethod
    def _programmed_macro(*, max_active_num: int | None, adc_bits: int) -> tuple[IdealCimMacro, torch.Tensor]:
        torch.manual_seed(42)
        macro = _make_macro(
            w_value_range=(-3, 3),
            x_value_range=(0, 1),
            input_num=8,
            output_num=4,
            adc_bits=adc_bits,
            max_active_num=max_active_num,
        )
        w = torch.randint(-3, 4, (macro.input_num, macro.output_num), dtype=torch.int32)
        macro.program(w)
        return macro, w

    def test_leading_axes_preserved(self) -> None:
        macro, _ = self._programmed_macro(max_active_num=2, adc_bits=8)
        x = torch.randint(0, 2, (3, 5, 8), dtype=torch.int32)
        y = macro.vec_mat_mul(x, adc_mode=0, adc_bits=8)
        assert y.shape == (3, 5, 4)

    def test_single_leading_axis(self) -> None:
        macro, _ = self._programmed_macro(max_active_num=None, adc_bits=8)
        x = torch.randint(0, 2, (3, 8), dtype=torch.int32)
        y = macro.vec_mat_mul(x, adc_mode=0, adc_bits=8)
        assert y.shape == (3, 4)

    def test_lossless_matches_whole_dot(self) -> None:
        macro, w = self._programmed_macro(max_active_num=2, adc_bits=0)
        x = torch.randint(0, 2, (5, 8), dtype=torch.int32)
        y = macro.vec_mat_mul(x, adc_mode=0, adc_bits=0)
        assert y.shape == (5, 4)
        expected = x.to(torch.int64) @ w.to(torch.int64)
        assert torch.equal(y, expected)


class TestProgramOwnership:
    """``program`` must take ownership of the weight tensor (clone + device)."""

    def test_program_does_not_alias_caller(self) -> None:
        macro = _make_macro(
            w_value_range=(-1, 1),
            x_value_range=(0, 1),
            input_num=4,
            output_num=2,
            adc_bits=8,
        )
        w = torch.zeros((macro.input_num, macro.output_num), dtype=torch.int32)
        macro.program(w)
        # Caller-side mutation must not affect the stored state.
        w.fill_(1)
        x = torch.ones(macro.input_num, dtype=torch.int32)
        out = macro.vec_mat_mul(x, adc_mode=0, adc_bits=0)
        assert torch.equal(out, torch.zeros(macro.output_num, dtype=torch.int64))
