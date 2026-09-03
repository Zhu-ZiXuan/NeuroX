"""IdealCimMacro per-plane quantization semantics.

One `vec_mat_mul` call is one independent ADC conversion per output per
WL plane, read through the selected mode scale; the output keeps the
leading order and the macro performs no accumulation. The caller presents each
sub-phase as its own zero-masked plane (engine mask formula), which preserves
quantize-then-accumulate semantics: `sum(Q(plane_dot)) != Q(sum(plane_dot))`
in general.
"""

from __future__ import annotations

import torch

from neurox.common.encoding import Encoding
from neurox.primitive.macro.cim import (
    CimMacroQuantizationScheme,
    IdealCimMacro,
    IdealCimMacroConfig,
    IdealCimMacroPolicy,
)


def _make_macro(
    *,
    input_num: int,
    max_active_num: int,
    output_num: int,
    rescale_factors: tuple[float, ...],
    adc_bits: int,
) -> IdealCimMacro:
    config = IdealCimMacroConfig(
        input_num=input_num,
        rescale_factors=rescale_factors,
        max_active_num=max_active_num,
        lane_num=1,
        scan_num=output_num,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        w_digit_num=2,
        w_digit_radix=2,
        w_encoding=Encoding.TRUE_FORM,
        x_digit_num=1,
        x_digit_radix=2,
        x_encoding=Encoding.UNSIGNED,
        x_value_range=(0, 1),
        w_value_range=(-3, 3),
        adc_bits=adc_bits,
        quantization_scheme=CimMacroQuantizationScheme.ZERO_POINT,
    )
    macro = IdealCimMacro(
        config=config,
        policy=IdealCimMacroPolicy(),
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    macro.eval()
    return macro


def _program_outputs(macro: IdealCimMacro, outputs: list[list[int]]) -> None:
    """Program output-major fixture values through the logical matrix API."""
    # Shape: [output_num, input_num] -> [input_num, output_num]
    w = torch.tensor(outputs, dtype=torch.int32).transpose(-1, -2)
    macro.program(w)


def _masked_planes(x: torch.Tensor, *, input_num: int, max_active_num: int) -> torch.Tensor:
    """Zero-masked WL planes via the engine mask formula."""
    p_num = input_num // max_active_num
    mask = torch.arange(input_num) // max_active_num == torch.arange(p_num).unsqueeze(-1)
    # Shape: [..., input_num] -> [..., P, input_num]
    return torch.where(mask, x.unsqueeze(-2), x.new_zeros(()))


def _zero_point_law(
    dot: torch.Tensor,
    *,
    factor: float,
    adc_bits: int,
    adc_active_bits: int,
) -> torch.Tensor:
    """Restate centered full-resolution quantization and arithmetic truncation."""
    zero_point = 1 << (adc_bits - 1)
    code = torch.floor(dot.to(torch.float32) / factor)
    code = code.clamp(-zero_point, zero_point - 1).to(torch.int32)
    return code >> (adc_bits - adc_active_bits)


class TestPerPlaneClampVsWholeSum:
    """A=2, R=4: one plane saturates positive, the other negative."""

    # A 1.5-MAC full-resolution scale at 3 bits yields codes in [-4, 3].
    _FACTOR = 1.5
    _BITS = 3

    def _saturating_macro(self) -> IdealCimMacro:
        macro = _make_macro(
            input_num=4,
            max_active_num=2,
            output_num=2,
            rescale_factors=(self._FACTOR,),
            adc_bits=self._BITS,
        )
        _program_outputs(macro, [[3, 3, -3, -3], [3, 0, -3, 0]])
        return macro

    def test_per_plane_codes_hit_conversion_extremes(self) -> None:
        macro = self._saturating_macro()
        planes = _masked_planes(torch.ones(4, dtype=torch.int32), input_num=4, max_active_num=2)
        y = macro.vec_mat_mul(planes, quantization_mode=0, adc_active_bits=self._BITS)
        assert y.dtype == torch.int32
        assert y.shape == (2, 2)
        # Plane dots: [[6, 3], [-6, -3]]; col 0 clips at both endpoints.
        plane_dot = torch.tensor([[6, 3], [-6, -3]], dtype=torch.int64)
        assert torch.equal(
            y,
            _zero_point_law(plane_dot, factor=self._FACTOR, adc_bits=self._BITS, adc_active_bits=self._BITS),
        )
        assert y[0, 0].item() == (1 << (self._BITS - 1)) - 1  # positive rail
        assert y[1, 0].item() == -(1 << (self._BITS - 1))  # negative rail

    def test_plane_code_sum_differs_from_whole_sum_quantization(self) -> None:
        """`sum(Q(plane_dot))` differs from `Q(sum(plane_dot))`."""
        macro = self._saturating_macro()
        planes = _masked_planes(torch.ones(4, dtype=torch.int32), input_num=4, max_active_num=2)
        y = macro.vec_mat_mul(planes, quantization_mode=0, adc_active_bits=self._BITS)
        # Caller-side digital accumulation.
        # Shape: [P, col] -> [col]
        plane_code_sum = y.sum(dim=0)
        # Whole dots are 0 for both cols, but the per-plane codes clip
        # asymmetrically before their caller-side sum.
        whole_dot = torch.tensor([0, 0], dtype=torch.int64)
        whole_code = _zero_point_law(
            whole_dot,
            factor=self._FACTOR,
            adc_bits=self._BITS,
            adc_active_bits=self._BITS,
        )
        assert torch.equal(whole_code, torch.zeros(2, dtype=torch.int32))
        assert not torch.equal(plane_code_sum, whole_code)


class TestExactOracle:
    """An explicit highest-precision request returns exact int64 plane dots."""

    def test_plane_dots_exact_and_sum_to_full_dot(self) -> None:
        torch.manual_seed(11)
        macro = _make_macro(
            input_num=4,
            max_active_num=2,
            output_num=2,
            rescale_factors=(2.0,),
            adc_bits=3,
        )
        w = torch.randint(-3, 4, (2, 4), dtype=torch.int32)
        _program_outputs(macro, w.tolist())
        x = torch.randint(0, 2, (3, 4), dtype=torch.int32)
        planes = _masked_planes(x, input_num=4, max_active_num=2)
        y = macro.vec_mat_mul(planes, quantization_mode=0, adc_active_bits=None)
        assert y.dtype == torch.int64
        assert y.shape == (3, 2, 2)
        w64 = w.to(torch.int64)
        x64 = x.to(torch.int64)
        for p in range(2):
            rows = slice(p * 2, (p + 1) * 2)
            expected_p = x64[:, rows] @ w64[:, rows].transpose(-1, -2)
            assert torch.equal(y[:, p, :], expected_p)
        # Sum over P is the full-row dot.
        # Shape: [..., P, col] -> [..., col]
        assert torch.equal(y.sum(dim=-2), x64 @ w64.transpose(-1, -2))


class TestFullActivationParity:
    """`max_active_num == input_num`: a full input is conformant and the
    macro adds no axis of its own."""

    def test_full_row_plane_matches_whole_sum_quantization(self) -> None:
        torch.manual_seed(13)
        factor = 4.0
        adc_bits = 3
        macro = _make_macro(
            input_num=4,
            max_active_num=4,
            output_num=2,
            rescale_factors=(factor,),
            adc_bits=adc_bits,
        )
        w = torch.randint(-3, 4, (2, 4), dtype=torch.int32)
        _program_outputs(macro, w.tolist())
        x = torch.randint(0, 2, (5, 4), dtype=torch.int32)
        y = macro.vec_mat_mul(x, quantization_mode=0, adc_active_bits=adc_bits)
        assert y.shape == (5, 2)  # no phase axis: leading order preserved
        whole_dot = x.to(torch.int64) @ w.to(torch.int64).transpose(-1, -2)
        assert torch.equal(
            y,
            _zero_point_law(whole_dot, factor=factor, adc_bits=adc_bits, adc_active_bits=adc_bits),
        )
