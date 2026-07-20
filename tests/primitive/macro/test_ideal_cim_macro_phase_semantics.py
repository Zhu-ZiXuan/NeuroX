"""IdealCimMacro per-plane quantization semantics.

One ``vec_mat_mul`` call is one independent ADC conversion per column per
WL plane, quantized against the per-conversion range
(``_max_plane_dot_abs``); the output keeps the leading order with primitive
trailing ``[col_num]`` and the macro performs no accumulation. The caller
presents each sub-phase as its own zero-masked plane (engine mask formula),
which preserves quantize-then-accumulate semantics:
``sum(Q(plane_dot)) != Q(sum(plane_dot))`` in general.
"""

from __future__ import annotations

import torch

from neurox.primitive.analog.adc_common import AdcOperationPoint
from neurox.primitive.macro.cim.ideal import IdealCimMacro, IdealCimMacroConfig, IdealCimMacroPolicy


def _make_xbar(
    *,
    row_num: int,
    active_row_num: int,
    col_num: int,
    adc_max_bits: int,
) -> IdealCimMacro:
    config = IdealCimMacroConfig(
        col_num=col_num,
        row_num=row_num,
        active_row_num=active_row_num,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        x_range=(0, 1),
        w_digit_count=1,
        w_digit_radix=2,
        w_digit_range=(-3, 3),
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


def _program_cols(xbar: IdealCimMacro, cols: list[list[int]]) -> None:
    """Program single-digit column weights ``cols[c][r]``."""
    w = torch.tensor(cols, dtype=torch.int32).unsqueeze(-2)  # [col, D=1, row]
    xbar.program(w)


def _masked_planes(x: torch.Tensor, *, row_num: int, max_active_rows: int) -> torch.Tensor:
    """Zero-masked WL planes via the engine mask formula.

    Shape: [..., row_num] -> [..., P, row_num]; plane ``p`` keeps exactly
    rows ``[p*max_active_rows, (p+1)*max_active_rows)``, zeros elsewhere.
    """
    p_num = row_num // max_active_rows
    mask = torch.arange(row_num) // max_active_rows == torch.arange(p_num).unsqueeze(-1)
    # Shape: [..., row_num] -> [..., P, row_num]
    return torch.where(mask, x.unsqueeze(-2), x.new_zeros(()))


class TestPerPlaneClampVsWholeSum:
    """A=2, R=4: one plane saturates positive, the other negative."""

    def _saturating_xbar(self) -> IdealCimMacro:
        xbar = _make_xbar(row_num=4, active_row_num=2, col_num=2, adc_max_bits=3)
        # col 0: plane 0 dot = +6 (positive per-conversion extreme), plane 1 = -6.
        # col 1: odd plane dots (+3 / -3) expose the floor asymmetry.
        _program_cols(xbar, [[3, 3, -3, -3], [3, 0, -3, 0]])
        return xbar

    def test_per_plane_codes_hit_conversion_extremes(self) -> None:
        xbar = self._saturating_xbar()
        planes = _masked_planes(torch.ones(4, dtype=torch.int32), row_num=4, max_active_rows=2)
        y = xbar.vec_mat_mul(planes, adc_operation_point=AdcOperationPoint(adc_mode=0, adc_bits=3))
        assert y.dtype == torch.int16
        assert y.shape == (2, 2)  # leading [P] preserved, trailing [col_num]
        # scale = ((1<<2)-1) / (A·max|w|·max|x|) = 3/6: codes floor(dot·0.5).
        expected = torch.tensor([[3, 1], [-3, -2]], dtype=torch.int16)
        assert torch.equal(y, expected)

    def test_plane_code_sum_differs_from_whole_sum_quantization(self) -> None:
        """``sum(Q(plane_dot))`` != ``Q(sum(plane_dot))`` at the same scale."""
        xbar = self._saturating_xbar()
        planes = _masked_planes(torch.ones(4, dtype=torch.int32), row_num=4, max_active_rows=2)
        y = xbar.vec_mat_mul(planes, adc_operation_point=AdcOperationPoint(adc_mode=0, adc_bits=3))
        # Shape: [P, col] -> [col]   caller-side digital accumulation
        plane_code_sum = y.to(torch.int64).sum(dim=0)
        # Whole dots are 0 for both cols; quantizing the whole sum at the
        # per-conversion scale yields 0 — but col 1's per-plane codes sum to -1.
        whole_dot = torch.tensor([0, 0], dtype=torch.int64)
        rescale = xbar.adc_rescale_factor(AdcOperationPoint(adc_mode=0, adc_bits=3))
        whole_code = torch.floor(whole_dot.to(torch.float32) * (1.0 / rescale)).to(torch.int64)
        assert torch.equal(plane_code_sum, torch.tensor([0, -1], dtype=torch.int64))
        assert not torch.equal(plane_code_sum, whole_code)


class TestLosslessSentinel:
    """``adc_bits == 0`` returns exact int64 plane dots."""

    def test_plane_dots_exact_and_sum_to_full_dot(self) -> None:
        torch.manual_seed(11)
        xbar = _make_xbar(row_num=4, active_row_num=2, col_num=2, adc_max_bits=0)
        w = torch.randint(-3, 4, (2, 4), dtype=torch.int32)
        _program_cols(xbar, w.tolist())
        x = torch.randint(0, 2, (3, 4), dtype=torch.int32)
        planes = _masked_planes(x, row_num=4, max_active_rows=2)
        y = xbar.vec_mat_mul(planes, adc_operation_point=AdcOperationPoint(adc_mode=0, adc_bits=0))
        assert y.dtype == torch.int64
        assert y.shape == (3, 2, 2)  # leading [batch, P] preserved, trailing [col_num]
        w64 = w.to(torch.int64)
        x64 = x.to(torch.int64)
        for p in range(2):
            rows = slice(p * 2, (p + 1) * 2)
            expected_p = x64[:, rows] @ w64[:, rows].transpose(-1, -2)
            assert torch.equal(y[:, p, :], expected_p)
        # Shape: [batch, P, col] -> [batch, col]   sum over P = full-row dot
        assert torch.equal(y.sum(dim=-2), x64 @ w64.transpose(-1, -2))


class TestFullActivationParity:
    """``active_row_num == row_num``: a full-row plane is conformant and the
    macro adds no axis of its own."""

    def test_full_row_plane_matches_whole_sum_quantization(self) -> None:
        torch.manual_seed(13)
        xbar = _make_xbar(row_num=4, active_row_num=4, col_num=2, adc_max_bits=3)
        w = torch.randint(-3, 4, (2, 4), dtype=torch.int32)
        _program_cols(xbar, w.tolist())
        x = torch.randint(0, 2, (5, 4), dtype=torch.int32)
        y = xbar.vec_mat_mul(x, adc_operation_point=AdcOperationPoint(adc_mode=0, adc_bits=3))
        assert y.shape == (5, 2)  # no phase axis: leading order preserved
        whole_dot = x.to(torch.int64) @ w.to(torch.int64).transpose(-1, -2)
        rescale = xbar._max_plane_dot_abs / ((1 << 2) - 1)
        expected = torch.floor(whole_dot.to(torch.float32) * (1.0 / rescale)).to(torch.int64).clamp(-4, 3)
        assert torch.equal(y.to(torch.int64), expected)
