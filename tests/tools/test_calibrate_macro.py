"""Pure helpers used by macro rescale calibration."""

from __future__ import annotations

import tomllib

import pytest
import torch

from neurox.tools._macro import unroll_active_positions
from neurox.tools.calibrate_macro._math import RescaleFit, fit_rescale_through_origin
from neurox.tools.calibrate_macro.rescale_fit import ModeFitResult, _fragment_lines


class TestFitRescaleThroughOrigin:
    def test_exact_recovery(self) -> None:
        code = torch.arange(0, 8, dtype=torch.float64).repeat(16)
        ideal = 2.5 * code
        fit = fit_rescale_through_origin(code, ideal)
        assert fit.rescale_factor == pytest.approx(2.5, abs=1e-12)
        assert fit.r2 == pytest.approx(1.0, abs=1e-12)
        assert fit.rmse == pytest.approx(0.0, abs=1e-12)
        assert fit.sample_num == code.numel()

    def test_least_squares_optimality_under_noise(self) -> None:
        generator = torch.Generator().manual_seed(7)
        code = torch.randint(0, 32, (512,), generator=generator).to(torch.float64)
        noise = torch.randn(512, generator=generator, dtype=torch.float64) * 0.05
        ideal = 1.3 * code + noise
        fit = fit_rescale_through_origin(code, ideal)
        expected = float((code * ideal).sum() / (code * code).sum())
        assert fit.rescale_factor == pytest.approx(expected, rel=1e-12)
        assert fit.rescale_factor == pytest.approx(1.3, abs=0.01)

    def test_integer_code_tensor_accepted(self) -> None:
        code = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        ideal = torch.tensor([0, 2, 4, 6], dtype=torch.int64)
        assert fit_rescale_through_origin(code, ideal).rescale_factor == pytest.approx(2.0, abs=1e-12)

    def test_all_zero_code_raises(self) -> None:
        with pytest.raises(ValueError, match="all zero"):
            fit_rescale_through_origin(torch.zeros(8), torch.ones(8))

    def test_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="numel"):
            fit_rescale_through_origin(torch.ones(4), torch.ones(5))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            fit_rescale_through_origin(torch.empty(0), torch.empty(0))


class TestActivePositionUnroll:
    def test_non_divisible_input_is_covered_once(self) -> None:
        x = torch.ones((2, 256), dtype=torch.long)
        planes = unroll_active_positions(x, input_num=256, max_active_num=9, inst_shape=())
        assert tuple(planes.shape) == (2, 29, 256)
        assert torch.equal(planes.sum(dim=1), x)

    def test_divisible_input_uses_exact_quotient(self) -> None:
        x = torch.ones((12,), dtype=torch.long)
        planes = unroll_active_positions(x, input_num=12, max_active_num=4, inst_shape=())
        assert planes.shape[-2] == 3
        assert torch.equal(planes.sum(dim=-2), x)

    def test_plane_axis_precedes_instance_axes(self) -> None:
        x = torch.ones((3, 1, 12), dtype=torch.long)
        planes = unroll_active_positions(x, input_num=12, max_active_num=4, inst_shape=(1,))
        assert tuple(planes.shape) == (3, 3, 1, 12)
        assert torch.equal(planes.sum(dim=1), x)


def _result(quantization_mode: int, rescale_factor: float) -> ModeFitResult:
    return ModeFitResult(
        quantization_mode=quantization_mode,
        adc_bits=3,
        fit=RescaleFit(
            rescale_factor=rescale_factor,
            sample_num=4,
            r2=1.0,
            rmse=0.0,
            max_abs_residual=0.0,
        ),
        code=torch.arange(4),
        ideal_value=torch.arange(4, dtype=torch.float64),
    )


class TestRescaleFragment:
    def test_emits_complete_factor_array(self) -> None:
        text = "\n".join(_fragment_lines([_result(1, 2.5)], (1.0, 2.0, 3.0)))
        assert tomllib.loads(text)["rescale_factors"] == pytest.approx([1.0, 2.5, 3.0])

    def test_result_order_does_not_change_mode_indexing(self) -> None:
        text = "\n".join(_fragment_lines([_result(1, 2.0), _result(0, 1.0)], (9.0, 9.0)))
        assert tomllib.loads(text)["rescale_factors"] == pytest.approx([1.0, 2.0])
