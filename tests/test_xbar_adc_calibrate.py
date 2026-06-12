"""Unit tests for ``neurox/tools/xbar_adc/calibrate.py``."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

from neurox.common import dataclass_from_file
from neurox.tools.xbar_adc.calibrate import (
    collect_calibration,
    derive_rescale_for_bits,
    fit_rescale_factor,
    log_calibration,
    plot_calibration,
    saturation_mask,
)
from neurox.tools.xbar_adc.calibrate import (
    main as calibrate_main,
)
from neurox.xbar import Offset1T1RXbarConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
XBAR_CONFIG = REPO_ROOT / "example" / "config" / "1t1r_28nm.toml"

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def xbar_cfg() -> Offset1T1RXbarConfig:
    if not XBAR_CONFIG.is_file():
        pytest.skip(f"missing test fixture: {XBAR_CONFIG}")
    return dataclass_from_file(Offset1T1RXbarConfig, XBAR_CONFIG, section="xbar")


def _make_run_config(tmp_path: Path, *, adc_mode: int = 0) -> Path:
    """Write a minimal calibrate-run TOML pointing at the chip preset."""
    cfg_path = tmp_path / "run.toml"
    cfg_path.write_text(
        f"""[xbar]
_neurox_use = "{XBAR_CONFIG}:xbar"

[workload]
weight_samples = 4
input_samples_per_weight = 8
batch_size = 4
seed = 0

[adc]
mode = {adc_mode}
"""
    )
    return cfg_path


# ---------------------------------------------------------------------------
# fit_rescale_factor — zero-through-origin LS, returns r_max (float)
# ---------------------------------------------------------------------------


class TestFitRescaleFactor:
    def test_perfect_linear(self) -> None:
        p = torch.arange(1, 11, dtype=torch.float64)
        y = 0.5 * p
        assert fit_rescale_factor(p, y) == pytest.approx(0.5, abs=1e-12)

    def test_negative_targets(self) -> None:
        p = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        y = torch.tensor([-2.0, -4.0, -6.0, -8.0], dtype=torch.float64)
        assert fit_rescale_factor(p, y) == pytest.approx(-2.0, abs=1e-12)

    def test_least_squares_solution(self) -> None:
        # Noisy y around y = 0.3 * p. Zero-through-origin LS = Σpy / Σp².
        p = torch.arange(1, 1001, dtype=torch.float64)
        torch.manual_seed(0)
        y = 0.3 * p + torch.randn(1000, dtype=torch.float64)
        expected = float((p * y).sum() / (p * p).sum())
        assert fit_rescale_factor(p, y) == pytest.approx(expected, abs=1e-12)

    def test_zero_denominator_raises(self) -> None:
        # All-zero phys_codes → Σp² == 0 → strictly an error in the new
        # zero-through-origin form (no intercept to absorb constant y).
        p = torch.zeros(5, dtype=torch.float64)
        y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        with pytest.raises(ValueError, match=r"all valid physical codes are zero"):
            fit_rescale_factor(p, y)


# ---------------------------------------------------------------------------
# derive_rescale_for_bits
# ---------------------------------------------------------------------------


class TestDeriveRescaleForBits:
    def test_power_of_two_scaling(self) -> None:
        derived = derive_rescale_for_bits(r_max=0.04, max_bits=4)
        assert derived[4] == pytest.approx(0.04)
        assert derived[3] == pytest.approx(0.08)
        assert derived[2] == pytest.approx(0.16)
        assert derived[1] == pytest.approx(0.32)

    def test_single_bit(self) -> None:
        derived = derive_rescale_for_bits(r_max=1.5, max_bits=1)
        assert derived == {1: 1.5}

    def test_rejects_zero_max_bits(self) -> None:
        with pytest.raises(ValueError, match=r"max_bits"):
            derive_rescale_for_bits(r_max=1.0, max_bits=0)


# ---------------------------------------------------------------------------
# saturation_mask
# ---------------------------------------------------------------------------


class TestSaturationMask:
    def test_signed_endpoints(self) -> None:
        codes = torch.tensor([-8, -7, 0, 6, 7, -8, 7], dtype=torch.float64)
        mask = saturation_mask(codes, signed_range=(-8, 7))
        assert mask.tolist() == [True, False, False, False, True, True, True]

    def test_no_endpoints(self) -> None:
        codes = torch.tensor([-3, -2, 0, 2], dtype=torch.float64)
        mask = saturation_mask(codes, signed_range=(-8, 7))
        assert mask.tolist() == [False, False, False, False]

    def test_eight_bit_endpoints(self) -> None:
        codes = torch.tensor([-128, 127, 0, 50, -100], dtype=torch.float64)
        mask = saturation_mask(codes, signed_range=(-128, 127))
        assert mask.tolist() == [True, True, False, False, False]

    def test_narrower_than_canonical(self) -> None:
        """GeneralADC-style narrower bounds — canonical endpoints must NOT
        register as saturated, only the realised bounds."""
        codes = torch.tensor([-100, -101, 100, 101, 127, -128], dtype=torch.float64)
        mask = saturation_mask(codes, signed_range=(-101, 101))
        assert mask.tolist() == [False, True, False, True, False, False]


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke_calibration(xbar_cfg):
    """Small but realistic calibration on the 28nm preset (adc_mode=0)."""
    return collect_calibration(
        xbar_config=xbar_cfg,
        distribution_path=None,
        adc_mode=0,
        weight_samples=4,
        input_samples_per_weight=8,
        batch_size=4,
        seed=0,
        device=CPU,
    )


class TestCollectCalibration:
    def test_returns_populated_object(self, smoke_calibration) -> None:
        r = smoke_calibration
        assert r.adc_mode == 0
        assert r.max_bits >= 1
        assert r.total_pairs > 0
        assert r.valid_pairs + r.saturated_pairs_excluded == r.total_pairs
        assert 0.0 <= r.saturation_rate <= 1.0
        assert r.r_max > 0.0
        assert len(r.derived_rescale) == r.max_bits
        for b, val in r.derived_rescale.items():
            assert val == pytest.approx(r.r_max * (2 ** (r.max_bits - b)))
        # batch_size=4 * n_groups=4 = 16.
        assert r.adc_instance_count == 16
        assert r.supports_flexible_bits is True

    def test_invalid_adc_mode(self, xbar_cfg) -> None:
        with pytest.raises(ValueError, match=r"adc_mode"):
            collect_calibration(
                xbar_config=xbar_cfg,
                distribution_path=None,
                adc_mode=99,
                weight_samples=4,
                input_samples_per_weight=8,
                batch_size=4,
                seed=0,
                device=CPU,
            )

    def test_rejects_bad_args(self, xbar_cfg) -> None:
        with pytest.raises(ValueError, match=r"weight_samples"):
            collect_calibration(
                xbar_config=xbar_cfg,
                distribution_path=None,
                adc_mode=0,
                weight_samples=0,
                input_samples_per_weight=8,
                batch_size=4,
                seed=0,
                device=CPU,
            )


# ---------------------------------------------------------------------------
# Logger output
# ---------------------------------------------------------------------------


class TestLogCalibration:
    def test_emits_expected_sections(self, smoke_calibration, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="neurox.tools.xbar_adc.calibrate"):
            log_calibration(smoke_calibration)

        text = caplog.text
        assert "ADC rescale calibration" in text
        assert "rescale_factor_at_max_bits" in text
        assert "Derived rescale factors" in text
        assert "NOT calibrated" in text
        assert "<- calibrated" in text
        assert "[[adc_calibration]]" in text
        assert "adc_mode = 0" in text
        assert "rescale_factor" in text
        assert "adc_instance_count" in text

    def test_emits_single_toml_line_per_mode(self, smoke_calibration, caplog: pytest.LogCaptureFixture) -> None:
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="neurox.tools.xbar_adc.calibrate"):
            log_calibration(smoke_calibration)
        count = caplog.text.count("[[adc_calibration]]")
        assert count == 1, f"expected exactly one [[adc_calibration]] block, got {count}"


# ---------------------------------------------------------------------------
# Plot smoke
# ---------------------------------------------------------------------------


class TestPlotCalibration:
    def test_writes_png(self, smoke_calibration, tmp_path: Path) -> None:
        path = tmp_path / "calib.png"
        plot_calibration(smoke_calibration, path)
        assert path.exists()
        assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_smoke(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        cfg = _make_run_config(tmp_path)
        argv = ["--config", str(cfg), "--device", "cpu"]
        with caplog.at_level(logging.INFO, logger="neurox.tools.xbar_adc.calibrate"):
            assert calibrate_main(argv) == 0
        assert "ADC rescale calibration" in caplog.text
        assert "[[adc_calibration]]" in caplog.text

    def test_cli_requires_config(self) -> None:
        with pytest.raises(SystemExit):
            calibrate_main([])

    def test_cli_rejects_bad_adc_mode(self, tmp_path: Path) -> None:
        cfg = _make_run_config(tmp_path, adc_mode=99)
        with pytest.raises(ValueError, match=r"adc_mode"):
            calibrate_main(["--config", str(cfg), "--device", "cpu"])
