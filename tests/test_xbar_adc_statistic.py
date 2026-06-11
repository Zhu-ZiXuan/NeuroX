"""Unit tests for ``neurox/tools/xbar_adc/statistic.py``."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

from neurox.common import dataclass_from_file
from neurox.tools.xbar_adc.statistic import (
    RangeCandidate,
    build_candidates,
    collect_statistics,
    log_statistics,
    plot_statistics,
)
from neurox.tools.xbar_adc.statistic import (
    main as statistic_main,
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


def _make_run_config(tmp_path: Path, *, distribution: str | None = None) -> Path:
    """Write a minimal statistic-run TOML pointing at the chip preset."""
    cfg_path = tmp_path / "run.toml"
    dist_line = f'distribution = "{distribution}"\n' if distribution else ""
    cfg_path.write_text(
        f"""[xbar]
_neurox_use = "{XBAR_CONFIG}:xbar"

[workload]
{dist_line}weight_samples = 8
input_samples_per_weight = 8
batch_size = 4
seed = 0

[statistic]
max_clip_rate_exp = 2

[plot]
bits = 4
bins_per_code = 4
"""
    )
    return cfg_path


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestRangeCandidate:
    def test_max_abs_derivations(self) -> None:
        c = RangeCandidate(clip_rate_exp=None, a__V=0.5, observed_clip_rate=0.0)
        assert c.label == "max_abs"
        assert c.nominal_clip_rate == 0.0
        assert c.filename == "spotlight_max_abs.png"

    def test_exp_2_derivations(self) -> None:
        c = RangeCandidate(clip_rate_exp=2, a__V=0.3, observed_clip_rate=0.01)
        assert c.label == "p99"
        assert c.nominal_clip_rate == pytest.approx(1e-2)
        assert c.filename == "spotlight_exp2.png"

    def test_exp_3_derivations(self) -> None:
        c = RangeCandidate(clip_rate_exp=3, a__V=0.4, observed_clip_rate=0.001)
        assert c.label == "p99.9"
        assert c.nominal_clip_rate == pytest.approx(1e-3)
        assert c.filename == "spotlight_exp3.png"

    def test_exp_12_no_float_drift(self) -> None:
        c = RangeCandidate(clip_rate_exp=12, a__V=0.49, observed_clip_rate=0.0)
        assert c.label == "p99.9999999999"
        assert c.filename == "spotlight_exp12.png"


# ---------------------------------------------------------------------------
# build_candidates math
# ---------------------------------------------------------------------------


class TestBuildCandidates:
    def test_uniform_observed_clip_rate(self) -> None:
        abs_diff = torch.linspace(0.0, 1.0, steps=100_000, dtype=torch.float64)
        cands = build_candidates(abs_diff, max_clip_rate_exp=3)
        labels = [c.label for c in cands]
        assert labels[0] == "max_abs"
        assert "p99" in labels
        assert "p99.9" in labels

        p99 = next(c for c in cands if c.label == "p99")
        assert p99.a__V == pytest.approx(0.99, abs=1e-3)
        assert p99.observed_clip_rate == pytest.approx(0.01, abs=2e-3)
        assert p99.nominal_clip_rate == pytest.approx(0.01)

        max_abs = cands[0]
        assert max_abs.a__V == pytest.approx(1.0, abs=1e-6)
        assert max_abs.observed_clip_rate == 0.0

    def test_insufficient_samples_raises(self) -> None:
        abs_diff = torch.linspace(0.0, 1.0, steps=100, dtype=torch.float64)
        with pytest.raises(ValueError, match=r"insufficient samples"):
            build_candidates(abs_diff, max_clip_rate_exp=4)

    def test_empty_input_raises(self) -> None:
        empty = torch.empty(0, dtype=torch.float64)
        with pytest.raises(ValueError, match=r"no ADC inputs captured"):
            build_candidates(empty, max_clip_rate_exp=2)


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke_stats(xbar_cfg):
    return collect_statistics(
        xbar_config=xbar_cfg,
        distribution_path=None,
        weight_samples=8,
        input_samples_per_weight=8,
        batch_size=4,
        max_clip_rate_exp=2,
        seed=0,
        device=CPU,
    )


class TestCollectStatistics:
    def test_returns_populated_object(self, smoke_stats) -> None:
        s = smoke_stats
        assert s.v_pos__V.numel() == s.v_neg__V.numel() == s.v_diff__V.numel()
        assert s.v_pos__V.numel() > 0
        assert s.captured_count == s.v_pos__V.numel()
        assert len(s.candidates) == 2
        assert s.candidates[0].label == "max_abs"
        assert s.candidates[0].clip_rate_exp is None
        assert s.candidates[1].label == "p99"
        assert s.candidates[1].clip_rate_exp == 2
        assert s.distribution_source == "uniform"
        # adc_instance_count = batch_size * n_groups = 4 * (col_num / ref_group_size) = 4 * 4 = 16.
        assert s.adc_instance_count == 16

    def test_rejects_bad_args(self, xbar_cfg) -> None:
        with pytest.raises(ValueError, match=r"weight_samples"):
            collect_statistics(
                xbar_config=xbar_cfg,
                distribution_path=None,
                weight_samples=0,
                input_samples_per_weight=4,
                batch_size=4,
                max_clip_rate_exp=2,
                seed=0,
                device=CPU,
            )

    def test_rejects_non_multiple_batch(self, xbar_cfg) -> None:
        with pytest.raises(ValueError, match=r"multiple of batch_size"):
            collect_statistics(
                xbar_config=xbar_cfg,
                distribution_path=None,
                weight_samples=5,
                input_samples_per_weight=4,
                batch_size=4,
                max_clip_rate_exp=2,
                seed=0,
                device=CPU,
            )

    def test_insufficient_samples_raises(self, xbar_cfg) -> None:
        with pytest.raises(ValueError, match=r"insufficient samples"):
            collect_statistics(
                xbar_config=xbar_cfg,
                distribution_path=None,
                weight_samples=4,
                input_samples_per_weight=4,
                batch_size=4,
                max_clip_rate_exp=4,
                seed=0,
                device=CPU,
            )


# ---------------------------------------------------------------------------
# Logger output
# ---------------------------------------------------------------------------


class TestLogStatistics:
    def test_emits_expected_headings(self, smoke_stats, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="neurox.tools.xbar_adc.statistic"):
            log_statistics(smoke_stats)

        text = caplog.text
        assert "ADC input statistics and range recommendation" in text
        assert "v_pos__V:" in text
        assert "v_neg__V:" in text
        assert "v_diff__V:" in text
        assert "|v_diff__V|:" in text
        assert "ADC input range candidates:" in text
        assert "max_abs" in text
        assert "p99" in text
        assert "adc_instance_count" in text


# ---------------------------------------------------------------------------
# Plot smoke
# ---------------------------------------------------------------------------


class TestPlotStatistics:
    def test_overview_plus_spotlights(self, smoke_stats, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        plot_statistics(smoke_stats, out_dir, bits=4, bins_per_code=4)
        assert (out_dir / "overview.png").exists()
        for c in smoke_stats.candidates:
            assert (out_dir / c.filename).exists()
            assert (out_dir / c.filename).stat().st_size > 0

    def test_one_bit_works(self, smoke_stats, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        plot_statistics(smoke_stats, out_dir, bits=1, bins_per_code=1)
        assert (out_dir / "overview.png").exists()
        assert (out_dir / "spotlight_max_abs.png").exists()

    def test_zero_bits_rejected(self, smoke_stats, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"bits = 0 invalid"):
            plot_statistics(smoke_stats, tmp_path / "out", bits=0, bins_per_code=4)

    def test_thirteen_bits_rejected(self, smoke_stats, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"bits = 13 invalid"):
            plot_statistics(smoke_stats, tmp_path / "out", bits=13, bins_per_code=4)

    def test_zero_bins_per_code_rejected(self, smoke_stats, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"bins_per_code = 0 invalid"):
            plot_statistics(smoke_stats, tmp_path / "out", bits=4, bins_per_code=0)


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_smoke(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        cfg = _make_run_config(tmp_path)
        with caplog.at_level(logging.INFO, logger="neurox.tools.xbar_adc.statistic"):
            assert statistic_main(["--config", str(cfg), "--device", "cpu"]) == 0
        assert "ADC input range candidates:" in caplog.text
        assert "resolved device: cpu" in caplog.text

    def test_cli_with_distribution(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        dist_path = tmp_path / "dist.toml"
        dist_path.write_text(
            "[w]\nvalues = [-1, 0, 1]\nprobs = [0.25, 0.5, 0.25]\n"
            "[x]\nvalues = [0, 1]\nprobs = [0.7, 0.3]\n"
        )
        cfg = _make_run_config(tmp_path, distribution=str(dist_path))
        with caplog.at_level(logging.INFO, logger="neurox.tools.xbar_adc.statistic"):
            assert statistic_main(["--config", str(cfg), "--device", "cpu"]) == 0
        assert str(dist_path) in caplog.text

    def test_cli_plot_dir_emits_overview_and_spotlights(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        out_dir = tmp_path / "out"
        cfg = _make_run_config(tmp_path)
        with caplog.at_level(logging.INFO, logger="neurox.tools.xbar_adc.statistic"):
            assert (
                statistic_main(["--config", str(cfg), "--plot-dir", str(out_dir), "--device", "cpu"])
                == 0
            )
        assert (out_dir / "overview.png").exists()
        assert (out_dir / "spotlight_max_abs.png").exists()
        assert (out_dir / "spotlight_exp2.png").exists()
        assert "wrote overview + 2 spotlights" in caplog.text
