"""Unit tests for the pure logic in :mod:`neurox.tools.calibrate_adc`.

Covers :mod:`._math`, the :mod:`._modes` file formats, the
:mod:`.mode_derive` derivation, and the emitted config fragments.
Synthetic-data tests only — no macro building, no probed runs.
"""

from __future__ import annotations

import tomllib
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from neurox.primitive.macro.cim import CimMacroMode
from neurox.tools.calibrate_adc._math import (
    RescaleFit,
    band_stats,
    cluster_values,
    filter_fit_samples,
    fit_linear,
    fit_rescale_through_origin,
    place_thresholds,
)
from neurox.tools.calibrate_adc._modes import (
    AdcMode,
    LayerRange,
    ModeSet,
    canonical_window,
    dump_mode_set,
    load_layer_ranges,
    load_mode_set,
)
from neurox.tools.calibrate_adc._testbench import (
    _unroll_sub_phase,
    grid_block_w,
    sample_capped_block_w,
    saturating_w,
)
from neurox.tools.calibrate_adc.mode_derive import derive_modes
from neurox.tools.calibrate_adc.rescale_fit import ModeFitResult, _fragment_lines


class TestFitRescaleThroughOrigin:
    def test_exact_recovery(self) -> None:
        """A noiseless proportional relation recovers the slope exactly."""
        code = torch.arange(0, 8, dtype=torch.float64).repeat(16)
        ideal = 2.5 * code
        fit = fit_rescale_through_origin(code, ideal)
        assert fit.rescale_factor == pytest.approx(2.5, abs=1e-12)
        assert fit.r2 == pytest.approx(1.0, abs=1e-12)
        assert fit.rmse == pytest.approx(0.0, abs=1e-12)
        assert fit.sample_num == code.numel()

    def test_least_squares_optimality_under_noise(self) -> None:
        """The fitted slope matches the closed-form LS estimate on noisy data."""
        gen = torch.Generator().manual_seed(7)
        code = torch.randint(0, 32, (512,), generator=gen).to(torch.float64)
        noise = torch.randn(512, generator=gen, dtype=torch.float64) * 0.05
        ideal = 1.3 * code + noise
        fit = fit_rescale_through_origin(code, ideal)
        expected = float((code * ideal).sum() / (code * code).sum())
        assert fit.rescale_factor == pytest.approx(expected, rel=1e-12)
        assert fit.rescale_factor == pytest.approx(1.3, abs=0.01)
        assert fit.max_abs_residual <= 0.05 * 4 + 0.05  # a few sigma of the injected noise

    def test_integer_code_tensor_accepted(self) -> None:
        """Codes arrive as int64 from the prober; the fit casts internally."""
        code = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        ideal = torch.tensor([0, 2, 4, 6], dtype=torch.int64)
        fit = fit_rescale_through_origin(code, ideal)
        assert fit.rescale_factor == pytest.approx(2.0, abs=1e-12)

    def test_all_zero_code_raises(self) -> None:
        with pytest.raises(ValueError, match="all zero"):
            fit_rescale_through_origin(torch.zeros(8), torch.ones(8))

    def test_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="numel"):
            fit_rescale_through_origin(torch.ones(4), torch.ones(5))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            fit_rescale_through_origin(torch.empty(0), torch.empty(0))


class TestBandStats:
    def test_groups_by_input_code(self) -> None:
        input_code = torch.tensor([0, 0, 1, 1, 2, 2])
        analog = torch.tensor([0.0, 0.1, 1.0, 1.2, 2.0, 2.5])
        bands = band_stats(input_code, analog, adc_input_code_range=(0, 2))
        assert [b.input_code for b in bands] == [0, 1, 2]
        assert bands[1].lo == pytest.approx(1.0)
        assert bands[1].hi == pytest.approx(1.2)
        assert bands[1].mean == pytest.approx(1.1)
        assert bands[1].count == 2

    def test_out_of_range_pairs_masked(self) -> None:
        """A code the converter resolves no tap for bounds no band."""
        input_code = torch.tensor([0, 1, 2, 9, -1])
        analog = torch.tensor([0.0, 1.0, 2.0, 9.0, -9.0])
        bands = band_stats(input_code, analog, adc_input_code_range=(0, 2))
        assert [b.input_code for b in bands] == [0, 1, 2]
        assert bands[2].hi == pytest.approx(2.0)
        assert sum(b.count for b in bands) == 3

    def test_grid_follows_the_range_bounds(self) -> None:
        """Bands cover exactly the inclusive range, one per code."""
        input_code = torch.arange(1, 4)
        analog = torch.tensor([1.0, 2.0, 3.0])
        bands = band_stats(input_code, analog, adc_input_code_range=(1, 3))
        assert [b.input_code for b in bands] == [1, 2, 3]

    def test_coverage_gap_raises(self) -> None:
        input_code = torch.tensor([0, 2])
        analog = torch.tensor([0.0, 2.0])
        with pytest.raises(ValueError, match=r"coverage gap.*\[1\]"):
            band_stats(input_code, analog, adc_input_code_range=(0, 2))

    def test_degenerate_range_raises(self) -> None:
        with pytest.raises(ValueError, match="upper"):
            band_stats(torch.tensor([0, 1]), torch.tensor([0.0, 1.0]), adc_input_code_range=(1, 1))


class TestPlaceThresholds:
    @staticmethod
    def _bands(code_num: int, *, width: float = 0.4) -> list:
        """Clean synthetic bands: band k spans [k, k + width]."""
        input_code = torch.arange(code_num).repeat_interleave(2)
        analog = torch.cat([torch.tensor([float(k), float(k) + width]) for k in range(code_num)])
        return list(band_stats(input_code, analog, adc_input_code_range=(0, code_num - 1)))

    def test_midpoint_placement(self) -> None:
        bands = self._bands(4, width=0.4)
        placement = place_thresholds(bands)
        # t[k] = (hi(k) + lo(k+1)) / 2 = (k + 0.4 + k + 1) / 2 = k + 0.7
        assert placement.thresholds == pytest.approx((0.7, 1.7, 2.7))
        assert placement.margins == pytest.approx((0.6, 0.6, 0.6))
        assert placement.min_margin == pytest.approx(0.6)
        assert placement.monotone is True

    def test_overlapping_bands_report_negative_margin(self) -> None:
        """Overlap is reported, not raised — the margin report is the gate."""
        input_code = torch.tensor([0, 0, 1, 1])
        analog = torch.tensor([0.0, 1.0, 0.8, 1.5])  # band 1 starts below band 0's hi
        placement = place_thresholds(band_stats(input_code, analog, adc_input_code_range=(0, 1)))
        assert placement.min_margin == pytest.approx(-0.2)
        assert placement.min_margin_boundary == 0

    def test_non_monotone_means_flagged(self) -> None:
        input_code = torch.tensor([0, 1, 2])
        analog = torch.tensor([2.0, 1.0, 3.0])  # mean(1) < mean(0)
        placement = place_thresholds(band_stats(input_code, analog, adc_input_code_range=(0, 2)))
        assert placement.monotone is False

    def test_single_band_raises(self) -> None:
        bands = self._bands(2)[:1]
        with pytest.raises(ValueError, match="at least 2 bands"):
            place_thresholds(bands)


class TestFitLinear:
    def test_exact_line(self) -> None:
        x = torch.arange(8, dtype=torch.float64)
        y = 3.0 * x + 1.5
        fit = fit_linear(x, y)
        assert fit.slope == pytest.approx(3.0, abs=1e-12)
        assert fit.intercept == pytest.approx(1.5, abs=1e-12)
        assert fit.r2 == pytest.approx(1.0, abs=1e-12)

    def test_degenerate_x_raises(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            fit_linear(torch.ones(4), torch.arange(4, dtype=torch.float64))


class TestClusterValues:
    def test_gap_split(self) -> None:
        values = [1.0, 1.02, 1.05, 8.0, 8.1]
        clusters = cluster_values(values, rel_tol=0.05, max_cluster_num=8)
        assert len(clusters) == 2
        assert clusters[0].member_idx == (0, 1, 2)
        assert clusters[1].member_idx == (3, 4)
        # Representative = largest member (a mode sized from it covers all).
        assert clusters[0].representative == pytest.approx(1.05)
        assert clusters[1].representative == pytest.approx(8.1)

    def test_all_equal_single_cluster(self) -> None:
        clusters = cluster_values([32.0] * 10, rel_tol=0.05, max_cluster_num=4)
        assert len(clusters) == 1
        assert clusters[0].representative == pytest.approx(32.0)
        assert len(clusters[0].member_idx) == 10

    def test_max_cluster_merging(self) -> None:
        """Over-split inputs merge at the smallest gaps until the cap holds."""
        values = [1.0, 2.0, 2.05, 4.0]
        clusters = cluster_values(values, rel_tol=0.001, max_cluster_num=2)
        assert len(clusters) == 2
        # 2.0/2.05 is the smallest gap, then 1.0 joins {2.0, 2.05} (gap 1.0
        # vs 1.95 to 4.0), leaving {1.0, 2.0, 2.05} and {4.0}.
        assert clusters[0].member_idx == (0, 1, 2)
        assert clusters[1].member_idx == (3,)

    def test_indices_partition_input(self) -> None:
        gen = torch.Generator().manual_seed(3)
        values = torch.rand(50, generator=gen).tolist()
        clusters = cluster_values(values, rel_tol=0.02, max_cluster_num=6)
        seen = sorted(i for c in clusters for i in c.member_idx)
        assert seen == list(range(50))

    def test_invalid_args_raise(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            cluster_values([], rel_tol=0.1, max_cluster_num=2)
        with pytest.raises(ValueError, match="rel_tol"):
            cluster_values([1.0], rel_tol=0.0, max_cluster_num=2)
        with pytest.raises(ValueError, match="max_cluster_num"):
            cluster_values([1.0], rel_tol=0.1, max_cluster_num=0)


class TestFilterFitSamples:
    def test_both_filters_and_counts(self) -> None:
        """Range and saturation drops apply jointly; counts report per cause."""
        code = torch.tensor([1, 7, 3, 7, 2], dtype=torch.int64)
        adc_input_code = torch.tensor([1, 9, 10, 12, 2], dtype=torch.int64)
        sel = filter_fit_samples(code, adc_input_code, adc_input_code_range=(0, 9), top_code=7)
        # idx 1: saturated only; idx 2: out of range only; idx 3: both.
        assert sel.keep.tolist() == [True, False, False, False, True]
        assert sel.range_dropped_num == 2
        assert sel.saturated_num == 2

    def test_range_bounds_inclusive(self) -> None:
        """Both endpoints of the input code range stay in the design domain."""
        code = torch.tensor([1, 2, 3], dtype=torch.int64)
        adc_input_code = torch.tensor([0, 4, 5], dtype=torch.int64)
        sel = filter_fit_samples(code, adc_input_code, adc_input_code_range=(0, 4), top_code=7)
        assert sel.keep.tolist() == [True, True, False]
        assert sel.range_dropped_num == 1
        assert sel.saturated_num == 0

    def test_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="numel"):
            filter_fit_samples(torch.ones(3), torch.ones(4), adc_input_code_range=(0, 1), top_code=3)


class TestLayerRangeMapping:
    @staticmethod
    def _write(tmp_path: Path, text: str) -> Path:
        path = tmp_path / "layer_ranges.toml"
        path.write_text(text)
        return path

    def test_parse(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            '"enc.0.q" = { range = [-12.5, 12.5] }\n"enc.0.ffn" = { range = [0.0, 31.0] }\n',
        )
        mapping = load_layer_ranges(path)
        assert mapping == {
            "enc.0.q": LayerRange(range=(-12.5, 12.5)),
            "enc.0.ffn": LayerRange(range=(0.0, 31.0)),
        }
        assert mapping["enc.0.q"].signed is True
        assert mapping["enc.0.ffn"].signed is False

    def test_empty_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least one layer"):
            load_layer_ranges(self._write(tmp_path, ""))

    def test_non_positive_upper_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"hi \(0.0\) > 0"):
            load_layer_ranges(self._write(tmp_path, '"a" = { range = [-1.0, 0.0] }\n'))

    def test_inverted_range_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"lo .* <= hi"):
            load_layer_ranges(self._write(tmp_path, '"a" = { range = [4.0, 1.0] }\n'))

    def test_non_finite_range_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="finite"):
            load_layer_ranges(self._write(tmp_path, '"a" = { range = [0.0, inf] }\n'))

    def test_scalar_range_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="2-element array"):
            load_layer_ranges(self._write(tmp_path, '"a" = { range = 1.0 }\n'))

    def test_unknown_key_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="exactly"):
            load_layer_ranges(self._write(tmp_path, '"a" = { range = [0.0, 1.0], signed = true }\n'))


class TestCanonicalWindow:
    def test_non_negative_range_maps_unsigned(self) -> None:
        assert canonical_window(LayerRange(range=(0.0, 12.2))) == (0, 13)

    def test_signed_range_maps_mid_zero(self) -> None:
        """A mid-zero window is asymmetric, so its top drives the size."""
        assert canonical_window(LayerRange(range=(-12.5, 12.5))) == (-14, 13)
        assert canonical_window(LayerRange(range=(-8.0, 7.0))) == (-8, 7)

    def test_negative_side_can_size_the_window(self) -> None:
        window = canonical_window(LayerRange(range=(-20.0, 1.0)))
        assert window == (-20, 19)

    def test_window_covers_the_range(self) -> None:
        for lo, hi in ((-12.5, 12.5), (-20.0, 1.0), (0.0, 7.5), (-0.5, 0.5)):
            lower, upper = canonical_window(LayerRange(range=(lo, hi)))
            assert lower <= lo and upper >= hi


class TestModeSet:
    @staticmethod
    def _mode_set() -> ModeSet:
        return ModeSet(
            modes=(
                AdcMode(quantization_mode=0, quantization_input_range=(0, 31), layer_num=1),
                AdcMode(quantization_mode=1, quantization_input_range=(-13, 12), layer_num=2),
            ),
            layers={"enc.0.ffn": 0, "enc.0.q": 1, "enc.0.k": 1},
        )

    def test_round_trip(self, tmp_path: Path) -> None:
        """dump_mode_set -> load_mode_set reproduces the spec exactly."""
        original = self._mode_set()
        path = tmp_path / "modes.toml"
        path.write_text(dump_mode_set(original))
        assert load_mode_set(path) == original

    def test_non_canonical_window_raises(self) -> None:
        """The mode set carries canonical windows only — no symmetric shape."""
        with pytest.raises(ValueError, match="canonical"):
            AdcMode(quantization_mode=0, quantization_input_range=(-8, 8), layer_num=1)

    def test_non_contiguous_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="contiguous"):
            ModeSet(
                modes=(AdcMode(quantization_mode=1, quantization_input_range=(0, 1), layer_num=1),),
                layers={"a": 1},
            )

    def test_unknown_layer_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown quantization_mode"):
            ModeSet(
                modes=(AdcMode(quantization_mode=0, quantization_input_range=(0, 1), layer_num=1),),
                layers={"a": 0, "b": 3},
            )

    def test_layer_num_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="layer_num"):
            ModeSet(
                modes=(AdcMode(quantization_mode=0, quantization_input_range=(0, 1), layer_num=2),),
                layers={"a": 0},
            )

    def test_load_rejects_scalar_window(self, tmp_path: Path) -> None:
        path = tmp_path / "modes.toml"
        path.write_text(
            '[[modes]]\nquantization_mode = 0\nquantization_input_range = 7\nlayer_num = 1\n\n[layers]\n"a" = 0\n'
        )
        with pytest.raises(ValueError, match="2-element array"):
            load_mode_set(path)

    def test_load_rejects_extra_top_level_key(self, tmp_path: Path) -> None:
        path = tmp_path / "modes.toml"
        path.write_text(dump_mode_set(self._mode_set()) + "\n[extra]\nx = 1\n")
        with pytest.raises(ValueError, match="top-level keys"):
            load_mode_set(path)


class TestStimulusCeilBlocking:
    """The stimulus generators + sub-phase mirror ceil-divide + mask, so a
    non-divisible row/active-row geometry (the paper 256/9) is covered by a
    short final block instead of raising on a divisibility assertion.
    """

    # Paper sub-array geometry: 256 % 9 != 0.
    _COL, _ROW, _ACTIVE = 4, 256, 9

    def _gen(self) -> torch.Generator:
        return torch.Generator().manual_seed(0)

    def test_grid_block_w_non_divisible_shape(self) -> None:
        w = grid_block_w(col_num=self._COL, row_num=self._ROW, active_row_num=self._ACTIVE, m_max=self._ACTIVE)
        assert tuple(w.shape) == (self._COL, 1, self._ROW)
        assert set(w.unique().tolist()) <= {-1, 0, 1}

    def test_capped_block_w_non_divisible_shape(self) -> None:
        w = sample_capped_block_w(
            self._gen(), col_num=self._COL, row_num=self._ROW, active_row_num=self._ACTIVE, cap=self._ACTIVE
        )
        assert tuple(w.shape) == (self._COL, 1, self._ROW)
        assert set(w.unique().tolist()) <= {-1, 0, 1}

    def test_saturating_w_non_divisible_shape(self) -> None:
        w = saturating_w(col_num=self._COL, row_num=self._ROW, active_row_num=self._ACTIVE)
        assert tuple(w.shape) == (self._COL, 1, self._ROW)

    def test_grid_block_w_divisible_leading_count(self) -> None:
        """At a divisible geometry every full block programs exactly ``count`` leading cells."""
        col_num, row_num, active = 3, 12, 4  # 12 % 4 == 0, 3 phases
        w = grid_block_w(col_num=col_num, row_num=row_num, active_row_num=active, m_max=active, col_stride=1)
        blocks = w.reshape(col_num, row_num // active, active)
        # Column c (grid_idx c, offset 0) programs c % (active + 1) leading cells per block.
        for c in range(col_num):
            for p in range(row_num // active):
                assert int(blocks[c, p].abs().sum()) == c % (active + 1)

    def test_unroll_mirror_covers_every_row_exactly_once(self) -> None:
        """Ceil sub-phase count: each real row is live in exactly one plane."""
        x = torch.ones((2, self._ROW), dtype=torch.long)
        planes = _unroll_sub_phase(x, row_num=self._ROW, max_active_num=self._ACTIVE, inst_rank=0)
        p_num = -(-self._ROW // self._ACTIVE)  # ceil(256 / 9) == 29
        assert tuple(planes.shape) == (2, p_num, self._ROW)
        # Sum over the P axis: every (batch, row) is active in exactly one plane.
        assert torch.equal(planes.sum(dim=1), x)

    def test_unroll_mirror_divisible_equals_exact_quotient(self) -> None:
        x = torch.ones((12,), dtype=torch.long)  # trailing row_num == 12
        planes = _unroll_sub_phase(x, row_num=12, max_active_num=4, inst_rank=0)
        assert planes.shape[-2] == 3  # 12 / 4 exact
        assert torch.equal(planes.sum(dim=-2), x)


class TestDeriveModes:
    def test_unsigned_first_global_mode_enumeration(self) -> None:
        """Unsigned group enumerates first; clusters ascend within a group."""
        layer_ranges = {
            "a.signed.small": LayerRange(range=(-1.0, 1.0)),
            "b.signed.large": LayerRange(range=(-8.0, 8.0)),
            "c.unsigned.only": LayerRange(range=(0.0, 4.0)),
        }
        modes, layer_to_mode = derive_modes(layer_ranges, rel_tol=0.05, max_modes_per_group=4)
        assert [(m.quantization_mode, m.signed) for m in modes] == [(0, False), (1, True), (2, True)]
        assert layer_to_mode == {"c.unsigned.only": 0, "a.signed.small": 1, "b.signed.large": 2}
        assert modes[0].quantization_input_range == (0, 4)
        assert modes[1].quantization_input_range == (-2, 1)
        assert modes[2].quantization_input_range == (-9, 8)

    def test_mode_window_covers_every_member(self) -> None:
        """A cluster's window covers each member layer's design range."""
        layer_ranges = {
            "s1": LayerRange(range=(-4.0, 3.0)),
            "s2": LayerRange(range=(-4.2, 3.5)),
            "s3": LayerRange(range=(-3.0, 2.0)),
        }
        modes, layer_to_mode = derive_modes(layer_ranges, rel_tol=0.5, max_modes_per_group=1)
        assert len(modes) == 1
        lower, upper = modes[0].quantization_input_range
        for name, spec in layer_ranges.items():
            assert layer_to_mode[name] == 0
            assert lower <= spec.range[0] and upper >= spec.range[1]

    def test_single_group_input(self) -> None:
        """An empty shape group contributes no mode; enumeration stays dense."""
        layer_ranges = {
            "x": LayerRange(range=(-2.0, 2.0)),
            "y": LayerRange(range=(-2.1, 2.1)),
        }
        modes, layer_to_mode = derive_modes(layer_ranges, rel_tol=0.3, max_modes_per_group=4)
        assert [(m.quantization_mode, m.signed) for m in modes] == [(0, True)]
        assert layer_to_mode == {"x": 0, "y": 0}

    def test_deterministic_under_input_order(self) -> None:
        """The derivation is invariant to the mapping's insertion order."""
        forward = {
            "u1": LayerRange(range=(0.0, 3.0)),
            "s1": LayerRange(range=(-1.0, 1.0)),
            "u2": LayerRange(range=(0.0, 9.0)),
            "s2": LayerRange(range=(-1.05, 1.05)),
        }
        backward = dict(reversed(forward.items()))
        assert derive_modes(forward, rel_tol=0.05, max_modes_per_group=4) == derive_modes(
            backward, rel_tol=0.05, max_modes_per_group=4
        )


class TestRescaleFragment:
    """The emitted fragment must paste verbatim into a macro config."""

    @staticmethod
    def _result(quantization_mode: int, *, rescale_factor: float) -> ModeFitResult:
        return ModeFitResult(
            quantization_mode=quantization_mode,
            adc_bits=3,
            quantization_input_range=(-8, 7),
            adc_input_code_range=(0, 7),
            fit=RescaleFit(
                rescale_factor=rescale_factor,
                sample_num=4,
                r2=1.0,
                rmse=0.0,
                max_abs_residual=0.0,
            ),
            total_num=4,
            range_dropped_num=0,
            saturated_num=0,
            code=torch.arange(4),
            ideal_code=torch.arange(4, dtype=torch.float64),
        )

    def test_fragment_matches_the_mode_config_schema(self) -> None:
        """Every emitted table builds a CimMacroMode: same keys, no extras."""
        text = "\n".join(_fragment_lines([self._result(0, rescale_factor=1.25)]))
        tables = tomllib.loads(text)["modes"]
        assert len(tables) == 1
        assert set(tables[0]) == {f.name for f in fields(CimMacroMode)}
        mode = CimMacroMode(
            quantization_input_range=tuple(tables[0]["quantization_input_range"]),
            adc_input_code_range=tuple(tables[0]["adc_input_code_range"]),
            max_bits_rescale_factor=tables[0]["max_bits_rescale_factor"],
        )
        assert mode.quantization_input_range == (-8, 7)
        assert mode.max_bits_rescale_factor == pytest.approx(1.25)

    def test_tables_emit_in_mode_order(self) -> None:
        """Table position is the mode index, so the order must be sorted."""
        results = [self._result(1, rescale_factor=2.0), self._result(0, rescale_factor=1.0)]
        tables = tomllib.loads("\n".join(_fragment_lines(results)))["modes"]
        assert [t["max_bits_rescale_factor"] for t in tables] == pytest.approx([1.0, 2.0])
