"""Unit tests for ADC-probe sampling and distribution statistics."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.nn import Module

from neurox.common import PolicyBase
from neurox.primitive.analog import AdcProber
from neurox.primitive.analog.current_adc import IadcRecord
from neurox.tools.calibrate_adc import AdcProbeData, _report, load_adc_probe_data, save_adc_probe_data
from neurox.tools.calibrate_adc._cli import parse_args
from neurox.tools.calibrate_adc._collection import (
    _ideal_value_range,
    _require_all_policy_toggles_off,
    _run_paired,
    ideal_value_counts,
    plan_target_batches,
)
from neurox.tools.calibrate_adc._report import summarize_probe
from neurox.tools.calibrate_adc._schema import ActiveRowSelection
from neurox.tools.calibrate_adc._stimulus import TargetStimulusSampler, feasible_ideal_values, sample_sparse_inputs


class _LeafPolicy(PolicyBase):
    noise: bool


class _MacroPolicy(PolicyBase):
    leaf: _LeafPolicy
    solve_chunk_size: int


class _PhysicalMacro(Module):
    adc_bits = 3
    inst_shape: tuple[int, ...] = ()

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("anchor", torch.empty(()))

    def vec_mat_mul(self, x: Tensor, *, quantization_mode: int, adc_active_bits: int) -> Tensor:
        assert quantization_mode == 0
        assert adc_active_bits == self.adc_bits
        AdcProber.submit(IadcRecord(i_in__uA=x.transpose(0, 1)))
        return x

    def restore_adc_layout(self, value: Tensor) -> Tensor:
        return value.transpose(0, 1)


class _IdealMacro:
    def vec_mat_mul(self, x: Tensor, *, quantization_mode: int, adc_active_bits: int) -> Tensor:
        assert quantization_mode == 0
        assert adc_active_bits == 0
        return x.to(torch.int64)


def test_sparse_input_sampling_never_exceeds_max_active_num() -> None:
    generator = torch.Generator().manual_seed(4)
    x = sample_sparse_inputs(
        torch.tensor((0, 1, 2, 3), dtype=torch.int64),
        batch_size=64,
        input_num=31,
        max_active_num=7,
        active_row_selection=ActiveRowSelection.SCATTERED,
        generator=generator,
    )
    assert tuple(x.shape) == (64, 31)
    assert bool(((x != 0).sum(dim=-1) <= 7).all())
    assert set(x.unique().tolist()) <= {0, 1, 2, 3}


def test_sparse_input_sampling_uses_distinct_positions() -> None:
    generator = torch.Generator().manual_seed(2)
    x = sample_sparse_inputs(
        torch.tensor((1,), dtype=torch.int64),
        batch_size=16,
        input_num=13,
        max_active_num=5,
        active_row_selection=ActiveRowSelection.SCATTERED,
        generator=generator,
    )
    assert torch.equal((x != 0).sum(dim=-1), torch.full((16,), 5))


def test_sparse_input_sampling_can_select_one_contiguous_window() -> None:
    generator = torch.Generator().manual_seed(3)
    x = sample_sparse_inputs(
        torch.tensor((1,), dtype=torch.int64),
        batch_size=16,
        input_num=13,
        max_active_num=5,
        active_row_selection=ActiveRowSelection.CONTIGUOUS,
        generator=generator,
    )
    positions = (x != 0).nonzero().view(16, 5, 2)[:, :, 1]
    assert bool(((positions[:, 1:] - positions[:, :-1]) == 1).all())


@pytest.mark.parametrize("target", [-5, 0, 5])
@pytest.mark.parametrize("selection", tuple(ActiveRowSelection))
def test_target_stimulus_sampling_realizes_every_requested_dot(
    target: int,
    selection: ActiveRowSelection,
) -> None:
    device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(5)
    sampler = TargetStimulusSampler(
        input_values=(0, 1, 2),
        weight_values=(-2, -1, 0, 1, 2),
        active_num=3,
        input_num=11,
        output_num=5,
        target_ideal_values=(-5, 0, 5),
        active_row_selection=selection,
        device=device,
        generator=generator,
    )

    w, x = sampler.sample(target, batch_size=64)
    ideal = (w * x.squeeze(0).unsqueeze(-1)).sum(dim=1)

    assert tuple(w.shape) == (64, 11, 5)
    assert tuple(x.shape) == (1, 64, 11)
    assert w.device == device
    assert x.device == device
    assert torch.equal(ideal, torch.full_like(ideal, target))
    assert bool(((x != 0).sum(dim=-1) <= 3).all())


def test_target_stimulus_sampling_rejects_an_impossible_target() -> None:
    with pytest.raises(ValueError, match="feasible support"):
        TargetStimulusSampler(
            input_values=(0, 1),
            weight_values=(-1, 1),
            active_num=2,
            input_num=4,
            output_num=2,
            target_ideal_values=(3,),
            active_row_selection=ActiveRowSelection.SCATTERED,
            device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(0),
        )


def test_feasible_ideal_values_exclude_integers_inside_the_numeric_envelope() -> None:
    support = feasible_ideal_values(
        input_values=(0, 1, 2, 3),
        weight_values=(-3, -2, -1, 0, 1, 2, 3),
        active_num=9,
    )

    assert support[0] == -81
    assert support[-1] == 81
    assert len(support) == 157
    assert -80 not in support
    assert 80 not in support


def test_target_stimulus_sampling_preserves_the_conditioned_input_distribution() -> None:
    sampler = TargetStimulusSampler(
        input_values=(0, 1),
        weight_values=(-1, 1),
        active_num=2,
        input_num=2,
        output_num=1,
        target_ideal_values=(0,),
        active_row_selection=ActiveRowSelection.CONTIGUOUS,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(8),
    )

    _, x = sampler.sample(0, batch_size=20_000)

    all_zero_fraction = float((x == 0).all(dim=-1).to(torch.float64).mean())
    assert all_zero_fraction == pytest.approx(2.0 / 3.0, abs=0.02)


def test_paired_run_uses_one_layout_restore_for_probe_and_code_positions() -> None:
    x = torch.tensor(((1, 2, 3), (4, 5, 6)), dtype=torch.int64)
    name, input_value, ideal_value = _run_paired(
        _PhysicalMacro(),
        _IdealMacro(),
        x,
        quantization_mode=0,
    )

    assert name == "i_in__uA"
    assert torch.equal(input_value, x.flatten().to(torch.float64))
    assert torch.equal(ideal_value, x.flatten())


def test_probe_result_round_trips_through_safe_data_file(tmp_path: Path) -> None:
    result = AdcProbeData(
        input_name="i_in__uA",
        input_value=torch.tensor((0.25, 0.75), dtype=torch.float32),
        ideal_value=torch.tensor((-1, 2), dtype=torch.int64),
        ideal_value_support=(-9, -7, -1, 0, 2, 9),
    )
    path = tmp_path / "probe.pt"

    save_adc_probe_data(result, path)
    loaded = load_adc_probe_data(path)

    assert loaded.input_name == result.input_name
    assert torch.equal(loaded.input_value, result.input_value)
    assert torch.equal(loaded.ideal_value, result.ideal_value)
    assert loaded.ideal_value_support == result.ideal_value_support


def test_probe_rejects_an_enabled_nonideality_toggle() -> None:
    _require_all_policy_toggles_off(_MacroPolicy(leaf=_LeafPolicy(noise=False), solve_chunk_size=32))
    with pytest.raises(ValueError, match=r"policy\.leaf\.noise"):
        _require_all_policy_toggles_off(_MacroPolicy(leaf=_LeafPolicy(noise=True), solve_chunk_size=32))


def test_theoretical_ideal_range_uses_value_domains_and_active_limit() -> None:
    assert _ideal_value_range((-3, 3), (0, 3), 9) == (-81, 81)
    assert _ideal_value_range((0, 7), (-2, 3), 4) == (-56, 84)


def test_target_batch_plan_fills_every_signed_ideal_value_deficit() -> None:
    random_data = AdcProbeData(
        input_name="i_in__uA",
        input_value=torch.zeros(8),
        ideal_value=torch.tensor((-2, -2, -1, -1, -1, -1, -1, 0)),
        ideal_value_support=(-2, -1, 0, 1, 2),
    )

    batches = plan_target_batches(
        random_data,
        min_samples_per_ideal_value=5,
        samples_per_batch=4,
    )

    assert torch.equal(ideal_value_counts(random_data), torch.tensor((2, 5, 1, 0, 0)))
    assert batches == {-2: 1, 0: 1, 1: 2, 2: 2}


def test_cli_uses_one_output_root_and_default_coverage_threshold() -> None:
    args = parse_args(("--config", "run.toml", "--output-dir", "logs"))

    assert args.output_dir == Path("logs")
    assert args.min_samples_per_ideal_value == 65_536
    assert not hasattr(args, "log_dir")
    assert not hasattr(args, "output")


def test_summary_keeps_every_observed_ideal_value() -> None:
    ideal = torch.tensor((-2, -1, -1, 0, 0, 0, 2), dtype=torch.int64)
    signal = torch.tensor((-2.1, -1.2, -0.8, -0.1, 0.0, 0.1, 2.2), dtype=torch.float64)
    summary = summarize_probe(ideal, signal)

    assert [cluster.ideal_value for cluster in summary.clusters] == [-2, -1, 0, 2]
    assert summary.observed_ideal_range == (-2, 2)


def test_summary_reports_complete_cluster_statistics() -> None:
    ideal = torch.tensor((3, 3, 3, 3), dtype=torch.int64)
    signal = torch.tensor((1.0, 2.0, 3.0, 4.0), dtype=torch.float64)
    cluster = summarize_probe(ideal, signal).clusters[0].signal

    assert cluster.count == 4
    assert cluster.minimum == pytest.approx(1.0)
    assert cluster.mean == pytest.approx(2.5)
    assert cluster.median == pytest.approx(2.5)
    assert cluster.maximum == pytest.approx(4.0)
    assert cluster.std == pytest.approx(float(signal.std(unbiased=False)))


def test_summary_rejects_misaligned_or_empty_streams() -> None:
    with pytest.raises(ValueError, match="numel"):
        summarize_probe(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError, match="at least one"):
        summarize_probe(torch.empty(0), torch.empty(0))


def test_large_sample_quantiles_keep_linear_interpolation(monkeypatch: pytest.MonkeyPatch) -> None:
    value = torch.tensor((9.0, 1.0, 5.0, 3.0), dtype=torch.float64)
    monkeypatch.setattr(_report, "_TORCH_QUANTILE_MAX_NUMEL", 3)

    actual = _report._quantiles(value)
    expected = torch.quantile(value, torch.tensor(_report._QUANTILES, dtype=torch.float64))

    assert torch.equal(actual, expected)
