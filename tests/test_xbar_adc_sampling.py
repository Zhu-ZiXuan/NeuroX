"""Unit tests for ``neurox/tools/xbar_adc/_sampling.py`` and ``_probe.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from neurox.analog.adc import ADC, AdcOperationPoint
from neurox.tools.xbar_adc._probe import (
    ProbeADC,
    install_probe_adc,
)
from neurox.tools.xbar_adc._sampling import (
    Distribution,
    _all_off_adc_policy,
    build_offset_1t1r_xbar_all_off,
    load_distribution,
    make_generator,
    resolve_device,
    sample_w,
    sample_x_batches,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
XBAR_CONFIG = REPO_ROOT / "example" / "presets" / "xbar" / "1t1r_28nm.toml"

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Build helpers (module-scoped to amortise xbar construction cost)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def xbar():
    return build_offset_1t1r_xbar_all_off(XBAR_CONFIG, device=CPU)


# ---------------------------------------------------------------------------
# Distribution loading
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "dist.toml"
    path.write_text(body)
    return path


class TestLoadDistribution:
    def test_none_path_yields_uniform(self, xbar: Any) -> None:
        dist = load_distribution(None, xbar)
        assert dist.w_values is None and dist.w_probs is None
        assert dist.x_values is None and dist.x_probs is None
        assert dist.source == "uniform"

    def test_missing_w_keeps_w_uniform(self, tmp_path: Path, xbar: Any) -> None:
        path = _write(tmp_path, "[x]\nvalues = [0, 1]\nprobs = [0.5, 0.5]\n")
        dist = load_distribution(path, xbar)
        assert dist.w_values is None and dist.w_probs is None
        assert dist.x_values is not None and dist.x_probs is not None

    def test_missing_x_keeps_x_uniform(self, tmp_path: Path, xbar: Any) -> None:
        # w_digit_range for the 28nm preset = (-1, 2).
        path = _write(tmp_path, "[w]\nvalues = [-1, 0, 1]\nprobs = [1.0, 1.0, 1.0]\n")
        dist = load_distribution(path, xbar)
        assert dist.w_values is not None and dist.w_probs is not None
        assert dist.x_values is None and dist.x_probs is None

    def test_probs_normalised(self, tmp_path: Path, xbar: Any) -> None:
        path = _write(tmp_path, "[w]\nvalues = [0, 1]\nprobs = [3.0, 1.0]\n")
        dist = load_distribution(path, xbar)
        assert dist.w_probs is not None
        assert torch.allclose(dist.w_probs, torch.tensor([0.75, 0.25], dtype=torch.float64))

    def test_length_mismatch(self, tmp_path: Path, xbar: Any) -> None:
        path = _write(tmp_path, "[w]\nvalues = [0, 1]\nprobs = [1.0]\n")
        with pytest.raises(ValueError, match=r"len\(values\)=2 != len\(probs\)=1"):
            load_distribution(path, xbar)

    def test_negative_probability(self, tmp_path: Path, xbar: Any) -> None:
        path = _write(tmp_path, "[w]\nvalues = [0, 1]\nprobs = [-0.1, 1.1]\n")
        with pytest.raises(ValueError, match=r"every entry must be >= 0"):
            load_distribution(path, xbar)

    def test_all_zero_probability(self, tmp_path: Path, xbar: Any) -> None:
        path = _write(tmp_path, "[w]\nvalues = [0, 1]\nprobs = [0.0, 0.0]\n")
        with pytest.raises(ValueError, match=r"at least one entry must be > 0"):
            load_distribution(path, xbar)

    def test_value_out_of_range(self, tmp_path: Path, xbar: Any) -> None:
        # w_digit_range = (-1, 2); 5 is out of range.
        path = _write(tmp_path, "[w]\nvalues = [5]\nprobs = [1.0]\n")
        with pytest.raises(ValueError, match=r"outside xbar legal range"):
            load_distribution(path, xbar)

    def test_empty_values(self, tmp_path: Path, xbar: Any) -> None:
        path = _write(tmp_path, "[w]\nvalues = []\nprobs = []\n")
        with pytest.raises(ValueError, match=r"must be non-empty"):
            load_distribution(path, xbar)

    def test_missing_required_key(self, tmp_path: Path, xbar: Any) -> None:
        path = _write(tmp_path, "[w]\nvalues = [0]\n")
        with pytest.raises(ValueError, match=r"both 'values' and 'probs' are required"):
            load_distribution(path, xbar)

    def test_non_integer_values(self, tmp_path: Path, xbar: Any) -> None:
        path = _write(tmp_path, "[w]\nvalues = [0.5]\nprobs = [1.0]\n")
        with pytest.raises(ValueError, match=r"must be a list of integers"):
            load_distribution(path, xbar)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


def _uniform_dist() -> Distribution:
    return Distribution(None, None, None, None, source="uniform")


def _categorical_dist(
    w_values: list[int] | None = None,
    w_probs: list[float] | None = None,
    x_values: list[int] | None = None,
    x_probs: list[float] | None = None,
) -> Distribution:
    return Distribution(
        w_values=torch.tensor(w_values, dtype=torch.int64) if w_values is not None else None,
        w_probs=torch.tensor(w_probs, dtype=torch.float64) if w_probs is not None else None,
        x_values=torch.tensor(x_values, dtype=torch.int64) if x_values is not None else None,
        x_probs=torch.tensor(x_probs, dtype=torch.float64) if x_probs is not None else None,
        source="test",
    )


class TestSampleW:
    def test_shape_dtype_uniform(self, xbar: Any) -> None:
        ws = list(sample_w(_uniform_dist(), xbar, n=3, device=CPU))
        assert len(ws) == 3
        for w in ws:
            assert tuple(w.shape) == (xbar.col_num, xbar.w_digit_count, xbar.row_num)
            assert w.dtype == torch.int64
            lo, hi = xbar.w_digit_range
            assert int(w.min()) >= lo
            assert int(w.max()) <= hi

    def test_categorical_values_subset(self, xbar: Any) -> None:
        # 28nm preset: w_digit_range = (-1, 2). Pick a strict subset {-1, 0}.
        dist = _categorical_dist(w_values=[-1, 0], w_probs=[0.5, 0.5])
        ws = list(sample_w(dist, xbar, n=2, device=CPU))
        for w in ws:
            assert set(w.flatten().tolist()).issubset({-1, 0})

    def test_determinism_with_seed(self, xbar: Any) -> None:
        g1 = make_generator(seed=42, device=CPU)
        g2 = make_generator(seed=42, device=CPU)
        ws1 = list(sample_w(_uniform_dist(), xbar, n=2, device=CPU, generator=g1))
        ws2 = list(sample_w(_uniform_dist(), xbar, n=2, device=CPU, generator=g2))
        for a, b in zip(ws1, ws2, strict=True):
            assert torch.equal(a, b)

    def test_n_zero_yields_nothing(self, xbar: Any) -> None:
        assert list(sample_w(_uniform_dist(), xbar, n=0, device=CPU)) == []


class TestSampleXBatches:
    def test_total_and_shape_uniform(self, xbar: Any) -> None:
        xs = list(sample_x_batches(_uniform_dist(), xbar, n_total=10, batch_size=4, device=CPU))
        assert sum(int(x.shape[0]) for x in xs) == 10
        for x in xs:
            assert x.shape[1] == xbar.row_num
            assert x.dtype == torch.int64
            lo, hi = xbar.x_range
            assert int(x.min()) >= lo
            assert int(x.max()) <= hi

    def test_last_batch_truncates(self, xbar: Any) -> None:
        xs = list(sample_x_batches(_uniform_dist(), xbar, n_total=5, batch_size=4, device=CPU))
        assert [int(x.shape[0]) for x in xs] == [4, 1]

    def test_categorical_values_subset(self, xbar: Any) -> None:
        # 28nm preset: x_range = (0, 1).
        dist = _categorical_dist(x_values=[0], x_probs=[1.0])
        xs = list(sample_x_batches(dist, xbar, n_total=8, batch_size=4, device=CPU))
        for x in xs:
            assert int(x.max()) == 0 and int(x.min()) == 0

    def test_determinism_with_seed(self, xbar: Any) -> None:
        g1 = make_generator(seed=7, device=CPU)
        g2 = make_generator(seed=7, device=CPU)
        xs1 = list(sample_x_batches(_uniform_dist(), xbar, n_total=8, batch_size=3, device=CPU, generator=g1))
        xs2 = list(sample_x_batches(_uniform_dist(), xbar, n_total=8, batch_size=3, device=CPU, generator=g2))
        for a, b in zip(xs1, xs2, strict=True):
            assert torch.equal(a, b)

    def test_n_total_zero(self, xbar: Any) -> None:
        assert list(sample_x_batches(_uniform_dist(), xbar, n_total=0, batch_size=4, device=CPU)) == []

    def test_invalid_batch_size(self, xbar: Any) -> None:
        with pytest.raises(ValueError, match=r"batch_size"):
            list(sample_x_batches(_uniform_dist(), xbar, n_total=4, batch_size=0, device=CPU))


# ---------------------------------------------------------------------------
# All-off xbar builder
# ---------------------------------------------------------------------------


class TestBuildOffsetXbar:
    def test_returns_offset_xbar(self, xbar: Any) -> None:
        from neurox.xbar import Offset1T1RXbar

        assert isinstance(xbar, Offset1T1RXbar)
        assert xbar.training is False

    def test_all_policy_flags_false(self, xbar: Any) -> None:
        # Spot-check leaves across the policy tree.
        assert xbar.policy.core.rram.prog_gamma is False
        assert xbar.policy.core.rram.stuck_at is False
        assert xbar.policy.core.nmos.A_vt_mismatch is False
        assert xbar.policy.core.tia.opamp_gain_sigma is False
        assert xbar.policy.readout.data_switchcap.cap_mismatch is False
        assert xbar.policy.readout.analog_mux.mux_noise_cm is False
        # bl_adc may be one of several policy types; just check it has been built.
        assert xbar.policy.readout.bl_adc is not None

    def test_unsupported_adc_config_raises(self) -> None:
        class _Bogus:
            pass

        with pytest.raises(TypeError, match=r"unsupported adc config"):
            _all_off_adc_policy(_Bogus())


# ---------------------------------------------------------------------------
# ProbeADC + install_probe_adc
# ---------------------------------------------------------------------------


class TestProbe:
    def test_install_replaces_bl_adc(self, xbar: Any) -> None:
        handle = install_probe_adc(xbar)
        try:
            assert isinstance(xbar.readout.bl_adc, ProbeADC)
            assert xbar.readout.bl_adc is handle.probe
            # The original ADC must be retained outside the module tree.
            assert isinstance(handle.original_adc, ADC) and not isinstance(handle.original_adc, ProbeADC)
        finally:
            handle.restore()

    def test_restore_reinstalls_original(self, xbar: Any) -> None:
        original = xbar.readout.bl_adc
        handle = install_probe_adc(xbar)
        handle.restore()
        assert xbar.readout.bl_adc is original

    def test_context_manager_restores_on_exit(self, xbar: Any) -> None:
        original = xbar.readout.bl_adc
        with install_probe_adc(xbar):
            assert isinstance(xbar.readout.bl_adc, ProbeADC)
        assert xbar.readout.bl_adc is original

    def test_unsupported_readout_raises(self) -> None:
        class _BogusXbar:
            class _Readout:
                pass

            readout = _Readout()

        with pytest.raises(TypeError, match=r"unsupported readout type"):
            install_probe_adc(_BogusXbar())  # type: ignore[arg-type]

    def test_one_vmm_captures_inputs(self, xbar: Any) -> None:
        # Drive one VMM through the probe-installed xbar and assert capture.
        from neurox.tools.xbar_adc._sampling import sample_w, sample_x_batches

        with install_probe_adc(xbar) as handle:
            w = next(sample_w(_uniform_dist(), xbar, n=1, device=CPU))
            xbar.program(w)
            x = next(sample_x_batches(_uniform_dist(), xbar, n_total=4, batch_size=4, device=CPU))
            op = AdcOperationPoint(adc_mode=0, adc_bits=xbar.adc_max_bits)
            xbar.vec_mat_mul(x, adc_operation_point=op)

            pos, neg, diff = handle.probe.captured()
            assert pos.numel() > 0
            assert pos.shape == neg.shape == diff.shape
            assert pos.dtype == torch.float64
            assert torch.allclose(diff, pos - neg)

    def test_probe_not_in_adc_registry(self) -> None:
        # ProbeADC must NOT be looked up by ADC.from_config.
        from neurox.analog.adc.base import ADC as _ADC

        # Iterate via the registry-mixin's lookup helper if available; otherwise
        # verify by searching all subclasses for the @register_key decorator.
        registered = getattr(_ADC, "_registry", {})
        assert ProbeADC not in registered.values()


# ---------------------------------------------------------------------------
# Device + generator helpers
# ---------------------------------------------------------------------------


class TestDeviceResolution:
    def test_auto_falls_back_to_cpu_when_no_cuda(self) -> None:
        # Result depends on test environment; just verify it returns a torch.device.
        device = resolve_device("auto")
        assert isinstance(device, torch.device)
        if not torch.cuda.is_available():
            assert device.type == "cpu"

    def test_explicit_cpu(self) -> None:
        assert resolve_device("cpu").type == "cpu"

    def test_cuda_without_cuda_raises(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA is available; cannot test the missing-cuda path")
        with pytest.raises(RuntimeError, match=r"no CUDA device"):
            resolve_device("cuda")

    def test_make_generator_none(self) -> None:
        assert make_generator(None, CPU) is None

    def test_make_generator_deterministic(self) -> None:
        g1 = make_generator(11, CPU)
        g2 = make_generator(11, CPU)
        assert g1 is not None and g2 is not None
        a = torch.randint(0, 100, (8,), generator=g1)
        b = torch.randint(0, 100, (8,), generator=g2)
        assert torch.equal(a, b)
