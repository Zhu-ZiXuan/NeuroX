"""Xue2020 end-to-end smoke tests."""

from __future__ import annotations

import pytest
import torch

from neurox import Profiler, Reporter

from ._utils import (
    MAG_MAX,
    QUANTIZATION_MODE,
    TINY_ADC_BITS,
    TINY_INPUT_NUM,
    TINY_K,
    TINY_OUTPUT_NUM,
    build_calibrated_macro,
    ideal_mac,
)

_CHANNEL_KEYS = (".cablc", ".dswct", ".sinwp_sc", ".pn_isub")
_MODULE_ROWS = ("array", "tmcsa", "control")


def _mixed_sign_weight() -> torch.Tensor:
    """Mixed-sign witness weight spanning both signs and a zero column."""
    return torch.tensor(
        [
            [1, 1, -1, 0],  # MAC(x=[1,2,1,0]) = 1 + 2 - 1 = 2
            [-2, 1, 0, 0],  # MAC = -2 + 2 = 0
            [3, -1, 0, 0],  # MAC = 3 - 2 = 1
            [0, 0, 0, 0],  # MAC = 0
        ],
        dtype=torch.long,
    ).transpose(-1, -2)


def test_xbar_end_to_end_and_profiler(device: torch.device) -> None:
    macro = build_calibrated_macro(device=device)
    w = _mixed_sign_weight()
    macro.program(w.to(device))

    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0], [0, 1, 2, 3]], dtype=torch.long)  # batch (3,)
    with Profiler() as prof, torch.no_grad():
        out = macro.vec_mat_mul(x.to(device), quantization_mode=QUANTIZATION_MODE, adc_active_bits=TINY_ADC_BITS)
    reporter = Reporter(macro)
    out = out.cpu()

    assert out.dtype in (torch.int64, torch.long)
    assert tuple(out.shape) == (x.shape[0], TINY_OUTPUT_NUM)
    assert -MAG_MAX <= int(out.min()) <= int(out.max()) <= MAG_MAX

    expected = ideal_mac(w, x)
    assert torch.equal(out, expected), f"MAC decode mismatch:\n{out.tolist()}\nvs\n{expected.tolist()}"
    assert int(out.min()) < 0 < int(out.max())

    by_name = reporter.by_name(prof)
    for key in _CHANNEL_KEYS:
        assert key in by_name, f"missing channel {key}; have {sorted(by_name)}"
        assert by_name[key] > 0.0
    for mod in _MODULE_ROWS:
        assert mod in by_name, f"missing module row {mod}; have {sorted(by_name)}"
        assert by_name[mod] > 0.0
    for absent in ("cell", "adc", "dswct", "sinwp_sc", "pn_isub"):
        assert absent not in by_name, f"unexpected energy row {absent}: {sorted(by_name)}"

    assert reporter.total_dynamic_energy__fJ(prof) > 0.0
    assert reporter.static.leakage__uW > 0.0

    cfg = macro.config
    for bits in range(1, TINY_ADC_BITS + 1):
        access__ns = (
            (cfg.x_bit_num - 1) * cfg.t_sample__ns + cfg.t_settle__ns + macro.tmcsa.latency__ns(active_bits=bits)
        )
        expected__ns = access__ns * macro.scan_num
        assert macro.latency__ns(adc_active_bits=bits) == pytest.approx(expected__ns)
    assert macro.latency__ns(adc_active_bits=TINY_ADC_BITS - 1) < macro.latency__ns(adc_active_bits=TINY_ADC_BITS)


def test_anonymous_leading_axes_broadcast(device: torch.device) -> None:
    macro = build_calibrated_macro(device=device)
    w = _mixed_sign_weight()
    macro.program(w.to(device))

    torch.manual_seed(3)
    x_flat = torch.randint(0, 1 << TINY_K, (6, TINY_INPUT_NUM), dtype=torch.long)
    with torch.no_grad():
        out_flat = macro.vec_mat_mul(
            x_flat.to(device), quantization_mode=QUANTIZATION_MODE, adc_active_bits=TINY_ADC_BITS
        ).cpu()
    assert tuple(out_flat.shape) == (6, TINY_OUTPUT_NUM)

    x_multi = x_flat.reshape(2, 3, TINY_INPUT_NUM)
    with torch.no_grad():
        out_multi = macro.vec_mat_mul(
            x_multi.to(device), quantization_mode=QUANTIZATION_MODE, adc_active_bits=TINY_ADC_BITS
        ).cpu()
    assert tuple(out_multi.shape) == (2, 3, TINY_OUTPUT_NUM)
    assert torch.equal(out_multi, out_flat.reshape(2, 3, TINY_OUTPUT_NUM))

    with torch.no_grad():
        out_scalar = macro.vec_mat_mul(
            x_flat[0].to(device), quantization_mode=QUANTIZATION_MODE, adc_active_bits=TINY_ADC_BITS
        ).cpu()
    assert tuple(out_scalar.shape) == (TINY_OUTPUT_NUM,)
    assert torch.equal(out_scalar, out_flat[0])

    with torch.no_grad():
        out_unit = macro.vec_mat_mul(
            x_flat[:1].to(device), quantization_mode=QUANTIZATION_MODE, adc_active_bits=TINY_ADC_BITS
        ).cpu()
    assert tuple(out_unit.shape) == (1, TINY_OUTPUT_NUM)
    assert torch.equal(out_unit, out_flat[:1])
