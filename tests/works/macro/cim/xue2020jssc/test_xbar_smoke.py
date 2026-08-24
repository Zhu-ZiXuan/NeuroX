"""Eager end-to-end smoke test for the xue2020jssc SINWP 1T1R CIM sub-array.

Builds the folded `Xue2020JsscCimMacro` (kernel pure array + composed
current-mode readout modules) from the hand-built near-ideal witness config with
its ladder calibrated in-code (`_utils.build_calibrated_macro`: `output_num = 4`
-> `io_num = 2` at `mux_factor = 2`, `input_num = max_active_num = 4`,
`input_bit_num = 2`, 3-bit ADC), programs a mixed-sign weight, and runs one
`vec_mat_mul` on an integer activation batch. Asserts:

  * the output is an integer signed-magnitude code tensor with the caller's
    leading order preserved, every value in `[-MAG_MAX, MAG_MAX]`, and
    bit-exactly the clamped ideal integer MAC;
  * a `Reporter` report is coherent: the two macro-billed channels
    (`cablc` / `control`) and the self-billing array + DSWCT / SINWP-SC /
    PN-ISUB + TMCSA module rows carry positive dynamic energy, and the totals
    are positive,
  * the macro's reported window is `conduction_span x mux_factor` — the access
    time of every column-MUX slot it serializes,
  * at `inst_shape = ()` every leading axis is anonymous broadcast batch:
    reshaping the batch dims is transparent (a multi-axis batch equals the
    flattened batch reshaped, a no-batch input yields `[output_num]`, and a
    size-1 leading axis broadcasts). The instance-axis contract at a non-empty
    `inst_shape` lives in `test_instance_axes`.

Runs eagerly (dynamo disabled) so the `@torch.compile` solver leaf is not
unrolled.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
import torch._dynamo

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

_CHANNEL_KEYS = (".cablc", ".control")
# The array, readout modules, and TMCSA billing module self-bill dynamic
# energy; the kernel ADC (`adc`) is energy-silent.
_MODULE_ROWS = ("array", "dswct", "sinwp_sc", "pn_isub", "tmcsa")


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is `@torch.compile`; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


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
    """Full VMM: shape / range / bit-exact decode, profiler coherence, latency law."""
    macro = build_calibrated_macro(device=device)
    w = _mixed_sign_weight()
    macro.program(w.to(device))

    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0], [0, 1, 2, 3]], dtype=torch.long)  # batch (3,)
    with Profiler() as prof, torch.no_grad():
        out = macro.vec_mat_mul(x.to(device), quantization_mode=QUANTIZATION_MODE, adc_bits=TINY_ADC_BITS)
    reporter = Reporter(macro)
    out = out.cpu()

    # --- 1. Integer signed-magnitude codes and shape ---
    assert out.dtype in (torch.int64, torch.long)
    assert tuple(out.shape) == (x.shape[0], TINY_OUTPUT_NUM)
    assert -MAG_MAX <= int(out.min()) <= int(out.max()) <= MAG_MAX

    # --- 2. Bit-exact decode of the clamped ideal integer MAC; both signs exercised ---
    expected = ideal_mac(w, x)
    assert torch.equal(out, expected), f"MAC decode mismatch:\n{out.tolist()}\nvs\n{expected.tolist()}"
    assert int(out.min()) < 0 < int(out.max())

    # --- 3. Report sanity: channels + module rows positive ---
    by_name = reporter.by_name(prof)
    for key in _CHANNEL_KEYS:
        assert key in by_name, f"missing channel {key}; have {sorted(by_name)}"
        assert by_name[key] > 0.0
    for mod in _MODULE_ROWS:
        assert mod in by_name, f"missing module row {mod}; have {sorted(by_name)}"
        assert by_name[mod] > 0.0
    # The array self-bills its capacitive row and the macro bills the whole input
    # branch under `.cablc`. The cell emits no dynamic row (the array logs its
    # caps), the kernel ADC is energy-silent (the `tmcsa` module row bills the
    # conversion), and the readout modules bill on module rows, not macro channels.
    for absent in ("cell", "adc", ".dswct", ".sinwp_sc", ".pn_isub"):
        assert absent not in by_name, f"unexpected energy row {absent}: {sorted(by_name)}"

    assert reporter.total_dynamic_energy__fJ(prof) > 0.0
    assert reporter.static.leakage__uW > 0.0

    # --- 4. Circuit latency and scheduled interval ---
    # The macro owns the WL sub-phases and the live-bit settle; the sensing tail
    # belongs to the converter that owns the search-step axis. mux_factor is the
    # macro's only time axis.
    cfg = macro.config
    for bits in range(1, TINY_ADC_BITS + 1):
        expected__ns = cfg.access_latency__ns(bits) * cfg.mux_factor
        assert macro.latency__ns(adc_bits=bits) == pytest.approx(expected__ns)
        assert macro.initiation_interval__ns(adc_bits=bits) == pytest.approx(cfg.t_cycle__ns * cfg.mux_factor)
        assert macro.initiation_interval__ns(adc_bits=bits) >= macro.latency__ns(adc_bits=bits)
    # A lowered resolution shortens the access by exactly the steps it drops.
    assert macro.latency__ns(adc_bits=TINY_ADC_BITS - 1) < macro.latency__ns(adc_bits=TINY_ADC_BITS)
    assert macro.latency__ns(adc_bits=TINY_ADC_BITS) == pytest.approx(
        cfg.access_latency__ns(TINY_ADC_BITS) * cfg.mux_factor
    )


def test_anonymous_leading_axes_broadcast(device: torch.device) -> None:
    """At `inst_shape = ()` every leading axis is anonymous broadcast batch: reshaping is transparent."""
    macro = build_calibrated_macro(device=device)
    w = _mixed_sign_weight()
    macro.program(w.to(device))

    torch.manual_seed(3)
    x_flat = torch.randint(0, 1 << TINY_K, (6, TINY_INPUT_NUM), dtype=torch.long)
    with torch.no_grad():
        out_flat = macro.vec_mat_mul(
            x_flat.to(device), quantization_mode=QUANTIZATION_MODE, adc_bits=TINY_ADC_BITS
        ).cpu()
    assert tuple(out_flat.shape) == (6, TINY_OUTPUT_NUM)

    # A multi-axis batch decodes each sample identically to the flattened batch.
    x_multi = x_flat.reshape(2, 3, TINY_INPUT_NUM)
    with torch.no_grad():
        out_multi = macro.vec_mat_mul(
            x_multi.to(device), quantization_mode=QUANTIZATION_MODE, adc_bits=TINY_ADC_BITS
        ).cpu()
    assert tuple(out_multi.shape) == (2, 3, TINY_OUTPUT_NUM)
    assert torch.equal(out_multi, out_flat.reshape(2, 3, TINY_OUTPUT_NUM))

    # A no-batch input yields the primitive trailing output axis alone.
    with torch.no_grad():
        out_scalar = macro.vec_mat_mul(
            x_flat[0].to(device), quantization_mode=QUANTIZATION_MODE, adc_bits=TINY_ADC_BITS
        ).cpu()
    assert tuple(out_scalar.shape) == (TINY_OUTPUT_NUM,)
    assert torch.equal(out_scalar, out_flat[0])

    # A size-1 leading axis broadcasts to the same single-sample decode.
    with torch.no_grad():
        out_unit = macro.vec_mat_mul(
            x_flat[:1].to(device), quantization_mode=QUANTIZATION_MODE, adc_bits=TINY_ADC_BITS
        ).cpu()
    assert tuple(out_unit.shape) == (1, TINY_OUTPUT_NUM)
    assert torch.equal(out_unit, out_flat[:1])
