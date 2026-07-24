"""Eager end-to-end smoke test for the xue2020jssc SINWP 1T1R CIM sub-array.

Builds the folded :class:`Xue2020JsscCimMacro` (kernel pure array + inline
current-mode readout chain) from the hand-built near-ideal witness config with
its ladder calibrated in-code (``_utils.build_calibrated_macro``: ``col_num = 4``
-> ``io_num = 2`` at ``mux_factor = 2``, ``row_num = active_row_num = 4``,
``input_bit_num = 2``, 3-bit ADC), programs a mixed-sign weight, and runs one
``vec_mat_mul`` on an integer activation batch. Asserts:

  * the output is an integer signed-magnitude code tensor with the caller's
    leading order preserved and primitive trailing ``[col_num]``, every value in
    ``[-MAG_MAX, MAG_MAX]``, and bit-exactly the clamped ideal integer MAC,
  * a :class:`NeuroxProfiler` report is coherent: the five macro-billed channels
    and the self-billing array + TMCSA module rows carry positive dynamic energy,
    the totals are positive, and the leakage energy reconciles as
    ``static.leakage_power__uW x total_latency__ns``,
  * the macro-root latency obeys ``t_cycle * serial_op_count`` (the static-energy
    time base; the macro is the sole latency emitter) with ``serial_op_count =
    numel(i_sub) // (inst_count x n_io) = mux_factor x batch``,
  * every leading axis is anonymous broadcast batch: reshaping the batch dims is
    transparent (a multi-axis batch equals the flattened batch reshaped, a
    no-batch input yields ``[col_num]``, and a size-1 leading axis broadcasts).

Runs eagerly (dynamo disabled) so the ``@torch.compile`` solver leaf is not
unrolled.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
import torch._dynamo

from neurox.common.profiler import NeuroxProfiler
from tests.works.macro.cim.xue2020jssc._utils import (
    ADC_MODE,
    MAG_MAX,
    TINY_ADC_BITS,
    TINY_COL_NUM,
    TINY_K,
    TINY_ROW_NUM,
    Xue2020JsscCimMacroConfig,
    build_calibrated_macro,
    build_config,
    build_macro,
    encode_weights,
    ideal_mac,
)

_CHANNEL_KEYS = (".cablc", ".dswct", ".sinwp_sc", ".pn_isub", ".control")
_MODULE_ROWS = ("array", "tmcsa")  # the restored array + the ADC self-bill dynamic energy (S4.1)


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is ``@torch.compile``; do not unroll it."""
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
    )


def test_xbar_end_to_end_and_profiler(device: torch.device) -> None:
    """Full VMM: shape / range / bit-exact decode, profiler coherence, latency law."""
    macro = build_calibrated_macro(device=device)
    w = _mixed_sign_weight()
    macro.program(encode_weights(w.to(device)))

    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0], [0, 1, 2, 3]], dtype=torch.long)  # batch (3,)
    with NeuroxProfiler() as prof, torch.no_grad():
        out = macro.vec_mat_mul(x.to(device), adc_mode=ADC_MODE, adc_bits=TINY_ADC_BITS)
    report = prof.report(macro)
    out = out.cpu()

    # --- 1. Integer signed-magnitude codes: leading [batch] preserved, trailing [col_num] ---
    assert out.dtype in (torch.int64, torch.long)
    assert tuple(out.shape) == (x.shape[0], TINY_COL_NUM)
    assert int(out.min()) >= -MAG_MAX and int(out.max()) <= MAG_MAX

    # --- 2. Bit-exact decode of the clamped ideal integer MAC; both signs exercised ---
    expected = ideal_mac(w, x)
    assert torch.equal(out, expected), f"MAC decode mismatch:\n{out.tolist()}\nvs\n{expected.tolist()}"
    assert int(out.min()) < 0 and int(out.max()) > 0

    # --- 3. Profiler report sanity: channels + module rows positive ---
    by_name = report.energy_by_name
    for key in _CHANNEL_KEYS:
        assert key in by_name, f"missing channel {key}; have {sorted(by_name)}"
        assert by_name[key] > 0.0
    for mod in _MODULE_ROWS:
        assert mod in by_name, f"missing module row {mod}; have {sorted(by_name)}"
        assert by_name[mod] > 0.0
    # S4.1: the array self-bills its cell-side row and the macro bills the
    # clamp-side under ``.cablc``; the non-reporter cell and the PN-ISUB seat (a
    # macro channel) emit no self-billed dynamic row.
    for absent in ("cell", "pn_isub"):
        assert absent not in by_name, f"unexpected self-billing module row {absent}: {sorted(by_name)}"

    assert prof.total_dynamic_energy__fJ > 0.0
    assert prof.total_latency__ns > 0.0

    # --- 3b. Leakage reconciliation: static power x total latency ---
    assert report.static.leakage_power__uW > 0.0
    assert report.leakage_energy__fJ == pytest.approx(report.static.leakage_power__uW * prof.total_latency__ns)

    # --- 4. Latency law: t_cycle * serial_op_count (macro sole latency emitter) ---
    # i_sub is [batch, group_size, n_io]; serial = numel // (inst_count x n_io) = mux_factor x batch.
    cfg = macro.config
    serial_op_count = cfg.mux_factor * x.shape[0]
    expected_latency = cfg.t_cycle__ns * serial_op_count
    macro_latency = report.latency_by_name.get("", 0.0)  # the macro root is named ""
    assert macro_latency == pytest.approx(expected_latency)
    # The macro is the sole non-zero latency emitter: the WL DAC has zero latency
    # and the macro builds the TMCSA with a zeroed step_latency (its honest sensing
    # durations feed t_other, not the profiled latency), so the total latency
    # equals the macro-root latency — the nonzero ADC step_latency never
    # double-counts against t_cycle.
    assert prof.total_latency__ns == pytest.approx(macro_latency)


def test_adc_step_latency_does_not_double_count(device: torch.device) -> None:
    """Nonzero ADC ``step_latency__ns`` widens ``t_other`` but never the profiled latency.

    The honest per-step SAR sensing durations feed the read-chain window
    ``t_other`` (and the ADC's own sensing-conduction energy), but the macro builds
    the TMCSA to emit NO latency and is the sole latency emitter. So two configs
    differing ONLY in ``step_latency__ns`` share the same ``t_cycle * serial`` total
    latency while their ``t_other`` differs — the sensing never double-counts
    against ``t_cycle``.
    """
    zero = build_config(step_latency__ns=(0.0, 0.0, 0.0))
    sensing = build_config(step_latency__ns=(3.16, 3.07, 3.11))
    assert sensing.t_other__ns > zero.t_other__ns  # sensing widens the read window
    assert zero.t_cycle__ns == sensing.t_cycle__ns

    w = _mixed_sign_weight()
    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0]], dtype=torch.long)  # batch (2,)

    def total_latency(cfg: Xue2020JsscCimMacroConfig) -> float:
        macro = build_macro(cfg, device=device)
        macro.program(encode_weights(w.to(device)))
        with NeuroxProfiler() as prof, torch.no_grad():
            macro.vec_mat_mul(x.to(device), adc_mode=ADC_MODE, adc_bits=TINY_ADC_BITS)
        return prof.total_latency__ns

    lat_zero = total_latency(zero)
    lat_sensing = total_latency(sensing)
    assert lat_sensing == pytest.approx(lat_zero)  # step_latency does not leak into latency
    assert lat_sensing == pytest.approx(zero.t_cycle__ns * zero.mux_factor * x.shape[0])


def test_anonymous_leading_axes_broadcast(device: torch.device) -> None:
    """Every leading axis is anonymous broadcast batch: reshaping batch dims is transparent."""
    macro = build_calibrated_macro(device=device)
    w = _mixed_sign_weight()
    macro.program(encode_weights(w.to(device)))

    torch.manual_seed(3)
    x_flat = torch.randint(0, 1 << TINY_K, (6, TINY_ROW_NUM), dtype=torch.long)  # batch (6,)
    with torch.no_grad():
        out_flat = macro.vec_mat_mul(x_flat.to(device), adc_mode=ADC_MODE, adc_bits=TINY_ADC_BITS).cpu()
    assert tuple(out_flat.shape) == (6, TINY_COL_NUM)

    # A multi-axis batch decodes each sample identically to the flattened batch.
    x_multi = x_flat.reshape(2, 3, TINY_ROW_NUM)
    with torch.no_grad():
        out_multi = macro.vec_mat_mul(x_multi.to(device), adc_mode=ADC_MODE, adc_bits=TINY_ADC_BITS).cpu()
    assert tuple(out_multi.shape) == (2, 3, TINY_COL_NUM)
    assert torch.equal(out_multi, out_flat.reshape(2, 3, TINY_COL_NUM))

    # A no-batch input yields the primitive trailing [col_num] alone.
    with torch.no_grad():
        out_scalar = macro.vec_mat_mul(x_flat[0].to(device), adc_mode=ADC_MODE, adc_bits=TINY_ADC_BITS).cpu()
    assert tuple(out_scalar.shape) == (TINY_COL_NUM,)
    assert torch.equal(out_scalar, out_flat[0])

    # A size-1 leading axis broadcasts to the same single-sample decode.
    with torch.no_grad():
        out_unit = macro.vec_mat_mul(x_flat[:1].to(device), adc_mode=ADC_MODE, adc_bits=TINY_ADC_BITS).cpu()
    assert tuple(out_unit.shape) == (1, TINY_COL_NUM)
    assert torch.equal(out_unit, out_flat[:1])
