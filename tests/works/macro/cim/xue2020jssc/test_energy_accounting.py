"""Eager energy-accounting laws for the xue2020jssc SINWP 1T1R CIM macro.

Pins the branch-ownership and channel-billing contract on the hand-built near-ideal witness macro
(`_utils.build_config`). Every check is a LAW read off the profiler report, not
a magic number: the coefficients are whatever the deterministic all-off analog
chain produces, and the assertions constrain how the billed energy MOVES.

Coverage:

  * the two macro-billed channels `cablc` / `control` appear under their
    exact dotted names (the macro root is named `""` so a channelled row reads
    `".<channel>"`), and the self-billing dynamic module rows are the array
    `array`, the readout modules `dswct` / `sinwp_sc` / `pn_isub`, and
    the TMCSA `tmcsa`; no dotted macro channel carries a readout module,
  * the static report seats the reporter leaves (control / adc_current_reference /
    cablc / sl_driver / dswct / sinwp_sc / pn_isub / tmcsa / array + the macro
    root),
  * **dynamic energy rides the conduction windows, NOT `t_cycle`**: doubling
    `t_cycle` (the leakage integration window) leaves the dynamic energy
    unchanged, while doubling a conduction window (`t_settle`) scales the read
    rows and leaves the window-invariant control channel untouched,
  * the input branch conduction is billed WHOLE by the macro on the `cablc`
    channel (`VDD * I_DL` over the per-bit window — the macro owns the
    conduction window), while the array module row bills ONLY its wire / node
    capacitive cycling: the channel matches the reconstructed whole branch
    exactly, the array row is strictly positive yet window-invariant (the cap
    oracle), and array + channel cover the whole branch plus the caps with no
    double-bill,
  * the array's capacitive row rides the shared core supply `vdd__V`,
  * the control channel fires once per access (`mux_factor` mux steps x batch),
  * each read row is LINEAR in every window knob (`t_sample[k]`,
    `t_settle`), the SC held-leg SUFFIX-SUM law (window
    `sum_{j>=k} t_sample[j] + t_other`) holds while cablc / dswct use the
    per-bit DIAGONAL window, and the live (K-1) bit conducts in `t_other`
    regardless of the sampling windows,
  * the `tmcsa` row is the scheme PHASE-BILLING module: it reduces to the
    pure fixed-energy model `adc_bits x tmcsa e_fixed` per converted element
    when both phase windows are all-zero, grows with `t_ph2` / `t_ph3`
    once nonzero, and the kernel ADC is energy-SILENT
    (`enable_energy_record=False`): no `adc` dynamic row exists and the
    kernel conduction knobs (`v_rail`, `t_conduct`, kernel `e_fixed`)
    move nothing.

Runs eagerly (dynamo disabled) so the `@torch.compile` solver leaf is not
unrolled.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import TypedDict

import pytest
import torch
import torch._dynamo
from torch import Tensor

from neurox import Profiler, Reporter

from ._utils import (
    QUANTIZATION_MODE,
    TINY_ADC_BITS,
    TINY_INPUT_NUM,
    TINY_OUTPUT_NUM,
    Xue2020JsscCimMacro,
    Xue2020JsscCimMacroConfig,
    build_config,
    build_macro,
)


class _AdcConfigUpdates(TypedDict, total=False):
    t_conduct_per_step__ns: tuple[float, ...]
    v_rail__V: float
    e_fixed_per_op__fJ: float


class _TmcsaConfigUpdates(TypedDict, total=False):
    t_ph2_per_step__ns: tuple[float, ...]
    t_ph3_per_step__ns: tuple[float, ...]


# Billed rows by slice name -> profiler energy-row key: the macro bills the two
# dotted channels; the readout modules self-bill on their own module rows.
_ROW_KEYS = {
    "cablc": ".cablc",  # macro channel
    "dswct": "dswct",  # module row
    "sinwp_sc": "sinwp_sc",  # module row
    "pn_isub": "pn_isub",  # module row
    "control": ".control",  # macro channel
}
_READ_ROWS = ("cablc", "dswct", "sinwp_sc")  # window-dependent conduction rows


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is `@torch.compile`; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _w_full(input_num: int = TINY_INPUT_NUM, output_num: int = TINY_OUTPUT_NUM) -> Tensor:
    """All-`+1` weights so every physical column / IO / mux slot conducts."""
    return torch.ones((input_num, output_num), dtype=torch.long)


def _x_full(k: int, input_num: int = TINY_INPUT_NUM) -> Tensor:
    """Full-scale K-bit input (every input bit set on every row) — all sub-phase legs conduct."""
    return torch.full((input_num,), (1 << k) - 1, dtype=torch.long)


def _run(
    config: Xue2020JsscCimMacroConfig,
    w: Tensor,
    x: Tensor,
    *,
    device: torch.device,
    quantization_mode: int = QUANTIZATION_MODE,
    adc_bits: int = TINY_ADC_BITS,
) -> tuple[Profiler, Reporter]:
    """Build + fabricate a fresh macro, program `w`, profile one VMM on `x`."""
    macro = build_macro(config, device=device)
    macro.program(w.to(device))
    with Profiler() as prof, torch.no_grad():
        macro.vec_mat_mul(x.to(device), quantization_mode=quantization_mode, adc_bits=adc_bits)
    return prof, Reporter(macro)


def _channels(by_name: dict[str, float]) -> dict[str, float]:
    """Per-row dynamic energy [fJ] keyed by slice name (macro channels dotted, module rows bare)."""
    return {name: by_name.get(key, 0.0) for name, key in _ROW_KEYS.items()}


def _channel_energies(
    config: Xue2020JsscCimMacroConfig, w: Tensor, x: Tensor, *, device: torch.device
) -> dict[str, float]:
    prof, reporter = _run(config, w, x, device=device)
    return _channels(reporter.by_name(prof))


def _with_adc(
    config: Xue2020JsscCimMacroConfig,
    *,
    t_conduct: tuple[float, ...] | None = None,
    v_rail: float | None = None,
    e_fixed: float | None = None,
) -> Xue2020JsscCimMacroConfig:
    """Replace only the KERNEL ADC conduction knobs (inert in this scheme — the deadness oracle)."""
    kw: _AdcConfigUpdates = {}
    if t_conduct is not None:
        kw["t_conduct_per_step__ns"] = t_conduct
    if v_rail is not None:
        kw["v_rail__V"] = v_rail
    if e_fixed is not None:
        kw["e_fixed_per_op__fJ"] = e_fixed
    return dataclasses.replace(config, adc_config=dataclasses.replace(config.adc_config, **kw))


def _with_tmcsa(
    config: Xue2020JsscCimMacroConfig,
    *,
    t_ph2: tuple[float, ...] | None = None,
    t_ph3: tuple[float, ...] | None = None,
) -> Xue2020JsscCimMacroConfig:
    """Replace only the TMCSA phase windows (energy-path only; value windows untouched)."""
    kw: _TmcsaConfigUpdates = {}
    if t_ph2 is not None:
        kw["t_ph2_per_step__ns"] = t_ph2
    if t_ph3 is not None:
        kw["t_ph3_per_step__ns"] = t_ph3
    return dataclasses.replace(config, tmcsa_config=dataclasses.replace(config.tmcsa_config, **kw))


def _whole_input_branch(macro: Xue2020JsscCimMacro, x: Tensor) -> float:
    """Reconstruct the WHOLE input-branch read energy from a re-solve, batch-summed [fJ].

    Re-runs the array DC solve (cells + wire IR drop) per WL plane against the same
    `window_array` windows `vec_mat_mul` uses, matching it bit-for-bit
    (deterministic under all-off + eval), and returns the whole input branch the
    macro bills on the `cablc` channel::

        whole = sum_k  VDD * I_DL * window_array[k]

    where `I_DL` is the per-column BL port current the solver returns and the
    conduction window `t` is the macro's, applied here post-solve. The array
    carries no conduction term, so there is no clamp / cell split to reconstruct.
    The re-solve runs OUTSIDE any profiler so it logs nothing of its own.

    Written for the `inst_shape = ()` macros this file builds: the clamp
    reference is expanded right-aligned onto the flat column axis, which a
    fabrication prefix would mis-seat.
    """
    cfg = macro.config
    vdd__V = cfg.vdd__V
    x_long = x.long()
    window = cfg.window_array__ns

    phys_col_num = macro.array.weight_grid_shape[-2]

    whole = 0.0
    for k in range(cfg.input_bit_num):
        # Shape: [..., row]
        plane = (x_long >> k) & 1
        # Shape: [..., row]
        v_wl = macro.wl_dac.convert(plane)
        # The array takes the line-level WL drive and both boundary snaps ready
        # made, exactly as vec_mat_mul builds them: one solve over every
        # physical column, the MUX slot living outside the array.
        leading = tuple(torch.broadcast_shapes(macro.inst_shape, v_wl.shape[:-1]))
        ref_shape = (*leading, phys_col_num)
        v_blc = macro.cablc_vref.v_out__V[..., 0, 0]
        steady = macro.array.solve_array(
            v_wl,
            bl_driver=macro.cablc,
            bl_driver_snap=macro.cablc.snapshot(v_ref__V=v_blc.expand(ref_shape), shape=ref_shape),
            sl_driver=macro.sl_driver,
            sl_driver_snap=macro.sl_driver.snapshot(
                v_ref__V=torch.zeros((), dtype=v_wl.dtype, device=v_wl.device).expand(ref_shape),
                shape=ref_shape,
            ),
        )
        # Shape: [..., phys_col] -> []
        step_energy = ((vdd__V * steady.i_bl_port__uA).sum(dim=-1) * window[k]).sum()
        whole += float(step_energy)
    return whole


# ---------------------------------------------------------------------------
# Channel presence + exact names
# ---------------------------------------------------------------------------


def test_billed_rows_present_with_exact_names(device: torch.device) -> None:
    """The two macro channels and the three readout module rows appear under their exact keys."""
    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0]], dtype=torch.long)  # batch (2,)
    prof, reporter = _run(build_config(), _w_full(), x, device=device)
    by_name = reporter.by_name(prof)
    for name, key in _ROW_KEYS.items():
        assert key in by_name, f"missing {name} row {key!r}; have {sorted(by_name)}"
        assert by_name[key] > 0.0, f"non-positive {key}: {by_name[key]}"
    # The readout modules self-bill: no macro channel carries their energy.
    for stale in (".dswct", ".sinwp_sc", ".pn_isub"):
        assert stale not in by_name, f"stale macro channel {stale}: {sorted(by_name)}"


def test_module_rows_self_bill_dynamic(device: torch.device) -> None:
    """Array, TMCSA, and the readout modules self-bill; only cablc/control stay macro channels."""
    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0]], dtype=torch.long)
    prof, reporter = _run(build_config(), _w_full(), x, device=device)
    by_name = reporter.by_name(prof)
    # The array bills its capacitive cycling (caps only); the TMCSA module
    # bills the conversion phases; the readout modules bill their rail
    # branches.
    for row in ("array", "tmcsa", "dswct", "sinwp_sc", "pn_isub"):
        assert by_name.get(row, 0.0) > 0.0, f"missing/empty {row} row; have {sorted(by_name)}"
    # The cell is a non-reporter (the array logs its caps), and the kernel ADC
    # is energy-SILENT (enable_energy_record=False): no adc dynamic row.
    for silent in ("cell", "adc"):
        assert silent not in by_name, f"unexpected self-billing module row {silent}: {sorted(by_name)}"


# ---------------------------------------------------------------------------
# Static report seats
# ---------------------------------------------------------------------------


def test_static_report_seats_reporters_only(device: torch.device) -> None:
    """The static walk contains every configured PPA-reporting module."""
    macro = build_macro(build_config(), device=device)
    static = {e.qualified_name: e.leakage__uW for e in Reporter(macro).static_entries}
    # Seats with nonzero witness leakage: the macro root (named ""), control
    # (UnmodeledBlock), adc_current_reference, the clamp drivers, the PN-ISUB
    # module, the kernel ADC (adc), and the TMCSA billing module (tmcsa).
    for seat in ("", "control", "adc_current_reference", "cablc", "sl_driver", "pn_isub", "tmcsa", "adc"):
        assert seat in static, f"missing static seat {seat!r}; have {sorted(static)}"
        assert static[seat] > 0.0, f"non-positive leakage seat {seat!r}: {static[seat]}"
    # The DSWCT / SINWP-SC modules are reporter leaves too; the witness ships
    # their leakage seats at 0.0, so they appear with exactly zero leakage. The
    # array itself holds no static conduction path (both scan modes rest
    # at zero cell bias), so its leakage seat is architecturally zero.
    for seat in ("dswct", "sinwp_sc", "array"):
        assert seat in static, f"missing static seat {seat!r}; have {sorted(static)}"
        assert static[seat] == 0.0, f"witness ships zero leakage for {seat!r}: {static[seat]}"


# ---------------------------------------------------------------------------
# Dynamic energy rides the conduction windows, NOT t_cycle (CRITICAL)
# ---------------------------------------------------------------------------


def test_dynamic_energy_scales_with_conduction_windows_not_t_cycle(device: torch.device) -> None:
    """Every dynamic channel bills over the conduction windows, never `t_cycle`.

    `t_cycle` is the leakage integration window, so doubling it leaves every
    dynamic channel unchanged; doubling a conduction window (`t_settle`) grows
    the read channels while the window-invariant control channel stands still.
    """
    w, x = _w_full(), _x_full(2)
    base_cfg = build_config(t_sample__ns=(1.0,), t_settle__ns=2.0, t_cycle__ns=50.0)

    prof_base, rep_base = _run(base_cfg, w, x, device=device)
    dyn_base = rep_base.total_dynamic_energy__fJ(prof_base)
    assert dyn_base > 0.0

    # --- Double t_cycle: dynamic unchanged ---
    prof_2t, rep_2t = _run(dataclasses.replace(base_cfg, t_cycle__ns=100.0), w, x, device=device)
    assert rep_2t.total_dynamic_energy__fJ(prof_2t) == pytest.approx(dyn_base)

    # --- Double a conduction window (t_settle -> t_other): the read channels grow ---
    prof_win, rep_win = _run(dataclasses.replace(base_cfg, t_settle__ns=4.0), w, x, device=device)
    assert rep_win.total_dynamic_energy__fJ(prof_win) > dyn_base  # dynamic grows with the window
    # The control channel is window-invariant; the read channels moved.
    ch_base = _channels(rep_base.by_name(prof_base))
    ch_win = _channels(rep_win.by_name(prof_win))
    assert ch_win["control"] == pytest.approx(ch_base["control"])
    for ch in _READ_ROWS:
        assert ch_win[ch] > ch_base[ch], f"read channel {ch} did not grow with t_settle"


# ---------------------------------------------------------------------------
# Input-branch ownership: cablc bills the whole branch, array bills caps only
# ---------------------------------------------------------------------------


def test_input_branch_billed_whole_by_cablc_array_bills_caps_only(device: torch.device) -> None:
    """The macro bills the whole input branch on `.cablc`; the array bills caps only.

    The `.cablc` channel bills the whole input branch `VDD * I_DL` over the
    per-bit conduction window (the macro owns the window); the array module row
    bills ONLY its wire / node capacitive cycling — no conduction. Reconciled
    against a re-solve: `.cablc` matches the reconstructed WHOLE branch EXACTLY
    (a clamp-side-only `(VDD - V_BL)` bill would fall strictly below it), the
    array row is strictly positive (caps) yet window-INVARIANT — the cap oracle: a
    conduction term would move it with the window and double-count the branch — and
    `array + cablc` covers the whole branch plus the caps with no double-bill.
    This is the `cablc` validation slice (array + channel).
    """
    cfg = build_config()
    w = _w_full()
    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0]], dtype=torch.long)  # batch (2,)

    def run(config: Xue2020JsscCimMacroConfig) -> tuple[float, float, float]:
        macro = build_macro(config, device=device)
        macro.program(w.to(device))
        with Profiler() as prof, torch.no_grad():
            macro.vec_mat_mul(x.to(device), quantization_mode=QUANTIZATION_MODE, adc_bits=TINY_ADC_BITS)
        by_name = Reporter(macro).by_name(prof)
        cablc = by_name.get(".cablc", 0.0)
        array = by_name.get("array", 0.0)
        # No cell row double-bills the branch (the cell is a non-reporter).
        assert "cell" not in by_name
        whole = _whole_input_branch(macro, x.to(device))
        return cablc, array, whole

    cablc, array, whole = run(cfg)
    # The witness must actually draw BL current at V_BLC > 0, else the whole branch
    # is zero and the collapse cannot be distinguished.
    assert whole > 0.0, f"witness draws no branch current: whole={whole}"
    # cablc bills the WHOLE input branch (VDD * I_DL), matching the re-solve — NOT
    # the clamp-side (VDD - V_BL) fraction alone.
    assert cablc == pytest.approx(whole), f"cablc {cablc} != reconstructed whole branch {whole}"
    # The array self-bills its capacitive cycling only (strictly positive, no conduction).
    assert array > 0.0, f"array row {array} must bill its capacitive cycling"
    # The two together cover the whole branch plus the caps — the cablc slice is
    # array + channel: no energy lost, no double-bill.
    assert cablc + array > whole, f"array + cablc {cablc + array} must exceed the whole branch {whole} by the caps"

    # The array row is the CAP oracle: purely capacitive, hence window-INVARIANT,
    # while cablc (the whole conduction branch) scales with the window. A conduction
    # term left in the array would move the array row and double-count the branch.
    cablc_wide, array_wide, whole_wide = run(dataclasses.replace(cfg, t_settle__ns=cfg.t_settle__ns + 3.0))
    assert array_wide == pytest.approx(array), (
        f"array row moved with the conduction window {array} -> {array_wide} (conduction leaked back into the array)"
    )
    assert cablc_wide == pytest.approx(whole_wide), f"cablc {cablc_wide} != reconstructed whole branch {whole_wide}"
    assert cablc_wide > cablc, f"cablc must grow with the conduction window {cablc} -> {cablc_wide}"


# ---------------------------------------------------------------------------
# Shared core supply
# ---------------------------------------------------------------------------


def test_array_cap_row_rides_the_shared_core_supply(device: torch.device) -> None:
    """Every array-node capacitance is billed against the shared `vdd__V`."""
    base = build_config()
    w, x = _w_full(), _x_full(2)

    def array_row(config: Xue2020JsscCimMacroConfig) -> float:
        prof, reporter = _run(config, w, x, device=device)
        return reporter.by_name(prof)["array"]

    e_base = array_row(base)
    raised = array_row(dataclasses.replace(base, vdd__V=2.0 * base.vdd__V))
    assert e_base > 0.0
    assert raised == pytest.approx(2.0 * e_base)


# ---------------------------------------------------------------------------
# Control per-access count
# ---------------------------------------------------------------------------


def test_control_channel_count_mux_times_batch(device: torch.device) -> None:
    """Control fires once per access: energy == `e_control_per_op * mux_factor * batch` (n_io-independent)."""
    cfg = build_config()  # K=2, mux_factor=2
    e_per_op = cfg.e_control_per_op__fJ
    mux = cfg.mux_factor
    w = _w_full()

    # Single sample (no batch axis): mux_factor accesses.
    e1 = _channel_energies(cfg, w, torch.tensor([1, 2, 1, 0], dtype=torch.long), device=device)["control"]
    assert e1 == pytest.approx(e_per_op * mux * 1)

    # Batch of 3: mux_factor x 3 accesses, independent of the io_num=2 lanes.
    x_batch = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0], [0, 1, 2, 3]], dtype=torch.long)
    e_batch = _channel_energies(cfg, w, x_batch, device=device)["control"]
    assert e_batch == pytest.approx(e_per_op * mux * 3)


# ---------------------------------------------------------------------------
# PN-ISUB module row: t_other conduction + per-op comparator decision
# ---------------------------------------------------------------------------


def test_pn_isub_row_present_and_uses_t_other(device: torch.device) -> None:
    """The `pn_isub` module row bills the three branches over `t_other` + the per-op decision.

    It rides `t_other` (via `t_settle`), so it grows with `t_settle` but is
    invariant to the sampled-bit windows `t_sample` (the PN-ISUB conducts only
    in the live/tail window).
    """
    w, x = _w_full(), _x_full(3)

    def pnisub(cfg: Xue2020JsscCimMacroConfig) -> float:
        return _channel_energies(cfg, w, x, device=device)["pn_isub"]

    base = pnisub(build_config(input_bit_num=3, t_sample__ns=(2.0, 3.0), t_settle__ns=1.0))
    more_settle = pnisub(build_config(input_bit_num=3, t_sample__ns=(2.0, 3.0), t_settle__ns=3.0))
    more_sample = pnisub(build_config(input_bit_num=3, t_sample__ns=(5.0, 7.0), t_settle__ns=1.0))
    assert base > 0.0
    assert more_settle > base, "pn_isub must grow with t_settle (rides t_other)"
    assert more_sample == pytest.approx(base), "pn_isub must be invariant to the sampled-bit windows"


# ---------------------------------------------------------------------------
# Window linearity (each knob) + control window-invariance
# ---------------------------------------------------------------------------


def test_read_channels_linear_in_t_sample(device: torch.device) -> None:
    """Each read channel is linear (collinear over 3 equally-spaced values) in `t_sample[0]`; control invariant."""
    w, x = _w_full(), _x_full(3)
    energies = [
        _channel_energies(
            build_config(input_bit_num=3, t_sample__ns=(ts, 3.0), t_settle__ns=1.0),
            w,
            x,
            device=device,
        )
        for ts in (2.0, 4.0, 6.0)  # equal spacing dt = 2
    ]
    for ch in _READ_ROWS:
        v = [e[ch] for e in energies]
        assert v[0] < v[1] < v[2], f"{ch} not increasing in t_sample[0]: {v}"
        # Equal spacing in the knob => equal spacing in energy (linear).
        assert (v[1] - v[0]) == pytest.approx(v[2] - v[1], rel=1e-9, abs=1e-9), f"{ch} non-linear: {v}"
    ctrl = [e["control"] for e in energies]
    assert ctrl[0] == pytest.approx(ctrl[1]), f"control not window-invariant: {ctrl}"
    assert ctrl[1] == pytest.approx(ctrl[2]), f"control not window-invariant: {ctrl}"


def test_read_channels_linear_in_t_settle(device: torch.device) -> None:
    """Each read channel is linear in `t_settle` (via `t_other`); control invariant."""
    w, x = _w_full(), _x_full(3)
    energies = [
        _channel_energies(
            build_config(input_bit_num=3, t_sample__ns=(2.0, 3.0), t_settle__ns=ts),
            w,
            x,
            device=device,
        )
        for ts in (1.0, 3.0, 5.0)  # equal spacing dt = 2
    ]
    for ch in _READ_ROWS:
        v = [e[ch] for e in energies]
        assert v[0] < v[1] < v[2], f"{ch} not increasing in t_settle: {v}"
        assert (v[1] - v[0]) == pytest.approx(v[2] - v[1], rel=1e-9, abs=1e-9), f"{ch} non-linear: {v}"
    ctrl = [e["control"] for e in energies]
    assert ctrl[0] == pytest.approx(ctrl[1]), f"control not window-invariant: {ctrl}"
    assert ctrl[1] == pytest.approx(ctrl[2]), f"control not window-invariant: {ctrl}"


def test_sc_held_leg_suffix_sum_law(device: torch.device) -> None:
    """The SINWP-SC held-leg window `sum_{j>=k} t_sample[j] + t_other` accumulates: later windows touch more legs.

    With `window_sc[k] = sum_{j>=k} t_sample[j] + t_other` (K=3): `t_sample[0]`
    rides only held leg 0, `t_sample[1]` rides legs 0 and 1, and `t_other`
    (via `t_settle`) rides all three. So the SC sensitivity strictly grows
    `d/dt_sample[0] < d/dt_sample[1] < d/dt_settle` (each step adds one more
    non-negative held-leg current) — the suffix-sum signature.
    """
    w, x = _w_full(), _x_full(3)

    def sc(cfg: Xue2020JsscCimMacroConfig) -> float:
        return _channel_energies(cfg, w, x, device=device)["sinwp_sc"]

    base = sc(build_config(input_bit_num=3, t_sample__ns=(2.0, 3.0), t_settle__ns=1.0))
    d_ts0 = (sc(build_config(input_bit_num=3, t_sample__ns=(4.0, 3.0), t_settle__ns=1.0)) - base) / 2.0
    d_ts1 = (sc(build_config(input_bit_num=3, t_sample__ns=(2.0, 5.0), t_settle__ns=1.0)) - base) / 2.0
    d_settle = (sc(build_config(input_bit_num=3, t_sample__ns=(2.0, 3.0), t_settle__ns=3.0)) - base) / 2.0

    # d_ts0 = S0, d_ts1 = S0 + S1, d_settle = S0 + S1 + S2 (all held-leg currents > 0).
    assert 0.0 < d_ts0 < d_ts1 < d_settle, f"suffix-sum ordering violated: {(d_ts0, d_ts1, d_settle)}"


def test_read_channel_per_bit_window_is_diagonal_not_suffix(device: torch.device) -> None:
    """cablc / dswct use PER-BIT windows: an isolated bit rides only its own `t_sample` — not the held suffix.

    The array/CABLC/DSWCT conduction is billed per input bit with `window_array`
    (sampled bit `k` -> `t_sample[k]`; live bit -> `t_other`), unlike the
    SINWP-SC held legs whose window is the suffix sum `sum_{j>=k} t_sample[j] +
    t_other`. Here only input bit 0 conducts (`x == 1`): its per-bit window is
    `t_sample[0]` alone, so the cablc/dswct channels MOVE with `t_sample[0]`
    yet are INVARIANT to `t_sample[1]` and `t_settle` — a diagonal signature
    the suffix-held window (bit 0 riding `t_sample[1]` and `t_other` too)
    breaks. The SINWP-SC channel is the positive control: its held bit-0 leg DOES
    ride the later windows.
    """
    w = _w_full(input_num=TINY_INPUT_NUM)
    x = torch.ones(TINY_INPUT_NUM, dtype=torch.long)  # bit 0 set, bits 1..K-1 zero

    def read(t_sample: tuple[float, ...], t_settle: float) -> dict[str, float]:
        return _channel_energies(
            build_config(input_bit_num=3, t_sample__ns=t_sample, t_settle__ns=t_settle), w, x, device=device
        )

    base = read((2.0, 3.0), 1.0)
    bump_ts0 = read((4.0, 3.0), 1.0)  # later t_sample[0]
    bump_ts1 = read((2.0, 5.0), 1.0)  # later t_sample[1] — no bit-1 conduction
    bump_settle = read((2.0, 3.0), 3.0)  # later t_other — no live-bit conduction

    for ch in ("cablc", "dswct"):
        assert bump_ts0[ch] > base[ch], f"{ch} must ride the conducting bit 0's window t_sample[0]: {base[ch]}"
        assert bump_ts1[ch] == pytest.approx(base[ch]), (
            f"{ch} bit-0 conduction leaked into t_sample[1] (suffix-held regression): {base[ch]} -> {bump_ts1[ch]}"
        )
        assert bump_settle[ch] == pytest.approx(base[ch]), (
            f"{ch} bit-0 conduction leaked into t_other/t_settle (suffix-held regression): "
            f"{base[ch]} -> {bump_settle[ch]}"
        )
    # Positive control: the SINWP-SC held bit-0 leg's window IS the suffix sum.
    assert bump_ts1["sinwp_sc"] > base["sinwp_sc"], (
        f"held bit-0 leg must ride the suffix window t_sample[1]: {base['sinwp_sc']} -> {bump_ts1['sinwp_sc']}"
    )


def test_live_bit_conducts_in_t_other_independent_of_sampling(device: torch.device) -> None:
    """The live (K-1) bit has no sample phase — its conduction window is `t_other` regardless of `t_sample`.

    Config-level: the last entry of both window vectors equals `t_other`. The
    array/CABLC legs are independent per bit, so the `t_settle` sensitivity of a
    read channel is exactly the live-bit leg — and, because that leg's current is
    a DC solve of the live plane (window-independent), the slope is INVARIANT to
    the sampled-bit windows.
    """
    cfg = build_config(input_bit_num=3, t_sample__ns=(2.0, 3.0), t_settle__ns=1.0)
    assert cfg.window_array__ns[-1] == cfg.t_other__ns
    assert cfg.window_sc__ns[-1] == cfg.t_other__ns

    w, x = _w_full(), _x_full(3)

    def cablc_settle_slope(t_sample: tuple[float, ...]) -> float:
        lo = _channel_energies(
            build_config(input_bit_num=3, t_sample__ns=t_sample, t_settle__ns=1.0), w, x, device=device
        )["cablc"]
        hi = _channel_energies(
            build_config(input_bit_num=3, t_sample__ns=t_sample, t_settle__ns=3.0), w, x, device=device
        )["cablc"]
        return (hi - lo) / 2.0

    slope_a = cablc_settle_slope((2.0, 3.0))
    slope_b = cablc_settle_slope((7.0, 5.0))  # very different sampling windows
    assert slope_a > 0.0, "the live bit must conduct in t_other"
    assert slope_a == pytest.approx(slope_b, rel=1e-9, abs=1e-9), (
        f"live-bit t_other leg leaked into sampling: {(slope_a, slope_b)}"
    )


# ---------------------------------------------------------------------------
# TMCSA phase billing: fixed model + phase-window growth + kernel ADC silence
# ---------------------------------------------------------------------------


def test_tmcsa_is_pure_fixed_when_phase_windows_zero(device: torch.device) -> None:
    """All-zero PH2/PH3 windows reduce the `tmcsa` row to `adc_bits * e_fixed` per converted element.

    The fixed constant is the TMCSA MODULE's `e_fixed_per_op__fJ` — the
    kernel `adc_config` constant is deliberately a DIFFERENT witness value,
    so a billing-duty regression (the kernel ADC billing again) is caught.
    """
    base = build_config()
    cfg = _with_tmcsa(base, t_ph2=(0.0, 0.0, 0.0), t_ph3=(0.0, 0.0, 0.0))
    assert cfg.tmcsa_config.e_fixed_per_op__fJ != cfg.adc_config.e_fixed_per_op__fJ
    w = _w_full()
    x = torch.tensor([1, 2, 3, 1], dtype=torch.long)  # single access (no batch axis)
    prof, reporter = _run(cfg, w, x, device=device)
    e_tmcsa = reporter.by_name(prof)["tmcsa"]

    adc_bits = cfg.adc_config.bits
    e_fixed = cfg.tmcsa_config.e_fixed_per_op__fJ
    # One converted element per (mux slot, io lane): i_sub is [gs, gn] with no batch axis.
    count = cfg.mux_factor * (TINY_OUTPUT_NUM // cfg.mux_factor)
    assert e_tmcsa == pytest.approx(adc_bits * e_fixed * count)


def test_tmcsa_grows_with_phase_windows_kernel_knobs_dead(device: torch.device) -> None:
    """The `tmcsa` row grows with `t_ph2` / `t_ph3`; the kernel ADC knobs move NOTHING.

    The kernel SarIadc is built with `enable_energy_record=False`: scaling
    its `v_rail` / `t_conduct` / `e_fixed` leaves the whole profile
    bit-identical (the deadness oracle), and no `adc` dynamic row exists.
    """
    w = _w_full()
    x = torch.tensor([1, 2, 3, 1], dtype=torch.long)
    base = build_config()  # witness phases: t_ph2 = 0.2 * step, t_ph3 = 0.3 * step

    def tmcsa(cfg: Xue2020JsscCimMacroConfig) -> float:
        prof, reporter = _run(cfg, w, x, device=device)
        by_name = reporter.by_name(prof)
        assert "adc" not in by_name, "kernel ADC must be energy-silent"
        return by_name["tmcsa"]

    base_t_ph2 = base.tmcsa_config.t_ph2_per_step__ns
    base_t_ph3 = base.tmcsa_config.t_ph3_per_step__ns
    e_fixed_only = tmcsa(_with_tmcsa(base, t_ph2=(0.0, 0.0, 0.0), t_ph3=(0.0, 0.0, 0.0)))
    e_base = tmcsa(base)
    e_more_ph2 = tmcsa(_with_tmcsa(base, t_ph2=tuple(2.0 * t for t in base_t_ph2)))
    e_more_ph3 = tmcsa(_with_tmcsa(base, t_ph3=tuple(2.0 * t for t in base_t_ph3)))

    assert e_base > e_fixed_only, "phase conduction must add on top of the fixed model"
    assert e_more_ph2 > e_base, "energy must grow with t_ph2"
    assert e_more_ph3 > e_base, "energy must grow with t_ph3"

    # Kernel-knob deadness: scaling the kernel conduction knobs changes nothing.
    e_kernel_scaled = tmcsa(_with_adc(base, t_conduct=(9.0, 9.0, 9.0), v_rail=5.0, e_fixed=123.0))
    assert e_kernel_scaled == pytest.approx(e_base), "kernel ADC energy knobs must be dead in this scheme"
