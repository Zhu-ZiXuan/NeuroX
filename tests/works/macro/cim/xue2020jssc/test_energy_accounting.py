"""Eager energy-accounting laws for the xue2020jssc SINWP 1T1R CIM sub-array.

Pins the branch-ownership / channel-billing contract of S4.3 (with the
t_cycle static-time-base ruling) on the hand-built near-ideal witness macro
(``_utils.build_config``). Every check is a LAW read off the profiler report, not
a magic number: the coefficients are whatever the deterministic all-off analog
chain produces, and the assertions constrain how the billed energy MOVES.

Coverage:

  * the five macro-billed channels ``cablc`` / ``dswct`` / ``sinwp_sc`` /
    ``pn_isub`` / ``control`` appear under their exact dotted names (the macro
    root is named ``""`` so a channelled row reads ``".<channel>"``), and the
    self-billing dynamic module rows are the restored array ``array`` and the
    TMCSA ``tmcsa``,
  * the dissolved DSWCT / SINWP-SC / PN-ISUB combining logs NOTHING of its own
    as a module — the macro bills those branches under the channels — and the
    removed kernel primitives (current mirror / adder / subtractor) plus the
    non-reporter cell leave no dynamic or static row,
  * the static report seats the reporter leaves (control / adc_current_reference /
    cablc / sl_driver / pn_isub / tmcsa / array + the macro root) and does NOT seat the
    dissolved DSWCT / SINWP-SC combining or the (non-target) cell,
  * **the static-energy time base is ``t_cycle`` (50 ns), NOT the conduction
    windows**: doubling ``t_cycle`` doubles the leakage (static) energy while the
    dynamic energy is unchanged; doubling a conduction window (``t_settle``)
    scales the read channels but leaves the static energy — and the
    window-invariant control channel — untouched,
  * the input branch conduction is billed WHOLE by the macro on the ``cablc``
    channel (``V_DD * I_DL`` over the per-bit window — the macro owns the
    conduction window), while the array module row bills ONLY its wire / node
    capacitive cycling: the channel matches the reconstructed whole branch
    exactly, the array row is strictly positive yet window-invariant (the cap
    oracle), and array + channel cover the whole branch plus the caps with no
    double-bill,
  * the control channel fires once per access (``mux_factor`` mux steps x batch),
  * each read channel is LINEAR in every window knob (``t_sample[k]``,
    ``t_settle``), the SC held-leg SUFFIX-SUM law (window
    ``sum_{j>=k} t_sample[j] + t_other``) holds while cablc / dswct use the
    per-bit DIAGONAL window, and the live (K-1) bit conducts in ``t_other``
    regardless of the sampling windows,
  * the TMCSA reduces to the pure fixed-energy model ``adc_bits x e_fixed`` per
    converted element when ``t_conduct_per_step`` is all-zero, and its energy
    grows with ``v_rail`` and ``t_conduct`` once nonzero.

Runs eagerly (dynamo disabled) so the ``@torch.compile`` solver leaf is not
unrolled.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest
import torch
import torch._dynamo
from torch import Tensor

from neurox.common.profiler import NeuroxProfiler, ProfilerReport
from tests.works.macro.cim.xue2020jssc._utils import (
    ADC_MODE,
    TINY_ADC_BITS,
    TINY_COL_NUM,
    TINY_ROW_NUM,
    Xue2020JsscCimMacro,
    Xue2020JsscCimMacroConfig,
    build_config,
    build_macro,
    encode_weights,
)

_CHANNELS = ("cablc", "dswct", "sinwp_sc", "pn_isub", "control")
_READ_CHANNELS = ("cablc", "dswct", "sinwp_sc")  # window-dependent conduction channels
# Kernel primitives removed by the rewrite (S3) + the dissolved combining blocks +
# the non-reporter cell: none of these may surface as a dynamic or static row. The
# array (S4.1) is NOT here — it self-bills its own capacitive module row.
_REMOVED_ROWS = (
    "dswct_msb",
    "dswct_lsb",
    "sc_adder",
    "current_mirror",
    "current_adder",
    "current_subtractor",
    "cell",
)


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is ``@torch.compile``; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _w_full(col: int = TINY_COL_NUM, row: int = TINY_ROW_NUM) -> Tensor:
    """All-``+1`` weights so every physical column / IO / mux slot conducts."""
    return torch.ones((col, row), dtype=torch.long)


def _x_full(k: int, row: int = TINY_ROW_NUM) -> Tensor:
    """Full-scale K-bit input (every input bit set on every row) — all sub-phase legs conduct."""
    return torch.full((row,), (1 << k) - 1, dtype=torch.long)


def _run(
    config: Xue2020JsscCimMacroConfig,
    w: Tensor,
    x: Tensor,
    *,
    device: torch.device,
    adc_mode: int = ADC_MODE,
    adc_bits: int = TINY_ADC_BITS,
) -> tuple[NeuroxProfiler, ProfilerReport]:
    """Build + fabricate a fresh macro, program ``w``, profile one VMM on ``x``."""
    macro = build_macro(config, device=device)
    macro.program(encode_weights(w.to(device)))
    with NeuroxProfiler() as prof, torch.no_grad():
        macro.vec_mat_mul(x.to(device), adc_mode=adc_mode, adc_bits=adc_bits)
    return prof, prof.report(macro)


def _channels(report: ProfilerReport) -> dict[str, float]:
    """Per-channel dynamic energy [fJ] keyed by bare channel name (root macro => ``".<ch>"``)."""
    by_name = report.energy_by_name
    return {ch: by_name.get(f".{ch}", 0.0) for ch in _CHANNELS}


def _channel_energies(
    config: Xue2020JsscCimMacroConfig, w: Tensor, x: Tensor, *, device: torch.device
) -> dict[str, float]:
    _prof, report = _run(config, w, x, device=device)
    return _channels(report)


def _with_adc(
    config: Xue2020JsscCimMacroConfig,
    *,
    t_conduct: tuple[float, ...] | None = None,
    v_rail: float | None = None,
) -> Xue2020JsscCimMacroConfig:
    """Replace only the TMCSA conduction knobs (energy-path only; windows untouched)."""
    kw: dict[str, object] = {}
    if t_conduct is not None:
        kw["t_conduct_per_step__ns"] = t_conduct
    if v_rail is not None:
        kw["v_rail__V"] = v_rail
    return dataclasses.replace(config, adc_config=dataclasses.replace(config.adc_config, **kw))


def _whole_input_branch(macro: Xue2020JsscCimMacro, x: Tensor) -> float:
    """Reconstruct the WHOLE input-branch read energy from a re-solve, batch-summed [fJ].

    Re-runs the array DC solve (cells + wire IR drop) per WL plane against the same
    ``window_array`` windows :meth:`vec_mat_mul` uses, matching it bit-for-bit
    (deterministic under all-off + eval), and returns the whole input branch the
    macro bills on the ``cablc`` channel::

        whole = sum_k  V_DD * I_DL * window_array[k]

    where ``I_DL`` is the per-column BL port current the solver returns and the
    conduction window ``t`` is the macro's, applied here post-solve. The array no
    longer carries any conduction term, so there is no clamp / cell split to
    reconstruct. The re-solve runs OUTSIDE any profiler so it logs nothing of its own.
    """
    cfg = macro.config
    device = x.device
    v_dd = cfg.v_dd__V
    bl_v_ref = torch.tensor(cfg.v_bl_clamp__V, dtype=torch.float64, device=device)
    sl_v_ref = torch.tensor(0.0, dtype=torch.float64, device=device)
    x_long = x.long()
    window = cfg.window_array__ns

    whole = 0.0
    for k in range(cfg.input_bit_num):
        plane = (x_long >> k) & 1  # [*batch, row]
        v_wl = macro.wl_dac.convert(plane)  # [*batch, row]
        steady = macro.array.solve_array(
            v_wl,
            bl_driver=macro.cablc,
            bl_v_ref__V=bl_v_ref,
            sl_driver=macro.sl_driver,
            sl_v_ref__V=sl_v_ref,
        )
        i_bl = steady.i_bl_port__uA  # [*batch, phys_col]
        whole += float(((v_dd * i_bl).sum(dim=-1) * window[k]).sum())
    return whole


# ---------------------------------------------------------------------------
# Channel presence + exact names
# ---------------------------------------------------------------------------


def test_channels_present_with_exact_dotted_names(device: torch.device) -> None:
    """The five macro-billed channels appear as ``".cablc/.dswct/.sinwp_sc/.pn_isub/.control"``."""
    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0]], dtype=torch.long)  # batch (2,)
    _prof, report = _run(build_config(), _w_full(), x, device=device)
    by_name = report.energy_by_name
    for ch in _CHANNELS:
        key = f".{ch}"
        assert key in by_name, f"missing channel {key}; have {sorted(by_name)}"
        assert by_name[key] > 0.0, f"non-positive {key}: {by_name[key]}"


def test_array_and_adc_self_bill_dynamic_rows(device: torch.device) -> None:
    """The array + the TMCSA self-bill dynamic rows; no cell / pn_isub row exists (S4.1)."""
    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0]], dtype=torch.long)
    _prof, report = _run(build_config(), _w_full(), x, device=device)
    by_name = report.energy_by_name
    # The array bills its capacitive cycling (caps only); the TMCSA bills sensing.
    assert by_name.get("array", 0.0) > 0.0, f"missing/empty array row; have {sorted(by_name)}"
    assert by_name.get("tmcsa", 0.0) > 0.0, f"missing/empty tmcsa row; have {sorted(by_name)}"
    # The whole input branch is a macro channel (.cablc), PN-ISUB is a macro
    # channel (.pn_isub); the non-reporter cell and the PN-ISUB seat emit no
    # self-billed dynamic row.
    for absent in ("cell", "pn_isub"):
        assert absent not in by_name, f"unexpected self-billing module row {absent}: {sorted(by_name)}"


def test_removed_primitives_emit_nothing(device: torch.device) -> None:
    """The dissolved combining + removed kernel primitives log neither dynamic nor static rows."""
    macro = build_macro(build_config(), device=device)
    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0]], dtype=torch.long)
    with NeuroxProfiler() as prof, torch.no_grad():
        macro.program(encode_weights(_w_full().to(device)))
        macro.vec_mat_mul(x.to(device), adc_mode=ADC_MODE, adc_bits=TINY_ADC_BITS)
    by_name = prof.report(macro).energy_by_name
    static = {r.qualified_name for r in NeuroxProfiler.collect_static(macro)}
    for name in _REMOVED_ROWS:
        assert name not in by_name, f"removed block {name} self-billed dynamic energy; have {sorted(by_name)}"
        assert not any(k.startswith(f"{name}.") for k in by_name), f"{name} emitted a channel row"
        assert name not in static, f"removed block {name} seated static PPA; have {sorted(static)}"


# ---------------------------------------------------------------------------
# Static report seats
# ---------------------------------------------------------------------------


def test_static_report_seats_reporters_only(device: torch.device) -> None:
    """The static walk seats the reporter leaves (+ the macro root) and NOT the dissolved combining."""
    macro = build_macro(build_config(), device=device)
    static = {r.qualified_name: r.leakage_power__uW for r in NeuroxProfiler.collect_static(macro)}
    # Seats: the macro root (named ""), control (UnmodeledBlock),
    # adc_current_reference, the clamp drivers, the PN-ISUB static seat
    # (UnmodeledBlock), the ADC, and the restored array (reporter leaf: cell grid +
    # wire infrastructure PPA).
    for seat in ("", "control", "adc_current_reference", "cablc", "sl_driver", "pn_isub", "tmcsa", "array"):
        assert seat in static, f"missing static seat {seat!r}; have {sorted(static)}"
        assert static[seat] > 0.0, f"non-positive leakage seat {seat!r}: {static[seat]}"
    # The dissolved combining + the non-target cell are absent from the walk.
    for absent in _REMOVED_ROWS:
        assert absent not in static, f"{absent} must not seat static PPA"


# ---------------------------------------------------------------------------
# Static-energy time base = t_cycle, NOT the conduction windows (CRITICAL)
# ---------------------------------------------------------------------------


def test_static_energy_scales_with_t_cycle_not_conduction_windows(device: torch.device) -> None:
    """Leakage (static) energy tracks ``t_cycle`` alone; the conduction windows drive only dynamic.

    ``leakage_energy = leakage_power x total_latency`` and the macro is the sole
    latency emitter (``total_latency = t_cycle x serial``). So doubling
    ``t_cycle`` doubles the static energy while every dynamic channel — which
    bills over the conduction windows, never ``t_cycle`` — is unchanged; and
    doubling a conduction window (``t_settle``) leaves the static energy (and the
    window-invariant control channel) untouched while the read channels grow.
    """
    w, x = _w_full(), _x_full(2)
    base_cfg = build_config(t_sample__ns=(1.0,), t_settle__ns=2.0, t_cycle__ns=50.0)

    prof_base, report_base = _run(base_cfg, w, x, device=device)
    static_base = report_base.leakage_energy__fJ
    dyn_base = prof_base.total_dynamic_energy__fJ
    assert static_base > 0.0 and dyn_base > 0.0

    # --- Double t_cycle: static energy doubles, dynamic unchanged ---
    prof_2t, report_2t = _run(dataclasses.replace(base_cfg, t_cycle__ns=100.0), w, x, device=device)
    assert report_2t.leakage_energy__fJ == pytest.approx(2.0 * static_base)
    assert prof_2t.total_dynamic_energy__fJ == pytest.approx(dyn_base)

    # --- Double a conduction window (t_settle -> t_other): static unchanged, read channels grow ---
    prof_win, report_win = _run(dataclasses.replace(base_cfg, t_settle__ns=4.0), w, x, device=device)
    assert report_win.leakage_energy__fJ == pytest.approx(static_base)  # static invariant to windows
    assert prof_win.total_dynamic_energy__fJ > dyn_base  # dynamic grows with the window
    # The control channel is window-invariant; the read channels moved.
    ch_base = _channels(report_base)
    ch_win = _channels(report_win)
    assert ch_win["control"] == pytest.approx(ch_base["control"])
    for ch in _READ_CHANNELS:
        assert ch_win[ch] > ch_base[ch], f"read channel {ch} did not grow with t_settle"


# ---------------------------------------------------------------------------
# Input-branch ownership: cablc bills the whole branch, array bills caps only
# ---------------------------------------------------------------------------


def test_input_branch_billed_whole_by_cablc_array_bills_caps_only(device: torch.device) -> None:
    """The macro bills the WHOLE input branch on ``.cablc``; the array bills caps only (S4.1 + S3).

    The ``.cablc`` channel bills the whole input branch ``V_DD * I_DL`` over the
    per-bit conduction window (the macro owns the window); the array module row
    bills ONLY its wire / node capacitive cycling — no conduction. Reconciled
    against a re-solve: ``.cablc`` matches the reconstructed WHOLE branch EXACTLY
    (a clamp-side-only ``(V_DD - V_BL)`` bill would fall strictly below it), the
    array row is strictly positive (caps) yet window-INVARIANT — the cap oracle: a
    conduction term would move it with the window and double-count the branch — and
    ``array + cablc`` covers the whole branch plus the caps with no double-bill.
    This is the ``cablc`` validation slice (array + channel).
    """
    cfg = build_config()
    w = _w_full()
    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0]], dtype=torch.long)  # batch (2,)

    def run(config: Xue2020JsscCimMacroConfig) -> tuple[float, float, float]:
        macro = build_macro(config, device=device)
        macro.program(encode_weights(w.to(device)))
        with NeuroxProfiler() as prof, torch.no_grad():
            macro.vec_mat_mul(x.to(device), adc_mode=ADC_MODE, adc_bits=TINY_ADC_BITS)
        report = prof.report(macro)
        cablc = report.energy_by_name.get(".cablc", 0.0)
        array = report.energy_by_name.get("array", 0.0)
        # No cell row double-bills the branch (the cell is a non-reporter).
        assert "cell" not in report.energy_by_name
        whole = _whole_input_branch(macro, x.to(device))
        return cablc, array, whole

    cablc, array, whole = run(cfg)
    # The witness must actually draw BL current at V_BLC > 0, else the whole branch
    # is zero and the collapse cannot be distinguished.
    assert whole > 0.0, f"witness draws no branch current: whole={whole}"
    # cablc bills the WHOLE input branch (V_DD * I_DL), matching the re-solve — NOT
    # the clamp-side (V_DD - V_BL) fraction alone.
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
# Control per-access count
# ---------------------------------------------------------------------------


def test_control_channel_count_mux_times_batch(device: torch.device) -> None:
    """Control fires once per access: energy == ``e_control_per_op * mux_factor * batch`` (n_io-independent)."""
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
# PN-ISUB channel: t_other conduction + per-op comparator decision
# ---------------------------------------------------------------------------


def test_pn_isub_channel_present_and_uses_t_other(device: torch.device) -> None:
    """The ``pn_isub`` channel bills the three branches over ``t_other`` + the per-op decision.

    It rides ``t_other`` (via ``t_settle``), so it grows with ``t_settle`` but is
    invariant to the sampled-bit windows ``t_sample`` (the PN-ISUB conducts only
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
    """Each read channel is linear (collinear over 3 equally-spaced values) in ``t_sample[0]``; control invariant."""
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
    for ch in _READ_CHANNELS:
        v = [e[ch] for e in energies]
        assert v[0] < v[1] < v[2], f"{ch} not increasing in t_sample[0]: {v}"
        # Equal spacing in the knob => equal spacing in energy (linear).
        assert (v[1] - v[0]) == pytest.approx(v[2] - v[1], rel=1e-9, abs=1e-9), f"{ch} non-linear: {v}"
    ctrl = [e["control"] for e in energies]
    assert ctrl[0] == pytest.approx(ctrl[1]) and ctrl[1] == pytest.approx(ctrl[2]), (
        f"control not window-invariant: {ctrl}"
    )


def test_read_channels_linear_in_t_settle(device: torch.device) -> None:
    """Each read channel is linear in ``t_settle`` (via ``t_other``); control invariant."""
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
    for ch in _READ_CHANNELS:
        v = [e[ch] for e in energies]
        assert v[0] < v[1] < v[2], f"{ch} not increasing in t_settle: {v}"
        assert (v[1] - v[0]) == pytest.approx(v[2] - v[1], rel=1e-9, abs=1e-9), f"{ch} non-linear: {v}"
    ctrl = [e["control"] for e in energies]
    assert ctrl[0] == pytest.approx(ctrl[1]) and ctrl[1] == pytest.approx(ctrl[2]), (
        f"control not window-invariant: {ctrl}"
    )


def test_sc_held_leg_suffix_sum_law(device: torch.device) -> None:
    """The SINWP-SC held-leg window ``sum_{j>=k} t_sample[j] + t_other`` accumulates: later windows touch more legs.

    With ``window_sc[k] = sum_{j>=k} t_sample[j] + t_other`` (K=3): ``t_sample[0]``
    rides only held leg 0, ``t_sample[1]`` rides legs 0 and 1, and ``t_other``
    (via ``t_settle``) rides all three. So the SC sensitivity strictly grows
    ``d/dt_sample[0] < d/dt_sample[1] < d/dt_settle`` (each step adds one more
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
    """cablc / dswct use PER-BIT windows: an isolated bit rides only its own ``t_sample`` — not the held suffix.

    The array/CABLC/DSWCT conduction is billed per input bit with ``window_array``
    (sampled bit ``k`` -> ``t_sample[k]``; live bit -> ``t_other``), unlike the
    SINWP-SC held legs whose window is the suffix sum ``sum_{j>=k} t_sample[j] +
    t_other``. Here only input bit 0 conducts (``x == 1``): its per-bit window is
    ``t_sample[0]`` alone, so the cablc/dswct channels MOVE with ``t_sample[0]``
    yet are INVARIANT to ``t_sample[1]`` and ``t_settle`` — a diagonal signature
    the suffix-held window (bit 0 riding ``t_sample[1]`` and ``t_other`` too)
    breaks. The SINWP-SC channel is the positive control: its held bit-0 leg DOES
    ride the later windows.
    """
    w = _w_full(row=TINY_ROW_NUM)
    x = torch.ones(TINY_ROW_NUM, dtype=torch.long)  # bit 0 set, bits 1..K-1 zero

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
    """The live (K-1) bit has no sample phase — its conduction window is ``t_other`` regardless of ``t_sample``.

    Config-level: the last entry of both window vectors equals ``t_other``. The
    array/CABLC legs are independent per bit, so the ``t_settle`` sensitivity of a
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
# TMCSA (ADC) energy: fixed model + conduction growth
# ---------------------------------------------------------------------------


def test_adc_energy_is_pure_fixed_when_t_conduct_zero(device: torch.device) -> None:
    """All-zero ``t_conduct_per_step`` reduces the TMCSA to ``adc_bits * e_fixed`` per converted element."""
    cfg = _with_adc(build_config(), t_conduct=(0.0, 0.0, 0.0))
    w = _w_full()
    x = torch.tensor([1, 2, 3, 1], dtype=torch.long)  # single access (no batch axis)
    _prof, report = _run(cfg, w, x, device=device)
    e_tmcsa = report.energy_by_name["tmcsa"]

    adc_bits = cfg.adc_config.bits
    e_fixed = cfg.adc_config.e_fixed_per_op__fJ
    # One converted element per (mux slot, io lane): i_sub is [gs, gn] with no batch axis.
    count = cfg.mux_factor * cfg.io_num
    assert e_tmcsa == pytest.approx(adc_bits * e_fixed * count)


def test_adc_energy_grows_with_v_rail_and_t_conduct(device: torch.device) -> None:
    """With nonzero conduction the TMCSA energy exceeds the fixed model and grows with ``v_rail`` / ``t_conduct``."""
    w = _w_full()
    x = torch.tensor([1, 2, 3, 1], dtype=torch.long)
    base = build_config()  # t_conduct (0.1, 0.1, 0.1), v_rail 1.0

    def tmcsa(cfg: Xue2020JsscCimMacroConfig) -> float:
        _prof, report = _run(cfg, w, x, device=device)
        return report.energy_by_name["tmcsa"]

    e_fixed_only = tmcsa(_with_adc(base, t_conduct=(0.0, 0.0, 0.0)))
    e_cond = tmcsa(base)
    e_more_conduct = tmcsa(_with_adc(base, t_conduct=(0.2, 0.2, 0.2)))
    e_more_vrail = tmcsa(_with_adc(base, v_rail=2.0))

    assert e_cond > e_fixed_only, "conduction term must add on top of the fixed model"
    assert e_more_conduct > e_cond, "energy must grow with t_conduct"
    assert e_more_vrail > e_cond, "energy must grow with v_rail"
