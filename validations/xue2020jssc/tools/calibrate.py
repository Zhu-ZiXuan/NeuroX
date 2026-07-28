"""Calibration-campaign orchestrator for the xue2020jssc CIM sub-array.

Re-derives the geometry-dependent DECLARED values and reports the energy-basis
result, then REPORTS the values for the operator to write back into ``params.toml``
(it never mutates the file; it fabricates nothing). Non-circular by construction:
the two peripheral seats are ADOPTED from Fig.18 shares (a declared adoption), the
whole read path stays pure physics (array IR-drop solve + declared windows), and
``p_zero`` is LOCKED to the read-path physics -- never solved against the total.

Campaign:
  a. Re-derive the reference ladder at THIS geometry -- DRIVE THE MACRO DIRECTLY
     (a single ``+1`` weight, MAC value 0..2**adc_bits-1 over the live rows) and
     capture the pre-ADC ``i_sub`` staircase through the ADC's own observation
     prober (no manual value-path replay, no to_ideal); set ``i_refs`` to the
     adjacent midpoints and verify the ladder makes the ADC code equal the MAC.
  b. Declare the ADOPTED peripheral seats from the Fig.18 shares (control 29.2 %
     as a pure per-op constant, reference 23.7 % as 100 % static).
  c. LOCK ``p_zero`` where the pure-physics read path conducts its Fig.18 read-path
     share (47.1 % x 32.06 = 15.10 pJ/access) -- locked to the READ PATH, NOT to
     the total. With the adopted seats at their Fig.18 shares the total then falls
     out near 32.06 pJ/access (aim to HIT, slightly over OK); this is a consequence,
     not a fit.
  d. Report the informational breakdown at the LOCKED p_zero and emit the
     ``[calibrated]`` ladder + adopted-seat block + locked p_zero for write-back.

Run:
    TORCH_COMPILE_DISABLE=1 uv run python validations/xue2020jssc/tools/calibrate.py --n 1000 --device cpu

The report is logged as it is built and written to ``--report``.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import tomllib
from itertools import pairwise
from pathlib import Path

import torch

from neurox.primitive.analog.current_adc.base import IadcProber
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy
from neurox.works.macro.cim.xue2020jssc import Xue2020JsscCimMacro

_LOG = logging.getLogger(__name__)

_VAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_VAL_DIR))
import validate as V  # noqa: E402  the sibling measurement engine (single source of the reduction)

_ADC_MODE = 0
_ADC_BITS = 3


# ---------------------------------------------------------------------------
# Step (a): reference-ladder re-derivation at this geometry
# ---------------------------------------------------------------------------


def unit_isub_staircase(macro: Xue2020JsscCimMacro) -> tuple[list[float], list[float]]:
    """Probe the unit ``i_sub`` staircase and return ``(grid, midpoint_refs)`` [uA].

    Programs logical column 0 to ``+1`` (others 0) and drives inputs whose live
    rows (``0..active_row_num-1``) sum to MAC value ``0..2**adc_bits-1`` (greedy
    fill), then DRIVES THE MACRO DIRECTLY (``vec_mat_mul``) and captures the pre-ADC
    magnitude current ``i_sub`` through the ADC's own
    :class:`IadcProber` (``i_in__uA`` per convert) -- the true
    IR-drop array-solve path, no manual replay, no to_ideal. Column 0 sits at mux
    slot 0 of IO 0 (grouped ``gs=0, gn=0``), so ``i_sub[m, 0, 0]`` is the staircase.
    The reference taps are the adjacent midpoints ``0.5 * (I(m) + I(m+1))``.
    """
    cfg = macro.config
    dev = next(macro.buffers()).device
    k_num = cfg.input_bit_num
    row_num, active_size = macro.row_num, cfg.max_active_num
    x_max = (1 << k_num) - 1
    m_max = (1 << cfg.adc_config.bits) - 1

    w = torch.zeros((*macro.inst_shape, macro.row_num, macro.col_num), dtype=torch.long, device=dev)
    w[:, 0] = 1
    macro.program(w)

    x = torch.zeros((m_max + 1, row_num), dtype=torch.long, device=dev)
    for m in range(m_max + 1):
        rem = m
        for r in range(active_size):
            v = min(x_max, rem)
            x[m, r] = v
            rem -= v
        assert rem == 0, f"cannot reach MAC value {m} with {active_size} selected inputs of max {x_max}"

    # Drive the full macro (array IR-drop solve + readout chain) once and capture
    # the pre-ADC magnitude current through the ADC's own observation prober. One
    # convert per vec_mat_mul, so records[-1].i_in__uA is this call's I_SUB.
    with IadcProber() as probe, torch.no_grad():
        macro.vec_mat_mul(x.float(), adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
    i_sub = probe.records[-1].i_in__uA  # [m_max + 1, group_size, group_num]

    grid = [float(v) for v in i_sub[:, 0, 0].cpu()]
    mids = [0.5 * (grid[m] + grid[m + 1]) for m in range(m_max)]
    return grid, mids


def verify_ladder(macro: Xue2020JsscCimMacro, mids: list[float]) -> tuple[bool, list[int]]:
    """Install ``mids`` and check the full macro produces code == MAC magnitude on the staircase."""
    cfg = macro.config
    dev = next(macro.buffers()).device
    m_max = (1 << cfg.adc_config.bits) - 1
    row_num = macro.row_num
    active_size = cfg.max_active_num
    x_max = (1 << cfg.input_bit_num) - 1

    w = torch.zeros((*macro.inst_shape, macro.row_num, macro.col_num), dtype=torch.long, device=dev)
    w[:, 0] = 1
    macro.program(w)
    x = torch.zeros((m_max + 1, row_num), dtype=torch.long, device=dev)
    for m in range(m_max + 1):
        rem = m
        for r in range(active_size):
            v = min(x_max, rem)
            x[m, r] = v
            rem -= v
    with torch.no_grad():
        codes = macro.vec_mat_mul(x.float(), adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
    got = [int(codes[m, 0]) for m in range(m_max + 1)]
    ok = got == list(range(m_max + 1))
    return ok, got


# ---------------------------------------------------------------------------
# Step (c): p_zero LOCK to the read-path physics
# ---------------------------------------------------------------------------


def read_path_pJ(m: V.Measurement, read_path: tuple[str, ...]) -> float:
    """Total per-access energy of the pure-physics read-path slices [pJ]."""
    return sum(m.slice(s).total__pJ for s in read_path)


def lock_p_zero(
    macro: Xue2020JsscCimMacro,
    anchors: dict,
    *,
    target_pJ: float,
    read_path: tuple[str, ...],
    n: int,
    seed: int,
    batch: int,
    grid: list[float],
) -> tuple[float | None, list[tuple[float, float]]]:
    """Find the ``p_zero`` where the read path conducts ``target_pJ`` per access.

    The read-path energy decreases monotonically with input sparsity, so a linear
    scan of ``grid`` brackets the crossing of ``target_pJ`` (= 47.1 % x 32.06 pJ,
    the Fig.18 read-path share). Returns ``(p_star, points)`` with ``p_star`` the
    interpolated crossing (or ``None`` if the target is not bracketed inside the
    grid). ``p_star`` is locked to the READ PATH physics, never to the total.
    """
    points: list[tuple[float, float]] = []
    for p in grid:
        m = V.measure(macro, anchors, n=n, p_zero=p, seed=seed, batch=batch)
        points.append((p, read_path_pJ(m, read_path)))
    for (pa, ea), (pb, eb) in pairwise(points):
        if (ea - target_pJ) * (eb - target_pJ) <= 0 and ea != eb:
            frac = (ea - target_pJ) / (ea - eb)
            return pa + frac * (pb - pa), points
    return None, points


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _rebuild(cfg: CimMacroConfig, policy: CimMacroPolicy, device: torch.device) -> Xue2020JsscCimMacro:
    macro = CimMacro.from_config(
        config=cfg,
        policy=policy,
        input_num=256,
        output_num=128,
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    assert isinstance(macro, Xue2020JsscCimMacro)
    macro.to(device)
    macro.eval()
    macro.fabricate()
    return macro


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", type=Path, default=_VAL_DIR / "params.toml")
    ap.add_argument("--policy", type=Path, default=_VAL_DIR / "policy.toml")
    ap.add_argument("--anchors", type=Path, default=_VAL_DIR / "anchors.toml")
    ap.add_argument("--n", type=int, default=1000, help="Draws per energy measurement (per lock-grid point).")
    ap.add_argument(
        "--p-zero",
        type=float,
        default=None,
        help="Override the breakdown p_zero (default = the LOCKED value from step c).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu", help="cpu, cuda[:idx], or auto (free GPU via nvidia-smi).")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lock-step", type=float, default=0.05, help="p_zero grid step for the read-path lock scan.")
    ap.add_argument("--report", type=Path, default=_VAL_DIR / "calibration.md")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = V._resolve_device(args.device)
    with args.anchors.open("rb") as fh:
        anchors = tomllib.load(fh)
    base_cfg = CimMacroConfig.from_file(args.params, section="cim_macro")
    policy = CimMacroPolicy.from_file(args.policy, section="policy")

    tgt = anchors["target"]
    target_pJ = tgt["per_access__pJ"]
    per_sub_uW = tgt["total_macro__mW"] * 1000.0 / tgt["sub_array_num"]  # 641.25 uW
    shares = anchors["fig18_shares"]
    conv = anchors["conventions"]
    read_path = tuple(conv["read_path"])
    read_path_target_pJ = shares["read_path_sum"] / 100.0 * target_pJ  # 47.1% x 32.06 = 15.10 pJ
    t_cycle = base_cfg.t_cycle__ns

    log: list[str] = []

    def emit(line: str = "") -> None:
        _LOG.info("%s", line)
        log.append(line)

    emit("# xue2020jssc calibration campaign")
    emit()
    emit(
        f"Energy-basis calibration of the geometry-dependent DECLARED values in `params.toml` "
        f"against the `anchors.toml` target. Device {device}, float32, `.eval()`, all-off policy. "
        f"N = {args.n} draws/point, seed {args.seed}. Non-circular: control + reference ADOPTED from Fig.18, "
        f"the read path is pure physics (array IR-drop solve), and p_zero is LOCKED to the read-path share "
        f"(never solved against the total)."
    )
    emit()

    # --- Step (a): re-derive the reference ladder ---
    emit("## Step a -- reference ladder re-derivation at this geometry")
    emit()
    macro0 = _rebuild(base_cfg, policy, device)
    grid, mids = unit_isub_staircase(macro0)
    monotonic = all(grid[i] < grid[i + 1] for i in range(len(grid) - 1))
    emit(
        f"Unit i_sub staircase I(MAC=0..{len(grid) - 1}) [uA] "
        f"over {base_cfg.max_active_num} selected inputs:"
    )
    emit("")
    emit("    " + ", ".join(f"{g:.4f}" for g in grid))
    emit("")
    emit(f"Adjacent-midpoint reference taps I_REF[0..{len(mids) - 1}] [uA]:")
    emit("")
    emit("    " + ", ".join(f"{v:.6f}" for v in mids))
    emit("")
    cal_cfg = dataclasses.replace(
        base_cfg, reference_config=dataclasses.replace(base_cfg.reference_config, i_refs__uA=(tuple(mids),))
    )
    macro_cal = _rebuild(cal_cfg, policy, device)
    ladder_ok, got = verify_ladder(macro_cal, mids)
    emit(
        f"Staircase monotonic: {monotonic} (driven through the full array IR-drop solve). "
        f"Ladder verification (code == MAC magnitude): {ladder_ok} (codes {got})."
    )
    emit()

    # --- Step (b): declare the ADOPTED peripheral seats from Fig.18 ---
    emit("## Step b -- adopted peripheral seats (Fig.18, declared not fitted)")
    emit()
    control_uW = shares["control"] / 100.0 * per_sub_uW
    reference_uW = shares["reference"] / 100.0 * per_sub_uW
    control_energy_fJ = control_uW * t_cycle  # per access
    e_control_fJ = conv["control_dynamic_fraction"] * control_energy_fJ
    control_leak_uW = conv["control_static_fraction"] * control_uW
    reference_leak_uW = conv["reference_static_fraction"] * reference_uW
    emit(
        f"per-sub-array budget = {tgt['total_macro__mW']} mW / {tgt['sub_array_num']} = {per_sub_uW:.4f} uW "
        f"(= {target_pJ:.4f} pJ/access at {tgt['op_frequency__MHz']} MHz)."
    )
    emit(
        f"- Control {shares['control']}% -> {control_uW:.4f} uW = {control_energy_fJ:.4f} fJ/access; "
        f"pure per-op caliber (100% dynamic, 0% static): `e_control_per_op__fJ = {e_control_fJ:.4f}`, "
        f"`control_config.leakage_per_inst__uW = {control_leak_uW:.4f}`."
    )
    emit(
        f"- Reference {shares['reference']}% -> 100% static: "
        f"`reference_config.leakage_per_inst__uW = {reference_leak_uW:.5f}` uW."
    )
    emit()
    emit(
        "Read-path seats stay pure physics: cablc / sl / adc / pn_isub static seats declared small/zero "
        "(NOT solved); g_map, V_BL_CLAMP, wire R, and the conduction windows stay declared structural "
        "constants; the capacitive / per-op remainders (array caps, SINWP-SC `c_hold`, TMCSA phase widths "
        "+ `e_fixed_per_op`) are the pair-caliber energy campaign's domain (their fitted values live in "
        "`params.toml` under each block's `[calibrated]` comment; the measured per-slice breakdown is in "
        "`results.md`), never reverse-solved to fill the total."
    )
    emit()

    # --- Step (c): LOCK p_zero to the read-path physics ---
    emit("## Step c -- p_zero LOCK to the read-path physics share (15.10 pJ/access)")
    emit()
    tol = anchors["gate"]["hard_tolerance_relative"]
    n_grid = max(2, int(round(1.0 / args.lock_step)))
    lock_grid = [round(i * args.lock_step, 4) for i in range(n_grid)]
    p_star, points = lock_p_zero(
        macro_cal,
        anchors,
        target_pJ=read_path_target_pJ,
        read_path=read_path,
        n=args.n,
        seed=args.seed,
        batch=args.batch,
        grid=lock_grid,
    )
    x_lo, x_hi = anchors["data"]["input_range"]
    f0 = 1.0 / (x_hi - x_lo + 1)
    emit(f"Read-path target = {shares['read_path_sum']}% x {target_pJ} = {read_path_target_pJ:.3f} pJ/access.")
    emit("")
    emit("| p_zero | read-path [pJ/acc] | vs 15.10 |")
    emit("|--:|--:|--:|")
    for p, e in points:
        emit(f"| {p:.2f} | {e:8.3f} | {e / read_path_target_pJ:5.3f}x |")
    emit("")
    if p_star is None:
        emit(
            f"The read path does not cross {read_path_target_pJ:.3f} pJ/access inside p_zero in "
            f"[{lock_grid[0]:.2f}, {lock_grid[-1]:.2f}] -- the lock is UNBRACKETED at this geometry (reported, "
            f"not forced). Falling back to the anchors declared p_zero = {anchors['data']['p_zero']:.3f} for the "
            f"breakdown."
        )
        locked_p = float(anchors["data"]["p_zero"])
    else:
        locked_p = p_star
        emit(
            f"**LOCKED p_zero = {locked_p:.4f}** (marginal P(x=0) ~= {f0 + (1.0 - f0) * locked_p:.3f}): the "
            f"p_zero at which the pure-physics read path conducts its Fig.18 {shares['read_path_sum']}% share. "
            f"Locked to the READ PATH, not the total."
        )
    emit()
    breakdown_p = args.p_zero if args.p_zero is not None else locked_p

    # --- Step (d): breakdown at the LOCKED p_zero + write-back ---
    emit("## Step d -- total + breakdown at the LOCKED p_zero, and write-back values")
    emit()
    gate_m = V.measure(macro_cal, anchors, n=args.n, p_zero=breakdown_p, seed=args.seed, batch=args.batch)
    within, rel = V.gate(gate_m, anchors)
    emit(
        f"At the LOCKED p_zero = {breakdown_p:.4f}: total = {gate_m.total__pJ:.3f} pJ/access = "
        f"{gate_m.total__pJ / target_pJ:.3f}x (err {rel * 100:+.1f}%), within +-{tol * 100:.0f}%: "
        f"{'PASS' if within else 'FAIL'} (aim to HIT; the total falls out of the read-path lock + adopted seats)."
    )
    emit()
    emit("Informational per-block breakdown at the LOCKED p_zero (calibrated ladder):")
    emit("")
    emit(V.energy_table(gate_m, anchors))
    emit()
    emit("Values to write back into params.toml / anchors.toml (print only; nothing is mutated):")
    emit("")
    emit(f"- `reference_config.i_refs__uA = [[{', '.join(f'{v:.6f}' for v in mids)}]]`  # calibrated ladder")
    emit(f"- `e_control_per_op__fJ = {e_control_fJ:.4f}`  # adopted (control 100% dynamic, pure per-op)")
    emit(f"- `control_config.leakage_per_inst__uW = {control_leak_uW:.4f}`  # adopted (control 0% static)")
    emit(f"- `reference_config.leakage_per_inst__uW = {reference_leak_uW:.5f}`  # adopted (reference 100% static)")
    emit(f"- `anchors.toml [data].p_zero = {locked_p:.4f}`  # LOCKED to the read-path share (15.10 pJ), NOT the total")
    emit()
    emit(
        "Read-path physical declarations (g_map, V_BL_CLAMP, wire R, windows) are left as declared; the "
        "informational breakdown documents where each read-path slice lands vs its Fig.18 share."
    )
    emit()

    args.report.write_text("\n".join(log) + "\n")
    _LOG.info("\n[wrote report -> %s]", args.report)


if __name__ == "__main__":
    main()
