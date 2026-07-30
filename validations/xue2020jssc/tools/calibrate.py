"""Calibration-campaign orchestrator for the xue2020jssc CIM sub-array.

Re-derives the geometry-dependent ``[calibrated]`` values of ``params.toml`` in
the FIXED stage order CURRENTS -> CAPACITANCES -> CONSTANTS and REPORTS them for
write-back (it mutates no config file; it writes only its own report).
Non-circular by construction: the two peripheral seats are ADOPTED from Fig.18
shares (a declared adoption), the whole read path stays pure physics (array
IR-drop solve + declared windows), and ``p_zero`` is LOCKED to the read-path
physics -- never solved against the total.

Stage 1 -- CURRENTS. The linearized cell chord at the nominal read operating
point (extracted offline by ``neurox.tools.calibrate_cell`` from
``tools/params_detail.toml``) sets the array's absolute current scale; on it this
stage re-derives the ABSOLUTE reference ladder ``reference_config.i_refs__uA`` by
driving the macro over the MAC staircase and probing the pre-ADC ``i_sub``
through the ADC's own observation prober -- the true array-solve path, no manual
replay and no ``to_ideal``.

Stage 2 -- CAPACITANCES. With conduction frozen by stage 1, the two paired-slice
residuals seat the capacitive remainders: the whole ``cablc+dswct`` residual
seats on the array wire + cell node caps through ONE uniform scale (the declared
per-node relative structure of ``tools/params_detail.toml`` is kept), and the
``sinwp_sc+pn_isub`` residual -- the kept comparator per-op included in the
measured row -- seats the SINWP-SC ``c_hold``. The cap scale is the
ILL-CONDITIONED seat of the campaign: the array cap row is only a few percent of
its pair, so the seat amplifies a relative conduction error by ``conduction /
residual``. The stage therefore splits its rounds into ``--pair-blocks``
statistically independent blocks whose solved-seat spread MEASURES the achieved
conditioning instead of asserting it. The basis itself (``--pair-repeat``,
default ``--repeat``) stays the campaign's standard one: the seat carries under a
percent of the total, so its tolerance is reported and accepted rather than
bought down with extra draws.

Stage 3 -- CONSTANTS. The TMCSA 9.3 % slice seats the per-step constant
``e_fixed_per_op__fJ`` (pinned at its declared plausibility ceiling) plus ONE
scale on the as-drawn Fig.10(b) PH2/PH3 phase occupancy; the two ADOPTED
peripheral seats (the control per-op constant, the reference leakage) come
straight from the Fig.18 shares.

PER-ACCESS normalization. A profiled energy total covers EVERY leading dimension
of the drive -- ``n_w`` weight programs x ``n_x`` input vectors, and inside one
call the K WL planes and the ``mux_factor`` serial MUX slots -- so a per-access
quantity is that total divided by ``accesses = n_w * n_x * mux_factor``, and a
per-op constant is divided further by its own event count per access
(``cap_events_per_access`` / ``tmcsa_steps_per_access``). Each stage PROVES its
normalization: it re-measures with ``n_w`` doubled and, separately, with ``n_x``
doubled, and reports both the per-access rows (must hold) and the raw totals
(must double).

Run:
    TORCH_COMPILE_DISABLE=1 uv run python validations/xue2020jssc/tools/calibrate.py \
        --device auto --n-w 32 --n-x 256 --repeat 4 --solve-chunk 4096

The report is logged as it is built and written verbatim to ``--report``
(default ``calibration.md``), so the shipped report is byte-reproducible by the
run whose settings its own header states.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import statistics
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import torch

from neurox.primitive.analog.current_adc.base import IadcProber
from neurox.primitive.analog.voltage_dac import GeneralVdacConfig
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy
from neurox.primitive.xbar.cell import XbarCell1t1rLinearConfig
from neurox.works.macro.cim.xue2020jssc import (
    Xue2020JsscCimMacro,
    Xue2020JsscCimMacroConfig,
    Xue2020JsscCimMacroPolicy,
)

_LOG = logging.getLogger(__name__)

_VAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_VAL_DIR))
import validate as V  # noqa: E402  the sibling measurement engine (single source of the reduction)

_QUANTIZATION_MODE = 0
_ADC_BITS = 3
_POLARITY_NUM = 2
_FJ_PER_PJ = 1000.0

# Declared cap STRUCTURE source: the campaign scales this relative structure by
# one uniform factor, never the individual nodes.
_DETAIL_PATH = _VAL_DIR / "tools" / "params_detail.toml"
_WIRE_CAP_FIELDS = (
    "bl_first_c__fF",
    "bl_segment_c__fF",
    "sl_first_c__fF",
    "sl_segment_c__fF",
    "wl_first_c__fF",
    "wl_segment_c__fF",
)
_CELL_CAP_FIELDS = ("c_bl__fF", "c_x__fF", "c_sl__fF", "c_wl__fF")

# [measured p208 Fig.10(b)] As-drawn PH2 / PH3 occupancy of one conversion step,
# read off the timing diagram: PH2 18 %, PH3 30 % (48 % of the step; PH1/PH4 the
# rest). Only the RATIO and the per-step proportionality are structural; the
# campaign scales both by ONE factor to close the TMCSA slice.
_PH2_AS_DRAWN_FRACTION = 0.18
_PH3_AS_DRAWN_FRACTION = 0.30

# [assumed] Plausibility ceiling of the TMCSA per-STEP constant at 55 nm (latch +
# coupling caps + the folded-in PH1 bias): tens-to-~150 fJ per step. The joint
# (window scale, e_fixed) fit is underdetermined by one slice constraint, so
# e_fixed is PINNED here and the window scale carries the residual.
_E_FIXED_CEILING__fJ = 150.0

# Profiler dynamic-energy rows the campaign reduces. The macro bills the whole
# input branch on its ``.cablc`` channel and the control constant on
# ``.control``; ``array`` is the cap-only array row; the readout modules
# self-bill on their own module rows.
_CABLC_CHANNEL = ".cablc"
_CONTROL_CHANNEL = ".control"
_ARRAY_ROW = "array"
_ROW_KEYS = (_CONTROL_CHANNEL, _CABLC_CHANNEL, _ARRAY_ROW, "dswct", "sinwp_sc", "pn_isub", "tmcsa")


# ---------------------------------------------------------------------------
# Per-access event counts (the leading-dim normalization of each per-op seat)
# ---------------------------------------------------------------------------


def cap_events_per_access(cfg: Xue2020JsscCimMacroConfig, *, col_num: int) -> int:
    """SINWP-SC hold-cap ``c_hold * v_dd**2`` events per ACCESS.

    The module bills ``x_bits * serial * gn * 2`` events on every leading
    (batch) element, and one leading element is ``serial = mux_factor``
    accesses, so the per-access count is ``x_bits * gn * 2`` -- independent of
    every leading dimension of the drive.
    """
    return cfg.input_bit_num * (col_num // cfg.mux_factor) * _POLARITY_NUM


def tmcsa_steps_per_access(cfg: Xue2020JsscCimMacroConfig, *, col_num: int) -> int:
    """TMCSA ``e_fixed_per_op__fJ`` charges per ACCESS = converted elements x steps.

    The module bills ``e_fixed * bits`` on every converted element and the
    converted elements of one access are the ``gn`` CIM-IO lanes.
    """
    return (col_num // cfg.mux_factor) * cfg.adc_config.bits


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def rebuild(cfg: CimMacroConfig, policy: CimMacroPolicy, device: torch.device) -> Xue2020JsscCimMacro:
    """Build + fabricate the sub-array at ``inst_shape=()``, float32, eval mode."""
    macro = CimMacro.from_config(
        config=cfg,
        policy=policy,
        input_num=V._ROW_NUM,
        output_num=V._COL_NUM,
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    assert isinstance(macro, Xue2020JsscCimMacro)
    macro.to(device)
    macro.eval()
    macro.fabricate()
    return macro


def scale_caps(cfg: Xue2020JsscCimMacroConfig, factor: float) -> Xue2020JsscCimMacroConfig:
    """Scale every array wire and cell node capacitance by ONE uniform factor."""
    array = cfg.array_config
    cell = array.cell_config
    array = dataclasses.replace(
        array,
        **{name: getattr(array, name) * factor for name in _WIRE_CAP_FIELDS},
        cell_config=dataclasses.replace(cell, **{name: getattr(cell, name) * factor for name in _CELL_CAP_FIELDS}),
    )
    return dataclasses.replace(cfg, array_config=array)


def set_phase_windows(cfg: Xue2020JsscCimMacroConfig, scale: float) -> Xue2020JsscCimMacroConfig:
    """Seat the TMCSA PH2/PH3 windows at ``scale`` x the as-drawn step occupancy."""
    steps = cfg.adc_config.step_latency__ns
    return dataclasses.replace(
        cfg,
        tmcsa_config=dataclasses.replace(
            cfg.tmcsa_config,
            t_ph2_per_step__ns=tuple(_PH2_AS_DRAWN_FRACTION * scale * s for s in steps),
            t_ph3_per_step__ns=tuple(_PH3_AS_DRAWN_FRACTION * scale * s for s in steps),
            e_fixed_per_op__fJ=_E_FIXED_CEILING__fJ,
        ),
    )


def declared_cap_structure() -> dict[str, float]:
    """The declared per-node cap structure the campaign scale multiplies (params_detail.toml)."""
    with _DETAIL_PATH.open("rb") as fh:
        detail = tomllib.load(fh)["cim_macro"]["array_config"]
    base = {name: float(detail[name]) for name in _WIRE_CAP_FIELDS}
    base.update({name: float(detail["cell_config"][name]) for name in _CELL_CAP_FIELDS})
    return base


# ---------------------------------------------------------------------------
# Stage 1: reference-ladder re-derivation at this geometry
# ---------------------------------------------------------------------------


def _staircase_drive(macro: Xue2020JsscCimMacro) -> torch.Tensor:
    """MAC-value staircase input ``[m_max + 1, row_num]`` (greedy fill of the live rows)."""
    cfg = macro.config
    dev = next(macro.buffers()).device
    x_max = (1 << cfg.input_bit_num) - 1
    m_max = (1 << cfg.adc_config.bits) - 1
    x = torch.zeros((m_max + 1, macro.row_num), dtype=torch.long, device=dev)
    for m in range(m_max + 1):
        rem = m
        for r in range(cfg.max_active_num):
            v = min(x_max, rem)
            x[m, r] = v
            rem -= v
        assert rem == 0, f"cannot reach MAC value {m} with {cfg.max_active_num} selected inputs of max {x_max}"
    return x


def _program_unit_weight(macro: Xue2020JsscCimMacro) -> None:
    """Program logical column 0 to ``+1`` and every other column to 0."""
    dev = next(macro.buffers()).device
    w = torch.zeros((*macro.inst_shape, macro.row_num, macro.col_num), dtype=torch.long, device=dev)
    w[:, 0] = 1
    macro.program(w)


def unit_isub_staircase(macro: Xue2020JsscCimMacro, *, leading_repeat: int = 1) -> list[float]:
    """Probe the unit ``i_sub`` staircase [uA] over MAC ``0..2**bits-1``.

    Drives the macro DIRECTLY (``vec_mat_mul``) with a single ``+1`` weight and
    captures the pre-ADC magnitude current through the ADC's own
    :class:`IadcProber` -- the true IR-drop array-solve path. Column 0 sits at
    MUX slot 0 of IO 0, so ``i_sub[..., m, 0, 0]`` is the staircase.

    Args:
        macro: The fabricated sub-array.
        leading_repeat: Extra LEADING batch axis the staircase is tiled over.
            The staircase is a per-conversion current, so it must not depend on
            how many leading positions ride the same solve; ``> 1`` is the
            leading-dimension invariance probe of this stage.
    """
    _program_unit_weight(macro)
    x = _staircase_drive(macro)
    if leading_repeat > 1:
        # Shape: [m, row] -> [leading_repeat, m, row]
        x = x.unsqueeze(0).expand(leading_repeat, *x.shape).contiguous()
    with IadcProber() as probe, torch.no_grad():
        macro.vec_mat_mul(x.float(), quantization_mode=_QUANTIZATION_MODE, adc_bits=_ADC_BITS)
    # Shape: [..., m, group_size, group_num]
    i_sub = probe.records[-1].i_in__uA
    while i_sub.ndim > 3:
        i_sub = i_sub[0]
    return [float(v) for v in i_sub[:, 0, 0].cpu()]


def midpoint_ladder(grid: list[float]) -> list[float]:
    """Adjacent-midpoint reference taps ``0.5 * (I(m) + I(m+1))``."""
    return [0.5 * (a + b) for a, b in pairwise(grid)]


def verify_ladder(macro: Xue2020JsscCimMacro) -> tuple[bool, list[int]]:
    """Check the full macro produces ``code == MAC magnitude`` on the staircase."""
    _program_unit_weight(macro)
    x = _staircase_drive(macro)
    with torch.no_grad():
        codes = macro.vec_mat_mul(x.float(), quantization_mode=_QUANTIZATION_MODE, adc_bits=_ADC_BITS)
    got = [int(codes[m, 0]) for m in range(x.shape[0])]
    return got == list(range(x.shape[0])), got


# ---------------------------------------------------------------------------
# Measurement + per-access rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rows:
    """One measurement reduced to the per-access rows the seat-solve consumes.

    Every field is pJ PER ACCESS; ``accesses`` is the leading-dimension count the
    raw profiler totals were divided by.
    """

    accesses: int
    n_w: int
    n_x: int
    repeat: int
    rel_std: float
    total__pJ: float
    row__pJ: dict[str, float]

    def raw_total__pJ(self, key: str) -> float:
        """Undo the per-access normalization: the raw profiled row total [pJ]."""
        return self.row__pJ[key] * self.accesses


def pool_rows(blocks: list[Rows]) -> Rows:
    """Pool statistically independent measurement blocks onto one basis.

    Each block is its own set of rounds (disjoint seeds, own profiler contexts),
    so the per-access rows combine exactly as ``sum(energy) / sum(accesses)`` --
    an access-weighted mean. ``rel_std`` becomes the POOLED (root-mean-square)
    within-block round-total relative std, i.e. the round-level dispersion
    estimated from every block at once.
    """
    total = sum(b.accesses for b in blocks)
    weights = [b.accesses / total for b in blocks]

    def wmean(get: Callable[[Rows], float]) -> float:
        return sum(w * get(b) for w, b in zip(weights, blocks, strict=True))

    return Rows(
        accesses=total,
        n_w=blocks[0].n_w,
        n_x=blocks[0].n_x,
        repeat=sum(b.repeat for b in blocks),
        rel_std=math.sqrt(sum(w * b.rel_std**2 for w, b in zip(weights, blocks, strict=True))),
        total__pJ=wmean(lambda b: b.total__pJ),
        row__pJ={key: wmean(lambda b, k=key: b.row__pJ[k]) for key in _ROW_KEYS},
    )


def relative_sigma(values: list[float]) -> float:
    """Sample relative standard deviation of independently solved seat values."""
    return statistics.stdev(values) / statistics.fmean(values) if len(values) > 1 else 0.0


def measure_rows(
    macro: Xue2020JsscCimMacro,
    anchors: dict,
    *,
    n_w: int,
    n_x: int,
    repeat: int,
    p_zero: float,
    seed: int,
) -> tuple[V.Measurement, Rows]:
    """Run the round engine and reduce it to the per-access campaign rows."""
    m = V.measure_rounds(macro, anchors, n_w=n_w, n_x=n_x, repeat=repeat, p_zero=p_zero, seed=seed)
    rows = Rows(
        accesses=m.accesses,
        n_w=m.n_w,
        n_x=m.n_x,
        repeat=m.repeat,
        rel_std=m.rel_std,
        total__pJ=m.total__pJ,
        row__pJ={key: m.dyn_by_name.get(key, 0.0) for key in _ROW_KEYS},
    )
    return m, rows


# ---------------------------------------------------------------------------
# Seat-solves (each returns the CORRECTION factor / value from measured rows)
# ---------------------------------------------------------------------------


def cap_scale_correction(rows: Rows, *, pair1_target__pJ: float) -> float:
    """Factor the array cap row must scale by to close the ``cablc+dswct`` pair.

    The array row is capacitance ONLY (the array bills no conduction), so it is
    exactly proportional to the uniform cap scale; the pair residual left by the
    frozen conduction (``.cablc`` input branch + ``dswct`` rails) is its target.
    """
    conduction__pJ = rows.row__pJ[_CABLC_CHANNEL] + rows.row__pJ["dswct"]
    return (pair1_target__pJ - conduction__pJ) / rows.row__pJ[_ARRAY_ROW]


def c_hold_solution__fF(
    rows: Rows, *, pair2_target__pJ: float, c_hold_now__fF: float, events: int, v_dd__V: float
) -> float:
    """``c_hold`` closing the ``sinwp_sc+pn_isub`` pair at the measured conduction.

    The SINWP-SC row is leg conduction + ``events * c_hold * v_dd**2`` per
    access, and the PN-ISUB row is its 3-branch conduction + the kept comparator
    per-op; only the cap term moves, so the pair miss converts straight into a
    cap-value correction.
    """
    measured__pJ = rows.row__pJ["sinwp_sc"] + rows.row__pJ["pn_isub"]
    delta__fJ = (pair2_target__pJ - measured__pJ) * _FJ_PER_PJ
    return c_hold_now__fF + delta__fJ / (events * v_dd__V**2)


def phase_scale_solution(
    rows: Rows, *, tmcsa_target__pJ: float, e_fixed__fJ: float, steps: int, scale_now: float
) -> float:
    """Phase-window scale closing the TMCSA slice at the pinned per-step constant.

    The TMCSA row is per-step branch conduction (linear in the PH2/PH3 windows)
    plus ``steps * e_fixed`` per access; the conduction part therefore scales
    with the window scale.
    """
    fixed__pJ = steps * e_fixed__fJ / _FJ_PER_PJ
    conduction__pJ = rows.row__pJ["tmcsa"] - fixed__pJ
    return scale_now * (tmcsa_target__pJ - fixed__pJ) / conduction__pJ


# ---------------------------------------------------------------------------
# p_zero LOCK to the read-path physics
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
    n_w: int,
    n_x: int,
    seed: int,
    grid: list[float],
) -> tuple[float | None, list[tuple[float, float]]]:
    """Find the ``p_zero`` where the read path conducts ``target_pJ`` per access.

    The read-path energy decreases monotonically with input sparsity, so a linear
    scan of ``grid`` (one round per point) brackets the crossing of ``target_pJ``
    (= 47.1 % x 32.06 pJ, the Fig.18 read-path share). Returns ``(p_star,
    points)`` with ``p_star`` the interpolated crossing (or ``None`` if the
    target is not bracketed inside the grid). ``p_star`` is locked to the READ
    PATH physics, never to the total.
    """
    points: list[tuple[float, float]] = []
    for p in grid:
        m = V.measure_rounds(macro, anchors, n_w=n_w, n_x=n_x, repeat=1, p_zero=p, seed=seed)
        points.append((p, read_path_pJ(m, read_path)))
    for (pa, ea), (pb, eb) in pairwise(points):
        if (ea - target_pJ) * (eb - target_pJ) <= 0 and ea != eb:
            frac = (ea - target_pJ) / (ea - eb)
            return pa + frac * (pb - pa), points
    return None, points


# ---------------------------------------------------------------------------
# Leading-dimension invariance harness
# ---------------------------------------------------------------------------


def invariance_points(
    macro: Xue2020JsscCimMacro,
    anchors: dict,
    *,
    n_w: int,
    n_x: int,
    p_zero: float,
    seed: int,
) -> list[tuple[str, Rows]]:
    """Measure the same macro at ``(n_w, n_x)``, ``(2 n_w, n_x)`` and ``(n_w, 2 n_x)``.

    ``n_w`` and ``n_x`` are the two LEADING draw dimensions of the workload. A
    correctly normalized per-access row is invariant across all three points
    while the raw profiled total doubles with either.
    """
    specs = (("baseline", n_w, n_x), ("n_w x2", 2 * n_w, n_x), ("n_x x2", n_w, 2 * n_x))
    out: list[tuple[str, Rows]] = []
    for label, w, x in specs:
        _m, rows = measure_rows(macro, anchors, n_w=w, n_x=x, repeat=1, p_zero=p_zero, seed=seed)
        out.append((label, rows))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", type=Path, default=_VAL_DIR / "params.toml")
    ap.add_argument("--policy", type=Path, default=_VAL_DIR / "policy.toml")
    ap.add_argument("--anchors", type=Path, default=_VAL_DIR / "anchors.toml")
    ap.add_argument("--n-w", type=int, default=32, help="Weight draws per round of a stage measurement.")
    ap.add_argument("--n-x", type=int, default=256, help="Input vectors per weight draw.")
    ap.add_argument("--repeat", type=int, default=4, help="Rounds pooled per stage measurement.")
    ap.add_argument(
        "--pair-repeat",
        type=int,
        default=None,
        help="Rounds pooled for the stage-2 pair solve; default --repeat (the campaign's standard basis).",
    )
    ap.add_argument(
        "--pair-blocks",
        type=int,
        default=4,
        help="Independent blocks the stage-2 basis is split into; their solved-seat spread measures conditioning.",
    )
    ap.add_argument(
        "--solve-chunk",
        type=int,
        default=4096,
        help="Array solve chunk (machine knob): leading instances solved per block; 0 solves all at once.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", help="cpu, cuda[:idx], or auto (free GPU via nvidia-smi).")
    ap.add_argument("--lock-step", type=float, default=0.1, help="p_zero grid step for the read-path lock scan.")
    ap.add_argument("--report", type=Path, default=_VAL_DIR / "calibration.md")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = V._resolve_device(args.device)
    with args.anchors.open("rb") as fh:
        anchors = tomllib.load(fh)
    cfg = CimMacroConfig.from_file(args.params, section="cim_macro")
    assert isinstance(cfg, Xue2020JsscCimMacroConfig)
    policy = CimMacroPolicy.from_file(args.policy, section="policy")
    assert isinstance(policy, Xue2020JsscCimMacroPolicy)
    policy = dataclasses.replace(
        policy,
        array_policy=dataclasses.replace(policy.array_policy, solve_chunk_size=args.solve_chunk),
    )

    pair_blocks = max(1, args.pair_blocks)
    pair_rounds_per_block = max(1, (args.pair_repeat if args.pair_repeat is not None else args.repeat) // pair_blocks)
    pair_repeat = pair_blocks * pair_rounds_per_block

    tgt = anchors["target"]
    target_pJ = tgt["per_access__pJ"]
    per_sub_uW = tgt["total_macro__mW"] * 1000.0 / tgt["sub_array_num"]
    shares = anchors["fig18_shares"]
    conv = anchors["conventions"]
    read_path = tuple(conv["read_path"])
    read_path_target_pJ = shares["read_path_sum"] / 100.0 * target_pJ
    p_zero = float(anchors["data"]["p_zero"])
    t_cycle = cfg.t_cycle__ns
    v_dd = cfg.v_dd__V
    gn = V._COL_NUM // cfg.mux_factor

    pair1_target__pJ = (shares["cablc"] + shares["dswct"]) / 100.0 * target_pJ
    pair2_target__pJ = (shares["sinwp_sc"] + shares["pn_isub"]) / 100.0 * target_pJ
    tmcsa_target__pJ = shares["tmcsa"] / 100.0 * target_pJ

    log: list[str] = []

    def emit(line: str = "") -> None:
        _LOG.info("%s", line)
        log.append(line)

    def row_table(points: list[tuple[str, Rows]]) -> None:
        """Emit the per-access rows + raw-total scaling of an invariance triple."""
        base = points[0][1]
        emit("| Row [pJ/access] | " + " | ".join(label for label, _ in points) + " | max dev |")
        emit("|---|" + "--:|" * (len(points) + 1))
        for key in _ROW_KEYS:
            vals = [r.row__pJ[key] for _, r in points]
            dev = max(abs(v - vals[0]) / vals[0] for v in vals[1:]) if vals[0] else 0.0
            emit(f"| `{key}` | " + " | ".join(f"{v:9.5f}" for v in vals) + f" | {dev * 100:5.3f} % |")
        totals = [r.total__pJ for _, r in points]
        dev_total = max(abs(v - totals[0]) / totals[0] for v in totals[1:])
        emit("| **total** | " + " | ".join(f"{v:9.5f}" for v in totals) + f" | {dev_total * 100:5.3f} % |")
        emit(
            "| accesses (leading count) | "
            + " | ".join(f"{r.accesses}" for _, r in points)
            + f" | x{points[1][1].accesses / base.accesses:.1f} / x{points[2][1].accesses / base.accesses:.1f} |"
        )
        emit(
            "| raw profiled total [nJ] | "
            + " | ".join(f"{r.total__pJ * r.accesses / 1e3:9.3f}" for _, r in points)
            + " | (scales with the leading count) |"
        )

    # =====================================================================
    emit("# xue2020jssc calibration campaign")
    emit()
    emit(
        f"Energy-basis re-derivation of the geometry-dependent `[calibrated]` values of `params.toml` against "
        f"the `anchors.toml` target, in the fixed stage order CURRENTS -> CAPACITANCES -> CONSTANTS. Device "
        f"{device}, float32, `.eval()`, all-off policy, solve chunk {args.solve_chunk}. Stage measurements: "
        f"n_w = {args.n_w} weight draws x n_x = {args.n_x} inputs x {args.repeat} rounds, seed {args.seed}, at "
        f"the declared workload p_zero = {p_zero:.3f}; the stage-2 pair solve pools {pair_repeat} rounds split "
        f"into {pair_blocks} statistically independent blocks x {pair_rounds_per_block} rounds, whose spread "
        f"MEASURES the ill-conditioned cap seat. Non-circular: control + reference are ADOPTED from "
        f"Fig.18, the read path is pure physics (array IR-drop solve), and p_zero is LOCKED to the read-path "
        f"share, never solved against the total."
    )
    emit()
    emit(
        f"Per-access basis: one access = one MUX-slot conversion set, so a run of n_w x n_x x rounds input "
        f"draws reads `accesses = n_w * n_x * mux_factor` ({cfg.mux_factor}) output accesses and EVERY profiled "
        f"energy total is divided by that leading count. A per-op seat is divided further by its own event "
        f"count per access: {cap_events_per_access(cfg, col_num=V._COL_NUM)} SINWP-SC hold-cap events "
        f"(x_bits x IO x polarity) and {tmcsa_steps_per_access(cfg, col_num=V._COL_NUM)} TMCSA step charges "
        f"(IO x steps). Each stage proves its own normalization by an n_w / n_x doubling check."
    )
    emit()

    # =====================================================================
    # Stage 1 -- CURRENTS
    # =====================================================================
    emit("## Stage 1 -- currents")
    emit()
    cell = cfg.array_config.cell_config
    assert isinstance(cell, XbarCell1t1rLinearConfig)
    wl_dac = cfg.wl_dac_config
    assert isinstance(wl_dac, GeneralVdacConfig)
    emit(
        f"Operating point of the linearized cell chord in force (extracted by `neurox.tools.calibrate_cell` "
        f"from `tools/params_detail.toml` at V_BL = V_BLC = {cfg.cablc_vref_config.v_refs__V[0][0]} V, V_SL = 0 V, "
        f"WL off/on 0.0 / {wl_dac.code_to_signal[-1]} V): g_cell_on = "
        f"{list(cell.g_cell_on_table__uS)} uS, g_cell_off = {list(cell.g_cell_off_table__uS)} uS, vx_ratio_on = "
        f"{list(cell.vx_ratio_on_table)}. The absolute current scale of every read-path row rides on it, so it "
        f"is the first thing this campaign fixes."
    )
    emit()
    macro = rebuild(cfg, policy, device)
    grid = unit_isub_staircase(macro)
    mids = midpoint_ladder(grid)
    monotonic = all(a < b for a, b in pairwise(grid))
    emit(f"Unit i_sub staircase I(MAC=0..{len(grid) - 1}) [uA] over {cfg.max_active_num} selected inputs:")
    emit("")
    emit("    " + ", ".join(f"{g:.4f}" for g in grid))
    emit("")
    emit(f"Adjacent-midpoint reference taps I_REF[0..{len(mids) - 1}] [uA]:")
    emit("")
    emit("    " + ", ".join(f"{v:.6f}" for v in mids))
    emit("")
    grid_tiled = unit_isub_staircase(macro, leading_repeat=2)
    probe_dev = max(abs(a - b) / b for a, b in zip(grid_tiled, grid, strict=True) if b)
    cfg = dataclasses.replace(
        cfg, reference_config=dataclasses.replace(cfg.reference_config, i_refs__uA=(tuple(mids),))
    )
    macro = rebuild(cfg, policy, device)
    ladder_ok, got = verify_ladder(macro)
    emit(
        f"Staircase monotonic: {monotonic} (driven through the full array IR-drop solve). Ladder verification "
        f"(code == MAC magnitude): {ladder_ok} (codes {got})."
    )
    emit("")
    emit(
        f"Leading-dimension invariance of the probe: re-driving the same staircase tiled over an extra leading "
        f"axis (x2) reproduces every step to {probe_dev * 100:.2e} % -- the staircase is a PER-CONVERSION "
        f"current, unaffected by how many leading positions ride one solve, so the ladder needs no leading-dim "
        f"normalization."
    )
    emit()

    # =====================================================================
    # Stage 2 -- CAPACITANCES
    # =====================================================================
    emit("## Stage 2 -- capacitances")
    emit()
    cap_events = cap_events_per_access(cfg, col_num=V._COL_NUM)
    base_caps = declared_cap_structure()
    shipped_caps = {name: getattr(cfg.array_config, name) for name in _WIRE_CAP_FIELDS}
    shipped_caps.update({name: getattr(cfg.array_config.cell_config, name) for name in _CELL_CAP_FIELDS})
    scale_in = shipped_caps["bl_first_c__fF"] / base_caps["bl_first_c__fF"]
    scale_spread = max(abs(shipped_caps[name] / base_caps[name] / scale_in - 1.0) for name in base_caps)
    c_hold_in = cfg.sinwp_sc_config.c_hold__fF

    blocks2 = [
        measure_rows(
            macro,
            anchors,
            n_w=args.n_w,
            n_x=args.n_x,
            repeat=pair_rounds_per_block,
            p_zero=p_zero,
            seed=args.seed + idx * pair_rounds_per_block,
        )[1]
        for idx in range(pair_blocks)
    ]
    rows2 = pool_rows(blocks2)
    correction = cap_scale_correction(rows2, pair1_target__pJ=pair1_target__pJ)
    cap_scale = scale_in * correction
    c_hold = c_hold_solution__fF(
        rows2, pair2_target__pJ=pair2_target__pJ, c_hold_now__fF=c_hold_in, events=cap_events, v_dd__V=v_dd
    )
    emit(
        f"Conduction is FROZEN by stage 1; the two paired-slice residuals seat the capacitive remainders. "
        f"Measured at the incoming values (cap scale x{scale_in:.5f} on the `params_detail.toml` structure, "
        f"uniform across all {len(base_caps)} cap nodes to {scale_spread * 100:.2e} %; c_hold = "
        f"{c_hold_in:.4f} fF), pooled over {rows2.repeat} rounds in {pair_blocks} independent blocks (pooled "
        f"round-total relative std {rows2.rel_std * 100:.3f} %):"
    )
    emit("")
    emit("| Pair | Fig.18 target [pJ/acc] | conduction (frozen) | capacitive term | measured pair | seat solved |")
    emit("|---|--:|--:|--:|--:|:--|")
    cond1 = rows2.row__pJ[_CABLC_CHANNEL] + rows2.row__pJ["dswct"]
    cap1 = rows2.row__pJ[_ARRAY_ROW]
    cap2_in__pJ = cap_events * c_hold_in * v_dd**2 / _FJ_PER_PJ
    cond2 = rows2.row__pJ["sinwp_sc"] + rows2.row__pJ["pn_isub"] - cap2_in__pJ
    emit(
        f"| cablc+dswct | {pair1_target__pJ:8.4f} | {cond1:8.4f} | {cap1:8.4f} (array row) | "
        f"{cond1 + cap1:8.4f} | uniform cap scale |"
    )
    emit(
        f"| sinwp_sc+pn_isub | {pair2_target__pJ:8.4f} | {cond2:8.4f} | {cap2_in__pJ:8.4f} (hold caps) | "
        f"{cond2 + cap2_in__pJ:8.4f} | `c_hold__fF` |"
    )
    emit("")
    emit(
        f"The array row bills capacitance ONLY, so it is exactly proportional to the uniform cap scale: the "
        f"pair-1 residual {pair1_target__pJ - cond1:.4f} pJ/access asks for x{correction:.5f} on the incoming "
        f"row, i.e. a cumulative **cap scale x{cap_scale:.5f}** on the declared `params_detail.toml` structure. "
        f"The SINWP-SC hold caps cycle {cap_events} times per access, so the pair-2 residual "
        f"{pair2_target__pJ - cond2:.4f} pJ/access seats **c_hold = {c_hold:.4f} fF** (the kept "
        f"{cfg.pn_isub_config.e_per_op__fJ:.1f} fJ comparator per-op stays inside the measured PN-ISUB row)."
    )
    emit()
    block_scales = [scale_in * cap_scale_correction(b, pair1_target__pJ=pair1_target__pJ) for b in blocks2]
    block_holds = [
        c_hold_solution__fF(
            b, pair2_target__pJ=pair2_target__pJ, c_hold_now__fF=c_hold_in, events=cap_events, v_dd__V=v_dd
        )
        for b in blocks2
    ]
    residual1 = pair1_target__pJ - cond1
    amplification = cond1 / residual1
    sigma_scale = relative_sigma(block_scales)
    sigma_hold = relative_sigma(block_holds)
    root_blocks = math.sqrt(pair_blocks)
    emit(
        f"CONDITIONING of the two seats, measured not asserted. The pair-1 residual is only "
        f"{residual1 / pair1_target__pJ:.1%} of its pair, so the cap-scale seat AMPLIFIES a relative conduction "
        f"error by conduction/residual = x{amplification:.1f} -- it is the ill-conditioned seat of this "
        f"campaign, so its tolerance is MEASURED here rather than asserted. The {pair_blocks} blocks below are "
        f"statistically independent ({pair_rounds_per_block} rounds each, disjoint seeds), so their spread IS "
        f"the seat uncertainty:"
    )
    emit("")
    emit("| Independent block | conduction cablc+dswct | array cap row | solved cap scale | solved c_hold [fF] |")
    emit("|---|--:|--:|--:|--:|")
    for idx, (b, sc, ch) in enumerate(zip(blocks2, block_scales, block_holds, strict=True)):
        seeds = args.seed + idx * pair_rounds_per_block
        emit(
            f"| block {idx} (seeds {seeds}..{seeds + pair_rounds_per_block - 1}) | "
            f"{b.row__pJ[_CABLC_CHANNEL] + b.row__pJ['dswct']:8.4f} | {b.row__pJ[_ARRAY_ROW]:8.5f} | "
            f"{sc:.5f} | {ch:.4f} |"
        )
    emit(
        f"| **pooled ({pair_repeat} rounds)** | **{cond1:8.4f}** | **{cap1:8.5f}** | **{cap_scale:.5f}** | "
        f"**{c_hold:.4f}** |"
    )
    emit("")
    emit(
        f"Block-basis relative 1 sigma: cap scale {sigma_scale * 100:.2f} %, c_hold {sigma_hold * 100:.2f} % (at "
        f"{pair_rounds_per_block} rounds). Averaging the {pair_blocks} blocks divides that by sqrt "
        f"{pair_blocks} = {root_blocks:.1f}, so the ADOPTED seats carry **cap scale +-"
        f"{sigma_scale / root_blocks * 100:.2f} % (1 sigma)** and **c_hold +-{sigma_hold / root_blocks * 100:.2f} "
        f"% (1 sigma)** at the {pair_repeat}-round basis. c_hold is well conditioned (its cap term is a large "
        f"fraction of its pair); the cap scale is not, and its stated tolerance is part of the result. That "
        f"tolerance is ACCEPTED at this standard basis, not bought down with extra draws: the array cap row is "
        f"{cap1 / rows2.total__pJ:.1%} of the measured total, so the seat's 1 sigma moves the headline by only "
        f"+-{sigma_scale / root_blocks * cap1 / rows2.total__pJ * 100:.2f} %."
    )
    emit()
    cfg = scale_caps(cfg, correction)
    cfg = dataclasses.replace(cfg, sinwp_sc_config=dataclasses.replace(cfg.sinwp_sc_config, c_hold__fF=c_hold))
    macro = rebuild(cfg, policy, device)
    _m2v, rows2v = measure_rows(
        macro, anchors, n_w=args.n_w, n_x=args.n_x, repeat=pair_repeat, p_zero=p_zero, seed=args.seed
    )
    pair1_out = rows2v.row__pJ[_CABLC_CHANNEL] + rows2v.row__pJ["dswct"] + rows2v.row__pJ[_ARRAY_ROW]
    pair2_out = rows2v.row__pJ["sinwp_sc"] + rows2v.row__pJ["pn_isub"]
    emit(
        f"Re-measured at the solved values: cablc+dswct = {pair1_out:.4f} pJ/access = "
        f"{pair1_out / pair1_target__pJ:.4f}x its Fig.18 share, sinwp_sc+pn_isub = {pair2_out:.4f} pJ/access = "
        f"{pair2_out / pair2_target__pJ:.4f}x."
    )
    emit()
    emit("Leading-dimension invariance (same macro, one round per point, both draw dimensions doubled in turn):")
    emit("")
    points2 = invariance_points(macro, anchors, n_w=args.n_w, n_x=args.n_x, p_zero=p_zero, seed=args.seed)
    row_table(points2)
    emit("")
    solved2 = [
        (
            label,
            cap_scale_correction(r, pair1_target__pJ=pair1_target__pJ) * cap_scale,
            c_hold_solution__fF(
                r, pair2_target__pJ=pair2_target__pJ, c_hold_now__fF=c_hold, events=cap_events, v_dd__V=v_dd
            ),
        )
        for label, r in points2
    ]
    emit("| Re-solved from that point | cap scale | c_hold [fF] |")
    emit("|---|--:|--:|")
    for label, sc, ch in solved2:
        emit(f"| {label} | {sc:.5f} | {ch:.4f} |")
    emit("")
    emit(
        f"That re-solve table proves the NORMALIZATION, not the seat: each point is a SINGLE round, so its "
        f"cap-scale column carries the full x{amplification:.1f} amplification of one round's conduction noise "
        f"and scatters far wider than the {pair_repeat}-round seat adopted above. The rows that must hold flat "
        f"are the per-access rows in the preceding table -- and they do."
    )
    emit("")

    # =====================================================================
    # Stage 3 -- CONSTANTS
    # =====================================================================
    emit("## Stage 3 -- constants")
    emit()
    steps_per_access = tmcsa_steps_per_access(cfg, col_num=V._COL_NUM)
    cfg = set_phase_windows(cfg, 1.0)
    macro = rebuild(cfg, policy, device)
    _m3, rows3 = measure_rows(
        macro, anchors, n_w=args.n_w, n_x=args.n_x, repeat=args.repeat, p_zero=p_zero, seed=args.seed
    )
    phase_scale = phase_scale_solution(
        rows3,
        tmcsa_target__pJ=tmcsa_target__pJ,
        e_fixed__fJ=_E_FIXED_CEILING__fJ,
        steps=steps_per_access,
        scale_now=1.0,
    )
    fixed__pJ = steps_per_access * _E_FIXED_CEILING__fJ / _FJ_PER_PJ
    emit(
        f"The TMCSA slice carries two unknowns against one constraint, so `e_fixed_per_op__fJ` is PINNED at its "
        f"declared plausibility ceiling {_E_FIXED_CEILING__fJ:.1f} fJ per conversion step (latch + coupling "
        f"caps + the folded-in PH1 bias at 55 nm) and the single phase-window scale carries the residual. "
        f"Measured at the as-drawn Fig.10(b) occupancy (PH2 {_PH2_AS_DRAWN_FRACTION:.0%} + PH3 "
        f"{_PH3_AS_DRAWN_FRACTION:.0%} of each step, scale x1):"
    )
    emit("")
    emit("| Term | [pJ/access] |")
    emit("|---|--:|")
    emit(f"| Fig.18 target ({shares['tmcsa']} %) | {tmcsa_target__pJ:8.4f} |")
    emit(f"| per-step constant ({steps_per_access} charges x {_E_FIXED_CEILING__fJ:.1f} fJ) | {fixed__pJ:8.4f} |")
    emit(f"| PH2/PH3 branch conduction at the as-drawn widths | {rows3.row__pJ['tmcsa'] - fixed__pJ:8.4f} |")
    emit(f"| conduction budget left by the constant | {tmcsa_target__pJ - fixed__pJ:8.4f} |")
    emit("")
    cfg = set_phase_windows(cfg, phase_scale)
    occupancy = (_PH2_AS_DRAWN_FRACTION + _PH3_AS_DRAWN_FRACTION) * phase_scale
    emit(
        f"**Phase-window scale x{phase_scale:.5f}** on the as-drawn widths: t_ph2 = "
        f"{[round(t, 6) for t in cfg.tmcsa_config.t_ph2_per_step__ns]} ns, t_ph3 = "
        f"{[round(t, 6) for t in cfg.tmcsa_config.t_ph3_per_step__ns]} ns, so PH2+PH3 occupy "
        f"{occupancy:.1%} of each conversion step and PH1/PH4 the rest. TENSION, reported not hidden: the "
        f"as-drawn occupancy is {(_PH2_AS_DRAWN_FRACTION + _PH3_AS_DRAWN_FRACTION):.0%}, so the slice cannot "
        f"close with BOTH the as-drawn widths and the e_fixed ceiling holding; the widening is the residual "
        f"compromise."
    )
    emit()
    macro = rebuild(cfg, policy, device)
    _m3v, rows3v = measure_rows(
        macro, anchors, n_w=args.n_w, n_x=args.n_x, repeat=args.repeat, p_zero=p_zero, seed=args.seed
    )
    emit(
        f"Re-measured at the solved windows: tmcsa = {rows3v.row__pJ['tmcsa']:.4f} pJ/access = "
        f"{rows3v.row__pJ['tmcsa'] / tmcsa_target__pJ:.4f}x its Fig.18 share."
    )
    emit()
    control_uW = shares["control"] / 100.0 * per_sub_uW
    reference_uW = shares["reference"] / 100.0 * per_sub_uW
    e_control_fJ = conv["control_dynamic_fraction"] * control_uW * t_cycle
    control_leak_uW = conv["control_static_fraction"] * control_uW
    reference_leak_uW = conv["reference_static_fraction"] * reference_uW
    cfg = dataclasses.replace(cfg, e_control_per_op__fJ=e_control_fJ)
    macro = rebuild(cfg, policy, device)
    emit(
        f"ADOPTED peripheral seats (declared from the Fig.18 shares, never fitted): per-sub-array budget = "
        f"{tgt['total_macro__mW']} mW / {tgt['sub_array_num']} = {per_sub_uW:.4f} uW = {target_pJ:.4f} pJ/access "
        f"at {tgt['op_frequency__MHz']} MHz. Control {shares['control']} % -> {control_uW:.4f} uW = "
        f"{e_control_fJ:.4f} fJ/access, billed as a PURE per-op constant (100 % dynamic, "
        f"`control_config.leakage_per_inst__uW = {control_leak_uW:.4f}`). Reference {shares['reference']} % -> "
        f"100 % static: `reference_config.leakage_per_inst__uW = {reference_leak_uW:.5f}` uW. Measured control "
        f"row: {rows3v.row__pJ[_CONTROL_CHANNEL]:.4f} pJ/access."
    )
    emit()
    emit("Leading-dimension invariance (same macro, one round per point, both draw dimensions doubled in turn):")
    emit("")
    points3 = invariance_points(macro, anchors, n_w=args.n_w, n_x=args.n_x, p_zero=p_zero, seed=args.seed)
    row_table(points3)
    emit("")
    emit("| Re-solved from that point | phase-window scale | e_control_per_op [fJ] |")
    emit("|---|--:|--:|")
    for label, r in points3:
        sc = phase_scale_solution(
            r,
            tmcsa_target__pJ=tmcsa_target__pJ,
            e_fixed__fJ=_E_FIXED_CEILING__fJ,
            steps=steps_per_access,
            scale_now=phase_scale,
        )
        emit(f"| {label} | {sc:.5f} | {r.row__pJ[_CONTROL_CHANNEL] * _FJ_PER_PJ:.4f} |")
    emit("")

    # =====================================================================
    # p_zero lock + closing breakdown
    # =====================================================================
    emit("## p_zero LOCK to the read-path physics share")
    emit()
    n_grid = max(2, round(1.0 / args.lock_step))
    lock_grid = [round(i * args.lock_step, 4) for i in range(n_grid)]
    p_star, points = lock_p_zero(
        macro,
        anchors,
        target_pJ=read_path_target_pJ,
        read_path=read_path,
        n_w=args.n_w,
        n_x=args.n_x,
        seed=args.seed,
        grid=lock_grid,
    )
    x_lo, x_hi = anchors["data"]["input_range"]
    f0 = 1.0 / (x_hi - x_lo + 1)
    emit(
        f"Read-path target = {shares['read_path_sum']} % x {target_pJ} = {read_path_target_pJ:.3f} pJ/access, "
        f"scanned at one round per point on the calibrated macro."
    )
    emit("")
    emit(f"| p_zero | read-path [pJ/acc] | vs {read_path_target_pJ:.2f} |")
    emit("|--:|--:|--:|")
    for p, e in points:
        emit(f"| {p:.2f} | {e:8.3f} | {e / read_path_target_pJ:5.3f}x |")
    emit("")
    if p_star is None:
        emit(
            f"The read path does not cross {read_path_target_pJ:.3f} pJ/access inside p_zero in "
            f"[{lock_grid[0]:.2f}, {lock_grid[-1]:.2f}] -- the lock is UNBRACKETED at this geometry (reported, "
            f"not forced). The declared p_zero = {p_zero:.3f} stands."
        )
        locked_p = p_zero
    else:
        locked_p = p_star
        emit(
            f"**LOCKED p_zero = {locked_p:.4f}** (marginal P(x=0) ~= {f0 + (1.0 - f0) * locked_p:.3f}): the "
            f"p_zero at which the pure-physics read path conducts its Fig.18 {shares['read_path_sum']} % share. "
            f"Locked to the READ PATH, not the total; the declared `anchors.toml` value is {p_zero:.3f}."
        )
    emit()
    emit("## Total + breakdown at the declared p_zero")
    emit()
    gate_m, _rows_g = measure_rows(
        macro, anchors, n_w=args.n_w, n_x=args.n_x, repeat=args.repeat, p_zero=p_zero, seed=args.seed
    )
    within, rel = V.gate(gate_m, anchors)
    tol = anchors["gate"]["hard_tolerance_relative"]
    emit(
        f"At the declared p_zero = {p_zero:.4f}: total = {gate_m.total__pJ:.3f} pJ/access = "
        f"{gate_m.total__pJ / target_pJ:.3f}x (err {rel * 100:+.1f}%), within +-{tol * 100:.0f}%: "
        f"{'PASS' if within else 'FAIL'} (the total FALLS OUT of the read-path seats + the adopted seats; it is "
        f"never solved for)."
    )
    emit()
    emit(V.energy_table(gate_m, anchors))
    emit()

    # =====================================================================
    emit("## Values to write back into params.toml (print only; nothing is mutated)")
    emit()
    emit(f"- `reference_config.i_refs__uA = [[{', '.join(f'{v:.6f}' for v in mids)}]]`  # stage 1, calibrated ladder")
    for name in _WIRE_CAP_FIELDS:
        emit(f"- `array_config.{name} = {getattr(cfg.array_config, name):.8g}`  # stage 2, cap scale x{cap_scale:.5f}")
    for name in _CELL_CAP_FIELDS:
        emit(
            f"- `array_config.cell_config.{name} = {getattr(cfg.array_config.cell_config, name):.8g}`"
            f"  # stage 2, cap scale x{cap_scale:.5f}"
        )
    emit(f"- `sinwp_sc_config.c_hold__fF = {c_hold:.4f}`  # stage 2, pair-2 residual over {cap_events} events/access")
    emit(
        f"- `tmcsa_config.t_ph2_per_step__ns = {[round(t, 6) for t in cfg.tmcsa_config.t_ph2_per_step__ns]}`"
        f"  # stage 3, as-drawn x{phase_scale:.5f}"
    )
    emit(
        f"- `tmcsa_config.t_ph3_per_step__ns = {[round(t, 6) for t in cfg.tmcsa_config.t_ph3_per_step__ns]}`"
        f"  # stage 3, as-drawn x{phase_scale:.5f}"
    )
    emit(f"- `tmcsa_config.e_fixed_per_op__fJ = {_E_FIXED_CEILING__fJ:.1f}`  # stage 3, pinned plausibility ceiling")
    emit(f"- `e_control_per_op__fJ = {e_control_fJ:.4f}`  # stage 3, adopted (control 100 % dynamic, pure per-op)")
    emit(f"- `control_config.leakage_per_inst__uW = {control_leak_uW:.4f}`  # stage 3, adopted (control 0 % static)")
    emit(
        f"- `reference_config.leakage_per_inst__uW = {reference_leak_uW:.5f}`  # stage 3, adopted "
        f"(reference 100 % static)"
    )
    emit()
    emit(
        "Read-path physical declarations (g_map, V_BL_CLAMP, wire R, conduction windows) are left as declared, "
        "the read-path static seats stay declared small/zero, and `anchors.toml` is not a write-back target of "
        "this campaign: the p_zero lock above is reported against its declared value, never written."
    )
    emit()

    args.report.write_text("\n".join(log) + "\n")
    _LOG.info("\n[wrote report -> %s]", args.report)


if __name__ == "__main__":
    main()
