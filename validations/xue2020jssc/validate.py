"""Energy-basis, total-only, non-circular validation for the xue2020jssc CIM sub-array.

Builds :class:`~neurox.works.macro.cim.xue2020jssc.Xue2020JsscCimMacro` from
``params.toml`` + ``policy.toml``, drives it DIRECTLY (rows ``0..active_row_num-1``
live, the rest zeroed) over a round-based random workload per the ``anchors.toml``
data conventions, and reduces the profiler to the ENERGY PER ACCESS. The one HARD
GATE is the total energy per access against the paper's 32.06 pJ/access (= 5.13 mW
/ 8 sub-arrays / 20 MHz) within +-5% at the declared ``p_zero``.

Energy-basis reduction. With ``accesses = n_samples * mux_factor`` (one VMM over
all ``col_num`` logical columns is ``mux_factor`` serial accesses):
  - static energy per access = ``leakage_power * leakage_window__ns`` (the
    declared integration window; independent of the draw count and ``mux_factor``);
  - dynamic energy per access = ``total_dynamic_energy / accesses`` = per-VMM
    dynamic / ``mux_factor``.
The total energy per access is their sum. 1 uA * 1 V * 1 ns = 1 fJ; 1 pJ = 1000 fJ.
The macro's access time is reported beside them and integrates nothing.

NON-CIRCULAR rigor: the ONLY seats declared to reproduce a Fig.18 share are
Control (29.2 %) + Reference (23.7 %) -- ADOPTED, because those two peripherals
are not modeled from physics. The whole read path (cablc, dswct, sinwp_sc,
pn_isub, tmcsa) is pure physics with declared structural constants (g_map,
V_BL_CLAMP, wire R, conduction windows); its static seats are declared small/zero
and NEVER reverse-solved to fill the total. ``p_zero`` is LOCKED to the read-path
physics -- the sparsity at which the pure-physics read path conducts its Fig.18
read-path share (47.1 % x 32.06 = 15.10 pJ/access) -- and is NEVER solved against
the total. The headline is the total energy per access at that locked ``p_zero``.

PAIRED-SLICE caliber for the Fig.18 comparison: the paper splits ONE series input
branch at node V_CMD (the drain of the DSWCT current-mirror input, p.207
Fig.9(a)) between the DSWCT and CABLC pie slices, and one series sink branch
between SINWP-SC (its sink transistors) and PN-ISUB (switches + comparator +
isub); the internal node voltages are not published, so only the PAIR SUMS are
well-defined comparison targets. The breakdown therefore compares ``cablc+dswct``
against 14.9 + 11.5 = 26.4 % and ``sinwp_sc+pn_isub`` against 8.0 + 3.4 =
11.4 %, with ``control`` / ``reference`` / ``tmcsa`` as singles; the four member
rows stay visible (informational, no per-member target).

The array module row bills only its wire / cell capacitive cycling, and the macro
``cablc`` channel bills the whole input branch ``V_DD * I_DL``; the ``cablc``
slice SUMS the two.

ROUND-based workload: ``--repeat`` rounds, each redrawing ``--n-w`` weight programs
and, per program, ``--n-x`` input vectors, and each profiled in its OWN context;
the per-access energies accumulate as an access-weighted mean and the report adds
the round-total relative std. Peak profiler memory stays that of one round.
``--solve-chunk`` bounds the array solve leading (a MACHINE knob; ``0`` solves all
at once); its default is sized against the PRE-FLATTENING leading, which still
carried the 32-slot column-MUX axis, so it needs retuning for the flattened
shapes together with the policy's own ``solve_chunk_size``. ``--device auto``
picks a free GPU via ``nvidia-smi`` (else CPU serial).

Per-block breakdown (INFORMATIONAL, not gated): each Fig.18 slice is reported in
pJ/access next to its ``share * 32.06 pJ`` reference; differences are labelled as
convention / node-voltage effects, not gated.

Run:
    make validate_xue2020jssc

The three TOML artifacts are FIXED files beside this script; only the run knobs
(device, seed, draw counts, solver chunk) are CLI-settable, by invoking the script
directly:

    TORCH_COMPILE_DISABLE=1 uv run python validations/xue2020jssc/validate.py \
        --device auto --n-w 64 --n-x 256 --repeat 8 --solve-chunk 4096

``results.md`` records the run whose report text this harness logs; the workload
``p_zero`` is read from ``anchors.toml``, never injected on the command line.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import statistics
import subprocess
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from neurox import Profiler, Reporter, stamp_names
from neurox.common import StaticEntry
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy
from neurox.works.macro.cim.xue2020jssc import Xue2020JsscCimMacro, Xue2020JsscCimMacroPolicy

_LOG = logging.getLogger(__name__)

_VAL_DIR = Path(__file__).resolve().parent
_PARAMS_PATH = _VAL_DIR / "params.toml"
_POLICY_PATH = _VAL_DIR / "policy.toml"
_ANCHORS_PATH = _VAL_DIR / "anchors.toml"

# Informational Fig.18 slices (never gated). Read path is pure physics; control /
# reference are the two ADOPTED peripheral seats.
_READ_PATH_SLICES = ("cablc", "dswct", "sinwp_sc", "pn_isub", "tmcsa")
_ADOPTED_SLICES = ("control", "reference")
_ALL_SLICES = (*_ADOPTED_SLICES, *_READ_PATH_SLICES)

# Paired-slice caliber: the paper splits one series input branch (at node V_CMD)
# between DSWCT and CABLC, and one series sink branch between SINWP-SC and
# PN-ISUB; only the pair sums are well-defined targets. Members keep no
# per-member target (rendered informationally).
_PAIRED_SLICES: dict[str, tuple[str, ...]] = {
    "cablc+dswct": ("cablc", "dswct"),
    "sinwp_sc+pn_isub": ("sinwp_sc", "pn_isub"),
}
_PAIR_MEMBERS = tuple(name for members in _PAIRED_SLICES.values() for name in members)

# Profiler energy-row keys per Fig.18 slice (dynamic side). A slice may pool
# several rows. The ``cablc`` slice pools the ``array`` module energy row (the array
# bills only its wire / cell capacitive cycling) PLUS the macro ``.cablc`` channel
# (the whole input branch ``V_DD * I_DL``). The DSWCT / SINWP-SC / PN-ISUB
# modules self-bill on their own module rows; ``reference`` has no dynamic row
# (100 % static); ``tmcsa`` is the scheme phase-billing MODULE row (the kernel
# ``adc`` is energy-silent). ``control`` is the one remaining macro channel
# besides ``.cablc``.
_DYN_NAMES: dict[str, tuple[str, ...]] = {
    "cablc": (".cablc", "array"),
    "dswct": ("dswct",),
    "sinwp_sc": ("sinwp_sc",),
    "pn_isub": ("pn_isub",),
    "control": (".control",),
    "tmcsa": ("tmcsa",),
}
# Static-record qualified names per slice (static side). The ``array`` static seat
# folds into ``cablc`` with the array's dynamic row (both belong to the input
# branch); the DSWCT / SINWP-SC / PN-ISUB module seats fold into their own slices;
# the ``tmcsa`` slice pools the billing-module seat and the kernel ``adc`` seat.
# Everything unmapped (wl_dac, sl_driver, the macro root) is pooled into an
# ``unmapped`` residual, kept visible in the breakdown so the slice sum reconciles
# with the true total.
_STATIC_NAMES: dict[str, tuple[str, ...]] = {
    "control": ("control",),
    "reference": ("adc_current_reference",),
    "cablc": ("cablc", "array"),
    "dswct": ("dswct",),
    "sinwp_sc": ("sinwp_sc",),
    "tmcsa": ("tmcsa", "adc"),
    "pn_isub": ("pn_isub",),
}

# The framework transcoder program() consumes: sign-magnitude, radix 2, two
# magnitude digits (LSB-first) -> a 3-bit signed weight.
_QUANTIZATION_MODE = 0
_ADC_BITS = 3
_ROW_NUM = 256
_COL_NUM = 128

_FJ_PER_PJ = 1000.0


# ---------------------------------------------------------------------------
# Build / draw
# ---------------------------------------------------------------------------


def build_macro(
    params_path: Path,
    policy_path: Path,
    *,
    device: torch.device,
    solve_chunk_size: int,
) -> Xue2020JsscCimMacro:
    """Build + fabricate the sub-array at ``inst_shape=()``, float32, eval mode.

    A module never knows its own name, so the assembled tree is stamped here:
    every energy record carries the name this walk hands out, and the reporter
    resolves its rows against the same tree.

    Args:
        params_path: Fixed ``params.toml``.
        policy_path: Fixed ``policy.toml``.
        device: Device the macro lives on.
        solve_chunk_size: Array solve chunk, a MACHINE knob overriding the policy
            file's field: it splits the broadcast leading (the input batch, the K
            WL sub-phases, and the serial MUX slots) to bound peak memory, and
            changes no modelled quantity.
    """
    config = CimMacroConfig.from_file(params_path, section="cim_macro")
    policy = CimMacroPolicy.from_file(policy_path, section="policy")
    assert isinstance(policy, Xue2020JsscCimMacroPolicy)
    policy = dataclasses.replace(
        policy,
        array_policy=dataclasses.replace(policy.array_policy, solve_chunk_size=solve_chunk_size),
    )
    macro = CimMacro.from_config(
        config=config,
        policy=policy,
        input_num=_ROW_NUM,
        output_num=_COL_NUM,
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    assert isinstance(macro, Xue2020JsscCimMacro)
    macro.to(device)
    macro.eval()
    macro.fabricate()
    stamp_names(macro)
    return macro


def leakage_window__ns(macro: Xue2020JsscCimMacro) -> float:
    """Duration one access's static power integrates over [ns].

    The measurement period ``t_cycle__ns`` = 1 / 20 MHz, the rate the paper's
    5.13 mW power figure was taken at. That period is a DUTY-CYCLE property of
    the measured setup, a distinct quantity from the ACCESS TIME the macro's
    ``latency__ns`` reports (the span one read chain settles over, the executed
    sensing included): a macro clocked at 20 MHz leaks for the whole period however
    small a fraction of it the read occupies. Which of the two a GENERAL
    workload should integrate over is an OPEN modelling choice — a duty-cycled
    deployment takes the period, a back-to-back one the access time. It is
    fixed here to the period, the basis the paper's figure was measured on, and
    it is declared here rather than derived so the choice cannot be made
    implicitly by whichever duration a consumer happens to reach for.
    """
    return macro.config.t_cycle__ns


def _draw_weight(gen: torch.Generator, *, input_num: int, output_num: int, lo: int, hi: int) -> torch.Tensor:
    """Value-uniform logical weights ``[input_num, output_num]``."""
    return torch.randint(
        lo,
        hi + 1,
        (input_num, output_num),
        generator=gen,
        dtype=torch.long,
        device=gen.device,
    )


def _draw_input(
    gen: torch.Generator,
    *,
    batch: int,
    input_num: int,
    max_active_num: int,
    lo: int,
    hi: int,
    p_zero: float,
) -> torch.Tensor:
    """Value-uniform inputs ``[batch, row]`` in ``[lo, hi]`` with an EXTRA Bernoulli zeroing at ``p_zero``.

    ``p_zero`` is a dropout probability applied ON TOP of the value-uniform draw
    (sparse-activation workload assumption), so the marginal zero fraction is
    ``P(x=0) = f0 + (1 - f0) * p_zero`` for the base rate ``f0 = 1/(hi-lo+1)``, NOT
    ``p_zero`` itself. Inactive rows (``>= active_row_num``) are forced to zero.
    """
    x = torch.randint(lo, hi + 1, (batch, input_num), generator=gen, dtype=torch.long, device=gen.device)
    if p_zero > 0.0:
        drop = torch.rand((batch, input_num), generator=gen, device=gen.device) < p_zero
        x = torch.where(drop, torch.zeros_like(x), x)
    x[:, max_active_num:] = 0
    return x


# ---------------------------------------------------------------------------
# Measure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SliceEnergy:
    """One Fig.18 slice: dynamic + static energy per access [pJ] vs its informational target.

    A pair MEMBER slice (cablc / dswct / sinwp_sc / pn_isub) carries
    ``target__pJ = 0.0`` — under the paired-slice caliber only the pair sums
    have well-defined targets; members are reported informationally.
    """

    name: str
    dynamic__pJ: float
    static__pJ: float
    target__pJ: float

    @property
    def total__pJ(self) -> float:
        return self.dynamic__pJ + self.static__pJ

    @property
    def ratio(self) -> float:
        return self.total__pJ / self.target__pJ if self.target__pJ else float("inf")


def paired_slices(slices: tuple[SliceEnergy, ...], shares: dict, target_total: float) -> tuple[SliceEnergy, ...]:
    """Aggregate the member slices into the paired-caliber comparison rows.

    Each pair row sums its members' dynamic / static energy and compares
    against the SUM of the members' Fig.18 shares (the only well-defined
    target: the paper splits one series branch between the two slices at an
    unpublished internal node voltage).
    """
    by_name = {s.name: s for s in slices}
    return tuple(
        SliceEnergy(
            name=pair,
            dynamic__pJ=sum(by_name[m].dynamic__pJ for m in members),
            static__pJ=sum(by_name[m].static__pJ for m in members),
            target__pJ=sum(shares[m] for m in members) / 100.0 * target_total,
        )
        for pair, members in _PAIRED_SLICES.items()
    )


@dataclass(frozen=True)
class Measurement:
    """Energy-per-access measurement: the gated total plus the informational breakdown.

    Attributes:
        total__pJ: The gated total energy per access.
        dynamic__pJ: Dynamic share of the total.
        static__pJ: Static (leakage) share of the total.
        slices: The informational Fig.18 slice breakdown.
        unmapped_static__pJ: Static residual belonging to no slice.
        access_latency__ns: The macro's modelled access time, per output
            access. Reported beside the energies; the static term integrates
            over :func:`leakage_window__ns`, never over this.
        window__ns: The declared leakage integration window.
        n_w: Weight programs drawn per round.
        n_x: Input vectors drawn per weight program.
        accesses: Output accesses read over every pooled round.
        p_zero: Input-sparsity point the round(s) ran at.
        seed: Generator seed of the first round.
        repeat: Pooled rounds, each redrawing the weights AND the inputs.
        rel_std: Relative standard deviation of the round totals.
        dyn_by_name: Every profiler dynamic-energy row in pJ PER ACCESS, keyed
            by qualified name. The gate reads only the slice aggregation above;
            this raw row view is what the calibration campaign
            (``tools/calibrate.py``) reduces its per-block residuals from.
    """

    total__pJ: float
    dynamic__pJ: float
    static__pJ: float
    slices: tuple[SliceEnergy, ...]
    unmapped_static__pJ: float
    access_latency__ns: float
    window__ns: float
    n_w: int
    n_x: int
    accesses: int
    p_zero: float
    seed: int
    repeat: int = 1
    rel_std: float = 0.0
    dyn_by_name: dict[str, float] = field(default_factory=dict)

    @property
    def draws(self) -> int:
        """Input draws over every pooled round: ``repeat * n_w * n_x``."""
        return self.repeat * self.n_w * self.n_x

    def slice(self, name: str) -> SliceEnergy:
        return next(s for s in self.slices if s.name == name)


def _per_access(
    static_entries: tuple[StaticEntry, ...],
    dynamic_by_name__fJ: dict[str, float],
    *,
    accesses: int,
    window__ns: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return ``(dynamic_by_name, static_by_name)`` energy PER ACCESS [pJ].

    Dynamic: each accumulated energy row divided by ``accesses``. Static: each
    block's ``leakage_power * window`` (the per-access integral of its leakage),
    which is fabrication-fixed and therefore read from the reporter's static
    rows, independently of any round. This is the SOLE static-energy derivation:
    the reported total is the sum of these rows, so the breakdown and the gated
    total cannot drift apart.
    """
    dyn = {k: v / accesses / _FJ_PER_PJ for k, v in dynamic_by_name__fJ.items()}
    stat = {e.qualified_name: e.leakage__uW * window__ns / _FJ_PER_PJ for e in static_entries}
    return dyn, stat


def measure(
    macro: Xue2020JsscCimMacro,
    reporter: Reporter,
    anchors: dict,
    *,
    n: int,
    p_zero: float,
    seed: int,
    batch: int = 8,
) -> Measurement:
    """Profile ONE round of ``n`` random draws and reduce to the energy per access.

    Draws ``n_w = ceil(n / batch)`` random weight matrices, each programmed and then
    driven by a fresh ``batch`` of random inputs, and reduces to the total / dynamic
    / static energy per access plus the informational Fig.18 slice breakdown. In the
    round vocabulary the two counts are ``n_w`` weight draws and ``n_x = batch``
    input vectors per weight draw, so the round reads ``n_w * n_x`` (input, weight)
    pairs.

    Each weight draw is profiled in its OWN context at ``leading_rank=1``: the
    caller's ``batch`` axis indexes independent unit operations, so each energy
    event resolves to ``[batch]`` and the dynamic energy rows are accumulated
    across the draws by hand. ``reporter`` is the one bound to ``macro``: it
    names every context's rows and carries the static rows. The two sides
    normalize differently: dynamic energy divides by the full ``accesses``
    count, while static energy is the fabrication-fixed ``leakage_power *
    window`` of ONE access.
    """
    cfg = macro.config
    device = next(macro.buffers()).device
    gen = torch.Generator(device=device).manual_seed(seed)

    data = anchors["data"]
    w_lo, w_hi = data["weight_range"]
    x_lo, x_hi = data["input_range"]
    target_total = anchors["target"]["per_access__pJ"]
    shares = anchors["fig18_shares"]

    mux = cfg.mux_factor
    window__ns = leakage_window__ns(macro)
    # One VMM serializes the `mux` column-MUX slots, so the duration the macro
    # reports divided by `mux` is the access time. Config-fixed, hence identical
    # for every draw.
    access_latency__ns = macro.latency__ns(adc_bits=_ADC_BITS) / mux

    n_w = max(1, -(-n // batch))  # ceil
    n_samples = 0
    dyn_by_name__fJ: dict[str, float] = {}
    total_dynamic__fJ = 0.0
    with torch.no_grad():
        for _ in range(n_w):
            w = _draw_weight(
                gen,
                input_num=macro.row_num,
                output_num=macro.col_num,
                lo=w_lo,
                hi=w_hi,
            )
            x = _draw_input(
                gen,
                batch=batch,
                input_num=macro.row_num,
                max_active_num=cfg.max_active_num,
                lo=x_lo,
                hi=x_hi,
                p_zero=p_zero,
            )
            # leading_rank=1: the `batch` axis indexes independent unit
            # operations, so each energy event resolves to [batch], one element
            # per input vector. device=None leaves the records where they were
            # recorded: the reporter reduces a whole book in one transfer.
            with Profiler(leading_rank=1, device=None) as prof:
                macro.program(w)
                macro.vec_mat_mul(x, quantization_mode=_QUANTIZATION_MODE, adc_bits=_ADC_BITS)
            for name, e__fJ in reporter.by_name(prof).items():
                dyn_by_name__fJ[name] = dyn_by_name__fJ.get(name, 0.0) + e__fJ
            total_dynamic__fJ += reporter.total_dynamic_energy__fJ(prof)
            n_samples += batch
    accesses = n_samples * mux

    dyn, stat = _per_access(reporter.static_entries, dyn_by_name__fJ, accesses=accesses, window__ns=window__ns)
    # The one static-energy derivation, shared by the total and the breakdown.
    static__pJ = sum(stat.values())
    dynamic__pJ = total_dynamic__fJ / accesses / _FJ_PER_PJ
    total__pJ = dynamic__pJ + static__pJ
    # Pair MEMBERS carry no per-member target (paired-slice caliber); the pair
    # rows derived by ``paired_slices`` carry the summed-share targets.
    slices = tuple(
        SliceEnergy(
            name=s,
            dynamic__pJ=sum(dyn.get(k, 0.0) for k in _DYN_NAMES.get(s, ())),
            static__pJ=sum(stat.get(k, 0.0) for k in _STATIC_NAMES.get(s, ())),
            target__pJ=0.0 if s in _PAIR_MEMBERS else shares[s] / 100.0 * target_total,
        )
        for s in _ALL_SLICES
    )
    mapped_static = sum(stat.get(k, 0.0) for s in _ALL_SLICES for k in _STATIC_NAMES.get(s, ()))
    unmapped_static = static__pJ - mapped_static

    return Measurement(
        total__pJ=total__pJ,
        dynamic__pJ=dynamic__pJ,
        static__pJ=static__pJ,
        slices=slices,
        unmapped_static__pJ=unmapped_static,
        access_latency__ns=access_latency__ns,
        window__ns=window__ns,
        n_w=n_w,
        n_x=batch,
        accesses=accesses,
        p_zero=p_zero,
        seed=seed,
        dyn_by_name=dyn,
    )


# ---------------------------------------------------------------------------
# Round pooling
# ---------------------------------------------------------------------------


def _pool_rounds(rounds: list[Measurement], *, p_zero: float, seed: int) -> Measurement:
    """Access-weighted mean of per-round measurements + the round-total relative std.

    Each round is an independent draw profiled in its own context, so the
    per-access energies combine as an access-weighted mean (= ``sum(energy) /
    sum(accesses)``) and the round-to-round dispersion of the total per-access is
    the statistical uncertainty. ``rel_std`` is the sample relative standard
    deviation of the round totals (0 for a single round).
    """
    first = rounds[0]
    total_accesses = sum(r.accesses for r in rounds)
    w = [r.accesses / total_accesses for r in rounds]

    def wmean(getter: Callable[..., float]) -> float:
        return sum(wi * getter(r) for wi, r in zip(w, rounds, strict=True))

    names = tuple(s.name for s in first.slices)
    slices = tuple(
        SliceEnergy(
            name=name,
            dynamic__pJ=wmean(lambda r, n=name: r.slice(n).dynamic__pJ),
            static__pJ=wmean(lambda r, n=name: r.slice(n).static__pJ),
            target__pJ=first.slice(name).target__pJ,
        )
        for name in names
    )
    round_totals = [r.total__pJ for r in rounds]
    mean_total = wmean(lambda r: r.total__pJ)
    rel_std = statistics.stdev(round_totals) / mean_total if len(round_totals) > 1 and mean_total else 0.0
    row_names = {name for r in rounds for name in r.dyn_by_name}
    dyn_by_name = {name: wmean(lambda r, n=name: r.dyn_by_name.get(n, 0.0)) for name in sorted(row_names)}

    return Measurement(
        total__pJ=mean_total,
        dynamic__pJ=wmean(lambda r: r.dynamic__pJ),
        static__pJ=wmean(lambda r: r.static__pJ),
        slices=slices,
        unmapped_static__pJ=wmean(lambda r: r.unmapped_static__pJ),
        access_latency__ns=first.access_latency__ns,
        window__ns=first.window__ns,
        n_w=first.n_w,
        n_x=first.n_x,
        accesses=total_accesses,
        p_zero=p_zero,
        seed=seed,
        repeat=len(rounds),
        rel_std=rel_std,
        dyn_by_name=dyn_by_name,
    )


def measure_rounds(
    macro: Xue2020JsscCimMacro,
    anchors: dict,
    *,
    n_w: int,
    n_x: int,
    repeat: int,
    p_zero: float,
    seed: int,
) -> Measurement:
    """Profile ``repeat`` rounds of ``n_w`` weight draws x ``n_x`` inputs and pool them.

    Every round redraws BOTH the weights and the inputs (its own seed, hence
    independent draws) and is profiled in its own context, so the peak
    event/tensor footprint stays that of ONE round however many rounds run, and
    the round-to-round spread of the total is the reported uncertainty.

    The reporter binds ``macro`` once, BEFORE any round runs: that walk is where
    a missing or stale name stamp is caught, and one binding then names the rows
    of every round and supplies the fabrication-fixed static rows.
    """
    reporter = Reporter(macro)
    rounds = [
        measure(macro, reporter, anchors, n=n_w * n_x, p_zero=p_zero, seed=seed + idx, batch=n_x)
        for idx in range(repeat)
    ]
    return _pool_rounds(rounds, p_zero=p_zero, seed=seed)


# ---------------------------------------------------------------------------
# Gate + tables
# ---------------------------------------------------------------------------


def gate(m: Measurement, anchors: dict) -> tuple[bool, float]:
    """Hard gate: total energy per access within +-tol of the target. Return ``(pass, rel_error)``."""
    target = anchors["target"]["per_access__pJ"]
    tol = anchors["gate"]["hard_tolerance_relative"]
    rel = (m.total__pJ - target) / target
    return abs(rel) <= tol, rel


def energy_table(m: Measurement, anchors: dict) -> str:
    """Informational per-block breakdown (pJ/access) + the gated total row.

    Paired-slice caliber: the target-bearing rows are the two pair sums
    (``cablc+dswct`` vs 26.4 %, ``sinwp_sc+pn_isub`` vs 11.4 %) and the
    singles ``control`` / ``reference`` / ``tmcsa``; the four member rows are
    kept visible without a per-member target (the paper's internal split node
    voltages are unpublished).
    """
    target = anchors["target"]["per_access__pJ"]
    tol = anchors["gate"]["hard_tolerance_relative"]
    shares = anchors["fig18_shares"]
    lines: list[str] = []
    lines.append("| Slice | Energy [pJ/acc] | dyn | static | Fig.18 x 32.06 [pJ] | pred/ref | basis |")
    lines.append("|---|--:|--:|--:|--:|--:|:--|")

    def row(s: SliceEnergy, *, basis: str) -> str:
        target_cell = f"{s.target__pJ:8.3f}" if s.target__pJ else "    -   "
        ratio_cell = f"{s.ratio:5.2f}x" if s.target__pJ else "  -   "
        return (
            f"| {s.name} | {s.total__pJ:8.3f} | {s.dynamic__pJ:7.3f} | {s.static__pJ:6.3f} | "
            f"{target_cell} | {ratio_cell} | {basis} |"
        )

    for name in _ADOPTED_SLICES:
        lines.append(row(m.slice(name), basis="adopted"))
    for pair in paired_slices(m.slices, shares, target):
        lines.append(row(pair, basis="physics pair"))
    lines.append(row(m.slice("tmcsa"), basis="physics"))
    for name in _PAIR_MEMBERS:
        lines.append(row(m.slice(name), basis="pair member"))
    if abs(m.unmapped_static__pJ) > 1e-9:
        lines.append(
            f"| (unmapped static) | {m.unmapped_static__pJ:8.3f} | {0.0:7.3f} | {m.unmapped_static__pJ:6.3f} | "
            f"{0.0:8.3f} |    -   | residual |"
        )
    within, rel = gate(m, anchors)
    lines.append(
        f"| **TOTAL (gated)** | **{m.total__pJ:8.3f}** | {m.dynamic__pJ:7.3f} | {m.static__pJ:6.3f} | "
        f"**{target:8.3f}** | **{m.total__pJ / target:5.3f}x** | {'PASS' if within else 'FAIL'} "
        f"(+-{tol * 100:.0f}%, err {rel * 100:+.1f}%) |"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render_report(m: Measurement, anchors: dict, *, device: torch.device) -> str:
    """Energy-basis gate report, the text ``results.md`` records."""
    target = anchors["target"]["per_access__pJ"]
    tol = anchors["gate"]["hard_tolerance_relative"]
    within, rel = gate(m, anchors)
    data = anchors["data"]
    x_lo, x_hi = data["input_range"]
    f0 = 1.0 / (x_hi - x_lo + 1)
    marginal = f0 + (1.0 - f0) * m.p_zero

    lines: list[str] = []
    lines.append("# xue2020jssc validation -- total energy per access")
    lines.append("")
    round_note = (
        f" Pooled over {m.repeat} rounds (relative std of the round totals = {m.rel_std * 100:.2f} %)."
        if m.repeat > 1
        else ""
    )
    lines.append(
        f"Energy-basis profiler run for `{_PARAMS_PATH.name}` + `{_POLICY_PATH.name}` on {device}. "
        f"n_w = {m.n_w} weight draws x n_x = {m.n_x} inputs x {m.repeat} rounds = {m.draws} draws "
        f"({m.accesses} accesses), seed {m.seed}, run p_zero = {m.p_zero:.3f} "
        f"(marginal P(x=0) = {marginal:.3f}); anchors-declared workload p_zero = {anchors['data']['p_zero']:.2f}."
        f"{round_note}"
    )
    lines.append("")
    declared_p = anchors["data"]["p_zero"]
    declared_marg = f0 + (1.0 - f0) * declared_p
    lines.append("## Hard gate -- total energy per access")
    lines.append("")
    std_note = f" +- {m.rel_std * 100:.2f} % (round-total relative std over {m.repeat} rounds)" if m.repeat > 1 else ""
    lines.append(
        f"Target 32.06 pJ/access (= 5.13 mW / 8 / 20 MHz); +-{tol * 100:.0f}%. At this run's p_zero = "
        f"{m.p_zero:.3f} (marginal P(x=0) = {marginal:.3f}): result **{m.total__pJ:.3f} pJ/access = "
        f"{m.total__pJ / target:.3f}x**{std_note} (err {rel * 100:+.1f}%), within +-{tol * 100:.0f}%: "
        f"{'yes' if within else 'no'}."
    )
    lines.append("")
    lines.append(
        f"The anchors-declared workload sparsity is p_zero = {declared_p:.2f} (marginal P(x=0) ~= "
        f"{declared_marg:.3f}), motivated INDEPENDENTLY by typical ~50%-zero post-ReLU CNN activations -- NOT "
        f"tuned to pass."
    )
    lines.append("")
    lines.append("## Energy breakdown (informational -- NOT gated)")
    lines.append("")
    lines.append(energy_table(m, anchors))
    lines.append("")
    lines.append(
        "The read-path slices are pure physics (g_map, V_BLC, conduction windows -- all declared); "
        "control + reference are the two ADOPTED Fig.18 seats. Paired-slice caliber: the paper splits one "
        "series input branch at node V_CMD (drain of the DSWCT current-mirror input, Fig.9(a)) between DSWCT "
        "and CABLC, and one series sink branch between SINWP-SC (its sink transistors) and PN-ISUB (switches "
        "+ comparator + isub); the internal node voltages are unpublished, so only the pair sums "
        "(cablc+dswct vs 26.4 %, sinwp_sc+pn_isub vs 11.4 %) are well-defined targets -- the member rows are "
        "informational. Differences from Fig.18 x 32.06 pJ are reported, not gated."
    )
    lines.append("")
    lines.append("## Declared conventions")
    lines.append("")
    lines.append(
        f"- Hard gate: total energy per access within +-{tol * 100:.0f}% of {target} pJ at the declared p_zero."
    )
    lines.append(
        "- Adopted seats (declared to reproduce a Fig.18 share, not fitted to the total): "
        "control 29.2 % (pure per-op, 100 % dynamic), reference 23.7 % (100 % static)."
    )
    lines.append(
        "- Read path (cablc, dswct, sinwp_sc, pn_isub, tmcsa): pure physics; static seats declared small/zero, "
        "NEVER reverse-solved to fill the total. Fig.18 comparison at the paired-slice caliber "
        "(cablc+dswct, sinwp_sc+pn_isub)."
    )
    lines.append(
        f"- Data: weights value-uniform in {data['weight_range']}, inputs value-uniform in {data['input_range']} "
        f"with an extra Bernoulli zeroing at p_zero (declared workload assumption); rows >= active_row_num zeroed."
    )
    lines.append(
        f"- Energy basis: static/access = leakage_power * leakage window ({m.window__ns:.1f} ns = the 20 MHz "
        f"measurement period, a duty-cycle property); dynamic/access = per-VMM dynamic / mux_factor; total/access "
        f"= their sum. The macro's access time ({m.access_latency__ns:.2f} ns) is the separate duration one read "
        f"chain settles over and integrates nothing."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _pick_free_gpu() -> int | None:
    """Return the index of a free CUDA GPU (lowest util, most free memory), or ``None``.

    Parses ``nvidia-smi``: use a free GPU if one shows, else CPU. A GPU counts as
    free at ``<= 10 %`` utilization; among those the one with
    the most free memory wins. Any failure (no ``nvidia-smi``, no visible / free
    GPU) returns ``None`` so the caller falls back to CPU serial.
    """
    if not torch.cuda.is_available():
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    best: tuple[int, float] | None = None  # (index, free_mib) among <=10% util
    for line in out.strip().splitlines():
        try:
            idx_s, util_s, free_s = (p.strip() for p in line.split(","))
            idx, util, free = int(idx_s), float(util_s), float(free_s)
        except ValueError:
            continue
        if util <= 10.0 and (best is None or free > best[1]):
            best = (idx, free)
    return best[0] if best is not None else None


def _resolve_device(name: str) -> torch.device:
    """Resolve a device name; ``auto`` picks a free GPU, else CPU."""
    if name == "auto":
        idx = _pick_free_gpu()
        dev = torch.device(f"cuda:{idx}") if idx is not None else torch.device("cpu")
        _LOG.info("[auto device -> %s]", dev)
        return dev
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    return torch.device(name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-w", type=int, default=64, help="Weight draws per round: the weight programs drawn per round.")
    ap.add_argument("--n-x", type=int, default=256, help="Input vectors per weight draw.")
    ap.add_argument("--repeat", type=int, default=8, help="Rounds; each redraws the weights AND the inputs.")
    ap.add_argument(
        "--solve-chunk",
        type=int,
        default=4096,
        help="Array solve chunk (machine knob): leading instances solved per block; 0 solves all at once.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cpu, cuda[:idx], or auto (pick a free GPU via nvidia-smi else cpu); default auto.",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with _ANCHORS_PATH.open("rb") as fh:
        anchors = tomllib.load(fh)
    p_zero = float(anchors["data"]["p_zero"])

    device = _resolve_device(args.device)
    macro = build_macro(_PARAMS_PATH, _POLICY_PATH, device=device, solve_chunk_size=args.solve_chunk)
    m = measure_rounds(
        macro,
        anchors,
        n_w=args.n_w,
        n_x=args.n_x,
        repeat=args.repeat,
        p_zero=p_zero,
        seed=args.seed,
    )

    _LOG.info("%s", render_report(m, anchors, device=device))


if __name__ == "__main__":
    main()
