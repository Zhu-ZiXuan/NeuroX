"""Energy-basis, total-only validation for the xue2020jssc CIM sub-array.

Builds :class:`~neurox.works.macro.cim.xue2020jssc.Xue2020JsscCimMacro` from
``params.toml`` + ``policy.toml``, drives it DIRECTLY (``max_active_num`` row
positions sampled independently for every input, the rest zeroed) over a
round-based random workload per the ``anchors.toml`` data conventions, and
reduces the profiler to the ENERGY PER ACCESS. The one HARD
GATE is the total energy per access against 32062.5 fJ/access (= 32.0625 pJ =
the paper's simulated 5.13 mW / 8 sub-arrays / 20 MHz) within +-5% at the
declared calibrated-activity workload.

Energy-basis reduction. With ``accesses = n_samples * mux_factor`` (one VMM over
all ``col_num`` logical columns is ``mux_factor`` serial accesses):
  - static energy per access = ``leakage_power * leakage_window__ns`` (the
    declared integration window; independent of the draw count and ``mux_factor``);
  - dynamic energy per access = ``total_dynamic_energy / accesses`` = per-VMM
    dynamic / ``mux_factor``.
The total energy per access is their sum. 1 uA * 1 V * 1 ns = 1 fJ; 1 pJ = 1000 fJ.
The macro's access time is reported beside them and integrates nothing.

ACCOUNTING BASIS: the seats declared to reproduce a Fig.18 share are
Control (29.2 %) + Reference (23.7 %) -- ADOPTED, because those two peripherals
are not modeled explicitly. The read path follows an explicit circuit model under
declared device, parasitic, timing, and workload assumptions; its static seats are
declared small/zero. The effective input and weight nonzero probabilities are
calibrated against the two read-path conduction seats; conditional on being
nonzero, sign-magnitude weights and unsigned inputs are uniform. TMCSA energy
follows its explicit branch-current and phase-duration model. With Control and Reference adopted from
the remaining Fig.18 shares, agreement with the total is therefore a consistency
check, not independent validation.

PAIRED-SLICE caliber for the Fig.18 comparison: the model bills ONE series input
branch across DSWCT and CABLC and one series sink branch across SINWP-SC and
PN-ISUB. Fig.18 reports separate circuit-block shares, but the internal node
voltages needed to reproduce that split are not published, so only the PAIR SUMS
are well-defined model-comparison targets. The breakdown therefore compares ``cablc+dswct``
against 14.9 + 11.5 = 26.4 % and ``sinwp_sc+pn_isub`` against 8.0 + 3.4 =
11.4 %, with ``control`` / ``reference`` / ``tmcsa`` as singles; the four member
rows stay visible (informational, no per-member target).

The array module row bills only its wire / cell capacitive cycling, and the macro
``cablc`` channel bills the whole input branch ``VDD * I_DL``; the ``cablc``
slice SUMS the two.

ROUND-based workload: ``--repeat`` rounds, each redrawing ``--n-w`` weight programs
and, per program, ``--n-x`` input vectors, and each profiled in its OWN context;
the per-access energies accumulate as an access-weighted mean and the report adds
the round-total relative std. Peak profiler memory stays that of one round.
``--solve-chunk`` bounds the array solve leading (a MACHINE knob; ``0`` solves all
at once); its default is sized against the PRE-FLATTENING leading, which still
carried the 32-slot column-MUX axis, so it needs retuning for the flattened
shapes together with the policy's own ``solve_chunk_size``. ``--device auto`` is
``cuda`` when a CUDA device is visible, else CPU serial; WHICH GPU that is stays
the operator's choice, through ``CUDA_VISIBLE_DEVICES``.

Per-block breakdown (INFORMATIONAL, not gated): each Fig.18 slice is reported in
fJ/access next to its ``share * target`` reference; differences are labelled as
convention / node-voltage effects, not gated.

Run:
    make validate_xue2020jssc

The three TOML artifacts are fixed files beside this script; the CLI accepts
runtime knobs and the output directory:

    TORCH_COMPILE_DISABLE=1 uv run --extra calib python validations/xue2020jssc/validate.py \
        --device auto --n-w 64 --n-x 256 --repeat 8 --solve-chunk 4096 \
        --output-dir log/validation/xue2020jssc

The report and an SVG breakdown comparison are written under ``--output-dir``;
no generated Markdown report is kept in the repository.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import statistics
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from neurox import Profiler, Reporter, stamp_names
from neurox.common import StaticEntry
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy
from neurox.works.macro.cim.xue2020jssc import Xue2020JsscCimMacro

_LOG = logging.getLogger(__name__)

_VAL_DIR = Path(__file__).resolve().parent
_PARAMS_PATH = _VAL_DIR / "params.toml"
_POLICY_PATH = _VAL_DIR / "policy.toml"
_ANCHORS_PATH = _VAL_DIR / "anchors.toml"

# Informational Fig.18 slices (never gated). Read path uses the explicit model; control /
# reference are the two ADOPTED peripheral seats.
_READ_PATH_SLICES = ("cablc", "dswct", "sinwp_sc", "pn_isub", "tmcsa")
_ADOPTED_SLICES = ("control", "reference")
_ALL_SLICES = (*_ADOPTED_SLICES, *_READ_PATH_SLICES)

# Paired-slice caliber: the model bills one series input branch across DSWCT and
# CABLC and one series sink branch across SINWP-SC and PN-ISUB; only the pair
# sums are well-defined comparisons without unpublished internal node voltages. Members keep no
# per-member target (rendered informationally).
_PAIRED_SLICES: dict[str, tuple[str, ...]] = {
    "cablc+dswct": ("cablc", "dswct"),
    "sinwp_sc+pn_isub": ("sinwp_sc", "pn_isub"),
}
_PAIR_MEMBERS = tuple(name for members in _PAIRED_SLICES.values() for name in members)

# Profiler energy-row keys per Fig.18 slice (dynamic side). A slice may pool
# several rows. The ``cablc`` slice pools the ``array`` module energy row (the array
# bills only its wire / cell capacitive cycling) PLUS the macro ``.cablc`` channel
# (the whole input branch ``VDD * I_DL``). The DSWCT / SINWP-SC / PN-ISUB
# modules self-bill on their own module rows; ``reference`` has no dynamic row
# (100 % static); ``tmcsa`` is the scheme phase-billing MODULE row (the kernel
# ``adc`` is energy-silent). ``control`` self-bills on its module row.
_DYN_NAMES: dict[str, tuple[str, ...]] = {
    "cablc": (".cablc", "array"),
    "dswct": ("dswct",),
    "sinwp_sc": ("sinwp_sc",),
    "pn_isub": ("pn_isub",),
    "control": ("control",),
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
# [reported p211 Fig.20(c)] 256 rows x 512 physical columns per sub-array.
ROW_NUM = 256
# [derived] Four physical columns encode each logical output: two magnitude
# digits x two polarities, so 512 / 4 = 128 logical outputs.
COL_NUM = 128
# [assumed] Nominal ambient operating point; the paper does not publish a
# temperature for the power simulation.
TEMPERATURE__K = 300.0


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
    policy = dataclasses.replace(
        policy,
        array_policy=dataclasses.replace(policy.array_policy, solve_chunk_size=solve_chunk_size),
    )
    macro = CimMacro.from_config(
        config=config,
        policy=policy,
        input_num=ROW_NUM,
        output_num=COL_NUM,
        inst_shape=(),
        dtype=torch.float32,
        T__K=TEMPERATURE__K,
    )
    macro.to(device)
    macro.eval()
    macro.fabricate()
    stamp_names(macro)
    return macro


def leakage_window__ns(macro: Xue2020JsscCimMacro) -> float:
    """Duration one access's static power integrates over [ns].

    The simulated operating period ``t_cycle__ns`` = 1 / 20 MHz, the rate of the
    paper's 5.13 mW power point. That period is a DUTY-CYCLE property, distinct
    from the ACCESS TIME the macro's ``access_latency__ns`` reports (the span one
    read chain settles over, the executed sensing included): a macro clocked at
    20 MHz leaks for the whole period however
    small a fraction of it the read occupies. The campaign therefore integrates
    every declared static seat over the complete period; the active phase times
    only govern dynamic conduction.
    """
    return macro.config.t_cycle__ns


def _draw_weight(
    gen: torch.Generator,
    *,
    input_num: int,
    output_num: int,
    max_magnitude: int,
    nonzero_probability: float,
) -> torch.Tensor:
    """Zero-inflated sign-magnitude weights, uniform conditional on nonzero."""
    nonzero = torch.rand((input_num, output_num), generator=gen, device=gen.device) < nonzero_probability
    magnitude = torch.randint(
        1,
        max_magnitude + 1,
        (input_num, output_num),
        generator=gen,
        dtype=torch.long,
        device=gen.device,
    )
    negative = torch.randint(
        0,
        2,
        (input_num, output_num),
        generator=gen,
        dtype=torch.bool,
        device=gen.device,
    )
    signed = torch.where(negative, -magnitude, magnitude)
    return torch.where(nonzero, signed, 0)


def _draw_input(
    gen: torch.Generator,
    *,
    batch: int,
    input_num: int,
    max_active_num: int,
    lo: int,
    hi: int,
    nonzero_probability: float,
) -> torch.Tensor:
    """At most `max_active_num` candidate rows, zero-inflated independently."""
    active_rows = torch.rand((batch, input_num), generator=gen, device=gen.device).topk(max_active_num, dim=1).indices
    nonzero = torch.rand((batch, max_active_num), generator=gen, device=gen.device) < nonzero_probability
    values = torch.randint(
        max(1, lo),
        hi + 1,
        (batch, max_active_num),
        generator=gen,
        dtype=torch.long,
        device=gen.device,
    )
    values = torch.where(nonzero, values, 0)
    return torch.zeros((batch, input_num), dtype=torch.long, device=gen.device).scatter(1, active_rows, values)


@dataclass(frozen=True)
class SliceEnergy:
    """One Fig.18 slice: dynamic + static energy per access [fJ] vs its informational target.

    A pair MEMBER slice (cablc / dswct / sinwp_sc / pn_isub) carries
    ``target__fJ = 0.0`` — under the paired-slice caliber only the pair sums
    have well-defined targets; members are reported informationally.
    """

    name: str
    dynamic__fJ: float
    static__fJ: float
    target__fJ: float

    @property
    def total__fJ(self) -> float:
        return self.dynamic__fJ + self.static__fJ

    @property
    def ratio(self) -> float:
        return self.total__fJ / self.target__fJ if self.target__fJ else float("inf")


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
            dynamic__fJ=sum(by_name[m].dynamic__fJ for m in members),
            static__fJ=sum(by_name[m].static__fJ for m in members),
            target__fJ=sum(shares[m] for m in members) / 100.0 * target_total,
        )
        for pair, members in _PAIRED_SLICES.items()
    )


def comparison_slices(m: Measurement, anchors: dict) -> tuple[SliceEnergy, ...]:
    """Return the five rows comparable with Fig.18 under the paired accounting basis."""
    pairs = {s.name: s for s in paired_slices(m.slices, anchors["fig18_shares"], anchors["target"]["per_access__fJ"])}
    return (
        m.slice("control"),
        m.slice("reference"),
        pairs["cablc+dswct"],
        pairs["sinwp_sc+pn_isub"],
        m.slice("tmcsa"),
    )


@dataclass(frozen=True)
class Measurement:
    """Energy-per-access measurement: the gated total plus the informational breakdown."""

    total__fJ: float
    """The gated total energy per access."""
    dynamic__fJ: float
    """Dynamic share of the total."""
    static__fJ: float
    """Static (leakage) share of the total."""
    slices: tuple[SliceEnergy, ...]
    """The informational Fig.18 slice breakdown."""
    unmapped_static__fJ: float
    """Static residual belonging to no slice."""
    access_latency__ns: float
    """The macro's modelled access time, per output access. Reported beside the
    energies; the static term integrates over `leakage_window__ns`, never over this."""
    window__ns: float
    """The declared leakage integration window."""
    n_w: int
    """Weight programs drawn per round."""
    n_x: int
    """Input vectors drawn per weight program."""
    accesses: int
    """Output accesses read over every pooled round."""
    seed: int
    """Generator seed of the first round."""
    repeat: int = 1
    """Pooled rounds, each redrawing the weights AND the inputs."""
    rel_std: float = 0.0
    """Relative standard deviation of the round totals."""
    dyn_by_name: dict[str, float] = field(default_factory=dict)
    """Every profiler dynamic-energy row in fJ PER ACCESS, keyed by qualified name.
    The gate reads only the slice aggregation above; this raw row view preserves
    the component-level accounting behind each reported slice."""

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
    """Return ``(dynamic_by_name, static_by_name)`` energy PER ACCESS [fJ].

    Dynamic: each accumulated energy row divided by ``accesses``. Static: each
    block's ``leakage_power * window`` (the per-access integral of its leakage),
    which is fabrication-fixed and therefore read from the reporter's static
    rows, independently of any round. This is the SOLE static-energy derivation:
    the reported total is the sum of these rows, so the breakdown and the gated
    total cannot drift apart.
    """
    dyn = {k: v / accesses for k, v in dynamic_by_name__fJ.items()}
    stat = {e.qualified_name: e.leakage__uW * window__ns for e in static_entries}
    return dyn, stat


def measure(
    macro: Xue2020JsscCimMacro,
    reporter: Reporter,
    anchors: dict,
    *,
    n: int,
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
    names every context's rows and carries the static rows.
    """
    cfg = macro.config
    device = next(macro.buffers()).device
    gen = torch.Generator(device=device).manual_seed(seed)

    data = anchors["data"]
    w_lo, w_hi = data["weight_range"]
    x_lo, x_hi = data["input_range"]
    target_total = anchors["target"]["per_access__fJ"]
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
                max_magnitude=max(abs(w_lo), abs(w_hi)),
                nonzero_probability=float(data["weight_nonzero_probability"]),
            )
            x = _draw_input(
                gen,
                batch=batch,
                input_num=macro.row_num,
                max_active_num=cfg.max_active_num,
                lo=x_lo,
                hi=x_hi,
                nonzero_probability=float(data["input_nonzero_probability"]),
            )
            # leading_rank=1: the `batch` axis indexes independent unit
            # operations, so each energy event resolves to [batch], one element
            # per input vector. The records stay where they were recorded: the
            # reporter reduces a whole book in one transfer.
            with Profiler(leading_rank=1) as prof:
                macro.program(w)
                macro.vec_mat_mul(x, quantization_mode=_QUANTIZATION_MODE, adc_bits=_ADC_BITS)
            for name, e__fJ in reporter.by_name(prof).items():
                dyn_by_name__fJ[name] = dyn_by_name__fJ.get(name, 0.0) + e__fJ
            total_dynamic__fJ += reporter.total_dynamic_energy__fJ(prof)
            n_samples += batch
    accesses = n_samples * mux

    dyn, stat = _per_access(reporter.static_entries, dyn_by_name__fJ, accesses=accesses, window__ns=window__ns)
    # The one static-energy derivation, shared by the total and the breakdown.
    static__fJ = sum(stat.values())
    dynamic__fJ = total_dynamic__fJ / accesses
    total__fJ = dynamic__fJ + static__fJ
    # Pair MEMBERS carry no per-member target (paired-slice caliber); the pair
    # rows derived by ``paired_slices`` carry the summed-share targets.
    slices = tuple(
        SliceEnergy(
            name=s,
            dynamic__fJ=sum(dyn.get(k, 0.0) for k in _DYN_NAMES.get(s, ())),
            static__fJ=sum(stat.get(k, 0.0) for k in _STATIC_NAMES.get(s, ())),
            target__fJ=0.0 if s in _PAIR_MEMBERS else shares[s] / 100.0 * target_total,
        )
        for s in _ALL_SLICES
    )
    mapped_static = sum(stat.get(k, 0.0) for s in _ALL_SLICES for k in _STATIC_NAMES.get(s, ()))
    unmapped_static = static__fJ - mapped_static

    return Measurement(
        total__fJ=total__fJ,
        dynamic__fJ=dynamic__fJ,
        static__fJ=static__fJ,
        slices=slices,
        unmapped_static__fJ=unmapped_static,
        access_latency__ns=access_latency__ns,
        window__ns=window__ns,
        n_w=n_w,
        n_x=batch,
        accesses=accesses,
        seed=seed,
        dyn_by_name=dyn,
    )


def _pool_rounds(rounds: list[Measurement], *, seed: int) -> Measurement:
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
            dynamic__fJ=wmean(lambda r, n=name: r.slice(n).dynamic__fJ),
            static__fJ=wmean(lambda r, n=name: r.slice(n).static__fJ),
            target__fJ=first.slice(name).target__fJ,
        )
        for name in names
    )
    round_totals = [r.total__fJ for r in rounds]
    mean_total = wmean(lambda r: r.total__fJ)
    rel_std = statistics.stdev(round_totals) / mean_total if len(round_totals) > 1 and mean_total else 0.0
    row_names = {name for r in rounds for name in r.dyn_by_name}
    dyn_by_name = {name: wmean(lambda r, n=name: r.dyn_by_name.get(n, 0.0)) for name in sorted(row_names)}

    return Measurement(
        total__fJ=mean_total,
        dynamic__fJ=wmean(lambda r: r.dynamic__fJ),
        static__fJ=wmean(lambda r: r.static__fJ),
        slices=slices,
        unmapped_static__fJ=wmean(lambda r: r.unmapped_static__fJ),
        access_latency__ns=first.access_latency__ns,
        window__ns=first.window__ns,
        n_w=first.n_w,
        n_x=first.n_x,
        accesses=total_accesses,
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
    rounds = [measure(macro, reporter, anchors, n=n_w * n_x, seed=seed + idx, batch=n_x) for idx in range(repeat)]
    return _pool_rounds(rounds, seed=seed)


def gate(m: Measurement, anchors: dict) -> tuple[bool, float]:
    """Hard gate: total energy per access within +-tol of the target. Return ``(pass, rel_error)``."""
    target = anchors["target"]["per_access__fJ"]
    tol = anchors["gate"]["hard_tolerance_relative"]
    rel = (m.total__fJ - target) / target
    return abs(rel) <= tol, rel


def energy_table(m: Measurement, anchors: dict) -> str:
    """Informational per-block breakdown (fJ/access) + the gated total row.

    Paired-slice caliber: the target-bearing rows are the two pair sums
    (``cablc+dswct`` vs 26.4 %, ``sinwp_sc+pn_isub`` vs 11.4 %) and the
    singles ``control`` / ``reference`` / ``tmcsa``; the four member rows are
    kept visible without a per-member target (the paper's internal split node
    voltages are unpublished).
    """
    target = anchors["target"]["per_access__fJ"]
    tol = anchors["gate"]["hard_tolerance_relative"]
    lines: list[str] = []
    lines.append("| Slice | Energy [fJ/acc] | dyn | static | Fig.18 x target [fJ] | pred/ref | basis |")
    lines.append("|---|--:|--:|--:|--:|--:|:--|")

    def row(s: SliceEnergy, *, basis: str) -> str:
        target_cell = f"{s.target__fJ:8.3f}" if s.target__fJ else "    -   "
        ratio_cell = f"{s.ratio:5.2f}x" if s.target__fJ else "  -   "
        return (
            f"| {s.name} | {s.total__fJ:8.3f} | {s.dynamic__fJ:7.3f} | {s.static__fJ:6.3f} | "
            f"{target_cell} | {ratio_cell} | {basis} |"
        )

    comparable = comparison_slices(m, anchors)
    lines.extend(row(s, basis="adopted") for s in comparable[:2])
    lines.extend(row(s, basis="modeled pair") for s in comparable[2:4])
    lines.append(row(comparable[4], basis="modeled"))
    lines.extend(row(m.slice(name), basis="pair member") for name in _PAIR_MEMBERS)
    if abs(m.unmapped_static__fJ) > 1e-9:
        lines.append(
            f"| (unmapped static) | {m.unmapped_static__fJ:8.3f} | {0.0:7.3f} | {m.unmapped_static__fJ:6.3f} | "
            f"{0.0:8.3f} |    -   | residual |"
        )
    within, rel = gate(m, anchors)
    lines.append(
        f"| **TOTAL (gated)** | **{m.total__fJ:8.3f}** | {m.dynamic__fJ:7.3f} | {m.static__fJ:6.3f} | "
        f"**{target:8.3f}** | **{m.total__fJ / target:5.3f}x** | {'PASS' if within else 'FAIL'} "
        f"(+-{tol * 100:.0f}%, err {rel * 100:+.1f}%) |"
    )
    return "\n".join(lines)


def plot_energy_breakdown(m: Measurement, anchors: dict, output_path: Path) -> None:
    """Plot NeuroX and Fig.18 energy breakdowns on the paired accounting basis."""
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    slices = comparison_slices(m, anchors)
    labels = ("Control", "Reference", "CABLC + DSWCT", "SINWP-SC + PN-ISUB", "TMCSA")
    colors = ("#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377")
    model__fJ = [s.total__fJ for s in slices]
    paper__fJ = [s.target__fJ for s in slices]

    modeled_sum__fJ = sum(s.total__fJ for s in slices)
    residual__fJ = m.total__fJ - modeled_sum__fJ
    residual_tol__fJ = max(1e-6, abs(m.total__fJ) * 1e-9)
    if abs(residual__fJ) > residual_tol__fJ:
        labels = (*labels, "Other")
        colors = (*colors, "#BBBBBB")
        model__fJ.append(residual__fJ)
        paper__fJ.append(0.0)

    fig, ax = plt.subplots(figsize=(14.0, 4.2))
    rows = (model__fJ, paper__fJ)
    max_segment__fJ = max(model__fJ + paper__fJ)
    for row_index, values__fJ in enumerate(rows):
        left__fJ = 0.0
        total__fJ = sum(values__fJ)
        for color, value__fJ in zip(colors, values__fJ, strict=True):
            if value__fJ == 0.0:
                continue
            ax.barh(
                row_index,
                value__fJ,
                left=left__fJ,
                height=0.58,
                color=color,
                edgecolor="white",
                linewidth=0.8,
            )
            if value__fJ / total__fJ >= 0.075:
                ax.text(
                    left__fJ + value__fJ / 2.0,
                    row_index,
                    f"{value__fJ / 1000.0:.2f} pJ\n{value__fJ / total__fJ * 100.0:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if color not in {"#CCBB44", "#BBBBBB"} else "black",
                    fontsize=8.5,
                    fontweight="bold",
                )
            left__fJ += value__fJ
        ax.text(
            left__fJ + max_segment__fJ * 0.01,
            row_index,
            f"{left__fJ / 1000.0:.3f} pJ",
            va="center",
            fontsize=9,
        )

    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color) for color in colors]
    ax.legend(
        legend_handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=False,
    )
    ax.set_yticks((0, 1), labels=("NeuroX", "Paper Fig.18"))
    ax.xaxis.set_major_formatter(lambda value__fJ, _: f"{value__fJ / 1000.0:g}")
    ax.set_xlabel("Energy per access [pJ]")
    ax.set_title("Xue2020 energy breakdown — paired accounting basis")
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    _LOG.info("breakdown plot: %s", output_path)


def render_report(m: Measurement, anchors: dict, *, device: torch.device) -> str:
    """Render the energy-basis gate report written to the run log."""
    target = anchors["target"]["per_access__fJ"]
    tol = anchors["gate"]["hard_tolerance_relative"]
    within, rel = gate(m, anchors)
    data = anchors["data"]
    input_nonzero_probability = float(data["input_nonzero_probability"])
    weight_nonzero_probability = float(data["weight_nonzero_probability"])

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
        f"({m.accesses} accesses), seed {m.seed}; calibrated nonzero probabilities "
        f"P(x!=0) = {input_nonzero_probability:.4f}, P(w!=0) = {weight_nonzero_probability:.4f}."
        f"{round_note}"
    )
    lines.append("")
    lines.append("## Hard gate -- total energy per access")
    lines.append("")
    std_note = f" +- {m.rel_std * 100:.2f} % (round-total relative std over {m.repeat} rounds)" if m.repeat > 1 else ""
    lines.append(
        f"Target {target:.1f} fJ/access (= {target / 1000.0:.4f} pJ = simulated 5.13 mW / 8 / 20 MHz); "
        f"+-{tol * 100:.0f}%. Under the declared calibrated-activity workload: result "
        f"**{m.total__fJ:.3f} fJ/access = "
        f"{m.total__fJ / target:.3f}x**{std_note} (err {rel * 100:+.1f}%), within +-{tol * 100:.0f}%: "
        f"{'yes' if within else 'no'}."
    )
    lines.append("")
    lines.append("## Energy breakdown (informational -- NOT gated)")
    lines.append("")
    lines.append(energy_table(m, anchors))
    lines.append("")
    lines.append(
        "The read-path slices follow the explicit circuit model under declared assumptions; "
        "control + reference are the two ADOPTED Fig.18 seats. Paired-slice caliber: the model bills one "
        "series input branch across DSWCT and CABLC and one series sink branch across SINWP-SC and PN-ISUB; "
        "the internal node voltages needed to reproduce Fig.18's separate block shares are unpublished, so only "
        "the pair sums "
        "(cablc+dswct vs 26.4 %, sinwp_sc+pn_isub vs 11.4 %) are well-defined targets -- the member rows are "
        "informational. Differences from Fig.18 x target are reported, not gated."
    )
    lines.append("")
    lines.append("## Declared conventions")
    lines.append("")
    lines.append(f"- Hard gate: total energy per access within +-{tol * 100:.0f}% of {target} fJ.")
    lines.append(
        "- Adopted seats (declared to reproduce a Fig.18 share, not fitted to the total): "
        "control 29.2 % (70 % per-op dynamic, 30 % leakage at 50 ns), reference 23.7 % (100 % static)."
    )
    lines.append(
        "- Read path (cablc, dswct, sinwp_sc, pn_isub, tmcsa): explicit circuit model under declared "
        "assumptions; static seats declared small/zero. Fig.18 comparison at the paired-slice caliber "
        "(cablc+dswct, sinwp_sc+pn_isub)."
    )
    lines.append(
        f"- Data: P(w!=0) = {weight_nonzero_probability:.4f}; conditional nonzero weights have equiprobable "
        f"sign and magnitude 1..{data['weight_range'][1]}. P(x!=0) = {input_nonzero_probability:.4f} among "
        f"at most nine candidate rows; conditional nonzero inputs are uniform over "
        f"1..{data['input_range'][1]}."
    )
    lines.append(
        f"- Energy basis: static/access = leakage_power * leakage window ({m.window__ns:.1f} ns = the 20 MHz "
        f"simulated operating period, a duty-cycle property); dynamic/access = per-VMM dynamic / mux_factor; "
        f"total/access "
        f"= their sum. The macro's access time ({m.access_latency__ns:.2f} ns) is the separate duration one read "
        f"chain settles over and integrates nothing."
    )
    lines.append("")
    return "\n".join(lines)


def resolve_device(name: str) -> torch.device:
    """Resolve a device name; ``auto`` is ``cuda`` when one is visible, else ``cpu``.

    Any other name is handed to :class:`torch.device` verbatim. This harness
    never chooses a GPU INDEX: which card the run lands on is the operator's
    choice, made outside the process through ``CUDA_VISIBLE_DEVICES``.
    """
    if name == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        _LOG.info("[auto device -> %s]", device)
        return device
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
        help="cpu, cuda[:idx], or auto (cuda if visible else cpu); default auto. Pick the card with "
        "CUDA_VISIBLE_DEVICES.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("log/validation/xue2020jssc"),
        help="Directory for validation.log and energy_breakdown.svg.",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "validation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=(logging.StreamHandler(), logging.FileHandler(log_path, mode="w", encoding="utf-8")),
        force=True,
    )
    _LOG.info("validation log: %s", log_path)

    with _ANCHORS_PATH.open("rb") as fh:
        anchors = tomllib.load(fh)

    device = resolve_device(args.device)
    macro = build_macro(_PARAMS_PATH, _POLICY_PATH, device=device, solve_chunk_size=args.solve_chunk)
    m = measure_rounds(
        macro,
        anchors,
        n_w=args.n_w,
        n_x=args.n_x,
        repeat=args.repeat,
        seed=args.seed,
    )

    _LOG.info("%s", render_report(m, anchors, device=device))
    plot_energy_breakdown(m, anchors, args.output_dir / "energy_breakdown.svg")


if __name__ == "__main__":
    main()
