"""Validate Xue2020 sub-array energy against the paper's targets."""

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

_READ_PATH_SLICES = ("cablc", "dswct", "sinwp_sc", "pn_isub", "tmcsa")
_ADOPTED_SLICES = ("control", "reference")
_ALL_SLICES = (*_ADOPTED_SLICES, *_READ_PATH_SLICES)

# Internal node voltages are unpublished, so series branches are compared in pairs.
_PAIRED_SLICES: dict[str, tuple[str, ...]] = {
    "cablc+dswct": ("cablc", "dswct"),
    "sinwp_sc+pn_isub": ("sinwp_sc", "pn_isub"),
}
_PAIR_MEMBERS = tuple(name for members in _PAIRED_SLICES.values() for name in members)

_DYN_NAMES: dict[str, tuple[str, ...]] = {
    "cablc": (".cablc", "array"),
    "dswct": (".dswct",),
    "sinwp_sc": (".sinwp_sc",),
    "pn_isub": (".pn_isub",),
    "control": ("control",),
    "tmcsa": ("tmcsa",),
}
_STATIC_NAMES: dict[str, tuple[str, ...]] = {
    "control": ("control",),
    "reference": ("tmcsa_iref",),
    "cablc": ("cablc", "array"),
    "dswct": (),
    "sinwp_sc": (),
    "tmcsa": ("tmcsa", "adc"),
    "pn_isub": (),
}

_QUANTIZATION_MODE = 0
_ADC_BITS = 3
# [reported p211 Fig.20(c)] 256 rows x 512 physical columns per sub-array.
ROW_NUM = 256
# [derived] 512 physical columns / two digits / two polarities.
COL_NUM = 128
# [assumed] The paper does not report simulation temperature.
TEMPERATURE__K = 300.0


def build_macro(
    params_path: Path,
    policy_path: Path,
    *,
    device: torch.device,
    solve_chunk_size: int,
) -> Xue2020JsscCimMacro:
    """Build and fabricate the configured macro."""
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
    """Leakage integration window of one access [ns]."""
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
    """One Fig.18 energy slice [fJ/access]."""

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
    """Aggregate series-branch members into comparable pairs."""
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
    """Energy validation result."""

    total__fJ: float
    dynamic__fJ: float
    static__fJ: float
    slices: tuple[SliceEnergy, ...]
    unmapped_static__fJ: float
    access_latency__ns: float
    window__ns: float
    n_w: int
    n_x: int
    accesses: int
    seed: int
    repeat: int = 1
    rel_std: float = 0.0
    dyn_by_name: dict[str, float] = field(default_factory=dict)

    @property
    def draws(self) -> int:
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
    """Reduce profiler rows to energy per access."""
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
    """Profile one round and reduce it to energy per access."""
    cfg = macro.config
    device = next(macro.buffers()).device
    gen = torch.Generator(device=device).manual_seed(seed)

    data = anchors["data"]
    w_lo, w_hi = data["weight_range"]
    x_lo, x_hi = data["input_range"]
    target_total = anchors["target"]["per_access__fJ"]
    shares = anchors["fig18_shares"]

    scan_num = cfg.scan_num
    window__ns = leakage_window__ns(macro)
    access_latency__ns = macro.latency__ns(adc_active_bits=_ADC_BITS) / scan_num

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
            with Profiler(leading_rank=1) as prof:
                macro.program(w)
                macro.vec_mat_mul(x, quantization_mode=_QUANTIZATION_MODE, adc_active_bits=_ADC_BITS)
            for name, e__fJ in reporter.by_name(prof).items():
                dyn_by_name__fJ[name] = dyn_by_name__fJ.get(name, 0.0) + e__fJ
            total_dynamic__fJ += reporter.total_dynamic_energy__fJ(prof)
            n_samples += batch
    accesses = n_samples * scan_num

    dyn, stat = _per_access(reporter.static_entries, dyn_by_name__fJ, accesses=accesses, window__ns=window__ns)
    static__fJ = sum(stat.values())
    dynamic__fJ = total_dynamic__fJ / accesses
    total__fJ = dynamic__fJ + static__fJ
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
    """Pool rounds by access count and compute their relative standard deviation."""
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
    """Profile and pool independent workload rounds."""
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
    """Render the energy breakdown and gated total."""
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
    return "\n".join(lines)


def resolve_device(name: str) -> torch.device:
    """Resolve `auto` to CUDA when available, otherwise CPU."""
    if name == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        _LOG.info("[auto device -> %s]", device)
        return device
    return torch.device(name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-w", type=int, default=64, help="Weight draws per round.")
    ap.add_argument("--n-x", type=int, default=256, help="Inputs per weight draw.")
    ap.add_argument("--repeat", type=int, default=8, help="Independent workload rounds.")
    ap.add_argument(
        "--solve-chunk",
        type=int,
        default=4096,
        help="Leading instances per array-solve chunk; 0 disables chunking.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cpu, cuda[:index], or auto.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("log/validation/xue2020jssc"),
        help="Output directory.",
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
