"""Validate Xue2020 sub-array energy against the paper's targets."""

from __future__ import annotations

import logging
import statistics
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from neurox import Profiler
from neurox.tools.validation.cim_macro import (
    ValidationArgs,
    build_macro,
    dynamic_energy_by_round__fJ,
    get_cli_args,
    mean_by_name,
    parse_cli_args,
    profile_vmm,
    render_run_summary,
    static_energy_by_round__fJ,
    validation_run,
)
from neurox.works.macro.cim.xue2020jssc import Xue2020JsscCimMacro

_LOG = logging.getLogger(__name__)

_VAL_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _VAL_DIR / "config.toml"
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
    "cablc": ("cablc", "array"),
    "dswct": ("dswct",),
    "sinwp_sc": ("sinwp_sc",),
    "pn_isub": ("pn_isub",),
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
# [assumed] The paper does not report simulation temperature.
TEMPERATURE__K = 300.0

type Anchors = dict[str, Any]


def measurement_cycle__ns(anchors: Anchors) -> float:
    """Paper measurement period used for leakage integration."""
    return float(anchors["measurement"]["cycle__ns"])


def _draw_weight(
    gen: torch.Generator,
    *,
    n_w: int,
    input_num: int,
    output_num: int,
    max_magnitude: int,
    nonzero_probability: float,
) -> torch.Tensor:
    """Zero-inflated sign-magnitude weights, uniform conditional on nonzero."""
    shape = (n_w, input_num, output_num)
    nonzero = torch.rand(shape, generator=gen, device=gen.device) < nonzero_probability
    magnitude = torch.randint(
        1,
        max_magnitude + 1,
        shape,
        generator=gen,
        dtype=torch.long,
        device=gen.device,
    )
    negative = torch.randint(
        0,
        2,
        shape,
        generator=gen,
        dtype=torch.bool,
        device=gen.device,
    )
    signed = torch.where(negative, -magnitude, magnitude)
    return torch.where(nonzero, signed, 0)


def _draw_input(
    gen: torch.Generator,
    *,
    n_x: int,
    n_w: int,
    input_num: int,
    max_active_num: int,
    lo: int,
    hi: int,
    nonzero_probability: float,
) -> torch.Tensor:
    """At most `max_active_num` candidate inputs, zero-inflated independently."""
    # Shape: [sample, weight, input] -> [sample, weight, active]
    active_inputs = (
        torch.rand((n_x, n_w, input_num), generator=gen, device=gen.device).topk(max_active_num, dim=2).indices
    )
    active_shape = (n_x, n_w, max_active_num)
    nonzero = torch.rand(active_shape, generator=gen, device=gen.device) < nonzero_probability
    values = torch.randint(
        max(1, lo),
        hi + 1,
        active_shape,
        generator=gen,
        dtype=torch.long,
        device=gen.device,
    )
    values = torch.where(nonzero, values, 0)
    # Shape: [sample, weight, active] -> [sample, weight, input]
    return torch.zeros((n_x, n_w, input_num), dtype=torch.long, device=gen.device).scatter(
        2,
        active_inputs,
        values,
    )


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


def paired_slices(
    slices: tuple[SliceEnergy, ...],
    shares: dict[str, float],
    target_total: float,
) -> tuple[SliceEnergy, ...]:
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


def comparison_slices(m: Measurement, anchors: Anchors) -> tuple[SliceEnergy, ...]:
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
    unmapped_dynamic__fJ: float
    unmapped_static__fJ: float
    rel_std: float

    def slice(self, name: str) -> SliceEnergy:
        return next(s for s in self.slices if s.name == name)


def measure(
    dynamic_by_group__fJ: dict[str, float],
    static_by_group__fJ: dict[str, float],
    round_total__fJ: tuple[float, ...],
    anchors: Anchors,
) -> Measurement:
    """Compare campaign-grouped circuit costs with the Fig.18 accounting slices."""
    target_total = anchors["target"]["per_access__fJ"]
    shares = anchors["fig18_shares"]
    dynamic = dynamic_by_group__fJ
    static = static_by_group__fJ
    slices = tuple(
        SliceEnergy(
            name=s,
            dynamic__fJ=dynamic.get(s, 0.0),
            static__fJ=static.get(s, 0.0),
            target__fJ=0.0 if s in _PAIR_MEMBERS else shares[s] / 100.0 * target_total,
        )
        for s in _ALL_SLICES
    )
    dynamic__fJ = sum(dynamic_by_group__fJ.values())
    static__fJ = sum(static_by_group__fJ.values())
    total__fJ = dynamic__fJ + static__fJ
    rel_std = statistics.stdev(round_total__fJ) / total__fJ if len(round_total__fJ) > 1 and total__fJ else 0.0

    return Measurement(
        total__fJ=total__fJ,
        dynamic__fJ=dynamic__fJ,
        static__fJ=static__fJ,
        slices=slices,
        unmapped_dynamic__fJ=dynamic.get("other", 0.0),
        unmapped_static__fJ=static.get("other", 0.0),
        rel_std=rel_std,
    )


def gate(m: Measurement, anchors: Anchors) -> tuple[bool, float]:
    """Return `(passed, relative_error)` against the configured energy tolerance."""
    target = anchors["target"]["per_access__fJ"]
    tol = anchors["gate"]["hard_tolerance_relative"]
    rel = (m.total__fJ - target) / target
    return abs(rel) <= tol, rel


def energy_table(m: Measurement, anchors: Anchors) -> str:
    """Render the energy breakdown and gated total."""
    target = anchors["target"]["per_access__fJ"]
    tol = anchors["gate"]["hard_tolerance_relative"]
    lines: list[str] = []
    lines.append("| Slice | Energy [fJ/acc] | dyn | static | Fig.18 x target [fJ] | pred/ref | basis |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | :--- |")

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
    unmapped__fJ = m.unmapped_dynamic__fJ + m.unmapped_static__fJ
    if abs(unmapped__fJ) > 1e-9:
        lines.append(
            f"| (unmapped) | {unmapped__fJ:8.3f} | {m.unmapped_dynamic__fJ:7.3f} | {m.unmapped_static__fJ:6.3f} | "
            f"{0.0:8.3f} |    -   | residual |"
        )
    within, rel = gate(m, anchors)
    lines.append(
        f"| **TOTAL (gated)** | **{m.total__fJ:8.3f}** | {m.dynamic__fJ:7.3f} | {m.static__fJ:6.3f} | "
        f"**{target:8.3f}** | **{m.total__fJ / target:5.3f}x** | {'PASS' if within else 'FAIL'} "
        f"(+-{tol * 100:.0f}%, err {rel * 100:+.1f}%) |"
    )
    return "\n".join(lines)


def plot_energy_breakdown(m: Measurement, anchors: Anchors, output_path: Path) -> None:
    """Plot NeuroX and Fig.18 energy breakdowns on the paired accounting basis."""
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    slices = comparison_slices(m, anchors)
    labels: tuple[str, ...] = ("Control", "Reference", "CABLC + DSWCT", "SINWP-SC + PN-ISUB", "TMCSA")
    colors: tuple[str, ...] = ("#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377")
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

    legend_handles = [Rectangle((0, 0), 1, 1, facecolor=color) for color in colors]
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


def render_report(
    m: Measurement,
    anchors: Anchors,
    *,
    run_summary: str,
    repeat: int,
) -> str:
    """Render the energy-basis gate report written to the run log."""
    target = anchors["target"]["per_access__fJ"]
    tol = anchors["gate"]["hard_tolerance_relative"]
    cycle__ns = measurement_cycle__ns(anchors)
    within, rel = gate(m, anchors)
    data = anchors["data"]
    input_nonzero_probability = float(data["input_nonzero_probability"])
    weight_nonzero_probability = float(data["weight_nonzero_probability"])

    lines: list[str] = []
    lines.append("# xue2020jssc validation -- total energy per access")
    lines.append("")
    round_note = (
        f" Pooled over {repeat} rounds (relative std of the round totals = {m.rel_std * 100:.2f} %)."
        if repeat > 1
        else ""
    )
    lines.append(
        f"Energy-basis profiler run for `{_CONFIG_PATH.name}` + `{_POLICY_PATH.name}`. "
        f"{run_summary} Calibrated nonzero probabilities "
        f"P(x!=0) = {input_nonzero_probability:.4f}, P(w!=0) = {weight_nonzero_probability:.4f}.{round_note}"
    )
    lines.append("")
    lines.append("## Hard gate -- total energy per access")
    lines.append("")
    std_note = f" +- {m.rel_std * 100:.2f} % (round-total relative std over {repeat} rounds)" if repeat > 1 else ""
    lines.append(
        f"Target {target:.1f} fJ/access (= {target / 1000.0:.4f} pJ = "
        f"simulated 5.13 mW * {cycle__ns:g} ns / 8); "
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


def validate(args: ValidationArgs) -> None:
    """Run the configured validation campaign."""
    # --- 1: load the paper anchors and construct the macro ensemble ---

    with _ANCHORS_PATH.open("rb") as fh:
        anchors = tomllib.load(fh)

    macro = build_macro(
        _CONFIG_PATH,
        _POLICY_PATH,
        inst_shape=(args.n_w,),
        device=args.device,
        dtype=torch.float32,
        T__K=TEMPERATURE__K,
    )
    if not isinstance(macro, Xue2020JsscCimMacro):
        raise TypeError(f"validation config built {type(macro).__name__}, expected Xue2020JsscCimMacro")
    # Each retained input/weight position is one physical macro's full VMM.
    macro.set_profile_leading_rank(2)
    profiler = Profiler(concat_dim=0)
    profiler.collect_static_data(macro)
    data = anchors["data"]
    w_lo, w_hi = data["weight_range"]
    x_lo, x_hi = data["input_range"]

    # --- 2: sample, program, and profile every independent round ---

    with torch.no_grad():
        for round_index in range(args.repeat):
            _LOG.info("profiling round %d/%d", round_index + 1, args.repeat)
            gen = torch.Generator(device=args.device).manual_seed(args.seed + round_index)
            weight = _draw_weight(
                gen,
                n_w=args.n_w,
                input_num=macro.input_num,
                output_num=macro.output_num,
                max_magnitude=max(abs(w_lo), abs(w_hi)),
                nonzero_probability=float(data["weight_nonzero_probability"]),
            )
            macro.program(weight)
            x = _draw_input(
                gen,
                n_x=args.n_x,
                n_w=args.n_w,
                input_num=macro.input_num,
                max_active_num=macro.config.max_active_num,
                lo=x_lo,
                hi=x_hi,
                nonzero_probability=float(data["input_nonzero_probability"]),
            )
            profile_vmm(
                macro,
                profiler,
                x,
                quantization_mode=_QUANTIZATION_MODE,
                adc_active_bits=_ADC_BITS,
            )

    # --- 3: interpret the profiler rows against the paper anchors ---

    profile_items = profiler.result
    torch.save(profile_items, args.output_dir / "profile.pt")
    report_items = {
        name: replace(
            item,
            area__um2=None if item.area__um2 is None else item.area__um2 / macro.inst_count,
            leakage__uW=None if item.leakage__uW is None else item.leakage__uW / macro.inst_count,
        )
        for name, item in profile_items.items()
    }
    dynamic_rounds = dynamic_energy_by_round__fJ(
        report_items,
        scan_num=macro.scan_num,
        n_x=args.n_x,
        groups={name: label for label, names in _DYN_NAMES.items() for name in names},
    )
    static_rounds = static_energy_by_round__fJ(
        report_items,
        scan_num=macro.scan_num,
        n_x=args.n_x,
        powered_duration__ns=torch.tensor(macro.scan_num * measurement_cycle__ns(anchors), dtype=torch.float64).expand(
            args.repeat * args.n_x, args.n_w
        ),
        groups={name: label for label, names in _STATIC_NAMES.items() for name in names},
    )
    round_total__fJ = tuple(
        sum(dynamic.values()) + sum(static.values())
        for dynamic, static in zip(dynamic_rounds, static_rounds, strict=True)
    )
    dynamic_by_group__fJ = mean_by_name(dynamic_rounds)
    static_by_group__fJ = mean_by_name(static_rounds)
    m = measure(
        dynamic_by_group__fJ,
        static_by_group__fJ,
        round_total__fJ,
        anchors,
    )
    run_summary = render_run_summary(
        device=args.device,
        n_w=args.n_w,
        n_x=args.n_x,
        repeat=args.repeat,
        scan_num=macro.scan_num,
        seed=args.seed,
    )

    # --- 4: emit the campaign-defined report, gate, and figure ---

    _LOG.info("%s", render_report(m, anchors, run_summary=run_summary, repeat=args.repeat))
    plot_energy_breakdown(m, anchors, args.output_dir / "energy_breakdown.svg")


def main(argv: list[str] | None = None) -> None:
    cli_args = get_cli_args(description=__doc__, campaign="xue2020jssc", argv=argv)
    args = parse_cli_args(cli_args)
    with validation_run(args, campaign="xue2020jssc") as run_args:
        validate(run_args)


if __name__ == "__main__":
    main()
