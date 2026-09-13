"""Validate the Ye2023 WH-2T1R macro against Fig. 19."""

from __future__ import annotations

import logging
import math
import statistics
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from neurox import Reporter
from neurox.tools.validation.cim_macro import (
    ValidationArgs,
    area_per_macro__um2,
    build_macro,
    get_cli_args,
    mean_by_name,
    parse_cli_args,
    profile_vmm,
    render_run_summary,
    static_energy_by_name__fJ,
    validation_run,
)
from neurox.works.macro.cim.ye2023jssc import Ye2023JsscCimMacro

_LOG = logging.getLogger(__name__)

_VAL_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _VAL_DIR / "config.toml"
_POLICY_PATH = _VAL_DIR / "policy.toml"
_ANCHORS_PATH = _VAL_DIR / "anchors.toml"

_ADC_ACTIVE_BITS = 4
_QUANTIZATION_MODE = 0
_GROUPS = ("array", "rscsa", "mux_driver", "timing_ctrl")

_DYNAMIC_GROUPS = {
    "array": "array",
    ".bl_conduction": "array",
    ".tbl_conduction": "array",
    "rscsa": "rscsa",
    "bl_driver": "mux_driver",
    "sl_driver": "mux_driver",
    "mux_driver": "mux_driver",
    "timing_ctrl": "timing_ctrl",
}

_STATIC_GROUPS = {
    "": "other",
    "array": "array",
    "rscsa": "rscsa",
    "rscsa_reference": "rscsa",
    "bl_driver": "mux_driver",
    "sl_driver": "mux_driver",
    "mux_driver": "mux_driver",
    "timing_ctrl": "timing_ctrl",
}


@dataclass(frozen=True)
class PointMeasurement:
    """One input-sparsity operating point."""

    input_sparsity: float
    paper_total__uW: float
    dynamic__fJ: dict[str, float]
    static__fJ: dict[str, float]
    round_total__uW: tuple[float, ...]

    @property
    def model_by_group__uW(self) -> dict[str, float]:
        return {
            group: (self.dynamic__fJ.get(group, 0.0) + self.static__fJ.get(group, 0.0)) / measurement_cycle__ns()
            for group in _GROUPS
        }

    @property
    def model_total__uW(self) -> float:
        return sum(self.model_by_group__uW.values())

    @property
    def relative_error(self) -> float:
        return (self.model_total__uW - self.paper_total__uW) / self.paper_total__uW

    @property
    def relative_std(self) -> float:
        if len(self.round_total__uW) < 2 or self.model_total__uW == 0.0:
            return 0.0
        return statistics.stdev(self.round_total__uW) / self.model_total__uW


def _load_anchors() -> dict[str, Any]:
    with _ANCHORS_PATH.open("rb") as file:
        return tomllib.load(file)


_ANCHORS = _load_anchors()


def measurement_cycle__ns() -> float:
    """Complete period used to convert energy and leakage into measured power."""
    return float(_ANCHORS["measurement"]["cycle__ns"])


def _build_macro(*, device: torch.device, n_w: int) -> Ye2023JsscCimMacro:
    """Build the validation ensemble."""
    macro = build_macro(
        _CONFIG_PATH,
        _POLICY_PATH,
        inst_shape=(n_w,),
        device=device,
        dtype=torch.float32,
        T__K=300.0,
    )
    if not isinstance(macro, Ye2023JsscCimMacro):
        raise TypeError(f"validation config built {type(macro).__name__}, expected Ye2023JsscCimMacro")
    access__ns = macro.latency__ns(adc_active_bits=_ADC_ACTIVE_BITS) / macro.scan_num
    cycle__ns = measurement_cycle__ns()
    if not math.isclose(access__ns, cycle__ns, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"validation access duration {access__ns} ns does not match measurement cycle {cycle__ns} ns")
    return macro


def _draw_weight(
    gen: torch.Generator,
    *,
    n_w: int,
    input_num: int,
    output_num: int,
    sparsity: float,
) -> torch.Tensor:
    nonzero = torch.rand((n_w, input_num, output_num), generator=gen, device=gen.device) >= sparsity
    value = torch.randint(
        1,
        8,
        (n_w, input_num, output_num),
        generator=gen,
        device=gen.device,
        dtype=torch.long,
    )
    return nonzero * value


def _draw_input_uniform(gen: torch.Generator, *, n_x: int, n_w: int, input_num: int) -> torch.Tensor:
    """Draw the shared uniform source for paired Fig.19 sparsity points."""
    return torch.rand((n_x, n_w, input_num), generator=gen, device=gen.device)


def _static_by_group__fJ(static_by_name__fJ: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, energy__fJ in static_by_name__fJ.items():
        group = _STATIC_GROUPS[name]
        result[group] = result.get(group, 0.0) + energy__fJ
    if result.get("other", 0.0) != 0.0:
        raise ValueError("validation breakdown leaves nonzero static energy unassigned")
    return result


def measure(
    dynamic_rounds: dict[float, list[dict[str, float]]],
    static_by_name__fJ: dict[str, float],
) -> tuple[PointMeasurement, ...]:
    """Measure and pool the two Fig. 19 workload points."""
    input_sparsities = tuple(float(value) for value in _ANCHORS["data"]["input_sparsity"])
    paper_totals = tuple(float(value) for value in _ANCHORS["paper"]["total_power__uW"])
    static = _static_by_group__fJ(static_by_name__fJ)
    static__fJ = sum(static_by_name__fJ.values())

    points: list[PointMeasurement] = []
    for input_sparsity, paper_total in zip(input_sparsities, paper_totals, strict=True):
        dynamic_by_name__fJ = mean_by_name(dynamic_rounds[input_sparsity])
        dynamic: dict[str, float] = {}
        for name, energy__fJ in dynamic_by_name__fJ.items():
            group = _DYNAMIC_GROUPS[name]
            dynamic[group] = dynamic.get(group, 0.0) + energy__fJ
        points.append(
            PointMeasurement(
                input_sparsity=input_sparsity,
                paper_total__uW=paper_total,
                dynamic__fJ=dynamic,
                static__fJ=static,
                round_total__uW=tuple(
                    (sum(round_data.values()) + static__fJ) / measurement_cycle__ns()
                    for round_data in dynamic_rounds[input_sparsity]
                ),
            )
        )
    return tuple(points)


def _paper_by_group__uW(point_index: int) -> dict[str, float]:
    key = "p875" if point_index == 0 else "p50"
    shares = _ANCHORS["paper"]["shares"][key]
    total = float(_ANCHORS["paper"]["total_power__uW"][point_index])
    return {group: total * float(shares[group]) / 100.0 for group in _GROUPS}


def _primary_point(points: tuple[PointMeasurement, ...]) -> PointMeasurement:
    target = float(_ANCHORS["gate"]["primary_input_sparsity"])
    return next(point for point in points if point.input_sparsity == target)


def gate(points: tuple[PointMeasurement, ...]) -> bool:
    point = _primary_point(points)
    tolerance = float(_ANCHORS["gate"]["total_power_tolerance_relative"])
    return abs(point.relative_error) <= tolerance


def render_report(
    points: tuple[PointMeasurement, ...],
    *,
    run_summary: str,
    macro_area__um2: float,
) -> str:
    """Render the validation log."""
    lines = [
        "# ye2023jssc validation",
        "",
        (f"{run_summary} Weight sparsity {float(_ANCHORS['data']['weight_sparsity']):.1%}."),
        "",
        "| Input sparsity | Model [uW] | Paper [uW] | Error | Round rel. std |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {point.input_sparsity:.1%} | {point.model_total__uW:.4f} | {point.paper_total__uW:.4f} | "
        f"{point.relative_error:+.2%} | {point.relative_std:.2%} |"
        for point in points
    )

    primary = _primary_point(points)
    tolerance = float(_ANCHORS["gate"]["total_power_tolerance_relative"])
    lines.extend(
        (
            "",
            (
                f"Hard gate: 50% input sparsity total power within +/-{tolerance:.0%}: "
                f"**{'PASS' if gate(points) else 'FAIL'}**."
            ),
            "",
            "## Breakdown (informational)",
            "",
            "| Input sparsity | Block | Model [uW] | Paper [uW] | Model/Paper |",
            "| ---: | :--- | ---: | ---: | ---: |",
        )
    )
    for index, point in enumerate(points):
        paper = _paper_by_group__uW(index)
        model = point.model_by_group__uW
        lines.extend(
            (
                f"| {point.input_sparsity:.1%} | {group} | {model[group]:.4f} | {paper[group]:.4f} | "
                f"{model[group] / paper[group]:.3f}x |"
            )
            for group in _GROUPS
        )
    lines.extend(
        (
            "",
            (
                f"Area: {macro_area__um2:.1f} um2 per macro "
                f"(paper {float(_ANCHORS['paper']['macro_area__um2']):.1f} um2)."
            ),
            f"Primary-point error: {primary.relative_error:+.2%}.",
        )
    )
    return "\n".join(lines)


def plot_breakdown(points: tuple[PointMeasurement, ...], output: Path) -> None:
    """Plot model and paper breakdowns for both sparsity points."""
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {"array": "#4477AA", "rscsa": "#EE6677", "mux_driver": "#228833", "timing_ctrl": "#CCBB44"}
    rows: list[tuple[str, dict[str, float]]] = []
    for index, point in enumerate(points):
        rows.append((f"NeuroX {point.input_sparsity:.1%}", point.model_by_group__uW))
        rows.append((f"Paper {point.input_sparsity:.1%}", _paper_by_group__uW(index)))

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    for row_index, (_, values) in enumerate(rows):
        left = 0.0
        total = sum(values.values())
        for group in _GROUPS:
            value = values[group]
            ax.barh(row_index, value, left=left, height=0.62, color=colors[group], edgecolor="white")
            if value / total >= 0.08:
                ax.text(
                    left + value / 2.0,
                    row_index,
                    f"{value:.2f}\n{value / total:.1%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
            left += value
        ax.text(left + 0.8, row_index, f"{left:.2f} uW", va="center", fontsize=9)

    handles = [Patch(color=colors[group]) for group in _GROUPS]
    ax.legend(
        handles,
        ("Array", "RS-CSA", "Mux & Driver", "Timing & Ctrl"),
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
    )
    ax.set_yticks(range(len(rows)), labels=[label for label, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Average power over the 85 ns measurement cycle [uW]")
    ax.set_title("Ye2023 Fig. 19 power breakdown")
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def validate(args: ValidationArgs) -> None:
    """Run the configured validation campaign."""
    # --- 1: construct the macro ensemble and static accounting ---

    macro = _build_macro(device=args.device, n_w=args.n_w)
    reporter = Reporter(macro)
    data = _ANCHORS["data"]
    input_sparsities = tuple(float(value) for value in data["input_sparsity"])
    static_by_name__fJ = static_energy_by_name__fJ(
        macro,
        reporter,
        measurement_cycle__ns=measurement_cycle__ns(),
    )
    dynamic_rounds: dict[float, list[dict[str, float]]] = {input_sparsity: [] for input_sparsity in input_sparsities}

    # --- 2: sample, program, and profile every round and operating point ---

    with torch.no_grad():
        for round_index in range(args.repeat):
            gen = torch.Generator(device=args.device).manual_seed(args.seed + round_index)
            weight = _draw_weight(
                gen,
                n_w=args.n_w,
                input_num=macro.input_num,
                output_num=macro.output_num,
                sparsity=float(data["weight_sparsity"]),
            )
            macro.program(weight)
            input_uniform = _draw_input_uniform(
                gen,
                n_x=args.n_x,
                n_w=args.n_w,
                input_num=macro.input_num,
            )
            for input_sparsity in input_sparsities:
                x = (input_uniform >= input_sparsity).long()
                profiled = profile_vmm(
                    macro,
                    reporter,
                    x,
                    quantization_mode=_QUANTIZATION_MODE,
                    adc_active_bits=_ADC_ACTIVE_BITS,
                )
                dynamic_rounds[input_sparsity].append(profiled.dynamic_by_name_per_access__fJ)

    # --- 3: interpret the profiler rows against the paper anchors ---

    points = measure(dynamic_rounds, static_by_name__fJ)
    run_summary = render_run_summary(
        device=args.device,
        n_w=args.n_w,
        n_x=args.n_x,
        repeat=args.repeat,
        scan_num=macro.scan_num,
        seed=args.seed,
    )

    # --- 4: emit the campaign-defined report, gate, and figure ---

    report = render_report(
        points,
        run_summary=run_summary,
        macro_area__um2=area_per_macro__um2(macro, reporter),
    )
    _LOG.info("%s", report)
    plot_path = args.output_dir / "power_breakdown.svg"
    plot_breakdown(points, plot_path)
    _LOG.info("breakdown plot: %s", plot_path)


def main(argv: list[str] | None = None) -> None:
    cli_args = get_cli_args(description=__doc__, campaign="ye2023jssc", argv=argv)
    args = parse_cli_args(cli_args)
    with validation_run(args, campaign="ye2023jssc") as run_args:
        validate(run_args)


if __name__ == "__main__":
    main()
