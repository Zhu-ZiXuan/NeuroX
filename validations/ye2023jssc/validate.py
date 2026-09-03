"""Validate the Ye2023 WH-2T1R macro against Fig. 19."""

from __future__ import annotations

import argparse
import logging
import statistics
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from neurox import Profiler, Reporter, stamp_names
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy
from neurox.works.macro.cim.ye2023jssc import (
    Ye2023JsscCimMacro,
    Ye2023JsscCimMacroConfig,
    Ye2023JsscCimMacroPolicy,
)

_LOG = logging.getLogger(__name__)

_VAL_DIR = Path(__file__).resolve().parent
_PARAMS_PATH = _VAL_DIR / "params.toml"
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


def build_macro(*, device: torch.device, n_w: int) -> Ye2023JsscCimMacro:
    """Build the validation ensemble."""
    config = cast(Ye2023JsscCimMacroConfig, CimMacroConfig.from_file(_PARAMS_PATH, section="cim_macro"))
    policy = cast(Ye2023JsscCimMacroPolicy, CimMacroPolicy.from_file(_POLICY_PATH, section="policy"))
    macro = cast(
        Ye2023JsscCimMacro,
        CimMacro.from_config(
            config=config,
            policy=policy,
            inst_shape=(n_w,),
            dtype=torch.float32,
            T__K=300.0,
        ),
    )
    macro.to(device)
    macro.eval()
    macro.fabricate()
    stamp_names(macro)
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
    return torch.rand((n_x, n_w, input_num), generator=gen, device=gen.device)


def _static_by_group__fJ(reporter: Reporter, *, n_w: int) -> dict[str, float]:
    result: dict[str, float] = {}
    for entry in reporter.static_entries:
        group = _STATIC_GROUPS[entry.qualified_name]
        result[group] = result.get(group, 0.0) + entry.leakage__uW / n_w * measurement_cycle__ns()
    if result.get("other", 0.0) != 0.0:
        raise ValueError("validation breakdown leaves nonzero static energy unassigned")
    return result


def _measure_round(
    macro: Ye2023JsscCimMacro,
    reporter: Reporter,
    *,
    n_w: int,
    n_x: int,
    seed: int,
    weight_sparsity: float,
    input_sparsities: tuple[float, ...],
) -> dict[float, dict[str, float]]:
    device = next(macro.buffers()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    weight = _draw_weight(
        gen,
        n_w=n_w,
        input_num=macro.input_num,
        output_num=macro.output_num,
        sparsity=weight_sparsity,
    )
    input_uniform = _draw_input_uniform(gen, n_x=n_x, n_w=n_w, input_num=macro.input_num)
    accesses = n_w * n_x * macro.output_num

    result: dict[float, dict[str, float]] = {}
    with torch.no_grad():
        macro.program(weight)
        for sparsity in input_sparsities:
            x = (input_uniform >= sparsity).to(torch.long)
            with Profiler(leading_rank=2, sync_device=torch.device("cpu")) as profiler:
                macro.vec_mat_mul(x, quantization_mode=_QUANTIZATION_MODE, adc_active_bits=_ADC_ACTIVE_BITS)
            grouped: dict[str, float] = {}
            for name, energy__fJ in reporter.by_name(profiler).items():
                group = _DYNAMIC_GROUPS[name]
                grouped[group] = grouped.get(group, 0.0) + energy__fJ / accesses
            result[sparsity] = grouped
    return result


def measure(
    macro: Ye2023JsscCimMacro,
    *,
    n_w: int,
    n_x: int,
    repeat: int,
    seed: int,
) -> tuple[PointMeasurement, ...]:
    """Measure and pool the two Fig. 19 workload points."""
    reporter = Reporter(macro)
    data = _ANCHORS["data"]
    input_sparsities = tuple(float(value) for value in data["input_sparsity"])
    paper_totals = tuple(float(value) for value in _ANCHORS["paper"]["total_power__uW"])
    static = _static_by_group__fJ(reporter, n_w=n_w)
    rounds = [
        _measure_round(
            macro,
            reporter,
            n_w=n_w,
            n_x=n_x,
            seed=seed + round_index,
            weight_sparsity=float(data["weight_sparsity"]),
            input_sparsities=input_sparsities,
        )
        for round_index in range(repeat)
    ]

    points: list[PointMeasurement] = []
    for sparsity, paper_total in zip(input_sparsities, paper_totals, strict=True):
        dynamic = {
            group: statistics.fmean(round_data[sparsity].get(group, 0.0) for round_data in rounds) for group in _GROUPS
        }
        round_totals = tuple(
            sum(round_data[sparsity].get(group, 0.0) + static.get(group, 0.0) for group in _GROUPS)
            / measurement_cycle__ns()
            for round_data in rounds
        )
        points.append(
            PointMeasurement(
                input_sparsity=sparsity,
                paper_total__uW=paper_total,
                dynamic__fJ=dynamic,
                static__fJ=static,
                round_total__uW=round_totals,
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
    macro: Ye2023JsscCimMacro,
    points: tuple[PointMeasurement, ...],
    *,
    device: torch.device,
    n_w: int,
    n_x: int,
    repeat: int,
) -> str:
    """Render the validation log."""
    lines = [
        "# ye2023jssc validation",
        "",
        (
            f"Device {device}; {n_w} weight draws x {n_x} inputs x {repeat} rounds; "
            f"weight sparsity {float(_ANCHORS['data']['weight_sparsity']):.1%}."
        ),
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
                f"Area: {Reporter(macro).static.area__um2 / n_w:.1f} um2 per macro "
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


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-w", type=int, default=64)
    parser.add_argument("--n-x", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("log/validation/ye2023jssc"))
    args = parser.parse_args()

    device = resolve_device(args.device)
    macro = build_macro(device=device, n_w=args.n_w)
    points = measure(macro, n_w=args.n_w, n_x=args.n_x, repeat=args.repeat, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "validation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=(logging.StreamHandler(), logging.FileHandler(log_path, mode="w", encoding="utf-8")),
        force=True,
    )
    report = render_report(
        macro,
        points,
        device=device,
        n_w=args.n_w,
        n_x=args.n_x,
        repeat=args.repeat,
    )
    _LOG.info("%s", report)
    plot_path = args.output_dir / "power_breakdown.svg"
    plot_breakdown(points, plot_path)
    _LOG.info("validation log: %s", log_path)
    _LOG.info("breakdown plot: %s", plot_path)


if __name__ == "__main__":
    main()
