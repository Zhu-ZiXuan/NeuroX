"""CLI: derive the ADC mode set from a per-layer range mapping file.

CLI: ``python -m neurox.tools.calibrate_adc.mode_derive --config <run.toml>
[--output <modes.toml>] [--plot-dir <dir>] [--log-dir <dir>]
[--log-level INFO]``

Reads the layer-range mapping TOML named by the run config (per-layer
design range + signedness; see :mod:`._modes` — producing that file, e.g.
from a training checkpoint's learned range params, is a consumer-side
step), partitions the layers by their ``signed`` flag, clusters the range
values within each sign group (deterministic 1-D relative-gap
agglomeration), and enumerates the resulting clusters as the ADC
operating-mode set. Group enumeration order is fixed: unsigned group
first, then signed; within a group, clusters ascend by representative
range. The mode-set TOML (mode tables + ``layer -> adc_mode`` mapping) is
always logged; ``--output`` writes it via :mod:`._modes`, and a cluster
plot goes to ``--plot-dir``.

CPU-only by design: the derivation touches a handful of scalars, so the
tool opts out of ``--device``.

See also:
    docs/guides/calibration/calibrate_adc.md
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from neurox.common import ConfigBase
from neurox.tools._config import add_standard_args, load_tool_config, resolve_relative_path, setup_logging

from ._math import ValueCluster, cluster_values
from ._modes import AdcMode, LayerRange, ModeSet, dump_mode_set, load_layer_ranges
from ._testbench import add_file_logging

logger = logging.getLogger(__name__)


# --- config schema ----------------------------------------------------------


@dataclass(frozen=True)
class _ClusterCfg:
    """``[cluster]`` section: knobs of the per-sign-group 1-D clustering."""

    rel_tol: float
    max_modes_per_group: int


class ModeDeriveToolConfig(ConfigBase):
    """Top-level config for :mod:`neurox.tools.calibrate_adc.mode_derive`.

    Attributes:
        mapping_file: Layer-range mapping TOML (see
            :func:`~neurox.tools.calibrate_adc._modes.load_layer_ranges`),
            relative to the tool TOML.
        cluster: Clustering knobs, applied per sign group.
    """

    mapping_file: Path
    cluster: _ClusterCfg


# --- derivation -------------------------------------------------------------


@dataclass(frozen=True)
class DerivedMode:
    """One derived ADC operating mode (a cluster within a sign group)."""

    adc_mode: int
    signed: bool
    cluster: ValueCluster


def derive_modes(
    layer_ranges: dict[str, LayerRange],
    *,
    rel_tol: float,
    max_modes_per_group: int,
) -> tuple[list[DerivedMode], dict[str, int]]:
    """Sign-group split, per-group clustering, and global mode enumeration.

    Deterministic: layers partition by their ``signed`` flag; the groups
    enumerate in the fixed order unsigned first then signed; within a
    group, members sort by layer name before clustering and the clusters
    ascend by representative (largest member). ``adc_mode`` indices run
    over that enumeration. An empty sign group contributes no mode.

    Returns:
        ``(modes, layer_to_mode)`` — every input layer mapped.
    """
    modes: list[DerivedMode] = []
    layer_to_mode: dict[str, int] = {}
    for signed in (False, True):
        members = sorted(n for n, spec in layer_ranges.items() if spec.signed == signed)
        if not members:
            logger.info("no %s layers; the group contributes no mode", "signed" if signed else "unsigned")
            continue
        values = [layer_ranges[n].range for n in members]
        clusters = cluster_values(values, rel_tol=rel_tol, max_cluster_num=max_modes_per_group)
        for cluster in clusters:
            adc_mode = len(modes)
            modes.append(DerivedMode(adc_mode=adc_mode, signed=signed, cluster=cluster))
            for idx in cluster.member_idx:
                layer_to_mode[members[idx]] = adc_mode
    return modes, layer_to_mode


def _build_mode_set(modes: list[DerivedMode], layer_to_mode: dict[str, int]) -> ModeSet:
    """Assemble the validated mode set from the derivation result."""
    return ModeSet(
        modes=tuple(
            AdcMode(
                adc_mode=m.adc_mode,
                signed=m.signed,
                range=m.cluster.representative,
                layer_num=len(m.cluster.member_idx),
            )
            for m in modes
        ),
        layers={name: layer_to_mode[name] for name in sorted(layer_to_mode)},
    )


# --- output -----------------------------------------------------------------


def _plot_clusters(
    modes: list[DerivedMode],
    layer_to_mode: dict[str, int],
    layer_ranges: dict[str, LayerRange],
    output_path: Path,
) -> None:
    """One PNG: per-layer range values colored by assigned mode."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(layer_ranges)
    cmap = plt.get_cmap("viridis")
    mode_color = {m.adc_mode: cmap(m.adc_mode / max(len(modes) - 1, 1)) for m in modes}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m in modes:
        xs = [i for i, n in enumerate(names) if layer_to_mode[n] == m.adc_mode]
        ys = [layer_ranges[names[i]].range for i in xs]
        sign_tag = "signed" if m.signed else "unsigned"
        marker = "o" if m.signed else "s"
        ax.scatter(
            xs,
            ys,
            s=26,
            color=mode_color[m.adc_mode],
            marker=marker,
            label=f"mode {m.adc_mode} ({sign_tag}, range {m.cluster.representative:g})",
        )
        ax.axhline(m.cluster.representative, color=mode_color[m.adc_mode], linestyle=":", linewidth=0.9)
    ax.set_xlabel("layer index (sorted by name)")
    ax.set_ylabel("layer range")
    ax.set_title("Per-layer ADC range by derived mode")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    logger.info("wrote cluster plot to %s", output_path)


# --- CLI --------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Derive the ADC mode set from a layer-range mapping (config-driven)")
    add_standard_args(parser, device=False, output_file=True)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("log/calibration/figures"),
        help="Directory for the cluster PNG (default log/calibration/figures)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("log/calibration"),
        help="Directory for the per-run log file (default log/calibration)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging(args.log_level)
    log_path = add_file_logging(args.log_dir, "mode_derive")
    logger.info("log file: %s", log_path)

    cfg = load_tool_config(ModeDeriveToolConfig, args.config)
    mapping_path = resolve_relative_path(cfg.mapping_file, args.config)
    assert mapping_path is not None
    layer_ranges = load_layer_ranges(mapping_path)
    logger.info("read %d layer ranges from %s", len(layer_ranges), mapping_path)

    modes, layer_to_mode = derive_modes(
        layer_ranges,
        rel_tol=cfg.cluster.rel_tol,
        max_modes_per_group=cfg.cluster.max_modes_per_group,
    )
    for m in modes:
        logger.info(
            "mode %d: signed=%-5s range %g (members %d, span %g .. %g)",
            m.adc_mode,
            m.signed,
            m.cluster.representative,
            len(m.cluster.member_idx),
            m.cluster.lo,
            m.cluster.hi,
        )

    mode_set = _build_mode_set(modes, layer_to_mode)
    text = dump_mode_set(mode_set)
    logger.info("%s", text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        logger.info("wrote mode-set TOML to %s", args.output)

    if args.plot_dir is not None:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        _plot_clusters(modes, layer_to_mode, layer_ranges, args.plot_dir / "mode_clusters.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
