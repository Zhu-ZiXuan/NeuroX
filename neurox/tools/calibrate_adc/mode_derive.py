"""Derive the quantization mode set from a per-layer range mapping file.

CLI: `python -m neurox.tools.calibrate_adc.mode_derive --config <run.toml>
[--output <modes.toml>] [--plot-dir <dir>] [--log-dir <dir>]
[--log-level INFO]`

Reads the layer-range mapping TOML named by the run config, maps every layer
onto the canonical window covering its range, partitions the layers by window
shape (unsigned / mid-zero), clusters the window extents within each group by
deterministic 1-D relative-gap agglomeration, and enumerates the resulting
clusters as the quantization mode set. Group enumeration order is fixed:
unsigned group first, then mid-zero; within a group, clusters ascend by
representative extent, and the cluster's window is its largest member's — the
one covering every member. The mode-set TOML is always logged; `--output`
writes it, and a cluster plot goes to `--plot-dir`.

CPU-only by design: the derivation touches a handful of scalars, so the tool
opts out of `--device`.

See Also:
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
from ._modes import AdcMode, LayerRange, ModeSet, canonical_window, dump_mode_set, load_layer_ranges
from ._testbench import add_file_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ClusterCfg:
    rel_tol: float
    max_modes_per_group: int


class ModeDeriveToolConfig(ConfigBase):
    mapping_file: Path
    """Layer-range mapping TOML, relative to the tool TOML."""
    cluster: _ClusterCfg


@dataclass(frozen=True)
class DerivedMode:
    """One derived quantization mode — a cluster within a window-shape group."""

    quantization_mode: int
    """Enumeration index of the mode."""
    quantization_input_range: tuple[int, int]
    """The cluster's canonical window — the largest member's, which covers
    every member."""
    cluster: ValueCluster
    """The clustered window extents backing the mode."""

    @property
    def signed(self) -> bool:
        """Whether the mode's window is mid-zero rather than unsigned."""
        return self.quantization_input_range[0] < 0


def _window_extent(window: tuple[int, int]) -> int:
    """Return the extent scalar ordering the windows of one shape group.

    Unsigned `[0, upper]` has extent `upper`, mid-zero `[-m, m - 1]` extent
    `m`. A larger extent covers a smaller one of the same shape, so a cluster
    sized from its largest member covers the whole cluster.
    """
    lower, upper = window
    return upper if lower == 0 else -lower


def _window_from_extent(extent: int, *, signed: bool) -> tuple[int, int]:
    """Rebuild one shape group's window from its extent scalar."""
    return (-extent, extent - 1) if signed else (0, extent)


def derive_modes(
    layer_ranges: dict[str, LayerRange],
    *,
    rel_tol: float,
    max_modes_per_group: int,
) -> tuple[list[DerivedMode], dict[str, int]]:
    """Shape-group split, per-group clustering, and global mode enumeration.

    Deterministic: every layer maps onto the canonical window covering its
    design range and the layers partition by that window's shape; the groups
    enumerate in the fixed order unsigned first then mid-zero; within a group,
    members sort by layer name before clustering and the clusters ascend by
    representative extent. `quantization_mode` indices run over that
    enumeration. An empty group contributes no mode.

    Returns:
        `(modes, layer_to_mode)`, with every input layer mapped.
    """
    windows = {name: canonical_window(spec) for name, spec in layer_ranges.items()}
    modes: list[DerivedMode] = []
    layer_to_mode: dict[str, int] = {}
    for signed in (False, True):
        members = sorted(n for n, window in windows.items() if (window[0] < 0) == signed)
        if not members:
            logger.info("no %s layers; the group contributes no mode", "mid-zero" if signed else "unsigned")
            continue
        extents = [float(_window_extent(windows[n])) for n in members]
        clusters = cluster_values(extents, rel_tol=rel_tol, max_cluster_num=max_modes_per_group)
        for cluster in clusters:
            quantization_mode = len(modes)
            modes.append(
                DerivedMode(
                    quantization_mode=quantization_mode,
                    quantization_input_range=_window_from_extent(int(cluster.representative), signed=signed),
                    cluster=cluster,
                )
            )
            for idx in cluster.member_idx:
                layer_to_mode[members[idx]] = quantization_mode
    return modes, layer_to_mode


def _build_mode_set(modes: list[DerivedMode], layer_to_mode: dict[str, int]) -> ModeSet:
    """Assemble the validated mode set from the derivation result."""
    return ModeSet(
        modes=tuple(
            AdcMode(
                quantization_mode=m.quantization_mode,
                quantization_input_range=m.quantization_input_range,
                layer_num=len(m.cluster.member_idx),
            )
            for m in modes
        ),
        layers={name: layer_to_mode[name] for name in sorted(layer_to_mode)},
    )


def _plot_clusters(
    modes: list[DerivedMode],
    layer_to_mode: dict[str, int],
    layer_ranges: dict[str, LayerRange],
    output_path: Path,
) -> None:
    """One PNG: per-layer design ranges + mode windows, colored by mode."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(layer_ranges)
    cmap = plt.get_cmap("viridis")
    mode_color = {m.quantization_mode: cmap(m.quantization_mode / max(len(modes) - 1, 1)) for m in modes}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m in modes:
        xs = [i for i, n in enumerate(names) if layer_to_mode[n] == m.quantization_mode]
        los = [layer_ranges[names[i]].range[0] for i in xs]
        his = [layer_ranges[names[i]].range[1] for i in xs]
        lower, upper = m.quantization_input_range
        shape_tag = "mid-zero" if m.signed else "unsigned"
        ax.vlines(xs, los, his, color=mode_color[m.quantization_mode], linewidth=2.0)
        ax.scatter(
            xs,
            his,
            s=26,
            color=mode_color[m.quantization_mode],
            marker="o" if m.signed else "s",
            label=f"mode {m.quantization_mode} ({shape_tag}, window [{lower}, {upper}])",
        )
        for bound in (lower, upper):
            ax.axhline(bound, color=mode_color[m.quantization_mode], linestyle=":", linewidth=0.9)
    ax.set_xlabel("layer index (sorted by name)")
    ax.set_ylabel("layer design range")
    ax.set_title("Per-layer design range by derived quantization mode")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    logger.info("wrote cluster plot to %s", output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive the quantization mode set from a layer-range mapping (config-driven)"
    )
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
    layer_ranges = load_layer_ranges(mapping_path)
    logger.info("read %d layer ranges from %s", len(layer_ranges), mapping_path)

    modes, layer_to_mode = derive_modes(
        layer_ranges,
        rel_tol=cfg.cluster.rel_tol,
        max_modes_per_group=cfg.cluster.max_modes_per_group,
    )
    for m in modes:
        logger.info(
            "mode %d: window [%d, %d] (members %d, extent span %g .. %g)",
            m.quantization_mode,
            *m.quantization_input_range,
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
