"""ADC boundary calibration utilities for the differential ADC family.

See also:
    docs/dev/modules/tools/README.md
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def compute_max_col_diff_current__uA(
    row_num: int,
    i_cell_max__uA: float,
    i_cell_min__uA: float,
) -> float:
    """Maximum BL difference current the ADC must resolve [uA].

    Args:
        row_num: Number of rows accumulating onto a single column.
        i_cell_max__uA: Per-cell on-state current at the LRS RRAM.
        i_cell_min__uA: Per-cell on-state current at the HRS RRAM.

    Returns:
        ``row_num * (I_cell_max - I_cell_min)`` [uA].
    """
    if row_num <= 0:
        raise ValueError(f"row_num ({row_num}) must be positive")
    if i_cell_max__uA < i_cell_min__uA:
        raise ValueError(f"i_cell_max ({i_cell_max__uA}) < i_cell_min ({i_cell_min__uA})")
    return row_num * (i_cell_max__uA - i_cell_min__uA)


def floor_boundaries_for_mode(max_signal: float, n_codes: int) -> list[float]:
    """Floor-style ADC boundaries for one operating mode.

    Codes ``0 … n_codes - 1`` cover ``[0, max_signal]`` with uniform
    LSB ``max_signal / n_codes``; boundaries are placed at code edges
    ``B_C = C · LSB`` for ``C ∈ {1, …, n_codes - 1}``.  A signal in
    ``[B_C, B_{C+1})`` floor-buckets to code ``C``.

    Args:
        max_signal: Upper edge of the measurement range, in input
            units.  Must be ``> 0``.
        n_codes: Number of distinct output codes.  Must be ``>= 2``.

    Returns:
        ``n_codes - 1`` boundary values in increasing order.
    """
    if n_codes < 2:
        raise ValueError(f"n_codes ({n_codes}) must be >= 2")
    if not (max_signal > 0.0):
        raise ValueError(f"max_signal ({max_signal}) must be > 0")
    lsb = max_signal / n_codes
    return [c * lsb for c in range(1, n_codes)]


def compute_adc_boundaries__uA(
    i_max_diff__uA: float,
    n_codes: int,
) -> list[float]:
    """Floor-style ADC boundaries — wrapper around :func:`floor_boundaries_for_mode`.

    Args:
        i_max_diff__uA: Full-scale input swing the ADC must span [uA].
        n_codes: Number of distinguishable output codes (``>= 2``).

    Returns:
        ``n_codes - 1`` floor-style boundary values [uA].
    """
    return floor_boundaries_for_mode(i_max_diff__uA, n_codes)


@dataclass(frozen=True)
class CalibrationResult:
    """Output of a single statistical calibration pass.

    Attributes:
        s_max: Empirical maximum signal observed (input units —
            uA for current-mode, V for voltage-mode).
        n_states_required: ``max(ideal_code) + 1`` — the minimum
            number of distinct integer states the ADC must resolve.
        n_bits_max: ``ceil(log2(n_states_required))`` — bit width of
            the highest-precision mode.
        boundaries: Floor-style threshold list for the highest-
            precision mode.
        signal_samples: Optional flat tensor of every signal value
            captured (for ``--visualize``).
        code_samples: Optional flat tensor of every ideal code value
            captured (for ``--visualize``).
    """

    s_max: float
    n_states_required: int
    n_bits_max: int
    boundaries: list[float]
    signal_samples: torch.Tensor | None = None
    code_samples: torch.Tensor | None = None


def derive_modes(
    n_bits_max: int,
    s_max: float,
    n_states_max: int,
    *,
    additional_modes: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int, float]]:
    """Build ``(n_bits, n_states, max_signal)`` mode tuples.

    Index 0 is the highest-precision mode. Each additional mode's
    ``max_signal`` shrinks as ``n_states / n_states_max``.

    Args:
        n_bits_max: Highest-precision bit width.
        s_max: Empirical maximum signal at the highest-precision mode.
        n_states_max: Empirical state count at the highest-precision mode.
        additional_modes: ``(n_bits, n_states)`` pairs for sub-modes,
            each satisfying ``n_states <= 2 ** n_bits <= 2 ** n_bits_max``.

    Returns:
        Mode tuples ordered with the highest-precision mode first.
    """
    modes: list[tuple[int, int, float]] = [(n_bits_max, n_states_max, s_max)]
    for nb, ns in additional_modes or []:
        if not (1 <= nb <= n_bits_max):
            raise ValueError(f"additional mode n_bits={nb} outside [1, {n_bits_max}]")
        if not (2 <= ns <= 1 << nb):
            raise ValueError(f"additional mode n_states={ns} outside [2, 2**{nb}]")
        sub_max = s_max * ns / n_states_max
        modes.append((nb, ns, sub_max))
    return modes


# ---------------------------------------------------------------------------
# Statistical calibration loop
# ---------------------------------------------------------------------------


def calibrate(
    *,
    config_path: Path,
    random_n: int,
    apply_noise: bool,
) -> CalibrationResult:
    """Run a real-vs-ideal-xbar comparison and fit floor-style boundaries.

    Args:
        config_path: Chip TOML.
        random_n: Number of random ``(weight, activation)`` groups to sample.
            ``<= 0`` enumerates corner cases.
        apply_noise: When True, build the physical xbar with all configured
            noise stages; when False, strip every noise sub-config.

    Returns:
        :class:`CalibrationResult`.

    Raises:
        ValueError: When ``config_path`` lacks the required sections.
    """
    from dataclasses import replace as dc_replace

    from neurox.common import T_ROOM__K, dataclass_from_file, dict_from_file
    from neurox.mapper.transcoder import Transcoder
    from neurox.xbar import Offset1T1RXbar, Offset1T1RXbarConfig

    xbar_cfg = dataclass_from_file(Offset1T1RXbarConfig, config_path, section="xbar")

    if not apply_noise:
        # Flip every device-level noise toggle off along the ownership chain.
        core = xbar_cfg.core_cfg
        readout = xbar_cfg.readout_cfg
        rram_cfg = dc_replace(
            core.rram_cfg,
            enable_prog_gamma=False,
            enable_read_telegraph=False,
            enable_read_thermal=False,
            enable_stuck_at=False,
        )
        nmos_cfg = dc_replace(
            core.nmos_cfg,
            enable_A_vt_mismatch=False,
            enable_A_beta_mismatch=False,
        )
        tia_nmos_cfg = dc_replace(
            core.tia_cfg.nmos_cfg,
            enable_A_vt_mismatch=False,
            enable_A_beta_mismatch=False,
        )
        tia_cfg = dc_replace(
            core.tia_cfg,
            enable_opamp_gain_sigma=False,
            nmos_cfg=tia_nmos_cfg,
        )
        core_cfg = dc_replace(core, rram_cfg=rram_cfg, nmos_cfg=nmos_cfg, tia_cfg=tia_cfg)
        xbar_cfg = dc_replace(xbar_cfg, core_cfg=core_cfg, readout_cfg=readout)

    # Build the xbar directly from the nested config tree.
    physical = Offset1T1RXbar(
        cfg=xbar_cfg,
        name="xbar",
        inst_shape=(),
        dtype=torch.float64,
        T__K=T_ROOM__K,
    )
    physical.eval()
    ideal = physical.to_ideal()
    ideal.eval()

    # Weight transcoder for the tool's calibration sweep.
    raw_full = dict_from_file(config_path)
    w_radix = xbar_cfg.w_digit_radix
    w_tc = Transcoder.create(
        raw_full["w_transcoder"]["encoding"],
        radix=w_radix,
        digit_num=xbar_cfg.w_digit_count,
    )

    col_num = physical.col_num
    row_num = physical.row_num
    # x_states = per-cycle input grid size.
    x_lo, x_hi = physical.x_range
    x_states = x_hi - x_lo + 1
    # symmetric signed-digit envelope r^D - 1 the transcoder targets.
    w_max = physical.w_digit_radix**physical.w_digit_count - 1

    if random_n > 0:
        torch.manual_seed(0)
        weights = torch.randint(
            -w_max,
            w_max + 1,
            (random_n, col_num, row_num),
            dtype=torch.int32,
        )
        activations = torch.randint(0, x_states, (random_n, row_num), dtype=torch.int32)
    else:
        # Corner cases: max-positive logical, min-negative logical,
        # checkerboard, all-zero — with all-on / all-off activations.
        n_corners = 4
        weights = torch.zeros((n_corners, col_num, row_num), dtype=torch.int32)
        weights[0] = w_max
        weights[1] = -w_max
        weights[2, ::2] = w_max
        weights[2, 1::2] = -w_max
        weights[3] = 0
        activations = torch.zeros((n_corners, row_num), dtype=torch.int32)
        activations[0] = 1
        activations[2] = 1
        activations[3] = 1

    signal_buf: list[torch.Tensor] = []
    code_buf: list[torch.Tensor] = []
    s_max = 0.0
    n_states_observed = 0
    # Grouped readout lattice constants.
    group_num = physical.n_ref_cols
    data_num = xbar_cfg.ref_group_size
    digit_num = xbar_cfg.w_digit_count

    for w_i, x_i in zip(weights, activations, strict=True):
        # Signed-digit-transcode the logical weight into the xbar-native digit grid.
        w_digits = w_tc.encode(w_i, dim=-2)
        physical.fabricate()
        physical.program(w_digits)
        core = physical.core

        # Build the execution shape from the fabricated layout.
        full_shape = (core.fabricated_col_num, core.fabricated_row_num)
        x_2d = x_i.unsqueeze(0).to(torch.float64)  # [1, row_num]
        wl_logic = x_2d.expand(1, core.fabricated_row_num).squeeze(-2)
        wl_drive = (wl_logic * core.v_dd_wl__V).unsqueeze(-2)

        # One-shot per-VMM runtime sampling.
        rram_snapshot = core.rram.snapshot(shape=full_shape)
        nmos_snapshot = core.nmos.snapshot(shape=full_shape)
        bl_driver_snapshot = core.tia.snapshot(shape=(core.fabricated_col_num,))
        sl_driver_snapshot = core.sl_driver.snapshot(shape=(core.fabricated_row_num,))

        # Reuse the core's solver and fabricated wire state.
        assert core.solver is not None
        result = core.solver.solve(
            wl_drive,
            rram_snapshot=rram_snapshot,
            nmos_snapshot=nmos_snapshot,
            bl_driver_snapshot=bl_driver_snapshot,
            sl_driver_snapshot=sl_driver_snapshot,
        )

        # Reuse the real readout chain end-to-end.
        from neurox.xbar._1t1r.offset import _split_logic_and_ref

        tia_dc = core.tia.solve_dc(
            result.i_bl_driver,
            bl_driver_snapshot,
            v_clamp_init__V=result.v_bl_clamp,
        )
        v_out_phys = tia_dc.v_out__V
        v_data_phys, v_ref_phys = _split_logic_and_ref(v_out_phys, physical.logic_phys_idx, physical.ref_phys_idx)
        # Drive the production readout submodules with the same grouped lattice.
        readout = physical.readout
        v_data_grouped = v_data_phys.unflatten(-1, (group_num, data_num, digit_num))
        v_pos__V, _ = readout.data_switchcap.sample_and_accumulate(v_data_grouped)
        v_ref_bank = v_ref_phys.unsqueeze(-1)
        v_ref_sampled__V, _ = readout.ref_switchcap.sample_and_accumulate(v_ref_bank)
        v_neg__V = v_ref_sampled__V.unsqueeze(-1).expand(*v_ref_sampled__V.shape, data_num)
        v_pos_muxed__V, v_neg_muxed__V, _ = readout.analog_mux.transport(v_pos__V, v_neg__V)
        signal__V = v_pos_muxed__V - v_neg_muxed__V  # [..., group_num, data_num]

        # Ideal integer dot product on the original logical weights.
        ideal.program(w_digits)
        x_int = x_i.to(torch.int64)
        dot = (w_i.to(torch.int64) * x_int.unsqueeze(0)).sum(dim=-1)

        signal_buf.append(signal__V.flatten().to(torch.float64))
        code_buf.append(dot.flatten().to(torch.int64))

        s_max = max(s_max, float(signal__V.max().item()))
        n_states_observed = max(
            n_states_observed,
            int(dot.abs().max().item()) + 1,
        )

    n_bits_max = max(math.ceil(math.log2(max(n_states_observed, 2))), 1)
    boundaries = floor_boundaries_for_mode(s_max, 1 << n_bits_max)

    return CalibrationResult(
        s_max=s_max,
        n_states_required=n_states_observed,
        n_bits_max=n_bits_max,
        boundaries=boundaries,
        signal_samples=torch.cat(signal_buf) if signal_buf else None,
        code_samples=torch.cat(code_buf) if code_buf else None,
    )


def visualize(result: CalibrationResult, output_path: Path) -> None:
    """Render a histogram of the calibrated signal-vs-code distribution.

    No-op when matplotlib is unavailable.
    """
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.stderr.write("[xbar_adc_boundaries] matplotlib not installed — skipping --visualize\n")
        return

    if result.signal_samples is None:
        sys.stderr.write("[xbar_adc_boundaries] no signal samples — skipping --visualize\n")
        return

    sigs = result.signal_samples.cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(sigs, bins=128, alpha=0.7, label="Signal")
    for b in result.boundaries:
        ax.axvline(b, color="red", linewidth=0.5, alpha=0.6)
    ax.set_xlabel("Signal")
    ax.set_ylabel("Count")
    ax.set_title(
        f"ADC calibration — s_max={result.s_max:.4g}, n_bits={result.n_bits_max}, n_states={result.n_states_required}"
    )
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_toml(result: CalibrationResult) -> str:
    """Render the calibrated result as a ``[bl_adc]`` TOML block."""
    lines = [
        "# ADC calibration result — paste under your chip TOML's [bl_adc] section.",
        f"# Empirical max signal:        {result.s_max:.6g}",
        f"# Required ideal-state count:  {result.n_states_required}",
        f"# Highest-precision bit width: {result.n_bits_max}",
        "",
        "[bl_adc]",
        "boundaries = [",
    ]
    for b in result.boundaries:
        lines.extend(f"    {b:.6g},")
    lines.extend("]")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="Chip config TOML path")
    parser.add_argument(
        "--random",
        type=int,
        default=256,
        help="N random input groups; <= 0 traverses corner cases",
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        help="Apply physical noise during sampling",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save a histogram PNG alongside --output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination TOML for the calibrated [bl_adc] block",
    )
    args = parser.parse_args(argv)

    result = calibrate(
        config_path=args.config,
        random_n=args.random,
        apply_noise=args.noise,
    )

    block = _format_toml(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(block)
        sys.stdout.write(f"Wrote calibrated [bl_adc] block to {args.output}\n")
    else:
        sys.stdout.write(block)

    if args.visualize:
        png_path = Path("xbar_adc_boundaries.png") if args.output is None else args.output.with_suffix(".png")
        visualize(result, png_path)
        sys.stdout.write(f"Wrote visualization to {png_path}\n")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
