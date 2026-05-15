"""ADC boundary calibration utilities for the differential ADC family.

Two-fold output for a calibrated ADC mode:

1. **Range** — empirical maximum BL signal (post-OpAmpTIA differential
   voltage ``v_pos - v_neg``, in V) across realistic
   ``(weight, activation)`` combinations.  Defines ``max_signal``
   for the highest-precision mode.
2. **Precision** — required code count derived from the maximum
   ideal integer state observed.  Defines ``n_bits`` and ``n_states``
   for the highest-precision mode.

Lower-precision modes are derived from the highest by halving codes
and shrinking ``max_signal`` proportionally; they share the same
underlying physical comparator thresholds.

The historical helpers :func:`compute_max_col_diff_current__uA` and
:func:`compute_adc_boundaries__uA` are preserved as convenience
wrappers.  Both now emit floor-style boundaries
(``B_C = C · LSB``) instead of round-style (``B_C = (C - 0.5) · LSB``).

The CLI :mod:`neurox.tools.xbar_adc_boundaries` runs a real-vs-ideal
xbar comparison over a sweep of inputs and prints a TOML block ready
to paste into a chip config.  CLI arguments mirror the user spec:

* ``--config <path>`` chip TOML.
* ``--random N`` ``N`` random input groups; ``N <= 0`` traverses corner
  cases.
* ``--noise`` apply physical noise during sampling.
* ``--visualize`` save a PNG of the signal-vs-code distribution.
* ``--output <path>`` write the calibrated ``[bl_adc]`` TOML block.

The CLI is intentionally minimal — chip-level studies that need
deeper post-processing should script around the public helpers
below.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from functools import partial
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
        ``row_num * (I_cell_max − I_cell_min)`` [uA].
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

    Note:
        Earlier versions placed boundaries at ``(C - 0.5) · LSB`` (round-
        to-nearest semantics).  The new ADC family uses floor-bucketize,
        so this helper now produces ``C · LSB`` to match.

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
    """Build a list of ``(n_bits, n_states, max_signal)`` mode tuples.

    Always emits the highest-precision mode at index 0.  Optional
    additional modes downsample bits and / or codes; their
    ``max_signal`` shrinks proportionally to ``n_states / n_states_max``.

    Args:
        n_bits_max: Highest-precision bit width.
        s_max: Empirical maximum signal at the highest-precision mode.
        n_states_max: Empirical state count at the highest-precision
            mode.
        additional_modes: Optional ``(n_bits, n_states)`` pairs for
            sub-modes.  Each pair must satisfy
            ``n_states <= 2 ** n_bits <= 2 ** n_bits_max``.

    Returns:
        List of ``(n_bits, n_states, max_signal)`` tuples ordered
        with the highest-precision mode first.
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

    Two-fold output (range + precision):

    * ``s_max`` is the maximum BL signal observed across the sweep.
    * ``n_states_required`` is ``max(ideal_code) + 1``.

    Args:
        config_path: Chip TOML.
        random_n: Number of random ``(weight, activation)`` groups to
            sample.  ``<= 0`` enumerates corner cases up to a bounded
            budget.
        apply_noise: When True, the physical xbar is built with its
            configured noise stages active; when False, every noise
            sub-config is skipped at module construction time.

    Returns:
        :class:`CalibrationResult`.

    Raises:
        ValueError: When ``config_path`` lacks the required sections.
    """
    # Late imports keep the module's lightweight helpers usable
    # without dragging in the heavy device / xbar dependency tree.
    from neurox.analog import (
        AnalogMux,
        AnalogMuxConfig,
        Decoder,
        DecoderConfig,
        Driver,
        DriverConfig,
        GeneralDAC,
        GeneralDACConfig,
        OpAmpTIA,
        OpAmpTIAConfig,
        SwitchCap,
        SwitchCapConfig,
    )
    from neurox.analog.adc import McsSarAdc, McsSarAdcConfig
    from neurox.analog.readout import OffsetSwitchCapMuxAdcReadOut, ReadOutConfig
    from neurox.common import T_ROOM__K, dict_configs_from_file, dict_from_file
    from neurox.device import NMOS, RRAM, NMOSConfig, RRAMConfig, Wire, WireConfig
    from neurox.mapper import SignedDigitTranscoder
    from neurox.xbar import (
        Core1T1R,
        Core1T1RConfig,
        Offset1T1RXbar,
        Offset1T1RXbarConfig,
    )

    specs = {
        "rram": RRAMConfig,
        "nmos": NMOSConfig,
        "tia": OpAmpTIAConfig,
        "tia_nmos": NMOSConfig,
        "bl_wire": WireConfig,
        "sl_wire": WireConfig,
        "wl_wire": WireConfig,
        "sl_driver": DriverConfig,
        "wl_decoder": DecoderConfig,
        "wl_dac": GeneralDACConfig,
        "bl_adc": McsSarAdcConfig,
        "analog_mux": AnalogMuxConfig,
        "data_switchcap": SwitchCapConfig,
        "ref_switchcap": SwitchCapConfig,
        "core": Core1T1RConfig,
        "readout": ReadOutConfig,
        "xbar": Offset1T1RXbarConfig,
    }
    typed = dict_configs_from_file(specs, config_path)

    if not apply_noise:
        # Strip every noise sub-config to drive the noise-free analytic
        # path.  Done by reconstructing each typed config without the
        # noise-bearing fields.
        rram_cfg = typed["rram"]
        typed["rram"] = type(rram_cfg)(
            **{**rram_cfg.__dict__, "prog_gamma": None, "read_telegraph": None, "read_thermal": None}
        )
        for nmos_key in ("nmos", "tia_nmos"):
            nmos_cfg = typed[nmos_key]
            typed[nmos_key] = type(nmos_cfg)(
                **{
                    **nmos_cfg.__dict__,
                    "A_vt__mV_um": None,
                    "A_beta_relative__um": None,
                }
            )
        tia_cfg = typed["tia"]
        typed["tia"] = type(tia_cfg)(**{**tia_cfg.__dict__, "opamp_gain_sigma": None})

    # Per-core device / circuit / wire module factories — each
    # ``Core1T1R`` gets its own ``RRAM`` / ``NMOS`` / ``OpAmpTIA``
    # (OpAmpTIA-internal pseudo-resistor NMOS included) and its own
    # ``BL`` / ``SL`` / ``WL`` :class:`~neurox.device.Wire`.  Wire
    # state is per-fabrication (see ``temp/wire.md``).
    rram_factory = partial(RRAM, typed["rram"], dtype=torch.float64)
    nmos_factory = partial(NMOS, typed["nmos"], T__K=T_ROOM__K, dtype=torch.float64)
    tia_nmos_factory = partial(NMOS, typed["tia_nmos"], T__K=T_ROOM__K, dtype=torch.float64)
    tia_factory = partial(OpAmpTIA, typed["tia"], nmos_factory=tia_nmos_factory, dtype=torch.float64)
    sl_wire_factory = partial(Wire, typed["sl_wire"], dtype=torch.float64)
    bl_wire_factory = partial(Wire, typed["bl_wire"], dtype=torch.float64)
    wl_wire_factory = partial(Wire, typed["wl_wire"], dtype=torch.float64)

    core_factory = partial(
        Core1T1R,
        typed["core"],
        rram_factory=rram_factory,
        nmos_factory=nmos_factory,
        tia_factory=tia_factory,
        sl_wire_factory=sl_wire_factory,
        bl_wire_factory=bl_wire_factory,
        wl_wire_factory=wl_wire_factory,
        sl_driver_factory=partial(Driver, typed["sl_driver"], dtype=torch.float64),
        wl_decoder_factory=partial(Decoder, typed["wl_decoder"]),
        wl_dac_factory=partial(GeneralDAC, typed["wl_dac"], dtype=torch.float64),
        dtype=torch.float64,
    )
    xbar_cfg = typed["xbar"]
    readout_factory = partial(
        OffsetSwitchCapMuxAdcReadOut,
        typed["readout"],
        data_switchcap_factory=partial(SwitchCap, typed["data_switchcap"], T__K=T_ROOM__K, dtype=torch.float64),
        ref_switchcap_factory=partial(SwitchCap, typed["ref_switchcap"], T__K=T_ROOM__K, dtype=torch.float64),
        analog_mux_factory=partial(AnalogMux, typed["analog_mux"], dtype=torch.float64),
        adc_factory=partial(McsSarAdc, typed["bl_adc"], T__K=T_ROOM__K, dtype=torch.float64),
        dtype=torch.float64,
    )
    physical = Offset1T1RXbar(
        cfg=xbar_cfg,
        core_factory=core_factory,
        readout_factory=readout_factory,
    )
    physical.eval()
    ideal = physical.to_ideal()
    ideal.eval()

    # Weight transcoder for the tool's calibration sweep — both
    # ``physical.fabricate`` and ``ideal.fabricate`` now consume an
    # xbar-native digit tensor, not a logical-weight tensor.  Read
    # the encoding policy from ``[w_transcoder]`` for parity with
    # ``example/common/macro_factory.py``.
    raw_full = dict_from_file(config_path)
    w_radix = xbar_cfg.w_digit_radix
    w_tc = SignedDigitTranscoder(raw_full["w_transcoder"]["encoding"], w_radix, xbar_cfg.w_digit_count)

    col_num = physical.col_num
    row_num = physical.row_num
    # Activation grid size is the xbar's per-cycle input range
    # (== 2 for the binary 1T1R WL pulse).
    x_lo, x_hi = physical.x_range
    x_states = x_hi - x_lo + 1
    # Algorithm-facing signed-weight bound used to seed the
    # calibration sweep.  Matches the symmetric signed-digit envelope
    # ``r^D - 1`` the transcoder fits into the xbar — the same value
    # the macro reports through ``XbarMacro.w_value_range``.  The
    # transcoder turns each logical value into the digit tensor the
    # xbar consumes; the xbar's physical ``w_digit_range`` may carry
    # extra one-sided headroom that the sweep does not exercise.
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
    # Grouped readout lattice constants — derived from the xbar's
    # ``Offset1T1RXbarConfig``.  The readout's ``data_switchcap`` was
    # fabricated against ``(*prefix, group_num, data_num, digit_num)``
    # and ``ref_switchcap`` against ``(*prefix, group_num, 1)``.
    group_num = physical.n_ref_cols
    data_num = xbar_cfg.ref_group_size
    digit_num = xbar_cfg.w_digit_count

    for w_i, x_i in zip(weights, activations, strict=True):
        # Fabricate routes through the offset xbar into Core1T1R; each
        # owned module fabricates its own internal state.  The
        # calibration tool reaches into ``physical.core.{rram, nmos,
        # tia, sl_driver, solver, v_dd_wl__V,
        # fabricated_col_num, fabricated_row_num}``,
        # ``physical.{logic_phys_idx, ref_phys_idx}`` and
        # ``physical.readout.{data_switchcap, ref_switchcap,
        # analog_mux}`` — calling the readout's leaf circuit kernels
        # directly so the calibration signal matches the production
        # chain block-for-block.
        # Signed-digit-transcode the logical weight into the xbar's
        # native digit grid before fabricating; both physical and
        # ideal twin share this contract.
        # Shape: w_digits -> [col_num, w_digit_count, row_num].
        w_digits = w_tc.encode(w_i, dim=-2)
        physical.fabricate(w_digits)
        core = physical.core

        # Build the execution shape from the fabricated layout — the
        # solver expects ``[*batch, phys_col_num, row_num]``.
        full_shape = (core.fabricated_col_num, core.fabricated_row_num)
        x_2d = x_i.unsqueeze(0).to(torch.float64)  # [1, row_num]
        wl_logic = x_2d.expand(1, core.fabricated_row_num).squeeze(-2)
        wl_drive = (wl_logic * core.v_dd_wl__V).unsqueeze(-2)

        # One-shot per-VMM runtime sampling.  Fresh per sweep iteration
        # since each iteration is a distinct VMM (independent dynamic
        # noise per ``temp/state_holding.md``).  Both BL (OpAmpTIA) and SL
        # (ideal Driver) clamp boundaries now sample once per VMM and
        # thread their snapshots through the solver.
        rram_snapshot = core.rram.snapshot(shape=full_shape)
        nmos_snapshot = core.nmos.snapshot(shape=full_shape)
        bl_driver_snapshot = core.tia.snapshot(shape=(core.fabricated_col_num,))
        sl_driver_snapshot = core.sl_driver.snapshot(shape=(core.fabricated_row_num,))

        # Reuse the core's solver — it holds the device modules and the
        # fabricated wire instances, rebuilt on every fabricate.  Wire
        # state lives on ``core.bl_wire`` / ``core.sl_wire`` and is
        # read inside the solver (see ``temp/wire.md``).
        assert core.solver is not None
        result = core.solver.solve(
            wl_drive,
            rram_snapshot=rram_snapshot,
            nmos_snapshot=nmos_snapshot,
            bl_driver_snapshot=bl_driver_snapshot,
            sl_driver_snapshot=sl_driver_snapshot,
        )

        # Reuse the real readout chain end-to-end so the calibration
        # signal matches block-for-block what the ADC sees in
        # production.  ``SolverResult`` carries only ``v_bl_clamp``;
        # re-invoke the OpAmpTIA's ``solve_dc`` with the *same*
        # ``bl_driver_snapshot`` at the solver's converged clamp-port
        # current to obtain the consistent ``v_out__V`` — same pattern
        # ``Core1T1R.forward`` uses.  ``result.i_bl_driver`` is the
        # boundary-KCL ``delta_v · g`` quantity (see
        # ``temp/solver.md``); we feed it directly rather than
        # ``i_cell.sum(-1)``, whose equality only holds on the current
        # 1-D BL ladder.
        from neurox.xbar._1t1r.offset_1t1r import _split_logic_and_ref

        tia_dc = core.tia.solve_dc(
            result.i_bl_driver,
            bl_driver_snapshot,
            v_clamp_init__V=result.v_bl_clamp,
        )
        v_out_phys = tia_dc.v_out__V
        v_data_phys, v_ref_phys = _split_logic_and_ref(v_out_phys, physical.logic_phys_idx, physical.ref_phys_idx)
        # Drive the production readout submodules with the same grouped
        # lattice the xbar's ``_vec_mat_mul_impl`` uses.  Every value-
        # domain step happens inside a leaf circuit module; the tool
        # itself only does shape ops (``unflatten`` / ``unsqueeze`` /
        # ``expand``) — exactly the project's
        # "shape-ops-only outside leaves" invariant.
        readout = physical.readout
        v_data_grouped = v_data_phys.unflatten(-1, (group_num, data_num, digit_num))
        v_pos__V, _ = readout.data_switchcap.sample_and_accumulate(v_data_grouped)
        v_ref_bank = v_ref_phys.unsqueeze(-1)
        v_ref_sampled__V, _ = readout.ref_switchcap.sample_and_accumulate(v_ref_bank)
        v_neg__V = v_ref_sampled__V.unsqueeze(-1).expand(*v_ref_sampled__V.shape, data_num)
        v_pos_muxed__V, v_neg_muxed__V, _ = readout.analog_mux.transport(v_pos__V, v_neg__V)
        signal__V = v_pos_muxed__V - v_neg_muxed__V  # [..., group_num, data_num]

        # Ideal arithmetic: per-column integer dot product on the
        # original logical weights — the ideal twin's digit tensor
        # collapses back to ``w_i`` via the same ``digit_weights``
        # it now stores.
        ideal.fabricate(w_digits)
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

    Skips the call gracefully when matplotlib is unavailable so the
    CLI's other paths still work in headless / minimal-deps
    environments.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
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
        lines.append(f"    {b:.6g},")
    lines.append("]")
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
        if args.output is None:
            png_path = Path("xbar_adc_boundaries.png")
        else:
            png_path = args.output.with_suffix(".png")
        visualize(result, png_path)
        sys.stdout.write(f"Wrote visualization to {png_path}\n")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
