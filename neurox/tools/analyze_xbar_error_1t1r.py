"""Analyse VMM accuracy of a 1T1R xbar against its lossless ideal twin.

See also:
    docs/dev/modules/tools/README.md
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch

from neurox.analog import Driver, GeneralDAC, OpAmpTIA, OpAmpTIAConfig
from neurox.analog.adc import GeneralADC, GeneralADCConfig
from neurox.analog.dac import GeneralDACConfig
from neurox.analog.driver import DriverConfig
from neurox.common import dict_configs_from_file, dict_from_file
from neurox.common.nonideality import (
    StateDependentGammaConfig,
    StuckAtFaultConfig,
    TelegraphConfig,
)
from neurox.device import NMOS, RRAM, Wire
from neurox.device.nmos import NMOSConfig
from neurox.device.rram import RRAMConfig
from neurox.device.wire import WireConfig
from neurox.xbar import IdealXbar, Offset1T1RXbar, Offset1T1RXbarConfig
from neurox.xbar.base import Xbar

logger = logging.getLogger(__name__)


_SPECS: dict[str, type] = {
    "rram": RRAMConfig,
    "nmos": NMOSConfig,
    "tia": OpAmpTIAConfig,
    "tia_nmos": NMOSConfig,
    "sl_driver": DriverConfig,
    "wl_dac": GeneralDACConfig,
    "bl_adc": GeneralADCConfig,
    "xbar": Offset1T1RXbarConfig,
}

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

_FMT = "{:.4e}"


# Preset noise overlays applied per category.  Magnitudes are chosen
# near the "1 % of native scale" mark — large enough that the per-source
# ranking is meaningful, small enough to stay in the regime real chips
# operate in.
_NOISE_PRESETS: dict[str, list[tuple[str, str, Any]]] = {
    "rram_saf": [
        ("rram", "stuck_at", StuckAtFaultConfig(p_at_min=0.001, p_at_max=0.001)),
    ],
    "rram_prog": [
        (
            "rram",
            "prog_gamma",
            StateDependentGammaConfig(k_slope=0.0, k_intercept=100.0, theta=1.0, min_val=0.01, max_val=0.1),
        ),
    ],
    "rram_read": [
        ("rram", "read_telegraph", TelegraphConfig(amplitude_mean=0.005, amplitude_std=0.001, p_high_state=0.01)),
        ("rram", "read_thermal", 0.002),
    ],
    "nmos_fab": [
        # Pelgrom matching coefficients: σ scales as 1/sqrt(W·L).
        # Typical 28 nm RVT values from Pelgrom-style measurements.
        ("nmos", "A_vt__V_um", 3.0e-3),
        ("nmos", "A_beta_relative__um", 3.0e-3),
    ],
    "periphery": [
        ("bl_adc", "sampling_noise", 0.001),
        ("bl_adc", "comparator_noise", 0.001),
        ("bl_adc", "drive_thermal", 0.0005),
        ("wl_dac", "drive_thermal", 0.005),
        ("sl_driver", "drive_thermal", 0.001),
    ],
}

# Catalog of every noise sub-config across every chip-cfg section the
# 1T1R xbar consumes.  Kept local to this tool because the section /
# field names are tied to the 1T1R config schema specifically — a
# different crossbar topology would carry a different catalog.
_NOISE_FIELDS: dict[str, list[str]] = {
    "rram": ["prog_gamma", "read_telegraph", "read_thermal", "stuck_at"],
    "nmos": ["A_vt__V_um", "A_beta_relative__um"],
    "bl_adc": ["sampling_noise", "comparator_noise", "drive_thermal"],
    "wl_dac": ["drive_thermal"],
    "sl_driver": ["drive_thermal"],
}


# ---------------------------------------------------------------------- #
# Config helpers                                                         #
# ---------------------------------------------------------------------- #


def strip_all_noise(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``cfg`` with every known noise sub-config set to ``None``."""
    out = dict(cfg)
    for section, fields in _NOISE_FIELDS.items():
        if section in out:
            out[section] = dataclasses.replace(out[section], **dict.fromkeys(fields))
    return out


def apply_noise_overlay(cfg: dict[str, Any], overlay: list[tuple[str, str, Any]]) -> dict[str, Any]:
    """Return a copy of ``cfg`` with the (section, field, value) tuples applied."""
    out = dict(cfg)
    section_updates: dict[str, dict[str, Any]] = {}
    for section, field, value in overlay:
        section_updates.setdefault(section, {})[field] = value
    for section, updates in section_updates.items():
        out[section] = dataclasses.replace(out[section], **updates)
    return out


# ---------------------------------------------------------------------- #
# Xbar construction                                                      #
# ---------------------------------------------------------------------- #


def _build_wire(raw_section: dict[str, Any] | None) -> Wire | None:
    """Build a :class:`Wire` from a raw TOML section, or ``None``."""
    if raw_section is None:
        return None
    return Wire(WireConfig(**raw_section))


def build_physical_xbar(
    cfg: dict[str, Any],
    raw: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Offset1T1RXbar:
    """Instantiate a :class:`Xbar1T1R` from a (cfg, raw) pair on ``device``.

    ``raw`` is the parsed-but-not-typed TOML dict (used for optional
    wire sections that aren't part of the strict-spec ``cfg``).
    """
    rram = RRAM(cfg=cfg["rram"], T__K=300.0, dtype=dtype, g_max__uS=cfg["core"].rram_g_max__uS)
    nmos = NMOS(cfg["nmos"], dtype=dtype)
    tia_nmos = NMOS(cfg["tia_nmos"], dtype=dtype)
    tia = OpAmpTIA(cfg["tia"], nmos=tia_nmos, dtype=dtype)
    sl_wire = _build_wire(raw.get("sl_wire"))
    bl_wire = _build_wire(raw.get("bl_wire"))
    wl_wire = _build_wire(raw.get("wl_wire"))
    return Offset1T1RXbar(
        config=cfg["xbar"],
        rram=rram,
        nmos=nmos,
        tia=tia,
        sl_wire=sl_wire,
        bl_wire=bl_wire,
        wl_wire=wl_wire,
        sl_driver=partial(Driver, cfg["sl_driver"], dtype=dtype),
        wl_dac=partial(GeneralDAC, cfg["wl_dac"], dtype=dtype),
        bl_adc=partial(GeneralADC, cfg["bl_adc"], dtype=dtype),
    ).to(device)


def build_ideal_twin(physical: Offset1T1RXbar, *, device: torch.device) -> IdealXbar:
    """Lossless reference matching ``physical``'s tile geometry and resolution."""
    return physical.to_ideal().to(device)


# ---------------------------------------------------------------------- #
# VMM measurement                                                        #
# ---------------------------------------------------------------------- #


def run_vmm(xbar: Xbar, w: torch.Tensor, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """One ``(w, x)`` VMM through ``xbar``; returns the column-code vector."""
    state = xbar.fabricate_unmapped(w)
    actual, _ = xbar.vec_mat_mul(x.unsqueeze(0).unsqueeze(0).to(dtype), state)
    return actual.squeeze().to(torch.int32)


def compare_against_ideal(
    physical: Offset1T1RXbar,
    ideal: IdealXbar,
    *,
    n_rand_w: int,
    n_rand_x: int,
    n_repeat: int,
    device: torch.device,
    dtype: torch.dtype,
    seed_offset: int,
) -> dict[str, float]:
    """Run ``n_rand_w * n_rand_x * n_repeat`` VMMs and stat the diff vs ideal.

    For each random ``W``, the ideal reference is computed once.  The
    physical xbar is re-fabricated ``n_repeat`` times per ``(W, x)``
    pair so noise contributions from programming-time stochasticity
    can vary between trials.
    """
    col = physical.col_num
    row = physical.row_num
    # Algorithm-facing signed-weight bound — symmetric signed-digit
    # envelope ``r^D - 1`` (matches ``XbarMacro.w_value_range`` / the
    # transcoder's natural output range).
    w_max = physical.w_digit_radix**physical.w_digit_count - 1
    x_max = physical.x_range[1]
    adc_levels = 1 << physical.config.adc_bits

    diffs: list[torch.Tensor] = []
    for w_trial in range(n_rand_w):
        torch.manual_seed(seed_offset + w_trial)
        w = torch.randint(-w_max, w_max + 1, (col, row), dtype=torch.int32, device=device)
        for x_trial in range(n_rand_x):
            torch.manual_seed(seed_offset + 10_000 * (w_trial + 1) + x_trial)
            x = torch.randint(0, x_max + 1, (row,), dtype=torch.int32, device=device)
            expected = run_vmm(ideal, w, x, dtype)
            for _ in range(n_repeat):
                actual = run_vmm(physical, w, x, dtype)
                diffs.append((actual - expected).to(torch.float64))

    all_diffs = torch.cat([d.flatten() for d in diffs])
    abs_diff = all_diffs.abs()
    return {
        "n": float(all_diffs.numel()),
        "exact_pct": (abs_diff == 0).float().mean().item() * 100.0,
        "mae": abs_diff.mean().item(),
        "rmse": torch.sqrt((all_diffs**2).mean()).item(),
        "max_abs": abs_diff.max().item(),
        "max_abs_pct_of_full_scale": abs_diff.max().item() / max(adc_levels - 1, 1) * 100.0,
        "mean_signed": all_diffs.mean().item(),
    }


# ---------------------------------------------------------------------- #
# Reporting                                                              #
# ---------------------------------------------------------------------- #


def _log_header(
    config: Path,
    device: torch.device,
    dtype: torch.dtype,
    n_rand_w: int,
    n_rand_x: int,
    n_repeat: int,
    use_noise: bool,
) -> None:
    """Emit a small banner describing the run setup."""
    logger.info("config: %s", config)
    logger.info("device: %s   dtype: %s", device, dtype)
    logger.info(
        "samples: %d random W * %d random x * %d repeats = %d VMMs per variant",
        n_rand_w,
        n_rand_x,
        n_repeat,
        n_rand_w * n_rand_x * n_repeat,
    )
    logger.info("noise variants: %s", "enabled" if use_noise else "disabled")
    logger.info("---")


def _log_variant(label: str, stats: dict[str, float], elapsed_s: float) -> None:
    """Emit the per-variant stats line."""
    logger.info(
        "%-30s  exact=%6.2f%%  MAE=%s  RMSE=%s  max|Δ|=%s  max%%FS=%5.2f%%  bias=%s  (%.1fs)",
        label,
        stats["exact_pct"],
        _FMT.format(stats["mae"]),
        _FMT.format(stats["rmse"]),
        _FMT.format(stats["max_abs"]),
        stats["max_abs_pct_of_full_scale"],
        _FMT.format(stats["mean_signed"]),
        elapsed_s,
    )


# ---------------------------------------------------------------------- #
# Variant orchestration                                                  #
# ---------------------------------------------------------------------- #


def measure_variant(
    label: str,
    cfg: dict[str, Any],
    raw: dict[str, Any],
    ideal: IdealXbar,
    *,
    n_rand_w: int,
    n_rand_x: int,
    n_repeat: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Build a physical xbar from ``cfg``, compare to ``ideal``, log + return stats."""
    physical = build_physical_xbar(cfg, raw, device=device, dtype=dtype)
    t0 = time.time()
    stats = compare_against_ideal(
        physical,
        ideal,
        n_rand_w=n_rand_w,
        n_rand_x=n_rand_x,
        n_repeat=n_repeat,
        device=device,
        dtype=dtype,
        seed_offset=hash(label) & 0xFFFF,
    )
    elapsed = time.time() - t0
    _log_variant(label, stats, elapsed)
    return stats


def build_noise_variants(noiseless_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return ``{label: cfg}`` for each isolated noise category + the all-combined.

    Every variant starts from the *noiseless* base ``cfg`` so each
    category is measured against the same control.
    """
    variants: dict[str, dict[str, Any]] = {}
    for label, overlay in _NOISE_PRESETS.items():
        variants[label] = apply_noise_overlay(noiseless_cfg, overlay)
    combined = [tup for overlay in _NOISE_PRESETS.values() for tup in overlay]
    variants["all_noise"] = apply_noise_overlay(noiseless_cfg, combined)
    return variants


def _run(
    config: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_rand_w: int,
    n_rand_x: int,
    use_noise: bool,
    n_repeat: int,
    quiet: bool,
) -> None:
    """Top-level driver: load config, build all variants, print summary."""
    raw = dict_from_file(config)
    full_cfg = dict_configs_from_file(_SPECS, config)

    noiseless_cfg = strip_all_noise(full_cfg)
    noiseless = build_physical_xbar(noiseless_cfg, raw, device=device, dtype=dtype)
    ideal = build_ideal_twin(noiseless, device=device)

    if not quiet:
        _log_header(config, device, dtype, n_rand_w, n_rand_x, n_repeat if use_noise else 1, use_noise)

    # Baseline: noiseless physical xbar vs ideal twin.  ``n_repeat = 1``
    # is sufficient — the path is deterministic.
    measure_variant(
        "noiseless_1t1r vs ideal",
        noiseless_cfg,
        raw,
        ideal,
        n_rand_w=n_rand_w,
        n_rand_x=n_rand_x,
        n_repeat=1,
        device=device,
        dtype=dtype,
    )

    if not use_noise:
        return

    for label, variant_cfg in build_noise_variants(noiseless_cfg).items():
        measure_variant(
            label,
            variant_cfg,
            raw,
            ideal,
            n_rand_w=n_rand_w,
            n_rand_x=n_rand_x,
            n_repeat=n_repeat,
            device=device,
            dtype=dtype,
        )


# ---------------------------------------------------------------------- #
# CLI                                                                    #
# ---------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    """CLI parser — every argument is required (no defaults)."""
    parser = argparse.ArgumentParser(
        description="Compare a 1T1R xbar's VMM output against its ideal twin (optionally with noise sweep).",
    )
    parser.add_argument("--config", type=Path, required=True, help="Chip TOML.")
    parser.add_argument("--device", type=str, required=True, help="Torch device (e.g. cuda:0, cpu).")
    parser.add_argument(
        "--dtype",
        type=str,
        required=True,
        choices=tuple(_DTYPE_MAP.keys()),
        help="Tensor floating-point dtype.",
    )
    parser.add_argument("--n-rand-w", type=int, required=True, help="Random W matrices.")
    parser.add_argument("--n-rand-x", type=int, required=True, help="Random x vectors per W.")
    parser.add_argument("--noise", action="store_true", help="Run the per-category noise sweep.")
    parser.add_argument(
        "--n-repeat",
        type=int,
        required=True,
        help="Re-fabrication count per (W, x) pair for noisy variants (ignored without --noise).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the run-setup banner.")
    return parser


def main() -> None:
    """Console entry point: parse CLI, configure logging, dispatch ``_run``."""
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_grad_enabled(False)
    _run(
        config=args.config,
        device=torch.device(args.device),
        dtype=_DTYPE_MAP[args.dtype],
        n_rand_w=args.n_rand_w,
        n_rand_x=args.n_rand_x,
        use_noise=args.noise,
        n_repeat=args.n_repeat,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
