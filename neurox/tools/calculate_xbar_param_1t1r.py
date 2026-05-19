"""1T1R-specific RRAM pre-distortion + ADC boundary calculator.

See also:
    docs/dev/modules/tools/README.md
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

from neurox.analog.adc import GeneralADCConfig
from neurox.analog.dac import GeneralDACConfig
from neurox.common import dict_configs_from_file
from neurox.device import NMOS
from neurox.device.nmos import NMOSConfig
from neurox.device.rram import RRAMConfig
from neurox.tools.xbar_adc_boundaries import (
    compute_adc_boundaries__uA,
    compute_max_col_diff_current__uA,
)
from neurox.xbar import CircuitCore1T1RConfig, Offset1T1RXbarConfig

logger = logging.getLogger(__name__)

_SPECS: dict[str, type] = {
    "rram": RRAMConfig,
    "nmos": NMOSConfig,
    "core": CircuitCore1T1RConfig,
    "wl_dac": GeneralDACConfig,
    "bl_adc": GeneralADCConfig,
    "xbar": Offset1T1RXbarConfig,
}

_FMT = "{:.4e}"


def _fmt_list(values: list[float]) -> str:
    """Render ``values`` as ``[v0, v1, …]`` with 5-sig-fig scientific notation."""
    return "[" + ", ".join(_FMT.format(v) for v in values) + "]"


def build_nmos(nmos_cfg: NMOSConfig, core_cfg: CircuitCore1T1RConfig) -> NMOS:
    """Build the NMOS device at ``T_ref`` (nominal β / V_th, no mismatch)."""
    return NMOS(
        nmos_cfg,
        W__um=core_cfg.access_nmos_W__um,
        L__um=core_cfg.access_nmos_L__um,
        T__K=nmos_cfg.T_ref__K,
    )


def compute_g_on__uS(nmos_cfg: NMOSConfig, core_cfg: CircuitCore1T1RConfig, v_dd_wl__V: float) -> float:
    """Deep-triode access-device conductance at the WL on-level.

    Mirrors the deep-triode limit of :meth:`NMOS.gds__uS` reduced to a
    noise-free scalar formula evaluated at ``T = T_ref``::

        β  = μ0 · C_ox · (W / L)
        g_on = β · (V_DD,WL − V_th0)

    Args:
        nmos_cfg: NMOS PDK process / spec configuration.
        core_cfg: 1T1R core config carrying the access-NMOS design
            parameters (``access_nmos_W/L__um``).
        v_dd_wl__V: WL drive voltage [V].

    Returns:
        Deep-triode conductance ``g_on`` [uS].  Clamped to a tiny
        positive epsilon to avoid sign flips in extreme tails.
    """
    # cm²/V/s · fF/μm² → 1e-1 uA/V² (matches NMOS.__init__).
    beta = (
        nmos_cfg.mu0__cm2_per_V_s
        * nmos_cfg.c_ox__fF_per_um2
        * 1e-1
        * (core_cfg.access_nmos_W__um / core_cfg.access_nmos_L__um)
    )
    return max(beta * (v_dd_wl__V - nmos_cfg.vth0__V), 1e-12)


def solve_cell_current__uA(
    g_rram__uS: float,
    g_on__uS: float,
    v_bl__V: float,
    alpha: float,
) -> float:
    """On-state cell current for a series RRAM/NMOS with sinh I-V.

    Solves the 1-D non-linear KCL

        g_rram · sinh(α · (V_BL − V_X)) / α = g_on · V_X

    by Newton iteration starting from ``V_X = V_BL / 2``.  ``α = 0``
    short-circuits to the closed-form ohmic solution.

    Args:
        g_rram__uS: RRAM conductance [uS].
        g_on__uS: Access-device deep-triode conductance [uS].
        v_bl__V: BL drive voltage [V] (SL is ground).
        alpha: RRAM sinh-IV non-linearity coefficient [1/V].

    Returns:
        Steady-state cell current ``I_cell`` [uA].
    """
    if alpha == 0.0:
        v_x = v_bl__V * g_rram__uS / (g_rram__uS + g_on__uS)
        return g_on__uS * v_x
    v_x = 0.5 * v_bl__V
    for _ in range(64):
        v_rram = v_bl__V - v_x
        i_rram = g_rram__uS * math.sinh(alpha * v_rram) / alpha
        di_drv = g_rram__uS * math.cosh(alpha * v_rram)
        f = g_on__uS * v_x - i_rram
        fp = g_on__uS + di_drv
        delta = f / fp if fp > 1e-30 else 0.0
        v_x -= delta
        if abs(delta) < 1e-15:
            break
    return g_on__uS * v_x


def compute_cell_current_range__uA(
    g_min__uS: float,
    g_max__uS: float,
    g_on__uS: float,
    v_bl__V: float,
    alpha: float,
) -> tuple[float, float]:
    """Per-cell on-state current at the (HRS, LRS) RRAM endpoints."""
    i_min__uA = solve_cell_current__uA(g_min__uS, g_on__uS, v_bl__V, alpha)
    i_max__uA = solve_cell_current__uA(g_max__uS, g_on__uS, v_bl__V, alpha)
    return i_min__uA, i_max__uA


def compute_target_currents__uA(
    i_cell_min__uA: float,
    i_cell_max__uA: float,
    n_states: int,
) -> list[float]:
    """Linear ``I_cell`` ladder from HRS to LRS with ``n_states`` rungs."""
    if n_states < 2:
        raise ValueError(f"n_states ({n_states}) must be >= 2")
    step = (i_cell_max__uA - i_cell_min__uA) / (n_states - 1)
    return [i_cell_min__uA + k * step for k in range(n_states)]


def invert_to_g_rram__uS(
    i_target__uA: float,
    g_on__uS: float,
    v_bl__V: float,
    alpha: float,
) -> float:
    """Closed-form ``g_rram`` that produces ``I_target`` through the cell.

    Inverts the cascade in one step: ``V_X`` is fixed by the NMOS
    branch (``V_X = I/g_on``), then ``g_rram`` follows directly from
    the RRAM I-V evaluated at ``V_rram = V_BL − V_X``.
    """
    v_x = i_target__uA / g_on__uS
    v_rram = v_bl__V - v_x
    if v_rram <= 0.0:
        raise ValueError(
            f"V_rram = V_BL - V_X = {v_rram:.3e} V <= 0; target current "
            f"{i_target__uA:.3e} uA exceeds the NMOS budget at V_BL={v_bl__V:.3e} V."
        )
    if alpha == 0.0:
        return i_target__uA / v_rram
    return i_target__uA * alpha / math.sinh(alpha * v_rram)


def compute_state_to_g__uS(
    target_currents__uA: list[float],
    g_on__uS: float,
    v_bl__V: float,
    alpha: float,
) -> list[float]:
    """Pre-distorted per-state RRAM conductance list [uS]."""
    return [invert_to_g_rram__uS(i, g_on__uS, v_bl__V, alpha) for i in target_currents__uA]


def _log_step_inputs(
    nmos_cfg: NMOSConfig,
    core_cfg: CircuitCore1T1RConfig,
    v_dd_wl__V: float,
    g_on__uS: float,
    g_min__uS: float,
    g_max__uS: float,
    v_bl__V: float,
    i_cell_min__uA: float,
    i_cell_max__uA: float,
    n_states: int,
    target_currents__uA: list[float],
    state_to_g__uS: list[float],
    row_num: int,
    i_max_diff__uA: float,
    n_codes: int,
    boundaries__uA: list[float],
) -> None:
    """Emit step-by-step derivation traces (idempotent, log-only)."""
    logger.info(
        "step 1: build NMOS model from config (W/L=%s/%s um, V_th=%s V)",
        _FMT.format(core_cfg.access_nmos_W__um),
        _FMT.format(core_cfg.access_nmos_L__um),
        _FMT.format(nmos_cfg.vth0__V),
    )

    logger.info(
        "step 2: nmos g_on at WL drive: V_DD,WL=%s V, g_on=%s uS", _FMT.format(v_dd_wl__V), _FMT.format(g_on__uS)
    )

    logger.info(
        "step 3: cell current range: g_min=%s uS, g_max=%s uS, V_BL=%s V, I_cell_min=%s uA, I_cell_max=%s uA",
        _FMT.format(g_min__uS),
        _FMT.format(g_max__uS),
        _FMT.format(v_bl__V),
        _FMT.format(i_cell_min__uA),
        _FMT.format(i_cell_max__uA),
    )

    logger.info("step 4: target current ladder (n_states=%d): %s", n_states, _fmt_list(target_currents__uA))

    logger.info("step 5: pre-distorted RRAM conductances (n_states=%d): %s", n_states, _fmt_list(state_to_g__uS))

    logger.info("step 6: max column diff current: row_num=%d, I_max_diff=%s uA", row_num, _FMT.format(i_max_diff__uA))

    logger.info("step 7: linear ADC boundaries (n_codes=%d): %s", n_codes, _fmt_list(boundaries__uA))


def _log_final(state_to_g__uS: list[float], boundaries__uA: list[float]) -> None:
    """Emit the two final paste-ready lists."""
    logger.info("---")
    logger.info("state_to_g__uS = %s  # %d-state", _fmt_list(state_to_g__uS), len(state_to_g__uS))
    logger.info("boundaries = %s  # %d-code", _fmt_list(boundaries__uA), len(boundaries__uA) + 1)


def _run(
    config: Path,
    g_min__uS: float,
    g_max__uS: float,
    n_states: int,
    n_codes: int,
    *,
    quiet: bool,
) -> None:
    """Top-level driver: load config, run all 7 steps, emit results."""
    cfg = dict_configs_from_file(_SPECS, config)
    nmos_cfg: NMOSConfig = cfg["nmos"]
    core_cfg: CircuitCore1T1RConfig = cfg["core"]
    wl_dac: GeneralDACConfig = cfg["wl_dac"]
    bl_adc: GeneralADCConfig = cfg["bl_adc"]
    xbar_cfg: Offset1T1RXbarConfig = cfg["xbar"]
    rram_cfg: RRAMConfig = cfg["rram"]

    v_dd_wl__V = wl_dac.code_to_signal[1]
    v_bl__V = bl_adc.drive_value
    alpha = rram_cfg.nonlinearity_alpha
    row_num = xbar_cfg.row_num

    # 1. Build NMOS model.
    build_nmos(nmos_cfg, core_cfg)

    # 2. NMOS deep-triode g_on at the configured WL on-level.
    g_on__uS = compute_g_on__uS(nmos_cfg, core_cfg, v_dd_wl__V)

    # 3. Cell current range at the (gmin, gmax) endpoints.
    i_cell_min__uA, i_cell_max__uA = compute_cell_current_range__uA(g_min__uS, g_max__uS, g_on__uS, v_bl__V, alpha)

    # 4. Linear current ladder.
    target_currents__uA = compute_target_currents__uA(i_cell_min__uA, i_cell_max__uA, n_states)

    # 5. Inverted per-state RRAM conductance.
    state_to_g__uS = compute_state_to_g__uS(target_currents__uA, g_on__uS, v_bl__V, alpha)

    # 6. Max column difference current.
    i_max_diff__uA = compute_max_col_diff_current__uA(row_num, i_cell_max__uA, i_cell_min__uA)

    # 7. Linear ADC boundaries (round-quantization).
    boundaries__uA = compute_adc_boundaries__uA(i_max_diff__uA, n_codes)

    if not quiet:
        _log_step_inputs(
            nmos_cfg,
            core_cfg,
            v_dd_wl__V,
            g_on__uS,
            g_min__uS,
            g_max__uS,
            v_bl__V,
            i_cell_min__uA,
            i_cell_max__uA,
            n_states,
            target_currents__uA,
            state_to_g__uS,
            row_num,
            i_max_diff__uA,
            n_codes,
            boundaries__uA,
        )
    _log_final(state_to_g__uS, boundaries__uA)


def _build_parser() -> argparse.ArgumentParser:
    """CLI parser — every argument is required (no defaults)."""
    parser = argparse.ArgumentParser(
        description="Compute pre-distorted RRAM state_to_g__uS and ADC boundaries for a 1T1R tile.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Chip TOML (NMOS + array + drive levels).")
    parser.add_argument("--gmin", type=float, required=True, help="HRS endpoint RRAM conductance [uS].")
    parser.add_argument("--gmax", type=float, required=True, help="LRS endpoint RRAM conductance [uS].")
    parser.add_argument("--n-states", type=int, required=True, help="Number of RRAM conductance states.")
    parser.add_argument("--n-codes", type=int, required=True, help="Number of ADC output codes.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step trace; print only final lists.")
    return parser


def main() -> None:
    """Console entry point: parse CLI, configure logging, dispatch ``_run``."""
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _run(
        config=args.config,
        g_min__uS=args.gmin,
        g_max__uS=args.gmax,
        n_states=args.n_states,
        n_codes=args.n_codes,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
