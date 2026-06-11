"""Single-cell 1T1R RRAM state-map optimizer.

See also:
    docs/dev/modules/tools/calculate_1t1r_states.md
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from neurox.device import (
    NMOS,
    RRAM,
    NMOSConfig,
    NMOSPolicy,
    NMOSSnapshot,
    RRAMConfig,
    RRAMPolicy,
    RRAMSnapshot,
)
from neurox.tools._config import add_standard_args, load_tool_config, setup_logging


# ---------------------------------------------------------------------------
# TOML config schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BiasCfg:
    """``[bias]`` section: per-cell read bias + operating temperature."""

    v_wl__V: float
    v_bl__V: float
    v_sl__V: float
    temperature__K: float


@dataclass(frozen=True)
class _DesignCfg:
    """``[design]`` section: RRAM ladder + access-NMOS sizing."""

    g_max__uS: float
    n_states: int
    access_nmos_W__um: float
    access_nmos_L__um: float


@dataclass(frozen=True)
class Calculate1T1RStatesConfig:
    """Top-level config for :mod:`neurox.tools.calculate_1t1r_states`.

    The ``[rram]`` and ``[nmos]`` sections typically use
    ``_neurox_use_preset`` to pull from ``neurox/presets/process/``.
    """

    rram: RRAMConfig
    nmos: NMOSConfig
    bias: _BiasCfg
    design: _DesignCfg

logger = logging.getLogger(__name__)

# Numerical constants — tight enough for float64; not exposed via CLI.
_MAX_ITER_VX = 100
_MAX_ITER_G = 100
_TOL_VX__V = 1e-12
_TOL_G__uS = 1e-9
_TOL_I__uA = 1e-9

_DTYPE = torch.float64


@dataclass(frozen=True)
class _CellBias:
    v_wl__V: float
    v_bl__V: float
    v_sl__V: float


@dataclass(frozen=True)
class _StateSolveResult:
    rram_g_max__uS: float
    state_to_g_map__uS: list[float]
    target_currents__uA: list[float]
    solved_currents__uA: list[float]
    solved_v_x__V: list[float]
    max_abs_current_error__uA: float


def _all_off_rram_policy() -> RRAMPolicy:
    return RRAMPolicy(prog_gamma=False, stuck_at=False, read_telegraph=False, read_thermal=False)


def _all_off_nmos_policy() -> NMOSPolicy:
    return NMOSPolicy(A_vt_mismatch=False, A_beta_mismatch=False)


def _program_and_snapshot(rram: RRAM, g__uS: float) -> RRAMSnapshot:
    """Program ``rram`` to ``g__uS`` (no drift / nonidealities) and snapshot once."""
    target = torch.tensor(g__uS, dtype=_DTYPE)
    rram.program(target, t_elapsed=0.0)
    return rram.snapshot(shape=())


def _solve_cell_current__uA(
    *,
    rram: RRAM,
    nmos: NMOS,
    nmos_snapshot: NMOSSnapshot,
    g__uS: float,
    bias: _CellBias,
) -> tuple[float, float]:
    """Solve KCL for one fixed RRAM conductance and return ``(I_cell, V_X)``.

    Uses bisection on ``V_X in [V_SL, V_BL]`` with
    ``f(V_X) = I_RRAM(V_BL - V_X; g) - I_NMOS(V_WL, V_X, V_SL)``.
    """
    rram_snapshot = _program_and_snapshot(rram, g__uS)

    def f(v_x__V: float) -> float:
        i_rram__uA = float(rram.solve_dc(torch.tensor(bias.v_bl__V - v_x__V, dtype=_DTYPE), rram_snapshot).i__uA.item())
        i_nmos__uA = float(
            nmos.solve_dc(
                vg__V=bias.v_wl__V,
                vd__V=v_x__V,
                vs__V=bias.v_sl__V,
                snapshot=nmos_snapshot,
            ).ids__uA.item()
        )
        return i_rram__uA - i_nmos__uA

    lo, hi = bias.v_sl__V, bias.v_bl__V
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        raise RuntimeError(
            f"V_X bracket has same-sign residual: g={g__uS} uS, bias=(WL={bias.v_wl__V}, "
            f"BL={bias.v_bl__V}, SL={bias.v_sl__V}), f(SL)={f_lo:.3e}, f(BL)={f_hi:.3e}"
        )

    f_mid = 0.0
    v_mid = 0.0
    for _ in range(_MAX_ITER_VX):
        v_mid = 0.5 * (lo + hi)
        f_mid = f(v_mid)
        if abs(f_mid) <= _TOL_I__uA or (hi - lo) <= _TOL_VX__V:
            v_x__V = v_mid
            i_cell__uA = float(
                nmos.solve_dc(
                    vg__V=bias.v_wl__V,
                    vd__V=v_x__V,
                    vs__V=bias.v_sl__V,
                    snapshot=nmos_snapshot,
                ).ids__uA.item()
            )
            return i_cell__uA, v_x__V
        if f_lo * f_mid < 0:
            hi, f_hi = v_mid, f_mid
        else:
            lo, f_lo = v_mid, f_mid

    raise RuntimeError(
        f"V_X bisection failed to converge in {_MAX_ITER_VX} iters: g={g__uS} uS, "
        f"residual={f_mid:.3e}, bracket width={hi - lo:.3e}"
    )


def _solve_g_for_target_current__uS(
    *,
    rram: RRAM,
    nmos: NMOS,
    nmos_snapshot: NMOSSnapshot,
    target_i__uA: float,
    g_lo__uS: float,
    g_hi__uS: float,
    bias: _CellBias,
) -> float:
    """Bisect on ``g in [g_lo, g_hi]`` to find ``I_cell(g) == target_i``."""

    def h(g__uS: float) -> float:
        i_cell__uA, _ = _solve_cell_current__uA(
            rram=rram, nmos=nmos, nmos_snapshot=nmos_snapshot, g__uS=g__uS, bias=bias
        )
        return i_cell__uA - target_i__uA

    lo, hi = g_lo__uS, g_hi__uS
    h_lo, h_hi = h(lo), h(hi)
    if h_lo * h_hi > 0:
        raise RuntimeError(
            f"g bracket has same-sign residual: target={target_i__uA} uA, h(g_min)={h_lo:.3e}, h(g_max)={h_hi:.3e}"
        )

    h_mid = 0.0
    for _ in range(_MAX_ITER_G):
        g_mid = 0.5 * (lo + hi)
        h_mid = h(g_mid)
        if abs(h_mid) <= _TOL_I__uA or (hi - lo) <= _TOL_G__uS:
            return g_mid
        if h_lo * h_mid < 0:
            hi, h_hi = g_mid, h_mid
        else:
            lo, h_lo = g_mid, h_mid

    raise RuntimeError(
        f"g bisection failed to converge in {_MAX_ITER_G} iters: target={target_i__uA} uA, "
        f"residual={h_mid:.3e}, bracket width={hi - lo:.3e}"
    )


def calculate_state_map(
    *,
    rram_config: RRAMConfig,
    nmos_config: NMOSConfig,
    bias: _CellBias,
    g_max__uS: float,
    n_states: int,
    access_nmos_W__um: float,
    access_nmos_L__um: float,
    temperature_K: float,
) -> _StateSolveResult:
    """Run the full state-map optimization and return the solved table."""
    rram = RRAM(
        config=rram_config,
        policy=_all_off_rram_policy(),
        inst_shape=(),
        dtype=_DTYPE,
        T__K=temperature_K,
        g_max__uS=g_max__uS,
    )
    nmos = NMOS(
        config=nmos_config,
        policy=_all_off_nmos_policy(),
        inst_shape=(),
        dtype=_DTYPE,
        T__K=temperature_K,
        W__um=access_nmos_W__um,
        L__um=access_nmos_L__um,
    )
    rram.eval()
    nmos.eval()
    nmos.fabricate()  # populate nominal buffers (mismatch disabled → deterministic)
    nmos_snapshot = nmos.snapshot(shape=())

    g_min__uS = rram_config.g_min__uS

    i_min__uA, vx_min__V = _solve_cell_current__uA(
        rram=rram, nmos=nmos, nmos_snapshot=nmos_snapshot, g__uS=g_min__uS, bias=bias
    )
    i_max__uA, vx_max__V = _solve_cell_current__uA(
        rram=rram, nmos=nmos, nmos_snapshot=nmos_snapshot, g__uS=g_max__uS, bias=bias
    )
    if not (i_max__uA > i_min__uA):
        raise RuntimeError(
            f"endpoint monotonicity failed: I_cell(g_min={g_min__uS}) = {i_min__uA:.4e}, "
            f"I_cell(g_max={g_max__uS}) = {i_max__uA:.4e}"
        )

    logger.info("endpoint @ g_min: V_X = %.10f V, I_cell = %.10f uA", vx_min__V, i_min__uA)
    logger.info("endpoint @ g_max: V_X = %.10f V, I_cell = %.10f uA", vx_max__V, i_max__uA)
    logger.info("I_min__uA = %.10f", i_min__uA)
    logger.info("I_max__uA = %.10f", i_max__uA)
    logger.info("I_max / I_min = %.6f", i_max__uA / i_min__uA)

    # End-slope diagnostic: dI/dg at g_max via a small backward step.
    g_probe__uS = g_max__uS - max(1e-3 * (g_max__uS - g_min__uS), 1e-9)
    i_probe__uA, _ = _solve_cell_current__uA(
        rram=rram, nmos=nmos, nmos_snapshot=nmos_snapshot, g__uS=g_probe__uS, bias=bias
    )
    dI_dg = (i_max__uA - i_probe__uA) / (g_max__uS - g_probe__uS)
    logger.info("dI/dg @ g_max ≈ %.6e uA/uS", dI_dg)

    targets__uA: list[float] = [i_min__uA + k * (i_max__uA - i_min__uA) / (n_states - 1) for k in range(n_states)]
    logger.info("target_currents__uA = %s", targets__uA)

    state_to_g__uS: list[float] = [0.0] * n_states
    state_to_g__uS[0] = g_min__uS
    state_to_g__uS[n_states - 1] = g_max__uS

    solved_i__uA: list[float] = [0.0] * n_states
    solved_vx__V: list[float] = [0.0] * n_states
    solved_i__uA[0] = i_min__uA
    solved_vx__V[0] = vx_min__V
    solved_i__uA[n_states - 1] = i_max__uA
    solved_vx__V[n_states - 1] = vx_max__V

    for k in range(1, n_states - 1):
        g_k__uS = _solve_g_for_target_current__uS(
            rram=rram,
            nmos=nmos,
            nmos_snapshot=nmos_snapshot,
            target_i__uA=targets__uA[k],
            g_lo__uS=g_min__uS,
            g_hi__uS=g_max__uS,
            bias=bias,
        )
        i_k__uA, vx_k__V = _solve_cell_current__uA(
            rram=rram, nmos=nmos, nmos_snapshot=nmos_snapshot, g__uS=g_k__uS, bias=bias
        )
        state_to_g__uS[k] = g_k__uS
        solved_i__uA[k] = i_k__uA
        solved_vx__V[k] = vx_k__V

    logger.info("state derivation trace:")
    for k in range(n_states):
        snap = " (snapped to g_min)" if k == 0 else " (snapped to g_max)" if k == n_states - 1 else ""
        logger.info(
            "  state %d: target = %.10f uA, g = %.10f uS, solved = %.10f uA, V_X = %.10f V%s",
            k,
            targets__uA[k],
            state_to_g__uS[k],
            solved_i__uA[k],
            solved_vx__V[k],
            snap,
        )

    # Solver-output validation.
    for k in range(1, n_states):
        if not (state_to_g__uS[k] > state_to_g__uS[k - 1]):
            raise RuntimeError(
                f"state_to_g_map is not strictly increasing at index {k}: "
                f"{state_to_g__uS[k - 1]} >= {state_to_g__uS[k]}"
            )
    for k in range(n_states):
        if not (g_min__uS <= state_to_g__uS[k] <= g_max__uS):
            raise RuntimeError(
                f"state_to_g_map[{k}] = {state_to_g__uS[k]} outside [g_min={g_min__uS}, g_max={g_max__uS}]"
            )

    abs_err__uA = [abs(solved_i__uA[k] - targets__uA[k]) for k in range(n_states)]
    max_err__uA = max(abs_err__uA)
    if not (max_err__uA <= _TOL_I__uA):
        raise RuntimeError(
            f"solver self-consistency failed: max |solved - target| = {max_err__uA:.3e} > tol = {_TOL_I__uA:.3e}"
        )

    return _StateSolveResult(
        rram_g_max__uS=g_max__uS,
        state_to_g_map__uS=state_to_g__uS,
        target_currents__uA=targets__uA,
        solved_currents__uA=solved_i__uA,
        solved_v_x__V=solved_vx__V,
        max_abs_current_error__uA=max_err__uA,
    )


def main(argv: list[str] | None = None) -> int:
    """Console entry point."""
    parser = argparse.ArgumentParser(
        description="Solve the per-state RRAM conductance ladder yielding linear 1T1R cell current.",
    )
    add_standard_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    cfg = load_tool_config(Calculate1T1RStatesConfig, args.config)

    design = cfg.design
    bias_cfg = cfg.bias
    if design.n_states < 2:
        raise SystemExit(f"[design].n_states ({design.n_states}) must be >= 2")
    if not (design.g_max__uS > 0):
        raise SystemExit(f"[design].g_max__uS ({design.g_max__uS}) must be > 0")
    if not (bias_cfg.v_bl__V > bias_cfg.v_sl__V):
        raise SystemExit(f"[bias].v_bl__V ({bias_cfg.v_bl__V}) must be > [bias].v_sl__V ({bias_cfg.v_sl__V})")
    if not (design.access_nmos_W__um > 0):
        raise SystemExit(f"[design].access_nmos_W__um ({design.access_nmos_W__um}) must be > 0")
    if not (design.access_nmos_L__um > 0):
        raise SystemExit(f"[design].access_nmos_L__um ({design.access_nmos_L__um}) must be > 0")
    if not (bias_cfg.temperature__K > 0):
        raise SystemExit(f"[bias].temperature__K ({bias_cfg.temperature__K}) must be > 0")

    bias = _CellBias(v_wl__V=bias_cfg.v_wl__V, v_bl__V=bias_cfg.v_bl__V, v_sl__V=bias_cfg.v_sl__V)

    logger.info("loaded config from %s", args.config)
    logger.info("disabled rram non-idealities: prog_gamma, stuck_at, read_telegraph, read_thermal")
    logger.info("disabled nmos non-idealities: A_vt_mismatch, A_beta_mismatch")
    logger.info("rram g_min__uS = %s", cfg.rram.g_min__uS)
    logger.info("rram g_max__uS = %s", design.g_max__uS)
    logger.info("bias V_WL = %s V, V_BL = %s V, V_SL = %s V", bias.v_wl__V, bias.v_bl__V, bias.v_sl__V)
    logger.info(
        "access NMOS W = %s um, L = %s um, T = %s K",
        design.access_nmos_W__um,
        design.access_nmos_L__um,
        bias_cfg.temperature__K,
    )

    result = calculate_state_map(
        rram_config=cfg.rram,
        nmos_config=cfg.nmos,
        bias=bias,
        g_max__uS=design.g_max__uS,
        n_states=design.n_states,
        access_nmos_W__um=design.access_nmos_W__um,
        access_nmos_L__um=design.access_nmos_L__um,
        temperature_K=bias_cfg.temperature__K,
    )

    logger.info("max_abs_current_error__uA = %.3e", result.max_abs_current_error__uA)
    logger.info("")
    logger.info("rram_g_max__uS = %s", result.rram_g_max__uS)
    logger.info("state_to_g_map__uS = %s", result.state_to_g_map__uS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
