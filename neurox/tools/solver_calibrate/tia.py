"""Calibrate the OpAmpTIA's Newton iteration count.

Single-axis sweep over ``n_newton`` using **step-ratio plateau detection**
(primary) plus a **relative residual guard** (sanity). Same chip-parameter-free
methodology as :mod:`solver_calibrate.full_jacobian` / :mod:`.nested`:

  * Plateau is read from the TIA's own ``V_clamp`` iterate sequence —
    when ``|V_CL,n − V_CL,n-1|`` stops shrinking, the inner Newton has
    hit fp round-off and further iterations don't change the answer.

  * Residual guard: ``|Ids − I_port| / |I_port,typ| < reltol``
    (default ``1e-2``) where ``I_port,typ = max|I_port|`` over the
    workload — workload-derived, not chip-tuned.

CLI: ``python -m neurox.tools.solver_calibrate.tia --help``
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor

from neurox.analog.tia import OpAmpTIA, OpAmpTIAConfig, OpAmpTIAPolicy
from neurox.device import NMOSPolicy
from neurox.tools._config import add_standard_args, load_tool_config, setup_logging
from neurox.xbar import Offset1T1RXbarConfig

from ._plateau import CandidateRow, WorkloadScale, pick_with_plateau_and_guard


# ---------------------------------------------------------------------------
# TOML config schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _WorkloadCfg:
    """``[workload]`` section: synthetic Gaussian port-current draw."""

    mean__uA: float
    sigma__uA: float
    n_sigma_span: float
    n_samples: int


@dataclass(frozen=True)
class _SweepCfg:
    """``[sweep]`` section: candidate iteration counts + plateau / guard knobs."""

    candidates: list[int]
    ratio_threshold: float
    reltol: float
    margin: int


@dataclass(frozen=True)
class _RuntimeCfg:
    """``[runtime]`` section: numerical floats reproducibility knobs."""

    dtype: str
    seed: int


@dataclass(frozen=True)
class SolverCalibrateTiaConfig:
    """Top-level config for :mod:`neurox.tools.solver_calibrate.tia`."""

    xbar: Offset1T1RXbarConfig
    workload: _WorkloadCfg
    sweep: _SweepCfg
    runtime: _RuntimeCfg

log = logging.getLogger(__name__)


def _build_tia(
    tia_config: OpAmpTIAConfig,
    *,
    n_newton: int,
    device: torch.device,
    dtype: torch.dtype,
) -> OpAmpTIA:
    """Build a noise-off TIA with the sweep's iteration count."""
    cfg = replace(tia_config, n_newton=n_newton)
    policy = OpAmpTIAPolicy(
        opamp_gain_sigma=False,
        nmos=NMOSPolicy(A_vt_mismatch=False, A_beta_mismatch=False),
    )
    tia = OpAmpTIA(
        config=cfg,
        policy=policy,
        name="probe",
        inst_shape=(1,),
        dtype=dtype,
        T__K=300.0,
    )
    tia.to(device)
    tia.eval()
    tia.fabricate()
    return tia


def _sample_port_currents(
    *,
    mean__uA: float,
    sigma__uA: float,
    n_sigma_span: float,
    n_samples: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Tensor:
    """Mix a linspace sweep (covers tails) + Gaussian draw (fills bulk)."""
    half_span = n_sigma_span * sigma__uA
    n_line = max(8, n_samples // 4)
    line = torch.linspace(mean__uA - half_span, mean__uA + half_span, n_line, dtype=dtype, device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    n_gauss = n_samples - n_line
    gauss = mean__uA + sigma__uA * torch.randn(n_gauss, dtype=dtype, device=device, generator=g)
    return torch.cat([line, gauss])


def sweep_n_newton(
    *,
    tia_config: OpAmpTIAConfig,
    candidates: list[int],
    port_currents__uA: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[CandidateRow], WorkloadScale]:
    """Run TIA at each candidate ``n_newton`` and collect step + residual stats.

    TIA has one unknown per sample (``V_clamp``), so the step-delta is a
    scalar comparison. The residual is the cell-style absolute
    ``|Ids − I_port|`` returned in ``OpAmpTIADCOP.residual__uA``.
    """
    rows: list[CandidateRow] = []
    i_port_typ__uA = float(port_currents__uA.abs().max().item())
    v_clamp_prev: Tensor | None = None
    for n_newton in candidates:
        tia = _build_tia(tia_config, n_newton=n_newton, device=device, dtype=dtype)
        snap = tia.snapshot(shape=(port_currents__uA.shape[0],))
        dcop = tia.solve_dc(port_currents__uA, snap, v_clamp_init__V=None)
        v_clamp = dcop.v_clamp__V.detach()
        residual_max__uA = float(dcop.residual__uA.abs().max().item())

        if v_clamp_prev is None:
            step_max__V: float | None = None
        else:
            step_max__V = float((v_clamp - v_clamp_prev).abs().max().item())

        rows.append(
            CandidateRow(
                iter_count=n_newton,
                step_max__V=step_max__V,
                step_per_class__V={"v_clamp": step_max__V if step_max__V is not None else 0.0},
                residual_max={"cell__uA": residual_max__uA},
            )
        )
        v_clamp_prev = v_clamp

    # TIA has no v_node concept; clamp-residual key is absent, so a
    # voltage scale isn't needed. Pass a dummy value to satisfy the
    # dataclass — the consumer ignores v_node_typ__V for this workload.
    scale = WorkloadScale(
        i_cell_typ__uA=i_port_typ__uA,
        v_node_typ__V=1.0,
    )
    return rows, scale


def _format_row(row: CandidateRow) -> str:
    step = f"{row.step_max__V:9.2e}" if row.step_max__V is not None else "     ---"
    return f"n_newton={row.iter_count:3d}  step_v_clamp={step}  residual.max={row.residual_max['cell__uA']:9.2e} μA"


def plot_sweep(
    rows: list[CandidateRow],
    scale: WorkloadScale,
    *,
    out_path: Path,
    reltol: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs_all = [r.iter_count for r in rows]
    fig, (ax_step, ax_res) = plt.subplots(1, 2, figsize=(11, 4))

    steps = [r.step_max__V for r in rows if r.step_max__V is not None]
    step_xs = [r.iter_count for r in rows if r.step_max__V is not None]
    ax_step.plot(step_xs, steps, marker="o")
    ax_step.set_yscale("log")
    ax_step.set_xlabel("n_newton")
    ax_step.set_ylabel("max |V_clamp,n − V_clamp,n−1| [V]")
    ax_step.set_title("V_clamp step (plateau ≡ convergence)")
    ax_step.grid(True, which="both", ls=":", lw=0.4)

    res_ys = [r.residual_max["cell__uA"] for r in rows]
    ax_res.plot(xs_all, res_ys, marker="o")
    ax_res.axhline(
        reltol * scale.i_cell_typ__uA,
        ls="--",
        color="gray",
        lw=0.7,
        label=f"guard ({reltol:.1e} × max|I_port| = {reltol * scale.i_cell_typ__uA:.2e} μA)",
    )
    ax_res.set_yscale("log")
    ax_res.set_xlabel("n_newton")
    ax_res.set_ylabel("max |Ids − I_port| [μA]")
    ax_res.set_title("Residual stats")
    ax_res.grid(True, which="both", ls=":", lw=0.4)
    ax_res.legend(fontsize="small")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate OpAmpTIA n_newton via step-ratio plateau.")
    add_standard_args(parser, plot_dir=True)
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    cfg = load_tool_config(SolverCalibrateTiaConfig, args.config)
    log.info("loaded config from %s", args.config)

    tia_config = cfg.xbar.core_config.tia_config
    if not isinstance(tia_config, OpAmpTIAConfig):
        raise SystemExit(f"TIA config must be OpAmpTIAConfig; got {type(tia_config).__name__}")
    dtype = torch.float32 if cfg.runtime.dtype == "float32" else torch.float64
    device = torch.device(args.device)

    log.info("=" * 80)
    log.info("OpAmpTIA — step-ratio plateau calibration")
    log.info(
        "workload: I_port ~ N(%.3f, %.3f) μA, span=±%.1fσ, %d samples",
        cfg.workload.mean__uA,
        cfg.workload.sigma__uA,
        cfg.workload.n_sigma_span,
        cfg.workload.n_samples,
    )
    log.info(
        "criteria: ratio_threshold=%.3f, reltol=%.1e, margin=%d",
        cfg.sweep.ratio_threshold,
        cfg.sweep.reltol,
        cfg.sweep.margin,
    )
    log.info("=" * 80)

    port_currents = _sample_port_currents(
        mean__uA=cfg.workload.mean__uA,
        sigma__uA=cfg.workload.sigma__uA,
        n_sigma_span=cfg.workload.n_sigma_span,
        n_samples=cfg.workload.n_samples,
        device=device,
        dtype=dtype,
        seed=cfg.runtime.seed,
    )

    rows, scale = sweep_n_newton(
        tia_config=tia_config,
        candidates=cfg.sweep.candidates,
        port_currents__uA=port_currents,
        device=device,
        dtype=dtype,
    )

    log.info("workload scale: max|I_port|=%.3e μA", scale.i_cell_typ__uA)
    log.info("")
    for r in rows:
        log.info(_format_row(r))
    log.info("")

    pick = pick_with_plateau_and_guard(
        rows,
        scale,
        ratio_threshold=cfg.sweep.ratio_threshold,
        reltol=cfg.sweep.reltol,
    )

    if pick.iter_count is None:
        log.error("Calibration failed: %s", pick.reason)
        return 2

    final = pick.iter_count + cfg.sweep.margin
    log.info("=" * 80)
    log.info("Picked n_newton = %d  (%s)", pick.iter_count, pick.reason)
    log.info("Recommended with margin %d: n_newton = %d", cfg.sweep.margin, final)
    log.info(
        "Residual guard ratio at pick: cell=%.3e  (reltol = %.1e)",
        pick.residual_guard_ratios.get("cell__uA", 0.0),
        cfg.sweep.reltol,
    )
    log.info("")
    log.info("TOML fragment for chip preset [xbar.core_config.tia_config]:")
    log.info("    n_newton = %d", final)
    log.info("=" * 80)

    if args.plot_dir is not None:
        plot_sweep(rows, scale, out_path=args.plot_dir / "tia_n_newton_sweep.png", reltol=cfg.sweep.reltol)
        log.info("Plot written to %s", args.plot_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
