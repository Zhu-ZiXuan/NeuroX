"""Shared helpers for the macro_tia tool.

Build an :class:`OpAmpTIA` from a parameter set, sweep its ``I_port → v_out``
DC transfer curve, and compute scoring metrics against a Gaussian workload
current model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neurox.primitive.analog.tia import OpAmpTIA, OpAmpTIAConfig, OpAmpTIAPolicy
from neurox.primitive.physical_constant import T_ROOM__K
from neurox.primitive.device import MOSFETPolicy


def build_tia(config: OpAmpTIAConfig, *, device: torch.device) -> OpAmpTIA:
    """Build a fabricated, nonideality-free :class:`OpAmpTIA` for sweeping."""
    tia_policy = OpAmpTIAPolicy(
        opamp_gain_sigma=False,
        nmos=MOSFETPolicy(A_vt_mismatch=False, A_beta_mismatch=False),
    )
    tia = OpAmpTIA(
        config=config,
        policy=tia_policy,
        name="probe",
        inst_shape=(1,),
        dtype=torch.float64,
        T__K=T_ROOM__K,
    )
    tia.to(device)
    tia.eval()
    tia.fabricate()
    return tia


@dataclass(frozen=True)
class TransferCurve:
    """DC ``I_port → v_out`` sweep result for one TIA instance.

    Attributes:
        i_uA: 1-D sweep currents [uA], shape ``(N,)``.
        v_out_V: Output voltages [V], shape ``(N,)``.
        v_min_V / v_max_V: ``softclip_center -/+ softclip_half_span`` for reference.
    """

    i_uA: torch.Tensor
    v_out_V: torch.Tensor
    v_min_V: float
    v_max_V: float

    def slope_at(self, i_uA: float) -> float:
        """Local derivative ``dv_out/dI`` [V/uA] at the closest grid point."""
        idx = int(torch.argmin((self.i_uA - i_uA).abs()).item())
        if idx == 0:
            return float((self.v_out_V[1] - self.v_out_V[0]) / (self.i_uA[1] - self.i_uA[0]))
        if idx == len(self.i_uA) - 1:
            return float((self.v_out_V[-1] - self.v_out_V[-2]) / (self.i_uA[-1] - self.i_uA[-2]))
        return float((self.v_out_V[idx + 1] - self.v_out_V[idx - 1]) / (self.i_uA[idx + 1] - self.i_uA[idx - 1]))

    def v_at(self, i_uA: float) -> float:
        """Output voltage at the closest sweep grid point."""
        idx = int(torch.argmin((self.i_uA - i_uA).abs()).item())
        return float(self.v_out_V[idx].item())


def sweep_transfer(
    tia: OpAmpTIA,
    *,
    v_ref__V: float,
    i_min_uA: float,
    i_max_uA: float,
    n_points: int,
    device: torch.device,
) -> TransferCurve:
    """Sweep DC ``I_port`` across ``[i_min_uA, i_max_uA]`` and capture ``v_out``.

    The reference clamp voltage is injected per call into
    :meth:`OpAmpTIA.snapshot` as a 0-d tensor built from ``v_ref__V``
    (the design tool's externally-fixed ``[hardware].v_ref__V`` knob).
    """
    if n_points < 2:
        raise ValueError(f"n_points ({n_points}) must be >= 2")
    v_ref_tensor = torch.tensor(v_ref__V, dtype=torch.float64, device=device)
    snap = tia.snapshot(v_ref__V=v_ref_tensor, shape=(1,), multi_coords=None)
    i_grid = torch.linspace(i_min_uA, i_max_uA, n_points, dtype=torch.float64, device=device)
    v_out_list: list[float] = []
    for i_val in i_grid:
        dcop = tia.solve_dc(i_val.expand(1), snap, v_clamp_init__V=None)
        v_out_list.append(float(dcop.v_out__V.item()))
    v_out = torch.tensor(v_out_list, dtype=torch.float64)
    v_center = tia.softclip_center__V
    v_half = tia.softclip_half_span__V
    return TransferCurve(
        i_uA=i_grid.cpu(),
        v_out_V=v_out,
        v_min_V=v_center - v_half,
        v_max_V=v_center + v_half,
    )


def linearity_r2(curve: TransferCurve, *, lo_uA: float, hi_uA: float) -> float:
    """Pearson R² of ``v_out ≈ a · I + b`` over an explicit ``[lo_uA, hi_uA]``.

    1.0 = perfectly linear over the chosen range; approaching 0 means the
    TIA's softclip / NMOS nonlinearity has bent the response so any linear
    fit leaves big residuals.
    """
    mask = (curve.i_uA >= lo_uA) & (curve.i_uA <= hi_uA)
    n = int(mask.sum().item())
    if n < 2:
        return 0.0
    i_b = curve.i_uA[mask].to(torch.float64)
    v_b = curve.v_out_V[mask].to(torch.float64)
    i_mean = i_b.mean()
    v_mean = v_b.mean()
    cov = ((i_b - i_mean) * (v_b - v_mean)).sum().item()
    var_i = ((i_b - i_mean) ** 2).sum().item()
    var_v = ((v_b - v_mean) ** 2).sum().item()
    if var_i <= 0.0 or var_v <= 0.0:
        return 0.0
    return float((cov * cov) / (var_i * var_v))


def saturation_onset(curve: TransferCurve, *, frac: float = 0.99) -> float:
    """Lowest ``I_port`` [uA] at which ``v_out`` reaches ``frac × (v_max - v_min)`` above ``v_min``."""
    threshold = curve.v_min_V + frac * (curve.v_max_V - curve.v_min_V)
    above = (curve.v_out_V >= threshold).nonzero()
    if above.numel() == 0:
        return float(curve.i_uA[-1].item())
    return float(curve.i_uA[int(above[0].item())].item())


@dataclass(frozen=True)
class WorkloadFit:
    """Scoring metrics for one TIA against a Gaussian workload current model."""

    workload_mean__uA: float
    workload_std__uA: float
    v_at_mean__V: float
    v_at_lo3sigma__V: float
    v_at_hi3sigma__V: float
    slope_at_mean__mV_per_uA: float
    saturation_onset__uA: float
    sat_margin_above_3sigma__uA: float
    linearity_r2_over_useful_range: float


def fit_to_workload(curve: TransferCurve, *, mean_uA: float, std_uA: float) -> WorkloadFit:
    """Compute single-config metrics for a Gaussian workload ``N(mean_uA, std_uA²)``."""
    if std_uA <= 0:
        raise ValueError(f"std_uA ({std_uA}) must be > 0")
    lo3 = mean_uA - 3 * std_uA
    hi3 = mean_uA + 3 * std_uA
    sat_at = saturation_onset(curve)
    return WorkloadFit(
        workload_mean__uA=mean_uA,
        workload_std__uA=std_uA,
        v_at_mean__V=curve.v_at(mean_uA),
        v_at_lo3sigma__V=curve.v_at(lo3),
        v_at_hi3sigma__V=curve.v_at(hi3),
        slope_at_mean__mV_per_uA=curve.slope_at(mean_uA) * 1e3,
        saturation_onset__uA=sat_at,
        sat_margin_above_3sigma__uA=sat_at - hi3,
        linearity_r2_over_useful_range=linearity_r2(curve, lo_uA=0.0, hi_uA=sat_at),
    )
