"""Shared hand-built config / probe / reference helpers for the isub_iadc_1t1r scheme tests.

Every solve-bearing test builds through :func:`build_tile` from the
hand-constructed :func:`build_tiny_config` witness — every config dataclass is
built directly in Python with explicit small values; only the device
sub-configs (RRAM / NMOS) come from the packaged library presets. Activations
expand into zero-masked WL planes via :func:`masked_planes` (the engine mask
formula — the macro consumes pre-expanded planes and adds no axis of its own),
and the per-plane ADC codes are checked against
:func:`per_phase_clamp_reference` — the CPU int64 unit-role reference
``clamp(x_p . w_p, -MAG_MAX, MAG_MAX)`` per WL sub-phase.

The analog ``I_SUB(M)`` grid at the ADC input depends on the whole electrical
config, so the witness ships a placeholder ladder and tests that need a
bit-exact decode calibrate in-code through :func:`build_calibrated_tile`:
probe the tile's own ``I_SUB(M)`` grid (:func:`probe_i_sub_grid`), install the
mid-point thresholds (:func:`midpoint_refs` + :func:`with_ref_levels`), and
rebuild — a law-level calibration derived from the config under test, not
from any shipped numbers.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor

from neurox.primitive.analog import (
    CurrentMirrorConfig,
    CurrentMirrorPolicy,
    CurrentReferenceConfig,
    CurrentReferencePolicy,
    CurrentSubtractorConfig,
    CurrentSubtractorPolicy,
    VoltageDriverConfig,
    VoltageDriverPolicy,
    VoltageReferenceConfig,
    VoltageReferencePolicy,
)
from neurox.primitive.analog.adc_common import AdcCalibrationRecord, AdcOperationPoint
from neurox.primitive.analog.current_adc import SarCurrentAdcConfig, SarCurrentAdcPolicy
from neurox.primitive.analog.voltage_dac import GeneralVoltageDacConfig, GeneralVoltageDacPolicy
from neurox.primitive.device import MosfetConfig, MosfetPolicy, RramConfig, RramPolicy
from neurox.primitive.macro.cim import CimMacro
from neurox.primitive.xbar.array import XbarArray1t1rConfig, XbarArray1t1rPolicy, XbarArraySteadyState
from neurox.primitive.xbar.cell import XbarCell1t1rDetailConfig, XbarCell1t1rDetailPolicy
from neurox.primitive.xbar.solver import NestedParallelRailSolverConfig
from neurox.works.macro.cim.isub_iadc_1t1r.macro import (
    IsubIadc1t1rCimMacro,
    IsubIadc1t1rCimMacroConfig,
    IsubIadc1t1rCimMacroPolicy,
)

# --- Tiny witness geometry: col_num = 2 -> 4 physical columns, P = 2 WL
# sub-phases, n_lane = 2 front-end lanes per polarity, n_io = 1. ---
TINY_COL_NUM = 2
TINY_ROW_NUM = 16
TINY_ACTIVE_ROW_NUM = 8
TINY_PHASE_NUM = TINY_ROW_NUM // TINY_ACTIVE_ROW_NUM  # 2
TINY_MUX_FACTOR = 1  # n_lane = col_num // mux_factor = 2
TINY_IO_COL_NUM = 2  # n_io = col_num // io_col_num = 1

TINY_ADC_BITS = 3
MAG_MAX = (1 << TINY_ADC_BITS) - 1  # 7 — the 3-bit magnitude saturation
ADC_OP = AdcOperationPoint(adc_mode=0, adc_bits=TINY_ADC_BITS)  # witness operating point

# Placeholder single-mode ladder (2**n_bits - 1 strictly increasing levels)
# carried by the witness config; decode-bearing tests replace it with the
# in-code probed mid-points (build_calibrated_tile).
_SEED_REF_LEVELS = tuple(float(k) for k in range(1, 1 << TINY_ADC_BITS))


def build_tiny_config() -> IsubIadc1t1rCimMacroConfig:
    """Hand-built tiny witness config, constructed entirely in Python.

    Small round values chosen to satisfy every config validator; the device
    sub-configs come from the packaged library presets (the sole file-backed
    inputs). The ADC / CurrentReference ladder is the placeholder
    ``_SEED_REF_LEVELS`` (kept equal on both copies per the single-source
    law); calibrate it via :func:`build_calibrated_tile` before asserting
    decode values.
    """
    cell_config = XbarCell1t1rDetailConfig(
        rram_config=RramConfig.from_preset("process/rram:default"),
        nmos_config=MosfetConfig.from_preset("process/mos:nmos_28_rvt"),
        state_to_g_map__uS=(10.0, 100.0),  # state 0 = HRS (preset g_min floor), 1 = LRS
        access_nmos_W__um=0.2,
        access_nmos_L__um=0.03,
        rram_g_max__uS=100.0,
        n_newton=5,
        c_bl__fF=0.2,
        c_x__fF=0.3,
        c_sl__fF=0.1,
        c_wl__fF=0.2,
    )
    array_config = XbarArray1t1rConfig(
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=1.0,
        latency_per_op__ns=1.0,
        row_first_space__um=1.0,
        row_cell_space__um=0.05,
        col_first_space__um=1.0,
        col_cell_space__um=0.05,
        bl_first_r__MOhm=1e-7,
        bl_first_c__fF=0.4,
        bl_segment_r__MOhm=1e-8,
        bl_segment_c__fF=0.02,
        sl_first_r__MOhm=1e-7,
        sl_first_c__fF=0.4,
        sl_segment_r__MOhm=1e-8,
        sl_segment_c__fF=0.02,
        wl_first_r__MOhm=1e-6,
        wl_first_c__fF=0.2,
        wl_segment_r__MOhm=1e-7,
        wl_segment_c__fF=0.01,
        cell_config=cell_config,
        solver_config=NestedParallelRailSolverConfig(n_outer=20, n_inner=4),
    )
    return IsubIadc1t1rCimMacroConfig(
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=8.0,  # macro-owned lump (Control + mirror roll-up)
        col_num=TINY_COL_NUM,
        row_num=TINY_ROW_NUM,
        active_row_num=TINY_ACTIVE_ROW_NUM,
        mux_factor=TINY_MUX_FACTOR,
        io_col_num=TINY_IO_COL_NUM,
        adc_calibration=(AdcCalibrationRecord(adc_mode=0, adc_bits=TINY_ADC_BITS, rescale_factor=1.0),),
        array_config=array_config,
        wl_dac_config=GeneralVoltageDacConfig(
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
            code_to_signal=(0.0, 0.9),  # 1-bit ON/OFF WL drive
            drive_thermal__V=0.0,
            energy_per_op__fJ=0.0,  # driver core only; WL load caps are array-billed
            latency_per_op__ns=0.0,
        ),
        bl_clamp_config=VoltageDriverConfig(
            r_out__MOhm=1e-5,  # small clamp regulation residual (solver well-posedness)
            offset_sigma__V=0.0,
            thermal_sigma__V=0.0,
            energy_per_op__fJ=2.0,  # per-column CMD interface-node cycle
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=2.0,
        ),
        sl_driver_config=VoltageDriverConfig(
            r_out__MOhm=0.0,  # ideal flat clamp
            offset_sigma__V=0.0,
            thermal_sigma__V=0.0,
            energy_per_op__fJ=0.0,  # physical zero — direct ground tie
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        clamp_ref_config=VoltageReferenceConfig(
            v_refs__V=(0.3, 0.0),  # tap 0 = BL-clamp V_BLC, tap 1 = SL ground
            tolerance_sigma_relative=0.0,
            noise_sigma_relative=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        p_mirror_config=CurrentMirrorConfig(mirror_ratio=0.5, ratio_sigma_relative=0.0),
        n_mirror_config=CurrentMirrorConfig(mirror_ratio=0.5, ratio_sigma_relative=0.0),
        subtractor_config=CurrentSubtractorConfig(
            gain=1.0,
            mismatch_sigma_relative=0.0,
            offset_sigma__uA=0.0,
            v_rail__V=1.0,  # == v_dd__V (validate_supply)
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=2.0,
        ),
        adc_config=SarCurrentAdcConfig(
            n_bits=TINY_ADC_BITS,
            margin_gain=3.0,
            ref_levels__uA=(_SEED_REF_LEVELS,),
            e_fixed_per_op__fJ=2.0,
            step_latency__ns=(1.0, 1.0, 1.0),  # t_conduct__ns = 3.0 derived
            comparator_offset_sigma__uA=0.0,
            coupling_mismatch_sigma__uA=0.0,
            mirror_mismatch_sigma_relative=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=3.0,
        ),
        reference_config=CurrentReferenceConfig(
            i_refs__uA=(_SEED_REF_LEVELS,),  # == adc ladder (single source of truth)
            tolerance_sigma_relative=0.0,
            noise_sigma_relative=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=4.0,
        ),
        v_dd__V=1.0,
        readout_latency_per_op__ns=1.0,
    )


def build_all_off_policy() -> IsubIadc1t1rCimMacroPolicy:
    """All-off (lossless baseline) policy, constructed entirely in Python."""
    return IsubIadc1t1rCimMacroPolicy(
        array=XbarArray1t1rPolicy(
            cell=XbarCell1t1rDetailPolicy(
                rram=RramPolicy(prog_gamma=False, stuck_at=False, read_telegraph=False, read_thermal=False),
                nmos=MosfetPolicy(A_vt_mismatch=False, A_beta_mismatch=False),
            ),
            solve_chunk_size=0,
        ),
        wl_dac=GeneralVoltageDacPolicy(drive_thermal=False),
        bl_clamp=VoltageDriverPolicy(offset=False, thermal=False),
        sl_driver=VoltageDriverPolicy(offset=False, thermal=False),
        clamp_ref=VoltageReferencePolicy(tolerance=False, noise=False),
        p_mirror=CurrentMirrorPolicy(mismatch=False),
        n_mirror=CurrentMirrorPolicy(mismatch=False),
        subtractor=CurrentSubtractorPolicy(mismatch=False, offset=False),
        adc=SarCurrentAdcPolicy(
            comparator_offset=False,
            replica_threshold_variation=False,
            mirror_mismatch=False,
            coupling_mismatch=False,
        ),
        reference=CurrentReferencePolicy(tolerance=False, noise=False),
    )


def tile_device(xbar: IsubIadc1t1rCimMacro) -> torch.device:
    """Device the tile lives on (first buffer of the module tree)."""
    return next(xbar.buffers()).device


def masked_planes(x: Tensor, *, row_num: int, max_active_rows: int, inst_rank: int = 0) -> Tensor:
    """Zero-masked WL planes via the engine mask formula.

    Shape: [..., *span, row_num] -> [..., P, *span, row_num]; the P axis
    inserts immediately left of the ``inst_rank``-wide inst-aligned span and
    plane ``p`` keeps exactly rows ``[p*max_active_rows, (p+1)*max_active_rows)``,
    zeros elsewhere (WL off).
    """
    p_num = row_num // max_active_rows
    mask = torch.arange(row_num, device=x.device) // max_active_rows == torch.arange(p_num, device=x.device).unsqueeze(
        -1
    )
    # Shape: [P, row_num] -> [P, 1*inst_rank, row_num]
    mask = mask.reshape(p_num, *(1,) * inst_rank, row_num)
    # Shape: [..., *span, row_num] -> [..., P, *span, row_num]
    return torch.where(mask, x.unsqueeze(max(-(inst_rank + 2), -(x.ndim + 1))), x.new_zeros(()))


def build_tile(
    config: IsubIadc1t1rCimMacroConfig,
    *,
    device: torch.device | None = None,
    policy: IsubIadc1t1rCimMacroPolicy | None = None,
    seed: int | None = None,
    inst_shape: tuple[int, ...] = (),
) -> IsubIadc1t1rCimMacro:
    """Build + fabricate one tile on ``device`` (all-off policy unless given)."""
    if policy is None:
        policy = build_all_off_policy()
    xbar = CimMacro.from_config(
        config=config,
        policy=policy,
        inst_shape=inst_shape,
        dtype=torch.float32,
        T__K=300.0,
    )
    assert isinstance(xbar, IsubIadc1t1rCimMacro)
    if device is not None:
        xbar.to(device)
    xbar.eval()
    if seed is not None:
        torch.manual_seed(seed)
    xbar.fabricate()
    return xbar


def with_ref_levels(
    config: IsubIadc1t1rCimMacroConfig, ref_levels__uA: tuple[float, ...]
) -> IsubIadc1t1rCimMacroConfig:
    """Install one single-mode ladder on BOTH threshold copies (kept consistent).

    The explicit outer tuple is the mode axis, so the per-mode consistency law
    keeps holding.
    """
    ref_bank__uA = (tuple(ref_levels__uA),)
    return dataclasses.replace(
        config,
        adc_config=dataclasses.replace(config.adc_config, ref_levels__uA=ref_bank__uA),
        reference_config=dataclasses.replace(config.reference_config, i_refs__uA=ref_bank__uA),
    )


def array_read(xbar: IsubIadc1t1rCimMacro, x_planes: Tensor) -> XbarArraySteadyState:
    """Boundary drive + array steady state for WL planes ``[..., row_num]``.

    Reproduces the ``vec_mat_mul`` array stage: WL DAC convert at the
    weight-grid full leading, one clamp-reference snapshot, and the kernel
    ``solve_array`` with the macro's lane-grouped BL clamp adapter.
    """
    _phys_col_num, row_num = xbar.array.weight_grid_shape[-2:]
    leading = torch.broadcast_shapes(xbar.array.weight_grid_shape, x_planes.unsqueeze(-2).shape)[:-2]
    with torch.no_grad():
        v_wl = xbar.wl_dac.convert(x_planes.expand(*leading, row_num))
        clamp_taps = xbar.clamp_ref.v_ref__V(xbar.clamp_ref.snapshot())
        return xbar.array.solve_array(
            v_wl,
            bl_driver=xbar._bl_clamp_lanes,
            bl_v_ref__V=clamp_taps[0],
            sl_driver=xbar.sl_driver,
            sl_v_ref__V=clamp_taps[1],
            t_conduct__ns=xbar.config.t_conduct__ns,
        )


def probe_i_sub(xbar: IsubIadc1t1rCimMacro, x_planes: Tensor) -> tuple[Tensor, Tensor]:
    """Analog ``(i_sub, sign)`` at the ADC input for WL planes ``[..., row_num]``.

    Reproduces the ``vec_mat_mul`` analog chain up to the ADC input (boundary
    drive + array solve -> polarity split + serial axes to leading ->
    p-mirror -> n-mirror -> subtractor), then moves the serial axes back
    trailing so the probe returns ``[..., n_io, io_col_num]`` with the
    columns of each IO in logical order. Leading dims of ``x_planes`` (batch
    and/or the sub-phase axis) ride through unchanged.
    """
    cfg = xbar.config
    lanes_per_io = cfg.io_col_num // cfg.mux_factor
    with torch.no_grad():
        steady = array_read(xbar, x_planes)
        i_lane = (
            steady.i_bl_port__uA.unflatten(-1, (xbar.col_num, 2))
            .movedim(-1, -2)
            .unflatten(-1, (xbar.n_lane, cfg.mux_factor))
            .movedim(-1, 0)
        )
        i_wdl = xbar.p_mirror.replicate(i_lane)  # [mux, ..., 2, n_lane]
        i_io = i_wdl.unflatten(-1, (xbar.n_io, lanes_per_io)).movedim(-1, 0)
        i_dl = xbar.n_mirror.replicate(i_io)  # [lpi, mux, ..., 2, n_io]
        i_sub, sign = xbar.subtractor.subtract(i_dl[..., 0, :], i_dl[..., 1, :], t_conduct__ns=cfg.t_conduct__ns)
        # [lpi, mux, ..., n_io] -> [..., n_io, lpi, mux] -> [..., n_io, io_col_num]
        i_sub = i_sub.movedim(0, -1).movedim(0, -1).flatten(-2)
        sign = sign.movedim(0, -1).movedim(0, -1).flatten(-2)
    return i_sub, sign


def probe_i_sub_grid(xbar: IsubIadc1t1rCimMacro, *, m_max: int) -> list[float]:
    """Probe the analog ``I_SUB(M)`` grid [uA] at the ADC input, M = 0..m_max.

    Programs logical column 0 all ``+1`` (others 0) and drives exactly ``M``
    sub-phase-0 rows per batch entry (``m_max <= max_active_rows``, so the
    whole grid lives inside one conversion — the per-conversion range). The
    zero-masked planes go through the modeled chain up to the ADC input
    via :func:`probe_i_sub`; plane 0 of IO 0 carries the grid. ``all_off``
    makes the probe deterministic. NOTE: reprograms the tile.
    """
    assert m_max <= xbar.max_active_rows
    device = tile_device(xbar)
    row_num = xbar.config.row_num

    w = torch.zeros((xbar.col_num, 1, row_num), dtype=torch.long, device=device)
    w[0, 0, :] = 1
    xbar.program(w)

    x = torch.zeros((m_max + 1, row_num), dtype=torch.long, device=device)
    for m in range(m_max + 1):
        x[m, :m] = 1

    # Shape: [m_max + 1, row_num] -> [m_max + 1, P, row_num]
    x_planes = masked_planes(x, row_num=row_num, max_active_rows=xbar.max_active_rows)
    i_sub, _sign = probe_i_sub(xbar, x_planes)  # [m_max + 1, P, n_io, col/n_io]
    return [float(v) for v in i_sub[:, 0, 0, 0].cpu()]


def midpoint_refs(grid: list[float]) -> tuple[float, ...]:
    """The ``2**n_bits - 1`` mid-point thresholds ``ref[k] = 0.5 * (I(k) + I(k+1))``."""
    level_num = (1 << TINY_ADC_BITS) - 1
    assert len(grid) >= level_num + 1
    return tuple(0.5 * (grid[k] + grid[k + 1]) for k in range(level_num))


def build_calibrated_tile(
    device: torch.device | None = None,
    *,
    inst_shape: tuple[int, ...] = (),
    seed: int | None = None,
) -> IsubIadc1t1rCimMacro:
    """Tiny tile with an in-code calibrated ladder: probe, install mid-points, rebuild.

    A first (placeholder-ladder) build probes the analog ``I_SUB(M)`` grid —
    the ladder does not matter before the ADC — and the rebuild installs the
    grid's mid-points as the calibrated thresholds on both threshold copies.
    The probe runs at ``inst_shape = ()``; under ``all_off`` the analog chain
    is deterministic, so the same ladder serves every instance.
    """
    config = build_tiny_config()
    probe_xbar = build_tile(config, device=device)
    grid = probe_i_sub_grid(probe_xbar, m_max=TINY_ACTIVE_ROW_NUM)
    calibrated = with_ref_levels(config, midpoint_refs(grid))
    return build_tile(calibrated, device=device, inst_shape=inst_shape, seed=seed)


def per_phase_clamp_reference(w: Tensor, x: Tensor, *, active_row_num: int, mag_max: int = MAG_MAX) -> Tensor:
    """CPU int64 unit-role reference: per-plane ``clamp(x_p . w_p)`` codes.

    Args:
        w: Ternary digit tensor ``[col_num, 1, row_num]`` (size-1 digit axis).
        x: Binary WL tensor ``[..., row_num]`` (full row; the reference
            derives the sub-phase partials itself).
        active_row_num: Rows per WL sub-phase (plane p owns rows
            ``[p * A, (p + 1) * A)``).
        mag_max: Signed-magnitude clip bound (3-bit -> 7).

    Returns:
        Expected signed per-plane codes ``[..., P, col_num]``
        (``P = row_num / active_row_num``) on CPU (int64; integer matmul
        stays on CPU by design).
    """
    w2 = w.squeeze(-2).cpu().long()  # (col, row)
    x2 = x.cpu().long()
    phase_num = w2.shape[-1] // active_row_num
    xp = x2.unflatten(-1, (phase_num, active_row_num))  # (..., P, A)
    wp = w2.unflatten(-1, (phase_num, active_row_num))  # (col, P, A)
    partial = torch.einsum("...pa,cpa->...pc", xp, wp)  # (..., P, col)
    return partial.clamp(-mag_max, mag_max)


def decode(
    xbar: IsubIadc1t1rCimMacro,
    w: Tensor,
    x: Tensor,
    *,
    adc_operation_point: AdcOperationPoint = ADC_OP,
) -> Tensor:
    """Program ``w``, expand ``x`` into masked WL planes, run one VMM.

    Returns the per-plane codes ``[..., P, col_num]`` on CPU.
    """
    device = tile_device(xbar)
    xbar.program(w.to(device))
    planes = masked_planes(x.to(device), row_num=xbar.config.row_num, max_active_rows=xbar.max_active_rows)
    with torch.no_grad():
        out = xbar.vec_mat_mul(planes, adc_operation_point=adc_operation_point)
    return out.cpu()
