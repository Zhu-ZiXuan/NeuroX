"""Validate the 1T1R xbar physical model at 64×64 scale.

Uses ``ideal_1t1r.toml`` (noise sub-configs omitted → skipped, access
switch configured as an ideal pass-gate).  Generates many random
(w, x) pairs within the valid signed-digit ranges derived from
``xbar.w_states`` and ``xbar.x_states``, runs the full ``fabricate`` →
``vec_mat_mul`` path, and compares against an analytic reference that
includes intrinsic ADC quantization.

The xbar computes vector-matrix multiplication where:
- w: signed digits in ``[-(w_states-1), w_states-1]``, internally
  sign-split into positive and negative sub-arrays.
- x: non-negative input in ``[0, x_states-1]``, drives both sub-arrays.
- output: ``ADC(pos_current) - ADC(neg_current)`` per column.
"""

import math
from functools import partial
from pathlib import Path as _Path

import pytest
import torch

from neurox.analog import (
    AnalogMux,
    AnalogMuxConfig,
    Driver,
    DriverConfig,
    SwitchCap,
    SwitchCapConfig,
)
from neurox.analog.adc import GeneralADC, GeneralADCConfig
from neurox.analog.dac import GeneralDAC, GeneralDACConfig
from neurox.analog.tia import OpAmpTIA, OpAmpTIAConfig
from neurox.common import T_ROOM__K, dict_configs_from_file, dict_from_file, thermal_voltage__V
from neurox.device import NMOS, RRAM, NMOSConfig, RRAMConfig
from neurox.digital import SubtractorConfig
from neurox.xbar import (
    CircuitCore1T1R,
    CircuitCore1T1RConfig,
    Offset1T1RXbar,
    Offset1T1RXbarConfig,
)
from neurox.xbar.readout import OffsetSwitchCapMuxAdcReadOut, ReadOutConfig

pytestmark = pytest.mark.skip(reason="Fixtures need rewriting to the Xbar.from_config-based construction path.")

CONFIG_FILE = _Path(__file__).parent / "fixtures" / "macro.toml"

_SPECS = {
    "rram": RRAMConfig,
    "nmos": NMOSConfig,
    "tia": OpAmpTIAConfig,
    "tia_nmos": NMOSConfig,
    "sl_driver": DriverConfig,
    "wl_dac": GeneralDACConfig,
    "bl_adc": GeneralADCConfig,
    "analog_mux": AnalogMuxConfig,
    "data_switchcap": SwitchCapConfig,
    "ref_switchcap": SwitchCapConfig,
    "digit_subtractor": SubtractorConfig,
    "core": CircuitCore1T1RConfig,
    "readout": ReadOutConfig,
    "xbar": Offset1T1RXbarConfig,
}


def _compute_nmos_ref_conductance(
    config_nmos: NMOSConfig,
    v_dd_wl: float,
    v_bl_clamp: float,
) -> tuple[float, float]:
    """Analytic-reference (g_on, g_off) for a noise-free NMOS.

    Mirrors :meth:`NMOS.fabricate_switch_state` with all mismatch
    sigmas ``None`` — returns the deterministic scalar conductances
    the test expects from the operating point.
    """
    beta = config_nmos.mu_Cox__mA_V2 * (config_nmos.W__nm / config_nmos.L__nm)
    V_T = thermal_voltage__V(T_ROOM__K)
    g_on = max(beta * (v_dd_wl - config_nmos.vth0__V), 1e-12)
    i_0 = beta * (config_nmos.n_factor - 1.0) * (V_T**2)
    i_off = i_0 * math.exp(-config_nmos.vth0__V / (config_nmos.n_factor * V_T))
    g_off = i_off / v_bl_clamp
    return g_on, g_off


def _build_xbar() -> Offset1T1RXbar:
    """Build a 64×64 xbar via the new three-layer factory chain.

    See ``docs/dev/modules/xbar/_1t1r/README.md`` + ``docs/dev/architecture/state_holding.md`` — ``CircuitCore1T1R``
    owns the array physics, ``ReadOut`` does the voltage-domain
    weighted-sum, and ``Offset1T1RXbar`` is the mapping layer.  Every
    device / circuit module is built per-core from its own factory so
    no per-instance state is shared across cores.
    """
    config = dict_configs_from_file(_SPECS, CONFIG_FILE)
    rram_factory = partial(RRAM, config["rram"], dtype=torch.float64)
    nmos_factory = partial(NMOS, config["nmos"], T__K=T_ROOM__K, dtype=torch.float64)
    tia_nmos_factory = partial(NMOS, config["tia_nmos"], T__K=T_ROOM__K, dtype=torch.float64)
    tia_factory = partial(OpAmpTIA, config["tia"], nmos_factory=tia_nmos_factory, dtype=torch.float64)
    core_factory = partial(
        CircuitCore1T1R,
        config["core"],
        rram_factory=rram_factory,
        nmos_factory=nmos_factory,
        tia_factory=tia_factory,
        sl_driver_factory=partial(Driver, config["sl_driver"], dtype=torch.float64),
        wl_dac_factory=partial(GeneralDAC, config["wl_dac"], dtype=torch.float64),
        dtype=torch.float64,
    )
    xbar_config = config["xbar"]
    readout_factory = partial(
        OffsetSwitchCapMuxAdcReadOut,
        config["readout"],
        data_switchcap_factory=partial(SwitchCap, config["data_switchcap"], T__K=T_ROOM__K, dtype=torch.float64),
        ref_switchcap_factory=partial(SwitchCap, config["ref_switchcap"], T__K=T_ROOM__K, dtype=torch.float64),
        analog_mux_factory=partial(AnalogMux, config["analog_mux"], dtype=torch.float64),
        adc_factory=partial(GeneralADC, config["bl_adc"], dtype=torch.float64),
        dtype=torch.float64,
    )
    return Offset1T1RXbar(
        config=xbar_config,
        core_factory=core_factory,
        readout_factory=readout_factory,
    )


def _solve_cell_current(
    g_rram: torch.Tensor,
    g_sw: torch.Tensor,
    v_bl: float,
    v_sl: float,
    alpha: float,
) -> torch.Tensor:
    """Per-cell current via inner Newton on v_x (analytic reference).

    Solves ``g_sw·(v_x − v_sl) = g_rram·sinh(alpha·(v_bl − v_x))/alpha``
    for v_x at each cell, then returns ``i_cell = g_sw·(v_x − v_sl)``.
    For alpha=0 falls back to ``g_rram·(v_bl − v_x) = g_sw·(v_x − v_sl)``.
    """
    v_x = torch.full_like(g_rram, (v_bl + v_sl) * 0.5)
    for _ in range(8):
        v_rram = v_bl - v_x
        if alpha == 0.0:
            i_rram = g_rram * v_rram
            di_rram = g_rram
        else:
            i_rram = g_rram * torch.sinh(alpha * v_rram) / alpha
            di_rram = g_rram * torch.cosh(alpha * v_rram)
        f = g_sw * (v_x - v_sl) - i_rram
        fp = (g_sw + di_rram).clamp(min=1e-12)
        v_x = v_x - f / fp
    return g_sw * (v_x - v_sl)


def _analytic_output(
    w: torch.Tensor,
    x: torch.Tensor,
    state_to_g: torch.Tensor,
    boundaries: torch.Tensor,
    bl_drive: float,
    sl_drive: float,
    alpha: float,
    g_on: float,
    g_off: float,
    ref_group_size: int = 0,
) -> torch.Tensor:
    """Compute the analytic xbar output with series RRAM + Switch cascade.

    For each sign-split sub-array, resolves the per-cell intermediate
    node v_x (RRAM–Switch junction) via Newton, computes the cell
    current from the switch side, sums per column, subtracts the
    state-0 dummy-column baseline current (if dummies are active),
    clamps to ``>= 0``, and bucketizes.

    Args:
        w: Signed weight digits [col_num, row_num].
        x: Non-negative input [row_num].
        state_to_g: Conductance LUT [w_states], in mS.
        boundaries: ADC boundaries, in mA.
        bl_drive: BL drive voltage [V].
        sl_drive: SL drive voltage [V].
        alpha: RRAM nonlinearity coefficient (1/V).
        g_on: Switch on-conductance [mS].
        g_off: Switch off-conductance [mS].
        ref_group_size: Number of logic columns sharing one dummy
            column.  ``0`` disables reference subtraction.

    Returns:
        Signed ADC codes [col_num].
    """
    w_pos = torch.clamp_min(w, 0)
    w_neg = torch.clamp_min(-w, 0)

    g_pos = state_to_g[w_pos.long()]
    g_neg = state_to_g[w_neg.long()]

    # Per-cell switch conductance: g_on for active cells (x>0), g_off otherwise.
    x_f = x.unsqueeze(-2).to(g_pos.dtype)
    g_sw = torch.where(x_f > 0.5, g_on, g_off)

    bl_i_pos = _solve_cell_current(g_pos, g_sw, bl_drive, sl_drive, alpha).sum(dim=-1)
    bl_i_neg = _solve_cell_current(g_neg, g_sw, bl_drive, sl_drive, alpha).sum(dim=-1)

    if ref_group_size > 0:
        # One dummy column per group, all state-0 cells (g_min).
        g_ref = state_to_g[0].expand_as(g_sw[..., :1, :])
        bl_i_ref = _solve_cell_current(g_ref, g_sw, bl_drive, sl_drive, alpha).sum(dim=-1)
        # All dummies see the same x, so they all produce the same current — one
        # scalar is enough to subtract from every logic column.
        bl_i_pos = (bl_i_pos - bl_i_ref).clamp_min(0.0)
        bl_i_neg = (bl_i_neg - bl_i_ref).clamp_min(0.0)

    code_pos = torch.bucketize(bl_i_pos, boundaries, out_int32=True)
    code_neg = torch.bucketize(bl_i_neg, boundaries, out_int32=True)

    return code_pos - code_neg


class TestXbar64x64:
    """Exhaustive solver accuracy tests on a 64×64 array with default config."""

    @pytest.fixture
    def xbar(self) -> Offset1T1RXbar:
        return _build_xbar()

    @pytest.fixture
    def reference(self):
        """Return analytic reference parameters from config."""
        raw = dict_from_file(CONFIG_FILE)
        config = dict_configs_from_file(_SPECS, CONFIG_FILE)
        v_dd_wl = float(config["wl_dac"].code_to_signal[-1])
        v_bl_clamp = config["bl_adc"].drive_value
        g_on, g_off = _compute_nmos_ref_conductance(config["nmos"], v_dd_wl, v_bl_clamp)
        return {
            "state_to_g": torch.tensor(config["rram"].state_to_g__mS, dtype=torch.float64),
            "boundaries": torch.tensor(config["bl_adc"].boundaries, dtype=torch.float64),
            "bl_drive": v_bl_clamp,
            "sl_drive": raw["sl_driver"]["drive_value"],
            "alpha": config["rram"].nonlinearity_alpha,
            "g_on": g_on,
            "g_off": g_off,
            "ref_group_size": config["xbar"].ref_group_size,
        }

    def test_many_random_pairs(
        self,
        xbar: Offset1T1RXbar,
        reference: tuple[torch.Tensor, torch.Tensor, float, float],
    ) -> None:
        """1000 random (w, x) pairs — report match rate and error stats."""
        state_to_g = reference["state_to_g"]
        boundaries = reference["boundaries"]
        bl_drive = reference["bl_drive"]
        sl_drive = reference["sl_drive"]
        alpha = reference["alpha"]
        g_on = reference["g_on"]
        g_off = reference["g_off"]
        ref_group_size = reference["ref_group_size"]
        col_num = xbar.col_num
        row_num = xbar.row_num
        w_max = xbar.w_states - 1
        x_max = xbar.x_states - 1
        n_trials = 1000

        print(f"\n{'=' * 70}")
        print(f"  64×64 xbar solver test: {n_trials} random (w, x) pairs")
        print(f"  w range: [{-w_max}, {w_max}]  (from w_states={xbar.w_states})")
        print(f"  x range: [0, {x_max}]  (from x_states={xbar.x_states})")
        print(f"  ADC levels: {xbar.output_levels}")
        print(f"{'=' * 70}")

        exact_matches = 0
        total_codes = 0
        max_diff_seen = 0
        total_abs_diff = 0
        diff_histogram: dict[int, int] = {}
        code_match_count = 0

        for trial in range(n_trials):
            torch.manual_seed(trial)
            w = torch.randint(-w_max, w_max + 1, (col_num, row_num), dtype=torch.int32)
            x = torch.randint(0, x_max + 1, (row_num,), dtype=torch.int32)

            expected = _analytic_output(
                w, x, state_to_g, boundaries, bl_drive, sl_drive, alpha, g_on, g_off, ref_group_size
            )

            physical_state = xbar.fabricate_unmapped(w)
            actual, _ = xbar.vec_mat_mul(
                x.unsqueeze(0).unsqueeze(0).to(torch.float64),
                physical_state,
            )

            diff = (actual.to(expected.dtype) - expected).abs()
            max_diff = diff.max().item()

            total_codes += col_num
            total_abs_diff += diff.sum().item()
            code_match_count += (diff == 0).sum().item()
            if max_diff > max_diff_seen:
                max_diff_seen = max_diff
            if torch.equal(actual.to(expected.dtype), expected):
                exact_matches += 1
            for d in diff.flatten().tolist():
                d_int = int(d)
                diff_histogram[d_int] = diff_histogram.get(d_int, 0) + 1

        match_rate = exact_matches / n_trials * 100
        code_match_rate = code_match_count / total_codes * 100
        avg_code_err = total_abs_diff / total_codes

        print("\n  Results:")
        print(f"    exact vector matches:   {exact_matches}/{n_trials} ({match_rate:.1f}%)")
        print(f"    per-code match rate:    {code_match_count}/{total_codes} ({code_match_rate:.2f}%)")
        print(f"    max code diff:          {max_diff_seen}")
        print(f"    avg abs code error:     {avg_code_err:.6f}")
        print("\n  Per-code diff histogram:")
        for d in sorted(diff_histogram):
            count = diff_histogram[d]
            pct = count / total_codes * 100
            print(f"    diff={d}: {count:6d} codes ({pct:6.2f}%)")
        print(f"{'=' * 70}")

    def test_edge_cases(
        self,
        xbar: Offset1T1RXbar,
        reference: tuple[torch.Tensor, torch.Tensor, float, float],
    ) -> None:
        """Specific edge cases: all-zero, all-max, checkerboard, single-active."""
        state_to_g = reference["state_to_g"]
        boundaries = reference["boundaries"]
        bl_drive = reference["bl_drive"]
        sl_drive = reference["sl_drive"]
        alpha = reference["alpha"]
        g_on = reference["g_on"]
        g_off = reference["g_off"]
        ref_group_size = reference["ref_group_size"]
        col_num = xbar.col_num
        row_num = xbar.row_num
        w_max = xbar.w_states - 1
        x_max = xbar.x_states - 1

        cases: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
            "w=0, x=0": (
                torch.zeros(col_num, row_num, dtype=torch.int32),
                torch.zeros(row_num, dtype=torch.int32),
            ),
            "w=max, x=max": (
                torch.full((col_num, row_num), w_max, dtype=torch.int32),
                torch.full((row_num,), x_max, dtype=torch.int32),
            ),
            "w=-max, x=max": (
                torch.full((col_num, row_num), -w_max, dtype=torch.int32),
                torch.full((row_num,), x_max, dtype=torch.int32),
            ),
            "w=max, x=0": (
                torch.full((col_num, row_num), w_max, dtype=torch.int32),
                torch.zeros(row_num, dtype=torch.int32),
            ),
            "diagonal w=max": (
                torch.zeros(col_num, row_num, dtype=torch.int32).fill_diagonal_(w_max),
                torch.full((row_num,), x_max, dtype=torch.int32),
            ),
            "checkerboard w": (
                torch.tensor(
                    [[w_max if (i + j) % 2 == 0 else -w_max for j in range(row_num)] for i in range(col_num)],
                    dtype=torch.int32,
                ),
                torch.full((row_num,), x_max, dtype=torch.int32),
            ),
            "single row active": (
                torch.full((col_num, row_num), w_max, dtype=torch.int32),
                torch.zeros(row_num, dtype=torch.int32).scatter_(0, torch.tensor([0]), x_max),
            ),
        }

        print(f"\n{'=' * 70}")
        print("  Edge case tests (64×64)")
        print(f"{'=' * 70}")

        all_pass = True
        for name, (w, x) in cases.items():
            expected = _analytic_output(
                w, x, state_to_g, boundaries, bl_drive, sl_drive, alpha, g_on, g_off, ref_group_size
            )

            physical_state = xbar.fabricate_unmapped(w)
            actual, _ = xbar.vec_mat_mul(
                x.unsqueeze(0).unsqueeze(0).to(torch.float64),
                physical_state,
            )

            match = torch.equal(actual.to(expected.dtype), expected)
            diff = (actual.to(expected.dtype) - expected).abs().max().item()

            status = "PASS" if match else f"FAIL (max diff={diff})"
            print(f"  {name:25s}  expected=[{expected.min().item():3}, {expected.max().item():3}]  {status}")
            if not match:
                all_pass = False

        assert all_pass, "Some edge cases failed"

    def test_output_distribution(
        self,
        xbar: Offset1T1RXbar,
        reference: tuple[torch.Tensor, torch.Tensor, float, float],
    ) -> None:
        """Profile the output code distribution over 500 random trials."""
        state_to_g = reference["state_to_g"]
        boundaries = reference["boundaries"]
        bl_drive = reference["bl_drive"]
        sl_drive = reference["sl_drive"]
        alpha = reference["alpha"]
        g_on = reference["g_on"]
        g_off = reference["g_off"]
        ref_group_size = reference["ref_group_size"]
        col_num = xbar.col_num
        row_num = xbar.row_num
        w_max = xbar.w_states - 1
        x_max = xbar.x_states - 1
        adc_max = xbar.output_levels - 1
        n_trials = 500

        all_codes: list[torch.Tensor] = []
        for trial in range(n_trials):
            torch.manual_seed(trial + 10000)
            w = torch.randint(-w_max, w_max + 1, (col_num, row_num), dtype=torch.int32)
            x = torch.randint(0, x_max + 1, (row_num,), dtype=torch.int32)

            expected = _analytic_output(
                w, x, state_to_g, boundaries, bl_drive, sl_drive, alpha, g_on, g_off, ref_group_size
            )
            all_codes.append(expected)

        codes = torch.cat(all_codes)
        print(f"\n{'=' * 70}")
        print("  Output code distribution (500 trials, 64×64)")
        print(f"{'=' * 70}")
        print(f"  possible range:   [{-adc_max}, {adc_max}]")
        print(f"  observed range:   [{codes.min().item()}, {codes.max().item()}]")
        print(f"  unique codes:     {codes.unique().numel()}")
        print(f"  mean:             {codes.float().mean():.3f}")
        print(f"  std:              {codes.float().std():.3f}")

        hist = torch.zeros(2 * adc_max + 1, dtype=torch.int64)
        for c in codes:
            hist[c.item() + adc_max] += 1
        print("\n  code histogram:")
        for i, count in enumerate(hist):
            code_val = i - adc_max
            if count > 0:
                bar = "#" * min(int(count / len(codes) * 200), 60)
                print(f"    code {code_val:+3d}: {count:6d} ({count / len(codes) * 100:5.1f}%) {bar}")
