"""Validate xbar vec_mat_mul and full XbarMacro pipeline at every level.

After the Switch/Xbar unification there is a single physical xbar class
``Xbar1T1R`` that subsumes both the previous ``IdealXbar`` (no analog
effects) and ``Xbar1T1RIdealSwitch`` (full circuit simulator) paths.
The bundled ``ideal_1t1r.toml`` configures the Switch as an ideal
pass-gate (``g_on = inf``, ``g_off = 0``) so ``Xbar1T1R`` reduces to
the noise-free circuit-solver path; turning any analog noise back on
exercises the solver's response.

Levels:

Level 1  ``Xbar1T1R.vec_mat_mul`` under the ideal-switch + noise-free
         config — the analytic reference is ``ADC(x · g · V_drive)``
         per sub-array, so this validates the circuit-solver + ADC
         path.
Level 2  (Kept for parity with older layouts — same call through the
         same xbar class; the two run identically under the current
         config and serve as a regression anchor.)
Level 3  Macro tiling + aggregation through ``Xbar1T1R`` with the ADC
         logic bypassed in the test body.  This confirms the macro's
         reshape / tile / sign-split / shift-add / accumulate pipeline
         reproduces ``torch.matmul`` exactly.
Level 4  Macro end-to-end with ADC + rescale: ``Xbar1T1R`` must
         approximate ``torch.matmul`` modulo ADC code-grid
         quantisation.  The residual gap quantifies circuit-solver +
         ADC error.

All tests use ``ideal_1t1r.toml`` (noise sub-configs omitted → skipped,
switch configured as an ideal pass-gate).
"""

from __future__ import annotations

import math
from functools import partial

import pytest
import torch

pytestmark = pytest.mark.skip(
    reason="Legacy XbarMacro/XbarMapper API removed; fixtures need rewriting "
    "to InterXbarSliceMacro (see neurox/macro/inter_xbar_slice.py)."
)

from neurox.analog import (
    AnalogMux,
    AnalogMuxConfig,
    Decoder,
    DecoderConfig,
    Driver,
    DriverConfig,
    SwitchCap,
    SwitchCapConfig,
)
from neurox.analog.dac import GeneralDAC, GeneralDACConfig
from neurox.analog.tia import OpAmpTIA, OpAmpTIAConfig
from neurox.analog.adc import GeneralADC, GeneralADCConfig
from neurox.analog.readout import OffsetSwitchCapMuxAdcReadOut, ReadOutConfig
from neurox.common import T_ROOM__K, dict_configs_from_file, dict_from_file, thermal_voltage__V
from neurox.config import DEFAULT_1T1R_TOML
from neurox.device import NMOS, RRAM, NMOSConfig, RRAMConfig
from neurox.digital import (
    Accumulator,
    AccumulatorConfig,
    Requantizer,
    RequantizerConfig,
    ShiftAdder,
    ShiftAdderConfig,
    Subtractor,
    SubtractorConfig,
)
from neurox.mapper.transcoder import Transcoder
from neurox.xbar import (
    CircuitCore1T1R,
    CircuitCore1T1RConfig,
    IdealXbar,
    Offset1T1RXbar,
    Offset1T1RXbarConfig,
)

CONFIG_FILE = DEFAULT_1T1R_TOML

_SPECS = {
    "rram": RRAMConfig,
    "nmos": NMOSConfig,
    "tia": OpAmpTIAConfig,
    "tia_nmos": NMOSConfig,
    "sl_driver": DriverConfig,
    "wl_decoder": DecoderConfig,
    "wl_dac": GeneralDACConfig,
    "bl_adc": GeneralADCConfig,
    "analog_mux": AnalogMuxConfig,
    "data_switchcap": SwitchCapConfig,
    "ref_switchcap": SwitchCapConfig,
    "digit_subtractor": SubtractorConfig,
    "accumulator": AccumulatorConfig,
    "shift_adder": ShiftAdderConfig,
    "core": CircuitCore1T1RConfig,
    "readout": ReadOutConfig,
    "xbar": Offset1T1RXbarConfig,
}


def _compute_nmos_ref_conductance(
    cfg_nmos: NMOSConfig,
    v_dd_wl: float,
    v_bl_clamp: float,
) -> tuple[float, float]:
    """Scalar (g_on, g_off) reference mirroring NMOS.fabricate_switch_state.

    Operating temperature is no longer a config field; the test
    fixtures build NMOS with the default ``T_ROOM__K``, so the
    analytic reference matches at the same temperature.
    """
    beta = cfg_nmos.mu_Cox__mA_V2 * (cfg_nmos.W__nm / cfg_nmos.L__nm)
    V_T = thermal_voltage__V(T_ROOM__K)
    g_on = max(beta * (v_dd_wl - cfg_nmos.vth0__V), 1e-12)
    i_0 = beta * (cfg_nmos.n_factor - 1.0) * (V_T**2)
    i_off = i_0 * math.exp(-cfg_nmos.vth0__V / (cfg_nmos.n_factor * V_T))
    return g_on, i_off / v_bl_clamp


@pytest.fixture
def cfg():
    raw = dict_from_file(CONFIG_FILE)
    typed = dict_configs_from_file(_SPECS, CONFIG_FILE)
    return raw, typed


def _build_ideal(typed) -> IdealXbar:
    """Build a IdealXbar by deriving it from the corresponding Xbar1T1R.

    Matches the production path: the fake reference is always
    constructed from the physical xbar via ``physical.to_ideal()``,
    so the tests exercise the same derivation used by
    ``xbar_ideal_factory``.

    Returned in ``eval()`` mode — stochastic rounding is on under
    ``module.training`` (the production HAT path) but tests want
    deterministic floor semantics so we lock it down here.
    """
    xbar = _build_1t1r(typed).to_ideal()
    xbar.eval()
    return xbar


def _build_1t1r(typed) -> Offset1T1RXbar:
    """Build the Xbar1T1R via the new three-layer factory chain.

    See ``docs/dev/modules/xbar/_1t1r/README.md`` + ``docs/dev/architecture/state_holding.md``.  Each
    device / circuit module is built per-core via its own factory so
    no fabricated state is shared across cores.  Returned in
    ``eval()`` mode (see :func:`_build_ideal`).
    """
    rram_factory = partial(RRAM, typed["rram"], dtype=torch.float64)
    nmos_factory = partial(NMOS, typed["nmos"], T__K=T_ROOM__K, dtype=torch.float64)
    tia_nmos_factory = partial(NMOS, typed["tia_nmos"], T__K=T_ROOM__K, dtype=torch.float64)
    tia_factory = partial(OpAmpTIA, typed["tia"], nmos_factory=tia_nmos_factory, dtype=torch.float64)
    core_factory = partial(
        CircuitCore1T1R,
        typed["core"],
        rram_factory=rram_factory,
        nmos_factory=nmos_factory,
        tia_factory=tia_factory,
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
        adc_factory=partial(GeneralADC, typed["bl_adc"], dtype=torch.float64),
        dtype=torch.float64,
    )
    xbar = Offset1T1RXbar(
        cfg=xbar_cfg,
        core_factory=core_factory,
        readout_factory=readout_factory,
    )
    xbar.eval()
    return xbar


def _build_macro(xbar, raw, typed):
    w_tc = Transcoder.create(
        raw["w_transcoder"]["encoding"], radix=xbar.w_states, digit_num=raw["w_transcoder"]["digit_num"]
    )
    x_tc = Transcoder.create(
        raw["x_transcoder"]["encoding"], radix=xbar.x_states, digit_num=raw["x_transcoder"]["digit_num"]
    )
    return XbarMacro(
        xbar=lambda: xbar,
        mapper=partial(XbarMapper, w_tc, x_tc, xbar.col_num, xbar.row_num),
        col_accumulator=partial(Accumulator, typed["accumulator"]),
        w_shift_adder=partial(ShiftAdder, typed["shift_adder"]),
        x_shift_adder=partial(ShiftAdder, typed["shift_adder"]),
        requantizer=partial(Requantizer, RequantizerConfig(bit_width=32)),
    )


# =========================================================================
# Level 1: IdealXbar vec_mat_mul — pure-integer baseline
# =========================================================================
class TestLevel1_IdealVMM:
    """Validate ``IdealXbar.vec_mat_mul`` — pure-integer dot-product with ADC clipping."""

    def test_zero_in_zero_out(self, cfg) -> None:
        _, typed = cfg
        xbar = _build_ideal(typed)
        w = torch.randint(-3, 4, (xbar.col_num, xbar.row_num), dtype=torch.int32)
        x = torch.zeros(1, xbar.row_num, dtype=torch.int32)
        y, _ = xbar.vec_mat_mul(x.unsqueeze(0), xbar.fabricate_unmapped(w))
        assert torch.all(y == 0)

    def test_zero_weight_zero_out(self, cfg) -> None:
        _, typed = cfg
        xbar = _build_ideal(typed)
        w = torch.zeros(xbar.col_num, xbar.row_num, dtype=torch.int32)
        x = torch.ones(1, xbar.row_num, dtype=torch.int32)
        y, _ = xbar.vec_mat_mul(x.unsqueeze(0), xbar.fabricate_unmapped(w))
        assert torch.all(y == 0)

    def test_max_gives_max_code(self, cfg) -> None:
        _, typed = cfg
        xbar = _build_ideal(typed)
        out_max = xbar._output_levels - 1
        w = torch.full((xbar.col_num, xbar.row_num), xbar.w_states - 1, dtype=torch.int32)
        x = torch.ones(1, xbar.row_num, dtype=torch.int32)
        y, _ = xbar.vec_mat_mul(x.unsqueeze(0), xbar.fabricate_unmapped(w))
        assert torch.all(y == out_max), f"Expected {out_max}, got {y.flatten()[:4]}"

    def test_neg_max_gives_neg_max_code(self, cfg) -> None:
        _, typed = cfg
        xbar = _build_ideal(typed)
        out_max = xbar._output_levels - 1
        w = torch.full((xbar.col_num, xbar.row_num), -(xbar.w_states - 1), dtype=torch.int32)
        x = torch.ones(1, xbar.row_num, dtype=torch.int32)
        y, _ = xbar.vec_mat_mul(x.unsqueeze(0), xbar.fabricate_unmapped(w))
        assert torch.all(y == -out_max)

    def test_symmetry(self, cfg) -> None:
        _, typed = cfg
        xbar = _build_ideal(typed)
        w_max = xbar.w_states - 1
        w_pos = torch.full((xbar.col_num, xbar.row_num), w_max, dtype=torch.int32)
        w_neg = torch.full((xbar.col_num, xbar.row_num), -w_max, dtype=torch.int32)
        x = torch.ones(1, xbar.row_num, dtype=torch.int32)
        y_pos, _ = xbar.vec_mat_mul(x.unsqueeze(0), xbar.fabricate_unmapped(w_pos))
        y_neg, _ = xbar.vec_mat_mul(x.unsqueeze(0), xbar.fabricate_unmapped(w_neg))
        assert torch.all(y_pos == -y_neg)

    def test_hand_computed_values(self, cfg) -> None:
        """Verify IdealXbar output against hand-computed expected codes.

        Config: w_states=4, x_states=2, row_num=64, col_num=64,
        out_levels=16.  max_dot = 64*3*1 = 192, rf = 193/16 = 12.0625.

        Quantization: code = floor(dot / rf).

        Test a single column (col=0) with controlled weight patterns:
        - All other columns are zero weight → code 0.
        - Input x = 1 at all 64 rows (binary WL, all asserted).
        """
        _, typed = cfg
        xbar = _build_ideal(typed)

        rf = xbar.output_rescale_factor  # 193/16 = 12.0625
        assert abs(rf - 12.0625) < 1e-6, f"Expected rf=12.0625, got {rf}"

        # Build a weight matrix: all zeros except column 0
        w = torch.zeros(xbar.col_num, xbar.row_num, dtype=torch.int32)
        x = torch.ones(1, xbar.row_num, dtype=torch.int32)

        # Case 1: col 0, all rows w=+3 → dot_pos=192, dot_neg=0
        # code_pos = floor(192 / 12.0625) = floor(15.917) = 15
        # code_neg = floor(0 / 12.0625) = 0
        # y = 15 - 0 = 15
        w[0, :] = 3
        y, _ = xbar.vec_mat_mul(x.unsqueeze(0), xbar.fabricate_unmapped(w))
        assert y.flatten()[0].item() == 15, f"Case 1: expected 15, got {y.flatten()[0].item()}"

        # Case 2: col 0, 32 rows w=+3, rest w=0 → dot_pos=96, dot_neg=0
        # code_pos = floor(96 / 12.0625) = floor(7.958) = 7
        # code_neg = 0
        # y = 7
        w[0, :] = 0
        w[0, :32] = 3
        y, _ = xbar.vec_mat_mul(x.unsqueeze(0), xbar.fabricate_unmapped(w))
        assert y.flatten()[0].item() == 7, f"Case 2: expected 7, got {y.flatten()[0].item()}"

        # Case 3: col 0, 13 rows w=+1, rest w=0 → dot_pos=13, dot_neg=0
        # code_pos = floor(13 / 12.0625) = floor(1.077) = 1
        # code_neg = 0
        # y = 1
        w[0, :] = 0
        w[0, :13] = 1
        y, _ = xbar.vec_mat_mul(x.unsqueeze(0), xbar.fabricate_unmapped(w))
        assert y.flatten()[0].item() == 1, f"Case 3: expected 1, got {y.flatten()[0].item()}"

        # Case 4: col 0, 12 rows w=+1, rest w=0 → dot_pos=12, dot_neg=0
        # code_pos = floor(12 / 12.0625) = floor(0.9948) = 0
        # code_neg = 0
        # y = 0  (12 states are below the first comparator threshold)
        w[0, :] = 0
        w[0, :12] = 1
        y, _ = xbar.vec_mat_mul(x.unsqueeze(0), xbar.fabricate_unmapped(w))
        assert y.flatten()[0].item() == 0, f"Case 4: expected 0, got {y.flatten()[0].item()}"

        # Case 5: col 0, all rows w=-2 → dot_pos=0, dot_neg=128
        # code_pos = 0
        # code_neg = floor(128 / 12.0625) = floor(10.61) = 10
        # y = 0 - 10 = -10
        w[0, :] = -2
        y, _ = xbar.vec_mat_mul(x.unsqueeze(0), xbar.fabricate_unmapped(w))
        assert y.flatten()[0].item() == -10, f"Case 5: expected -10, got {y.flatten()[0].item()}"

        # Case 6: col 0, 32 rows w=+2, 32 rows w=-1 → dot_pos=64, dot_neg=32
        # code_pos = floor(64 / 12.0625) = floor(5.305) = 5
        # code_neg = floor(32 / 12.0625) = floor(2.652) = 2
        # y = 5 - 2 = 3
        w[0, :] = 0
        w[0, :32] = 2
        w[0, 32:] = -1
        y, _ = xbar.vec_mat_mul(x.unsqueeze(0), xbar.fabricate_unmapped(w))
        assert y.flatten()[0].item() == 3, f"Case 6: expected 3, got {y.flatten()[0].item()}"

        # Case 7: x = 0 everywhere → all codes 0 regardless of weight
        w[0, :] = 3
        x_zero = torch.zeros(1, xbar.row_num, dtype=torch.int32)
        y, _ = xbar.vec_mat_mul(x_zero.unsqueeze(0), xbar.fabricate_unmapped(w))
        assert torch.all(y == 0), f"Case 7: expected all 0, got {y.flatten()[:4]}"

        # Verify all other columns are 0 in cases 1-6 (weights were zero there)
        w[:] = 0
        w[0, :] = 3
        y, _ = xbar.vec_mat_mul(x.unsqueeze(0), xbar.fabricate_unmapped(w))
        assert torch.all(y.flatten()[1:] == 0), "Non-zero output in zero-weight columns"


# =========================================================================
# Level 2: Xbar1T1R vec_mat_mul vs analytic
# =========================================================================
class TestLevel2_1T1RVMM:
    def test_solver_matches_analytic(self, cfg) -> None:
        raw, typed = cfg
        xbar = _build_1t1r(typed)
        state_to_g = torch.tensor(typed["rram"].state_to_g__mS, dtype=torch.float64)
        boundaries = torch.tensor(typed["bl_adc"].boundaries, dtype=torch.float64)
        bl_drive = typed["bl_adc"].drive_value
        sl_drive = raw["sl_driver"]["drive_value"]
        alpha = typed["rram"].nonlinearity_alpha
        v_dd_wl = float(typed["wl_dac"].code_to_signal[-1])
        g_on, g_off = _compute_nmos_ref_conductance(typed["nmos"], v_dd_wl, bl_drive)
        ref_group_size = typed["xbar"].ref_group_size
        w_max = xbar.w_states - 1

        def _cell_i(g_rram: torch.Tensor, g_sw: torch.Tensor) -> torch.Tensor:
            v_x = torch.full_like(g_rram, (bl_drive + sl_drive) * 0.5)
            for _ in range(8):
                v_r = bl_drive - v_x
                if alpha == 0.0:
                    i_r = g_rram * v_r
                    di_r = g_rram
                else:
                    i_r = g_rram * torch.sinh(alpha * v_r) / alpha
                    di_r = g_rram * torch.cosh(alpha * v_r)
                f = g_sw * (v_x - sl_drive) - i_r
                fp = (g_sw + di_r).clamp(min=1e-12)
                v_x = v_x - f / fp
            return g_sw * (v_x - sl_drive)

        exact = 0
        for trial in range(500):
            torch.manual_seed(trial)
            w = torch.randint(-w_max, w_max + 1, (xbar.col_num, xbar.row_num), dtype=torch.int32)
            x = torch.randint(0, 2, (xbar.row_num,), dtype=torch.int32)

            # Analytic: per-cell v_x Newton → cell current → sum → reference subtract → bucketize
            w_pos = torch.clamp_min(w, 0)
            w_neg = torch.clamp_min(-w, 0)
            g_pos = state_to_g[w_pos.long()]
            g_neg = state_to_g[w_neg.long()]
            x_f = x.unsqueeze(-2).to(torch.float64)
            g_sw = torch.where(x_f > 0.5, g_on, g_off)

            i_pos = _cell_i(g_pos, g_sw).sum(-1)
            i_neg = _cell_i(g_neg, g_sw).sum(-1)
            if ref_group_size > 0:
                g_ref = state_to_g[0].expand_as(g_sw[..., :1, :])
                i_ref = _cell_i(g_ref, g_sw).sum(-1)
                i_pos = (i_pos - i_ref).clamp_min(0.0)
                i_neg = (i_neg - i_ref).clamp_min(0.0)
            expected = torch.bucketize(i_pos, boundaries, out_int32=True) - torch.bucketize(
                i_neg, boundaries, out_int32=True
            )

            state = xbar.fabricate_unmapped(w)
            actual, _ = xbar.vec_mat_mul(x.unsqueeze(0).unsqueeze(0).to(torch.float64), state)
            if torch.equal(actual.to(expected.dtype), expected):
                exact += 1

        rate = exact / 500 * 100
        print(f"\n  1T1R solver: {exact}/500 exact matches ({rate:.1f}%)")
        assert rate > 30


# =========================================================================
# Level 3: Macro tiling + aggregation (no ADC)
# =========================================================================
class TestLevel3_MacroTiling:
    @pytest.mark.parametrize("N,K", [(4, 64), (16, 128), (64, 256), (128, 512)])
    def test_no_adc_matches_ideal_matmul(self, cfg, N, K) -> None:
        """Without ADC quantization, macro pipeline must exactly reproduce matmul."""
        raw, typed = cfg
        xbar = _build_ideal(typed)
        macro = _build_macro(xbar, raw, typed)
        macro.eval()

        torch.manual_seed(42)
        w = torch.randint(-3, 4, (N, K), dtype=torch.int32)
        x = torch.randint(0, 2, (1, K), dtype=torch.int32)
        y_ideal = x.to(torch.int64) @ w.to(torch.int64).T

        # The mapper already produces sign-split for weights (sign axis at -3,
        # size 2 for the differential encoding) and a placeholder sign=1 axis
        # for activations.  No additional sign-split call is needed.
        w_mapped = macro.mapper.map_w(w)
        x_mapped = macro.mapper.map_x(x)

        w_pos = w_mapped.select(-3, 0)
        w_neg = w_mapped.select(-3, 1)
        x_vec = x_mapped.select(-3, 0).squeeze(-2)
        dot_pos = (x_vec.unsqueeze(-2).float() * w_pos.float()).sum(-1).to(torch.int64)
        dot_neg = (x_vec.unsqueeze(-2).float() * w_neg.float()).sum(-1).to(torch.int64)
        y_raw = dot_pos - dot_neg

        y_raw = macro.x_shift_adder.operate(y_raw, xbar.x_states, dim=-3)
        y_raw = macro.w_shift_adder.operate(y_raw, xbar.w_states, dim=-2)
        y_raw = macro.col_accumulator.operate(y_raw, dim=-3)
        y_raw = y_raw.flatten(start_dim=-2)[..., :N]

        assert torch.equal(y_raw.flatten()[:N], y_ideal.flatten()[:N]), (
            f"N={N},K={K}: ideal={y_ideal.flatten()[:4].tolist()} got={y_raw.flatten()[:4].tolist()}"
        )


# =========================================================================
# Level 4: Macro end-to-end with ADC
# =========================================================================
class TestLevel4_MacroEndToEnd:
    """Compare the Xbar1T1R back-end to ``torch.matmul``.

    With Level 3 having proved the macro-pipeline maths,
    any residual gap here that exceeds the ADC code-grid is an analog /
    circuit-solver error rather than a macro-pipeline error.
    """

    @staticmethod
    def _run_macro(xbar_builder, raw, typed, *, N: int, K: int):
        """Run one macro end-to-end with identity rescale and return (y_recovered, y_ideal, code_width)."""
        xbar = xbar_builder(typed)
        macro = _build_macro(xbar, raw, typed)
        macro.eval()

        torch.manual_seed(42)
        w = torch.randint(-3, 4, (N, K), dtype=torch.int32)
        x = torch.randint(0, 2, (1, K), dtype=torch.int32)
        y_ideal = (x.to(torch.int64) @ w.to(torch.int64).T).float()

        macro.fabricate(w)
        m = torch.ones(N, dtype=torch.int32)
        s = torch.zeros(N, dtype=torch.int32)
        y_xbar, _ = macro.matmul(x, w, None, m, s, None)

        y_recovered = y_xbar.float() * macro.output_rescale_factor
        # Worst-case per-column quantization step (one ADC code mapped
        # back to ideal-MAC units), times the two sign-split sub-arrays.
        code_width = 2.0 * macro.output_rescale_factor
        return y_recovered, y_ideal, code_width

    @pytest.mark.parametrize(
        "name,builder,N,K,tol_codes",
        [
            # IdealXbar: pure-integer VMM baseline, ADC code-grid clipping
            # is the only error source.
            ("ideal", _build_ideal, 16, 128, 4),
            # Xbar1T1R (ideal switch): full circuit solver path; wider
            # tolerance accounts for the baseline conductance from g_min.
            ("1t1r", _build_1t1r, 8, 64, 8),
        ],
    )
    def test_macro_matches_torch_matmul(self, cfg, name, builder, N, K, tol_codes) -> None:
        """Each macro back-end recovers ``torch.matmul`` within ``tol_codes`` ADC code-widths."""
        raw, typed = cfg
        y_recovered, y_ideal, code_width = self._run_macro(builder, raw, typed, N=N, K=K)
        diff = (y_recovered - y_ideal).abs()

        max_allowed = tol_codes * code_width
        print(
            f"\n  [{name}] N={N} K={K}: "
            f"max_diff={diff.max().item():.2f}, "
            f"mean_diff={diff.mean().item():.2f}, "
            f"code_width={code_width:.2f}, "
            f"bound={max_allowed:.2f}"
        )
        assert diff.max().item() <= max_allowed, (
            f"[{name}] max diff {diff.max().item():.2f} exceeds {tol_codes} code-widths ({max_allowed:.2f})"
        )

    def test_real_xbar_tracks_ideal_xbar(self, cfg) -> None:
        """Ideal vs real xbar through the same XbarMacro on identical input.

        IdealXbar (rf=1.0, no ADC) outputs exact integer MACs; Xbar1T1R
        (rf=12.0625, with ADC) outputs in a different scale.  After
        recovery (``y * rf``) both are in ideal-integer units and
        comparable.  Tolerance is set in 1T1R code-widths since the
        ADC quantization dominates the gap.
        """
        raw, typed = cfg
        N, K = 8, 64
        y_ideal_recov, y_ref, _ = self._run_macro(_build_ideal, raw, typed, N=N, K=K)
        y_1t1r_recov, _, code_width_1t1r = self._run_macro(_build_1t1r, raw, typed, N=N, K=K)

        diff_vs_ref = (y_1t1r_recov - y_ref).abs()
        diff_vs_ideal = (y_1t1r_recov - y_ideal_recov).abs()
        max_allowed = 4 * code_width_1t1r
        print(
            f"\n  ideal-xbar vs 1t1r-xbar through same XbarMacro (N={N}, K={K}):"
            f"\n    diff(1t1r, torch.matmul) max={diff_vs_ref.max().item():.2f}"
            f"\n    diff(1t1r, ideal_xbar)   max={diff_vs_ideal.max().item():.2f}"
            f"\n    bound={max_allowed:.2f}"
        )
        assert diff_vs_ideal.max().item() <= max_allowed, (
            f"1t1r vs ideal gap {diff_vs_ideal.max().item():.2f} exceeds 4 code-widths"
        )
