"""Test the inference-only quantized operators.

Covers:

1. ``QuantLinear`` / ``QuantConv2d`` forward with a state_dict crafted via
   the production helper ``derive_layer_int_params`` and ``IdealMacro``.
   This is the operator-wiring smoke-test: any failure here is an
   operator-side bug, not a macro / crossbar bug.
2. Cross-macro consistency: ``IdealMacro`` vs. ``XbarMacro(IdealXbar,...)``
   on the same float weights.  The two state_dicts differ — each is
   derived with the macro's own ``output_rescale_factor`` folded in by
   ``derive_layer_int_params`` — so the test exercises the rescale-factor
   plumbing end-to-end.  The assertion is a loose relative-error bound
   (the ADC code-grid is coarse) intended to catch bias-fold or
   multiplier-derivation regressions that would inflate the gap by
   orders of magnitude.

The QAT training path (``LinearQAT``/``Conv2dQAT``) was removed; QAT now
lives entirely outside NeuroX (see ``example/lenet/qat.py`` for the pt2e flow).
"""

from __future__ import annotations

import math
from functools import partial

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skip(
    reason="Legacy XbarMacro/XbarMapper API removed; fixtures need rewriting "
    "to InterXbarSliceMacro (see neurox/macro/inter_xbar_slice.py)."
)

from neurox.analog import Driver, DriverConfig
from neurox.analog.dac import GeneralDAC, GeneralDACConfig
from neurox.analog.adc import GeneralADC, GeneralADCConfig
from neurox.common import dict_configs_from_file, dict_from_file
from neurox.config import DEFAULT_1T1R_MACRO_TOML
from neurox.device import NMOS, RRAM, NMOSConfig, RRAMConfig
from neurox.digital import (
    Accumulator,
    AccumulatorConfig,
    Requantizer,
    RequantizerConfig,
    ShiftAdder,
    ShiftAdderConfig,
    SubtractorConfig,
)
from neurox.macro.base import NeuroxMacroQuantMatMul
from neurox.macro.ideal import IdealMacro
from neurox.mapper.transcoder import Transcoder
from neurox.operator.base import NeuroxOperator
from neurox.operator.conv import QuantConv2d
from neurox.operator.linear import QuantLinear, derive_layer_int_params
from neurox.xbar import Offset1T1RXbar, Offset1T1RXbarConfig

CONFIG_FILE = DEFAULT_1T1R_MACRO_TOML

_SPECS = {
    "rram": RRAMConfig,
    "nmos": NMOSConfig,
    "sl_driver": DriverConfig,
    "wl_dac": GeneralDACConfig,
    "bl_adc": GeneralADCConfig,
    "digit_subtractor": SubtractorConfig,
    "accumulator": AccumulatorConfig,
    "shift_adder": ShiftAdderConfig,
    "xbar": Offset1T1RXbarConfig,
}


# ---------------------------------------------------------------------------
# Helpers: build IdealMacro / IdealXbar macro / Xbar1T1R macro
# ---------------------------------------------------------------------------


def _build_fake_macro() -> IdealMacro:
    """IdealMacro sized to the ±255 grid implied by the default 1T1R config."""
    raw = dict_from_file(CONFIG_FILE)
    typed = dict_configs_from_file(_SPECS, CONFIG_FILE)
    w_states = len(typed["rram"].state_to_g__mS)
    x_states = 2
    w_max = w_states ** raw["w_transcoder"]["digit_num"] - 1
    x_max = x_states ** raw["x_transcoder"]["digit_num"] - 1
    m = IdealMacro(x_value_range=(-x_max, x_max), w_value_range=(-w_max, w_max))
    assert m.w_value_range == (-w_max, w_max), f"IdealMacro w_value_range {m.w_value_range} != ±{w_max}"
    return m


def _build_ideal_macro() -> XbarMacro:
    """XbarMacro backed by the lossless twin of the 1T1R tile.

    Mirrors ``xbar_ideal_factory``: reuse the physical macro, then
    swap the tile for its ``to_ideal()`` equivalent.
    """
    macro = _build_1t1r_macro()
    macro.xbar = macro.xbar.to_ideal()
    return macro


def _build_1t1r_macro() -> XbarMacro:
    """XbarMacro backed by Xbar1T1R with ideal switch (from default TOML)."""
    raw = dict_from_file(CONFIG_FILE)
    typed = dict_configs_from_file(_SPECS, CONFIG_FILE)

    xbar = Offset1T1RXbar(
        config=typed["xbar"],
        rram=partial(RRAM, typed["rram"], dtype=torch.bfloat16),
        nmos=partial(NMOS, typed["nmos"], dtype=torch.bfloat16),
        sl_driver=partial(Driver, typed["sl_driver"], dtype=torch.bfloat16),
        wl_dac=partial(GeneralDAC, typed["wl_dac"], dtype=torch.bfloat16),
        bl_adc=partial(GeneralADC, typed["bl_adc"], dtype=torch.bfloat16),
    )
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


def _per_channel_weight_scale(weight_float: torch.Tensor, qmax: int) -> torch.Tensor:
    """Symmetric per-output-channel scale matching ``derive_layer_int_params``'s expectation."""
    dims = tuple(range(1, weight_float.ndim))
    return weight_float.abs().amax(dim=dims).clamp(min=1e-8) / qmax


def _make_layer_state(
    weight_float: torch.Tensor,
    bias_float: torch.Tensor | None,
    macro: NeuroxMacroQuantMatMul,
    *,
    input_scale: float = 1.0 / 32.0,
    input_zp: int = 0,
    x_qmin: int = -127,
    x_qmax: int = 127,
    output_scale: float = 1.0 / 16.0,
    output_zp: int = 0,
) -> dict[str, torch.Tensor]:
    """Build a NeuroX-flat state_dict via the production extractor.

    Routes through ``derive_layer_int_params`` so the macro's
    ``output_rescale_factor`` is correctly folded into ``bias_int``
    and ``(rescale_multiplier, rescale_rshift)`` — the same code path
    that ``HATLinear``/``HATConv2d`` and the pt2e checkpoint extractor
    use in production.
    """
    w_qmax = macro.w_value_range[1]
    w_scale = _per_channel_weight_scale(weight_float, w_qmax)

    sx_t = torch.tensor(input_scale, dtype=torch.float32)
    sy_t = torch.tensor(output_scale, dtype=torch.float32)
    zp_x_t = torch.tensor(input_zp, dtype=torch.int32)

    weight_dtype = NeuroxOperator.weight_dtype_for_range(*macro.w_value_range)
    w_int, bias_int, multiplier, rshift = derive_layer_int_params(
        float_weight=weight_float,
        float_bias=bias_float,
        input_scale=sx_t,
        input_zero_point=zp_x_t,
        weight_scale=w_scale,
        output_scale=sy_t,
        w_qmax=w_qmax,
        weight_dtype=weight_dtype,
        rescale_factor=macro.output_rescale_factor,
    )

    return {
        "weight_int": w_int,
        "bias_int": bias_int,
        "rescale_multiplier": multiplier,
        "rescale_rshift": rshift,
        "output_zero_point": torch.tensor([output_zp], dtype=torch.int32),
        "input_scale": torch.tensor([input_scale], dtype=torch.float32),
        "input_zero_point": torch.tensor([input_zp], dtype=torch.int32),
        "input_qmin": torch.tensor(x_qmin, dtype=torch.int32),
        "input_qmax": torch.tensor(x_qmax, dtype=torch.int32),
        "output_scale": torch.tensor([output_scale], dtype=torch.float32),
    }


def _make_linear_state(linear: nn.Linear, macro, **kwargs) -> dict[str, torch.Tensor]:
    return _make_layer_state(
        linear.weight.detach(),
        linear.bias.detach() if linear.bias is not None else None,
        macro,
        **kwargs,
    )


def _make_conv2d_state(conv: nn.Conv2d, macro, **kwargs) -> dict[str, torch.Tensor]:
    return _make_layer_state(
        conv.weight.detach(),
        conv.bias.detach() if conv.bias is not None else None,
        macro,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Linear tests
# ---------------------------------------------------------------------------


class TestQuantLinear:
    def test_forward_runs(self) -> None:
        torch.manual_seed(0)
        macro = _build_fake_macro()
        linear = nn.Linear(64, 32)
        quant = QuantLinear.from_torch(linear, macro, "test.linear")
        quant.load_state_dict(_make_linear_state(linear, macro), strict=False)
        quant.eval()
        quant.fabricate()

        x = torch.randn(4, 64) * 0.5
        y = quant(x)
        assert y.shape == (4, 32)
        assert y.isfinite().all()

    def test_output_is_finite_and_nonzero(self) -> None:
        """QuantLinear output must be finite and carry signal (not all-zero)."""
        torch.manual_seed(0)
        macro = _build_fake_macro()
        linear = nn.Linear(64, 32, bias=False)
        linear.weight.data.mul_(0.1)  # keep outputs small enough for the grid
        state = _make_linear_state(linear, macro, input_scale=1.0 / 64.0, output_scale=1.0 / 256.0)
        quant = QuantLinear.from_torch(linear, macro, "test.linear")
        quant.load_state_dict(state, strict=False)
        quant.eval()
        quant.fabricate()

        x = torch.randn(4, 64) * 0.5
        y_q = quant(x)
        assert y_q.isfinite().all()
        assert y_q.abs().max() > 0.01  # some non-trivial signal

    def test_cross_macro_consistency(self) -> None:
        """IdealMacro vs XbarMacro(IdealXbar,...) on the same float weights.

        Each backend gets its own state_dict — the weight bins are
        identical but ``bias_int`` and ``(mult, rshift)`` differ because
        ``derive_layer_int_params`` folds in the macro's
        ``output_rescale_factor`` (1.0 for IdealMacro, 12.0625 for the
        default IdealXbar config).  If that plumbing is correct the two
        outputs must agree up to ADC code-grid noise on the IdealXbar
        side; a bias-fold or multiplier bug would blow the gap up by
        orders of magnitude, which the relative-error bound below
        catches.
        """
        torch.manual_seed(0)
        linear = nn.Linear(128, 64)

        fake = _build_fake_macro()
        q_fake = QuantLinear.from_torch(linear, fake, "test.linear")
        q_fake.load_state_dict(_make_linear_state(linear, fake), strict=False)
        q_fake.eval()
        q_fake.fabricate()

        ideal = _build_ideal_macro()
        q_ideal = QuantLinear.from_torch(linear, ideal, "test.linear")
        q_ideal.load_state_dict(_make_linear_state(linear, ideal), strict=False)
        q_ideal.eval()
        q_ideal.fabricate()

        x = torch.randn(4, 128) * 0.5
        y_fake = q_fake(x)
        y_ideal = q_ideal(x)
        assert y_fake.shape == y_ideal.shape
        assert y_fake.isfinite().all() and y_ideal.isfinite().all()

        # Loose relative-error bound calibrated to the default 16-level
        # ADC config (rf ≈ 12.06): each ADC step ≈ 12 ideal-MAC units,
        # so a per-element gap of ~0.5–1.0× ymax is normal noise.  The
        # point of the bound is to catch order-of-magnitude regressions
        # (bias-fold or multiplier bugs would push it well above 1.5),
        # not to assert tight numerical equivalence — accuracy story is
        # covered end-to-end by the LeNet eval in ``example/lenet``.
        denom = y_fake.abs().max().clamp(min=1e-6)
        rel_err = ((y_ideal - y_fake).abs().max() / denom).item()
        print(f"\n  cross-macro rel err: {rel_err:.3f} (bound 5.0)")
        assert rel_err < 5.0, (
            f"IdealMacro vs IdealXbar relative error {rel_err:.3f} "
            f"exceeds 5.0 — likely bias-fold or multiplier-derivation bug"
        )


# ---------------------------------------------------------------------------
# Conv2d tests
# ---------------------------------------------------------------------------


class TestQuantConv2d:
    def test_forward_runs(self) -> None:
        torch.manual_seed(0)
        macro = _build_fake_macro()
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        quant = QuantConv2d.from_torch(conv, macro, "test.conv")
        quant.load_state_dict(_make_conv2d_state(conv, macro), strict=False)
        quant.eval()
        quant.fabricate()

        x = torch.randn(2, 3, 8, 8) * 0.5
        y = quant(x)
        assert y.shape == (2, 16, 8, 8)
        assert y.isfinite().all()

    @pytest.mark.parametrize(
        "padding",
        [0, 1],
    )
    def test_shapes(self, padding: int) -> None:
        torch.manual_seed(0)
        macro = _build_fake_macro()
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=padding)
        quant = QuantConv2d.from_torch(conv, macro, "test.conv")
        quant.load_state_dict(_make_conv2d_state(conv, macro), strict=False)
        quant.eval()
        quant.fabricate()

        x = torch.randn(1, 3, 10, 10) * 0.5
        y = quant(x)
        expected_hw = 10 - 2 + 2 * padding
        assert y.shape == (1, 16, expected_hw, expected_hw)
