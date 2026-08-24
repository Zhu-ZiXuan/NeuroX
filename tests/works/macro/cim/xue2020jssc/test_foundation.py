"""CPU-only eager foundation test for the xue2020jssc SINWP 1T1R CIM sub-array.

Covers the build / dispatch / derivation / validation foundation of
`neurox/works/macro/cim/xue2020jssc` on the hand-built near-ideal witness
config (`_utils.build_config`):

  * `CimMacro.from_config` registry dispatch reaches
    `Xue2020JsscCimMacro`, and the `to_dict` / `from_dict` reflection
    round trip re-selects `Xue2020JsscCimMacroConfig` through the family
    discriminator,
  * the derived-geometry laws (never stored): `phys_col_num = output_num *
    w_digit_num * 2` and `io_num = output_num // mux_factor`, computed from the
    witness config's own values,
  * the derived DSWCT / SINWP-SC ratio anchors and the runtime-resolution
    conduction-window laws,
  * the logical value-domain contract and the quantization surface: the
    published windows, the mode-index guard, the `r_b = r_B * 2**(B - b)`
    rescale law with its lossless-oracle sentinel, the magnitude input-code
    map, and the one-bit-wider ideal twin,
  * config validation rejects the exact-reshape `output_num % mux_factor`
    violation, a `w_digit_radix < 2` weight structure, a negative input-phase
    duration, and a mode count that does not match the reference ladder rows,
  * the GENERALIZED weight / input geometry — no fixed `w_digit_num` or
    `w_digit_radix` is imposed: `w_digit_num = 1` (a single polarity digit,
    ratios degenerate to the MSB anchor), `w_digit_num = 3`, and
    `input_bit_num = 1` all validate and derive the right geometry / windows,
  * a `max_active_num` that does not divide `input_num`.

Construction / validation / derivation only — no DC solve. Runs eagerly
(dynamo disabled) so the `@torch.compile` solver leaf is not unrolled.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest
import torch
import torch._dynamo

from neurox.primitive.macro.cim import CimMacro, CimMacroConfig
from neurox.works.macro.cim.xue2020jssc import Xue2020JsscCimMacro, Xue2020JsscCimMacroConfig

from ._utils import (
    TINY_ADC_BITS,
    TINY_INPUT_NUM,
    TINY_OUTPUT_NUM,
    build_all_off_policy,
    build_config,
    build_macro,
)


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is `@torch.compile`; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


# ---------------------------------------------------------------------------
# Registry dispatch + reflection
# ---------------------------------------------------------------------------


def test_registry_dispatch() -> None:
    """`CimMacro.from_config` dispatch reaches the scheme class."""
    config = build_config()
    macro = CimMacro.from_config(
        config=config,
        policy=build_all_off_policy(),
        input_num=TINY_INPUT_NUM,
        output_num=TINY_OUTPUT_NUM,
        inst_shape=(),
        dtype=torch.float64,
        T__K=300.0,
    )
    assert isinstance(macro, Xue2020JsscCimMacro)


def test_dict_reflection_round_trip() -> None:
    """`to_dict` / `from_dict` re-selects the scheme config through the family tag."""
    config = build_config()
    restored = CimMacroConfig.from_dict(config.to_dict())
    assert isinstance(restored, Xue2020JsscCimMacroConfig)
    assert restored == config


# ---------------------------------------------------------------------------
# Derived geometry / ratios (never stored)
# ---------------------------------------------------------------------------


def test_derived_geometry_laws() -> None:
    """Physical geometry derives from the constructed macro dimensions."""
    macro = build_macro(build_config(mux_factor=4), output_num=8)
    assert macro.col_num // macro.config.mux_factor == 2
    assert macro.array.weight_grid_shape[-2:] == (32, macro.row_num)

    macro_d1 = build_macro(build_config(w_digit_num=1))
    macro_d3 = build_macro(build_config(w_digit_num=3))
    assert macro_d1.array.weight_grid_shape[-2:] == (8, macro_d1.row_num)
    assert macro_d3.array.weight_grid_shape[-2:] == (24, macro_d3.row_num)


def test_derived_ratio_anchors() -> None:
    """DSWCT / SINWP-SC ratios derive DOWNWARD from the two MSB anchors (LSB-first)."""
    config = build_config()
    # r_d = dswct_ratio_msb * radix**(d - (D-1)); D = 2 -> [0.25, 0.5].
    assert config.digit_ratios == (0.25, 0.5)
    # s_k = sc_ratio_msb * 2**(k - (K-1)); K = 2 -> [0.25, 0.5].
    assert config.x_bit_ratios == (0.25, 0.5)
    # Composite r_d * s_k per (digit, bit) == the paper's net (1/16, 1/8, 1/8, 1/4).
    composite = sorted(r * s for r in config.digit_ratios for s in config.x_bit_ratios)
    assert composite == pytest.approx([1 / 16, 1 / 8, 1 / 8, 1 / 4])


def test_derived_ratios_degenerate_to_anchor_at_size_one() -> None:
    """A single digit / a single input bit collapses each ratio tuple to its MSB anchor.

    With `w_digit_num = 1` the DSWCT digit sum is a single leg and the ratio is
    the anchor alone (`r_0 = dswct_ratio_msb`); likewise `input_bit_num = 1`
    reduces the SINWP-SC combine to the MSB anchor with the sample-and-hold leg
    off. This is the digit-sum / bit-sum identity that makes the compute path
    degenerate cleanly.
    """
    d1 = build_config(w_digit_num=1)
    assert d1.digit_ratios == (0.5,)  # MSB anchor alone, no cross-digit combine
    k1 = build_config(input_bit_num=1)
    assert k1.x_bit_ratios == (0.5,)  # MSB anchor alone, hold leg off


# ---------------------------------------------------------------------------
# Window laws (checked over two parameterisations => a law, not a number)
# ---------------------------------------------------------------------------


def test_window_laws_default() -> None:
    """K=2 windows include the runtime ADC width in the live-bit tail."""
    config = build_config(t_sample__ns=1.0, t_settle__ns=2.0, latency_per_step__ns=3.0)
    for bits in (1, 3):
        tail = 2.0 + bits * 3.0
        assert config.tail__ns(bits) == tail
        assert config.array_windows__ns(bits) == (1.0, tail)
        assert config.sinwp_windows__ns(bits) == (1.0 + tail, tail)


def test_window_laws_three_bit() -> None:
    """K=3 repeats one physical input phase and forms SINWP suffix windows."""
    config = build_config(
        input_bit_num=3,
        t_sample__ns=2.0,
        t_settle__ns=1.0,
        latency_per_step__ns=3.0,
    )
    tail = 1.0 + 3 * 3.0
    assert config.array_windows__ns(3) == (2.0, 2.0, tail)
    assert config.sinwp_windows__ns(3) == (4.0 + tail, 2.0 + tail, tail)


def test_t_cycle_contains_the_max_resolution_access() -> None:
    """The scheduled period contains the maximum-resolution circuit latency."""
    config = build_config(input_bit_num=3, t_sample__ns=2.0, t_settle__ns=1.0)
    access__ns = config.access_latency__ns(config.adc_config.bits)
    assert config.t_cycle__ns >= access__ns
    with pytest.raises(ValueError, match="t_cycle__ns"):
        dataclasses.replace(config, t_cycle__ns=access__ns - 1.0)


def test_tmcsa_phases_fit_one_adc_decision_step() -> None:
    """The macro schedule must contain both explicit TMCSA conduction phases."""
    config = build_config(latency_per_step__ns=1.0)
    with pytest.raises(ValueError, match=r"tmcsa_config\.t_ph2__ns \+ tmcsa_config\.t_ph3__ns"):
        dataclasses.replace(
            config,
            tmcsa_config=dataclasses.replace(config.tmcsa_config, t_ph2__ns=0.8, t_ph3__ns=0.3),
        )


# ---------------------------------------------------------------------------
# Value-domain contract
# ---------------------------------------------------------------------------


def test_value_domain_contract() -> None:
    """The macro exposes the fixed K-bit input / sign-magnitude weight contract."""
    config = build_config()
    macro = build_macro(config)
    assert macro.x_value_range == (0, (1 << config.input_bit_num) - 1) == (0, 3)
    assert macro.w_value_range == (-3, 3)
    assert macro.adc_max_bits == config.adc_config.bits == TINY_ADC_BITS
    # One published window per declared mode; the mode count IS the ladder-row count.
    assert macro.quantization_input_ranges == tuple(m.quantization_input_range for m in config.modes)
    assert len(macro.quantization_input_ranges) == config.reference_config.mode_num == 1


def test_quantization_mode_out_of_range_rejected() -> None:
    """Every mode-indexed entry point rejects an index outside the declared modes."""
    macro = build_macro(build_config())
    beyond = len(macro.quantization_input_ranges)
    with pytest.raises(ValueError, match="quantization_mode"):
        macro.rescale_factor(quantization_mode=beyond, adc_bits=macro.adc_max_bits)
    with pytest.raises(ValueError, match="quantization_mode"):
        macro.map_quantization_input_code(torch.zeros(2, dtype=torch.long), quantization_mode=beyond)


def test_rescale_factor_bit_width_law() -> None:
    """`r_b = r_B * 2**(B - b)`: dropping a bit doubles what one code carries."""
    macro = build_macro(build_config())
    max_bits = macro.adc_max_bits
    r_max = macro.rescale_factor(quantization_mode=0, adc_bits=max_bits)
    assert r_max > 0.0
    for bits in range(1, max_bits):
        assert macro.rescale_factor(quantization_mode=0, adc_bits=bits) == pytest.approx(
            r_max * 2.0 ** (max_bits - bits)
        )
    # The lossless oracle sits outside the chain, and a converting width beyond
    # the physical resolution has no meaning.
    assert macro.rescale_factor(quantization_mode=0, adc_bits=None) == 1.0
    with pytest.raises(ValueError, match="adc_bits"):
        macro.rescale_factor(quantization_mode=0, adc_bits=max_bits + 1)


def test_quantization_input_code_map_is_magnitude() -> None:
    """The sign-magnitude readout discriminates `|code|` over its ladder span."""
    config = build_config()
    macro = build_macro(config)
    code = torch.tensor([-5, -1, 0, 3], dtype=torch.long)
    mapped, code_range = macro.map_quantization_input_code(code, quantization_mode=0)
    assert torch.equal(mapped, code.abs())
    # Circuit knowledge, declared by config: the taps span the magnitudes the
    # TMCSA resolves, not the window's own span — the sign never enters the
    # converter, so a mid-zero window is twice as wide as the code axis.
    window_lower, window_upper = config.modes[0].quantization_input_range
    assert code_range == config.modes[0].adc_input_code_range
    assert code_range != (0, window_upper - window_lower)


def test_ideal_twin_is_one_bit_wider() -> None:
    """The sign-magnitude twin publishes `adc_max_bits + 1` signed bits, same windows.

    A sign plus B magnitude bits spans a signed code range a zero-point
    quantizer only reaches at B + 1 bits; the twin inherits every other domain.
    """
    config = build_config()
    macro = build_macro(config)
    twin = macro.to_ideal()
    assert twin.adc_max_bits == macro.adc_max_bits + 1
    assert twin.quantization_input_ranges == macro.quantization_input_ranges
    assert twin.x_value_range == macro.x_value_range
    assert twin.w_value_range == macro.w_value_range
    assert twin.max_active_num == macro.max_active_num
    # The twin is the rescale reference: its own codes need no correction.
    assert twin.rescale_factor(quantization_mode=0, adc_bits=twin.adc_max_bits) == 1.0


def test_validate_rejects_mode_count_mismatch() -> None:
    """A mode is one threshold ladder row: `len(modes) == reference_config.mode_num`."""
    config = build_config()
    with pytest.raises(ValueError, match="len\\(modes\\)"):
        dataclasses.replace(config, modes=(*config.modes, *config.modes))


# ---------------------------------------------------------------------------
# Geometry validation errors
# ---------------------------------------------------------------------------


def test_validate_rejects_bad_mux_blocking() -> None:
    """`col_num % mux_factor != 0` breaks the exact CIM-IO reshape."""
    with pytest.raises(ValueError, match=r"col_num \(4\) % mux_factor"):
        build_macro(build_config(mux_factor=3))


def test_validate_rejects_sub_binary_radix() -> None:
    """A sign-magnitude digit needs at least the `{0, 1}` a polarity pair encodes: `w_digit_radix >= 2`."""
    config = build_config()
    with pytest.raises(ValueError, match=r"w_digit_radix \(1\) >= 2"):
        dataclasses.replace(config, w_digit_radix=1)


def test_validate_rejects_negative_input_phase_duration() -> None:
    """The repeated input phase duration cannot be negative."""
    config = build_config()
    with pytest.raises(ValueError, match="t_sample__ns"):
        dataclasses.replace(config, t_sample__ns=-1.0)


# ---------------------------------------------------------------------------
# Generalized weight and input geometry
# ---------------------------------------------------------------------------


def test_generalized_w_digit_num_accepted() -> None:
    """One and three magnitude digits both validate and build.

    The SINWP-SC digit sum degenerates to identity at a single digit, so a
    general `w_digit_num` needs no special case. A single polarity
    digit (`w_digit_num = 1`, weights in `{-1, 0, 1}`) and three digits both
    construct without raising and expose the right per-weight geometry.
    """
    d1 = build_config(w_digit_num=1)
    assert d1.w_digit_num == 1
    macro1 = build_macro(d1)
    assert macro1.w_value_range == (-1, 1)
    assert macro1.array.weight_grid_shape[-2:] == (macro1.col_num * 1 * 2, macro1.row_num)

    d3 = build_config(w_digit_num=3)  # also accepted
    assert d3.w_digit_num == 3
    macro3 = build_macro(d3)
    assert macro3.array.weight_grid_shape[-2:] == (macro3.col_num * 3 * 2, macro3.row_num)


def test_generalized_w_digit_radix_accepted() -> None:
    """A radix-three digit validates, builds, and exposes its signed range."""
    d3 = build_config(w_digit_radix=3)
    assert d3.w_digit_radix == 3
    macro = build_macro(d3)
    assert macro.w_value_range == (-8, 8)


def test_input_bit_num_one_accepted() -> None:
    """At K=1 the live bit alone occupies the tail window."""
    config = build_config(input_bit_num=1)
    assert config.input_bit_num == 1
    tail = config.tail__ns(config.adc_config.bits)
    assert config.array_windows__ns(config.adc_config.bits) == (tail,)
    assert config.sinwp_windows__ns(config.adc_config.bits) == (tail,)
    macro = build_macro(config)
    assert macro.x_value_range == (0, 1)  # single input bit


# ---------------------------------------------------------------------------
# Non-divisible max_active_num
# ---------------------------------------------------------------------------


def test_non_divisible_max_active_num_accepted() -> None:
    """The paper 256-input / 9-position geometry validates."""
    assert 256 % 9 != 0  # max_active_num need not divide input_num
    config = dataclasses.replace(build_config(), max_active_num=9)
    macro = build_macro(config, input_num=256)
    assert config.max_active_num == 9
    assert macro.row_num == 256

    # A tiny non-divisible geometry also builds a live macro (5 % 3 != 0).
    macro = build_macro(build_config(max_active_num=3), input_num=5, output_num=4)
    assert macro.max_active_num == 3
    assert isinstance(macro, Xue2020JsscCimMacro)
