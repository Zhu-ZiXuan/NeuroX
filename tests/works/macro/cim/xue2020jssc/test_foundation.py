"""CPU-only eager foundation test for the xue2020jssc SINWP 1T1R CIM sub-array.

Covers the build / dispatch / derivation / validation foundation of
``neurox/works/macro/cim/xue2020jssc`` on the hand-built near-ideal witness
config (``_utils.build_config``):

  * ``CimMacro.from_config`` registry dispatch reaches
    :class:`Xue2020JsscCimMacro`, and the ``to_dict`` / ``from_dict`` reflection
    round trip re-selects :class:`Xue2020JsscCimMacroConfig` through the family
    discriminator,
  * the derived-geometry laws (never stored): ``phys_col_num = output_num *
    w_digit_num * 2`` and ``io_num = output_num // mux_factor``, computed from the
    witness config's own values,
  * the derived DSWCT / SINWP-SC ratio anchors and the conduction-window laws
    (``t_other``, ``window_array``, ``window_sc``) as laws — checked over two
    distinct window parameterisations,
  * the logical value-domain contract and ADC mode / bit surface,
  * config validation rejects the exact-reshape ``output_num % mux_factor``
    violation, a ``w_digit_radix < 2`` weight structure, and a ``t_sample__ns``
    length that is not ``input_bit_num - 1``,
  * the GENERALIZED weight / input geometry — the fixed ``w_digit_num == 2`` /
    ``w_digit_radix == 2`` guards are GONE: ``w_digit_num = 1`` (a single P/N
    digit, ratios degenerate to the MSB anchor), ``w_digit_num = 3``, and
    ``input_bit_num = 1`` all validate and derive the right geometry / windows,
  * a ``max_active_num`` that does not divide ``input_num``.

Construction / validation / derivation only — no DC solve. Runs eagerly
(dynamo disabled) so the ``@torch.compile`` solver leaf is not unrolled.
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
    """Run eagerly — the solver leaf is ``@torch.compile``; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


# ---------------------------------------------------------------------------
# Registry dispatch + reflection
# ---------------------------------------------------------------------------


def test_registry_dispatch() -> None:
    """``CimMacro.from_config`` dispatch reaches the scheme class."""
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
    """``to_dict`` / ``from_dict`` re-selects the scheme config through the family tag."""
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

    With ``w_digit_num = 1`` the DSWCT digit sum is a single leg and the ratio is
    the anchor alone (``r_0 = dswct_ratio_msb``); likewise ``input_bit_num = 1``
    reduces the SINWP-SC combine to the MSB anchor with the sample-and-hold leg
    off. This is the digit-sum / bit-sum identity that makes the compute path
    degenerate cleanly.
    """
    d1 = build_config(w_digit_num=1)
    assert d1.digit_ratios == (0.5,)  # MSB anchor alone, no cross-digit combine
    k1 = build_config(input_bit_num=1, t_sample__ns=())
    assert k1.x_bit_ratios == (0.5,)  # MSB anchor alone, hold leg off


# ---------------------------------------------------------------------------
# Window laws (checked over two parameterisations => a law, not a number)
# ---------------------------------------------------------------------------


def test_window_laws_default() -> None:
    """K=2 windows: ``t_other`` = settle + sum(step latency); array / SC suffix sums.

    The witness sets a nonzero ``step_latency__ns`` (the honest per-step SAR
    sensing durations feed ``t_other``), so ``t_other`` strictly exceeds
    ``t_settle`` — the sensing is included in the read window while ``t_settle``
    stays the pure non-sensing settle.
    """
    config = build_config(t_sample__ns=(1.0,), t_settle__ns=2.0)
    assert config.t_other__ns == 2.0 + sum(config.adc_config.step_latency__ns)
    assert config.t_other__ns > config.t_settle__ns  # sensing widens the read window
    assert config.window_array__ns == (1.0, config.t_other__ns)
    # SC leg k window = sum(t_sample[k:]) + t_other.
    assert config.window_sc__ns == (1.0 + config.t_other__ns, config.t_other__ns)


def test_window_laws_three_bit() -> None:
    """The same window laws hold at K=3 with distinct sample windows (law, not number)."""
    config = build_config(input_bit_num=3, t_sample__ns=(2.0, 3.0), t_settle__ns=1.0)
    t_other = 1.0 + sum(config.adc_config.step_latency__ns)  # settle + sum(step_latency)
    assert config.t_other__ns == t_other
    # window_array: sampled bits use their own window; the live bit uses t_other.
    assert config.window_array__ns == (2.0, 3.0, t_other)
    # window_sc: suffix sums of the sample windows plus the tail.
    assert config.window_sc__ns == (2.0 + 3.0 + t_other, 3.0 + t_other, t_other)


def test_t_cycle_at_least_conduction_span() -> None:
    """``t_cycle__ns`` is the static time base and must contain the whole conduction span."""
    config = build_config(input_bit_num=3, t_sample__ns=(2.0, 3.0), t_settle__ns=1.0)
    # conduction_span = sum(t_sample) + t_other.
    assert config.conduction_span__ns == sum(config.t_sample__ns) + config.t_other__ns
    assert config.t_cycle__ns >= config.conduction_span__ns
    # A t_cycle shorter than the conduction span is rejected.
    with pytest.raises(ValueError, match="t_cycle__ns"):
        dataclasses.replace(config, t_cycle__ns=config.conduction_span__ns - 1.0)


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
    assert macro.adc_mode_num == config.reference_config.mode_num == 1
    for mode in range(config.reference_config.mode_num):
        assert macro.adc_rescale_factor(adc_mode=mode, adc_bits=config.adc_config.bits) > 0.0
    with pytest.raises(KeyError, match="adc_calibration"):
        macro.adc_rescale_factor(adc_mode=config.reference_config.mode_num, adc_bits=config.adc_config.bits)


# ---------------------------------------------------------------------------
# Geometry validation errors
# ---------------------------------------------------------------------------


def test_validate_rejects_bad_mux_blocking() -> None:
    """``col_num % mux_factor != 0`` breaks the exact CIM-IO reshape."""
    with pytest.raises(ValueError, match=r"col_num \(4\) % mux_factor"):
        build_macro(build_config(mux_factor=3))


def test_validate_rejects_sub_binary_radix() -> None:
    """A sign-magnitude digit needs at least the ``{0, 1}`` a P/N pair encodes: ``w_digit_radix >= 2``."""
    config = build_config()
    with pytest.raises(ValueError, match=r"w_digit_radix \(1\) >= 2"):
        dataclasses.replace(config, w_digit_radix=1)


def test_validate_rejects_bad_t_sample_length() -> None:
    """One sample window per SAMPLED bit: ``len(t_sample__ns) == input_bit_num - 1``."""
    config = build_config(input_bit_num=2, t_sample__ns=(1.0,))
    # K=2 wants exactly one sample window; two is rejected.
    with pytest.raises(ValueError, match="t_sample__ns"):
        dataclasses.replace(config, t_sample__ns=(1.0, 2.0))


# ---------------------------------------------------------------------------
# Generalized weight and input geometry
# ---------------------------------------------------------------------------


def test_generalized_w_digit_num_accepted() -> None:
    """One and three magnitude digits both validate and build.

    The DSWCT digit sum is a plain ``.sum(-1)`` that degenerates to identity at a
    single digit, so a general ``w_digit_num`` needs no special case. A single P/N
    digit (``w_digit_num = 1``, weights in ``{-1, 0, 1}``) and three digits both
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
    """A single input bit (``input_bit_num = 1``) validates: no sample window, live bit alone.

    K = 1 runs the live bit alone (the sample-and-hold leg off): ``t_sample__ns``
    is empty and both window vectors collapse to the single ``t_other`` entry.
    """
    config = build_config(input_bit_num=1, t_sample__ns=())
    assert config.input_bit_num == 1
    assert config.t_sample__ns == ()  # one window per sampled bit; none are sampled
    assert config.window_array__ns == (config.t_other__ns,)
    assert config.window_sc__ns == (config.t_other__ns,)
    macro = build_macro(config)
    assert macro.x_value_range == (0, 1)  # single input bit


# ---------------------------------------------------------------------------
# Non-divisible max_active_num
# ---------------------------------------------------------------------------


def test_non_divisible_max_active_num_accepted() -> None:
    """The paper 256-input / 9-position geometry validates."""
    assert 256 % 9 != 0  # not a divisor — the base rule would reject this
    config = dataclasses.replace(build_config(), max_active_num=9)
    macro = build_macro(config, input_num=256)
    assert config.max_active_num == 9
    assert macro.row_num == 256

    # A tiny non-divisible geometry also builds a live macro (5 % 3 != 0).
    macro = build_macro(build_config(max_active_num=3), input_num=5, output_num=4)
    assert macro.max_active_num == 3
    assert isinstance(macro, Xue2020JsscCimMacro)
