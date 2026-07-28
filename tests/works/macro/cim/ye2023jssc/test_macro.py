"""Eager end-to-end laws for the ye2023jssc WH-2T1R CIM macro.

Covers the whole macro contract on the hand-built analytic witness
(``_utils.build_config``):

  * registry dispatch reaches :class:`Ye2023JsscCimMacro` and the ``to_dict`` /
    ``from_dict`` reflection round trip re-selects the scheme config,
  * the logical geometry and the value-domain surface: the weight envelope spans
    the WEIGHT planes only while the PHYSICAL grid also carries the redundant
    (SUBA4) plane,
  * the readout runs ONE operating point: any ``adc_mode`` but 0 and any
    ``adc_bits`` but the RS-CSA's physical resolution are rejected,
  * the PH0 compensation is DERIVED from the model's own all-off floor —
    ``floor * row_num * sum(weight_radix + redundant_radix)``, the redundant
    plane included — so a zero-input access lands on code 0 exactly,
  * ``to_ideal()`` at ``adc_bits = 0`` equals the in-code UNSIGNED integer MAC
    oracle (property self-consistency: the geometric radix ladder matches the
    scheme's ``weight_radix``),
  * end-to-end: known weights + 1-bit inputs -> codes trailing ``[output_num]``,
    monotone in the true MAC, and a mid-range input decoding to the MAC,
  * the LSB-first asymmetric-weight regression: the ``m = 1`` plane and the
    ``m = 4`` plane decode to DIFFERENT outputs (a reversed digit order swaps
    them) — an exact expected code list,
  * the scheme models no mismatch / noise / jitter: no scheme config declares a
    sigma, no scheme policy carries a toggle, and decoding is bit-identical in
    ``train()`` mode,
  * the Fig.19 energy blocks appear under their exact channel / module names and
    the macro emits latency exactly once.

Runs eagerly (dynamo disabled) so the ``@torch.compile`` solver leaf is not
unrolled.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest
import torch
import torch._dynamo

from neurox.common import PolicyBase
from neurox.common.profiler import NeuroxProfiler
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig
from neurox.works.macro.cim.ye2023jssc.array import Ye2023Jssc2t1rArrayConfig
from neurox.works.macro.cim.ye2023jssc.cell import Ye2023Jssc2t1rCellConfig
from neurox.works.macro.cim.ye2023jssc.rscsa import RsCsaIadcConfig

from ._utils import (
    ADC_MODE,
    TINY_ADC_BITS,
    TINY_INPUT_NUM,
    TINY_OUTPUT_NUM,
    TINY_PLANE_NUM,
    TINY_REDUNDANT_RADIX,
    TINY_WEIGHT_RADIX,
    W_MAX,
    Ye2023JsscCimMacro,
    Ye2023JsscCimMacroConfig,
    build_all_off_policy,
    build_config,
    build_macro,
    decode,
    expected_ph0__uA,
    ideal_mac,
)


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is ``@torch.compile``; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


# ---------------------------------------------------------------------------
# Registry dispatch + reflection
# ---------------------------------------------------------------------------


def test_registry_dispatch(device: torch.device) -> None:
    """``CimMacro.from_config`` dispatch reaches the scheme class."""
    macro = CimMacro.from_config(
        config=build_config(),
        policy=build_all_off_policy(),
        input_num=TINY_INPUT_NUM,
        output_num=TINY_OUTPUT_NUM,
        inst_shape=(),
        dtype=torch.float64,
        T__K=300.0,
    )
    assert isinstance(macro, Ye2023JsscCimMacro)


def test_dict_reflection_round_trip() -> None:
    """``to_dict`` / ``from_dict`` re-selects the scheme config through the family tag."""
    config = build_config()
    restored = CimMacroConfig.from_dict(config.to_dict())
    assert isinstance(restored, Ye2023JsscCimMacroConfig)
    assert restored == config


# ---------------------------------------------------------------------------
# Derived geometry / value-domain surface
# ---------------------------------------------------------------------------


def test_geometry_and_properties(device: torch.device) -> None:
    """The derived geometry and the ADC / digit properties are as expected."""
    macro = build_macro(build_config(), device=device)
    assert macro.row_num == TINY_INPUT_NUM
    assert macro.col_num == TINY_OUTPUT_NUM
    assert macro.max_active_num == TINY_INPUT_NUM

    assert macro.x_value_range == (0, 1)
    # The weight envelope spans the WEIGHT planes only; the redundant plane is
    # not part of the encodable range.
    assert macro.w_value_range == (0, W_MAX)
    assert macro.config.w_digit_num == len(TINY_WEIGHT_RADIX)
    assert macro.adc_mode_num == 1
    assert macro.adc_max_bits == TINY_ADC_BITS
    assert macro.adc_rescale_factor(adc_mode=ADC_MODE, adc_bits=TINY_ADC_BITS) == 1.0
    with pytest.raises(KeyError):
        macro.adc_rescale_factor(adc_mode=0, adc_bits=1)


def test_physical_grid_includes_the_redundant_plane(device: torch.device) -> None:
    """The physical column count is ``row_num * (weight planes + redundant planes)``."""
    macro = build_macro(build_config(), device=device)
    phys_col_num = TINY_INPUT_NUM * TINY_PLANE_NUM
    assert macro.array.weight_grid_shape == (phys_col_num, TINY_OUTPUT_NUM)
    assert len(TINY_REDUNDANT_RADIX) > 0


def test_single_operating_point_is_enforced(device: torch.device) -> None:
    """Any ``adc_mode`` but 0, or any ``adc_bits`` but the physical one, is rejected."""
    macro = build_macro(build_config(), device=device)
    w = torch.zeros((TINY_INPUT_NUM, TINY_OUTPUT_NUM), dtype=torch.long, device=device)
    macro.program(w)
    x = torch.ones(TINY_INPUT_NUM, dtype=torch.long, device=device)

    with pytest.raises(ValueError):
        macro.vec_mat_mul(x, adc_mode=1, adc_bits=TINY_ADC_BITS)
    with pytest.raises(ValueError):
        macro.vec_mat_mul(x, adc_mode=ADC_MODE, adc_bits=TINY_ADC_BITS - 1)


# ---------------------------------------------------------------------------
# Derived PH0 compensation
# ---------------------------------------------------------------------------


def test_ph0_is_derived_from_the_all_off_floor(device: torch.device) -> None:
    """PH0 == ``floor * row_num * sum(weight_radix + redundant_radix)`` — no config field."""
    macro = build_macro(build_config(), device=device)
    assert macro.rscsa.i_ph0_comp__uA == pytest.approx(expected_ph0__uA())
    # The redundant plane is part of the sum: dropping it would shrink PH0.
    weight_only = macro.config.cell_config.i_t2_table__uA[0][0] * TINY_INPUT_NUM * sum(TINY_WEIGHT_RADIX)
    assert macro.rscsa.i_ph0_comp__uA > weight_only
    assert not any("ph0" in field.name.lower() for field in dataclasses.fields(macro.config.adc_config))


def test_zero_input_decodes_code_zero(device: torch.device) -> None:
    """A zero-MAC access lands on code 0 exactly, for any programmed weights."""
    macro = build_macro(build_config(), device=device)
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)
    code = decode(macro, w, torch.zeros(TINY_INPUT_NUM, dtype=torch.long, device=device))
    assert torch.equal(code.long(), torch.zeros_like(code.long()))


# ---------------------------------------------------------------------------
# to_ideal at adc_bits = 0 == UNSIGNED integer MAC oracle
# ---------------------------------------------------------------------------


def test_to_ideal_bits0_matches_unsigned_oracle(device: torch.device) -> None:
    """``to_ideal().vec_mat_mul(adc_bits=0)`` equals the UNSIGNED integer MAC.

    Validates that ``to_ideal`` preserves the logical weight contract.
    """
    macro = build_macro(build_config(), device=device)
    ideal = macro.to_ideal()

    w_val = torch.tensor(
        [[0, 1], [1, 1], [2, 3], [7, 0]],
        dtype=torch.long,
        device=device,
    ).transpose(-1, -2)
    x = torch.tensor([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=torch.long, device=device)  # batch (4,)

    ideal.program(w_val)
    got = ideal.vec_mat_mul(x, adc_mode=ADC_MODE, adc_bits=0).cpu()
    expected = ideal_mac(w_val, x, clamp=False)
    assert torch.equal(got.long(), expected)


# ---------------------------------------------------------------------------
# End-to-end: shape, monotonicity, mid-range decode
# ---------------------------------------------------------------------------


def test_end_to_end_shape_and_mac(device: torch.device) -> None:
    """Codes trail ``[output_num]`` and equal the unsigned MAC."""
    macro = build_macro(build_config(), device=device)
    w_val = torch.tensor(
        [[0, 1], [1, 1], [2, 3], [7, 0]],
        dtype=torch.long,
        device=device,
    ).transpose(-1, -2)
    x = torch.tensor([1, 1], dtype=torch.long, device=device)  # single access

    code = decode(macro, w_val, x)
    assert code.shape == (TINY_OUTPUT_NUM,)
    # Analytic chain: code == clamped UNSIGNED MAC exactly.
    assert torch.equal(code.long(), ideal_mac(w_val, x))


def test_decode_monotone_in_mac(device: torch.device) -> None:
    """The decoded code is monotone non-decreasing in the true MAC of one output."""
    macro = build_macro(build_config(), device=device)
    # Output 0 holds the max weight (7) on input 0; sweep the input on/off gives
    # MAC 0 -> 7. A second input row raises it further.
    w_val = torch.tensor(
        [[7, 1], [0, 0], [0, 0], [0, 0]],
        dtype=torch.long,
        device=device,
    ).transpose(-1, -2)
    xs = [
        torch.tensor([0, 0], dtype=torch.long, device=device),  # MAC 0
        torch.tensor([0, 1], dtype=torch.long, device=device),  # MAC 1
        torch.tensor([1, 0], dtype=torch.long, device=device),  # MAC 7
        torch.tensor([1, 1], dtype=torch.long, device=device),  # MAC 8
    ]
    codes = [int(decode(macro, w_val, x)[0]) for x in xs]
    assert codes == sorted(codes), f"not monotone: {codes}"
    assert codes[0] < codes[-1], f"no dynamic range: {codes}"
    # Mid-range input decodes to the MAC.
    assert codes[1] == 1 and codes[3] == 8


# ---------------------------------------------------------------------------
# LSB-first asymmetric-weight regression
# ---------------------------------------------------------------------------


def test_lsb_first_asymmetric_weight_regression(device: torch.device) -> None:
    """The ``m = 1`` plane and the ``m = 4`` plane decode to DIFFERENT outputs.

    Output 0 carries weight 1 (only the m=1 plane, digit 0) and output 1 carries
    weight 4 (only the m=4 plane, digit 2). With a single input high, output 0
    reads 1 and output 1 reads 4. A reversed digit order (digit 0 -> m=4) would
    swap them — the exact expected list pins the LSB-first convention.
    """
    macro = build_macro(build_config(), device=device)
    # [out, in]: out0 = weight 1, out1 = weight 4, out2/3 = 0.
    w_val = torch.tensor(
        [[1, 0], [4, 0], [0, 0], [0, 0]],
        dtype=torch.long,
        device=device,
    ).transpose(-1, -2)
    x = torch.tensor([1, 0], dtype=torch.long, device=device)  # only input 0 high

    code = decode(macro, w_val, x)
    assert code.tolist() == [1, 4, 0, 0], f"LSB-first place-value broken: {code.tolist()}"


# ---------------------------------------------------------------------------
# No mismatch / noise / jitter anywhere in the scheme
# ---------------------------------------------------------------------------


def test_no_scheme_config_declares_a_statistical_spread() -> None:
    """No scheme-owned config field names a sigma / mismatch / noise / jitter knob."""
    banned = ("sigma", "mismatch", "noise", "jitter")
    for config_type in (
        Ye2023JsscCimMacroConfig,
        Ye2023Jssc2t1rArrayConfig,
        Ye2023Jssc2t1rCellConfig,
        RsCsaIadcConfig,
    ):
        for field in dataclasses.fields(config_type):
            assert not any(token in field.name.lower() for token in banned), (
                f"{config_type.__name__}.{field.name} declares a statistical spread"
            )


def test_scheme_policies_carry_no_toggles() -> None:
    """The array / cell / RS-CSA policies are source-free; only the kernel clamps hold toggles."""
    policy = build_all_off_policy()
    for child in (policy.array_policy, policy.array_policy.cell_policy, policy.adc_policy):
        assert isinstance(child, PolicyBase)
        bool_fields = [f.name for f in dataclasses.fields(child) if isinstance(getattr(child, f.name), bool)]
        assert bool_fields == [], f"{type(child).__name__} carries toggles {bool_fields}"


def test_decode_is_deterministic_in_training_mode(device: torch.device) -> None:
    """No conversion jitter is wired: ``train()`` decodes exactly like ``eval()``."""
    macro = build_macro(build_config(), device=device)
    w = torch.tensor(
        [[1, 3], [7, 0], [2, 5], [4, 4]],
        dtype=torch.long,
        device=device,
    ).transpose(-1, -2)
    x = torch.tensor([1, 1], dtype=torch.long, device=device)
    eval_code = decode(macro, w, x)
    macro.train()
    assert torch.equal(decode(macro, w, x), eval_code)
    assert torch.equal(decode(macro, w, x), eval_code)


# ---------------------------------------------------------------------------
# Energy blocks + single latency
# ---------------------------------------------------------------------------


def test_energy_channels_and_single_latency(device: torch.device) -> None:
    """Every Fig.19 block appears under its exact name, all positive; latency once."""
    macro = build_macro(build_config(), device=device)
    w_val = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)
    x = torch.ones(TINY_INPUT_NUM, dtype=torch.long, device=device)

    macro.program(w_val)
    with NeuroxProfiler() as prof, torch.no_grad():
        macro.vec_mat_mul(x, adc_mode=ADC_MODE, adc_bits=TINY_ADC_BITS)
    report = prof.report(macro)
    by_name = report.energy_by_name

    # Array block: self-billed per-access caps ("array") + the macro-billed
    # conduction / per-vector-cap channels; then the readout and the flat seats.
    for key in ("array", ".bl_cond", ".dl_cond", ".bl_cap", "rscsa", ".mux_driver", ".timing_ctrl"):
        assert key in by_name, f"missing energy block {key!r}; have {sorted(by_name)}"
        assert by_name[key] > 0.0, f"non-positive energy block {key!r}: {by_name[key]}"

    # The macro is the sole latency emitter: exactly one latency event.
    assert len(prof.latency_events) == 1, f"expected one latency event, got {len(prof.latency_events)}"
    assert prof.total_latency__ns > 0.0


def test_static_report_seats(device: torch.device) -> None:
    """The static walk seats every configured PPA reporter (macro root + children)."""
    macro = build_macro(build_config(), device=device)
    static = {r.qualified_name: r.leakage_power__uW for r in NeuroxProfiler.collect_static(macro)}
    for seat in ("", "array", "rscsa", "bl_driver", "sl_driver", "mux_driver", "timing_ctrl"):
        assert seat in static, f"missing static seat {seat!r}; have {sorted(static)}"
        assert static[seat] > 0.0, f"non-positive leakage seat {seat!r}: {static[seat]}"
