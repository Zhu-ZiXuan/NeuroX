"""Eager end-to-end laws for the ye2023jssc WH-2T1R CIM macro.

Covers the whole macro contract on the hand-built analytic witness
(``_utils.build_config``):

  * registry dispatch reaches :class:`Ye2023JsscCimMacro` and the ``to_dict`` /
    ``from_dict`` reflection round trip re-selects the scheme config,
  * the logical geometry and the value-domain surface: the weight envelope spans
    the WEIGHT planes only while the PHYSICAL grid also carries the redundant
    (SUBA4) plane,
  * the quantization surface: the declared mode is the only index accepted, the
    rescale factor doubles per dropped bit, and the mapped input codes are the
    identity of an unsigned window,
  * every bit width rides the ONE injected reference: the code at ``b`` bits
    equals the max-bits code right-shifted by the bit deficit, and only
    ``adc_bits`` in ``[1, adc_max_bits]`` is accepted (the readout has no
    lossless oracle),
  * the PH0 compensation is a REQUIRED macro config field the readout is handed
    verbatim; the witness seats it at the model's own all-off floor —
    ``floor * row_num * sum(weight_radix + redundant_radix)``, the redundant
    plane included — so a zero-input access lands on code 0 exactly,
  * ``to_ideal()`` publishes the macro's own windows and, driven losslessly,
    equals the UNSIGNED integer MAC oracle (property self-consistency: the
    geometric radix ladder matches the scheme's ``weight_radix``),
  * end-to-end: known weights + 1-bit inputs -> codes trailing ``[output_num]``,
    monotone in the true MAC, and a mid-range input decoding to the MAC,
  * the LSB-first asymmetric-weight regression: the ``m = 1`` plane and the
    ``m = 4`` plane decode to DIFFERENT outputs (a reversed digit order swaps
    them) — an exact expected code list,
  * the scheme models no mismatch / noise / jitter: no scheme config declares a
    sigma, no scheme policy carries a toggle, and decoding is bit-identical in
    ``train()`` mode,
  * the Fig.19 energy blocks appear under their exact channel / module names,
  * the die ensemble (``inst_shape=(die_num,)``) crossed with an input batch: one
    call over ``x [n_x, 1, row]`` reads every (input, weight) pair bit-exactly as
    the single-die macro does and bills the SUM of those dies' energy per channel.

Runs eagerly (dynamo disabled) so the ``@torch.compile`` solver leaf is not
unrolled.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Iterator

import pytest
import torch
import torch._dynamo

from neurox.common import PolicyBase
from neurox.common.profiler import NeuroxProfiler, ProfilerReport
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig
from neurox.works.macro.cim.ye2023jssc.array import Ye2023Jssc2t1rArrayConfig
from neurox.works.macro.cim.ye2023jssc.cell import Ye2023Jssc2t1rCellConfig
from neurox.works.macro.cim.ye2023jssc.rscsa import RsCsaIadcConfig

from ._utils import (
    QUANTIZATION_MODE,
    TINY_ADC_BITS,
    TINY_INPUT_NUM,
    TINY_OUTPUT_NUM,
    TINY_PLANE_NUM,
    TINY_REDUNDANT_RADIX,
    TINY_WEIGHT_RADIX,
    W_MAX,
    FLOOR__uA,
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
    assert macro.adc_max_bits == TINY_ADC_BITS
    # The witness declares the single mode its config carries.
    assert len(macro.quantization_input_ranges) == len(macro.config.modes) == 1


def test_physical_grid_includes_the_redundant_plane(device: torch.device) -> None:
    """The physical column count is ``row_num * (weight planes + redundant planes)``."""
    macro = build_macro(build_config(), device=device)
    phys_col_num = TINY_INPUT_NUM * TINY_PLANE_NUM
    assert macro.array.weight_grid_shape == (phys_col_num, TINY_OUTPUT_NUM)
    assert len(TINY_REDUNDANT_RADIX) > 0


def test_only_declared_modes_and_converting_bit_widths_are_accepted(device: torch.device) -> None:
    """An undeclared mode, the lossless sentinel, and out-of-range bits are rejected."""
    macro = build_macro(build_config(), device=device)
    w = torch.zeros((TINY_INPUT_NUM, TINY_OUTPUT_NUM), dtype=torch.long, device=device)
    macro.program(w)
    x = torch.ones(TINY_INPUT_NUM, dtype=torch.long, device=device)

    mode_num = len(macro.quantization_input_ranges)
    with pytest.raises(ValueError):
        macro.vec_mat_mul(x, quantization_mode=mode_num, adc_bits=TINY_ADC_BITS)
    with pytest.raises(ValueError):
        macro.vec_mat_mul(x, quantization_mode=-1, adc_bits=TINY_ADC_BITS)
    # The physical readout converts; it has no lossless oracle.
    with pytest.raises(ValueError):
        macro.vec_mat_mul(x, quantization_mode=QUANTIZATION_MODE, adc_bits=None)
    with pytest.raises(ValueError):
        macro.vec_mat_mul(x, quantization_mode=QUANTIZATION_MODE, adc_bits=0)
    with pytest.raises(ValueError):
        macro.vec_mat_mul(x, quantization_mode=QUANTIZATION_MODE, adc_bits=TINY_ADC_BITS + 1)


# ---------------------------------------------------------------------------
# Quantization surface: windows, rescale chain, input-code map
# ---------------------------------------------------------------------------


def test_published_windows_are_the_configured_ones(device: torch.device) -> None:
    """``quantization_input_ranges`` republishes the config's mode windows in order."""
    macro = build_macro(build_config(), device=device)
    assert macro.quantization_input_ranges == tuple(m.quantization_input_range for m in macro.config.modes)


def test_rescale_factor_doubles_per_dropped_bit(device: torch.device) -> None:
    """``r_b = r_B * 2**(B - b)``: one code carries twice as much per bit dropped."""
    macro = build_macro(build_config(), device=device)
    factors = [
        macro.rescale_factor(quantization_mode=QUANTIZATION_MODE, adc_bits=b) for b in range(1, TINY_ADC_BITS + 1)
    ]
    for coarse, fine in itertools.pairwise(factors):
        assert coarse == pytest.approx(2.0 * fine)
    # The witness ladder steps one MAC unit per code, so max bits is the identity.
    assert factors[-1] == pytest.approx(macro.config.modes[QUANTIZATION_MODE].max_bits_rescale_factor)
    with pytest.raises(ValueError):
        macro.rescale_factor(quantization_mode=QUANTIZATION_MODE, adc_bits=TINY_ADC_BITS + 1)


def test_quantization_input_code_map_is_the_unsigned_identity(device: torch.device) -> None:
    """The scheme is unsigned: the zero-point map moves no code.

    The published range is the mode's declared ``adc_input_code_range`` — the
    calibration artifact itself, never a value recomputed from the window.
    """
    macro = build_macro(build_config(), device=device)
    lower, upper = macro.quantization_input_ranges[QUANTIZATION_MODE]
    assert lower == 0  # the scheme converts unsigned MACs only
    code = torch.arange(lower, upper + 1, device=device)
    mapped, code_range = macro.map_quantization_input_code(code, quantization_mode=QUANTIZATION_MODE)
    assert torch.equal(mapped, code)
    assert code_range == macro.config.modes[QUANTIZATION_MODE].adc_input_code_range


def test_lowered_bits_ride_the_shared_ladder(device: torch.device) -> None:
    """The code at ``b`` bits is the max-bits code right-shifted by the bit deficit.

    All bit widths ride the ONE injected reference current, from which the
    readout derives its whole max-bits ladder, so lowering the width drops the
    code's low bits instead of re-scaling the transfer.
    """
    macro = build_macro(build_config(), device=device)
    # Column j holds weight j % (W_MAX + 1) on every input, so one access with
    # both inputs high sweeps MACs across the whole 4-bit code range.
    values = torch.arange(TINY_OUTPUT_NUM, device=device) % (W_MAX + 1)
    w = values.expand(TINY_INPUT_NUM, TINY_OUTPUT_NUM).contiguous().long()
    x = torch.ones(TINY_INPUT_NUM, dtype=torch.long, device=device)

    full = decode(macro, w, x, adc_bits=TINY_ADC_BITS).long()
    assert int(full.max()) > 1, f"the witness sweep must exercise the ladder: {full.tolist()}"
    for bits in range(1, TINY_ADC_BITS + 1):
        lowered = decode(macro, w, x, adc_bits=bits).long()
        assert torch.equal(lowered, full >> (TINY_ADC_BITS - bits)), f"bits {bits}: {lowered.tolist()}"


# ---------------------------------------------------------------------------
# Seated PH0 compensation
# ---------------------------------------------------------------------------


def test_ph0_is_the_configured_seat(device: torch.device) -> None:
    """The macro hands the readout its own ``i_ph0_comp__uA`` field, unmodified."""
    config = build_config()
    macro = build_macro(config, device=device)
    assert macro.rscsa.i_ph0_comp__uA == config.i_ph0_comp__uA
    # The witness seats the all-off row floor, redundant plane included: dropping
    # that plane would shrink the seat.
    assert config.i_ph0_comp__uA == pytest.approx(expected_ph0__uA())
    weight_only = FLOOR__uA * TINY_INPUT_NUM * sum(TINY_WEIGHT_RADIX)
    assert config.i_ph0_comp__uA > weight_only
    # The compensation is the MACRO's seat; the readout config declares no field.
    assert not any("ph0" in field.name.lower() for field in dataclasses.fields(macro.config.adc_config))


def test_ph0_seat_is_required_and_non_negative() -> None:
    """``i_ph0_comp__uA`` is a required physical field, rejected when negative."""
    config = build_config()
    assert "i_ph0_comp__uA" in {field.name for field in dataclasses.fields(config)}
    with pytest.raises(ValueError):
        dataclasses.replace(config, i_ph0_comp__uA=-1.0)


def test_zero_input_decodes_code_zero(device: torch.device) -> None:
    """A zero-MAC access lands on code 0 exactly, for any programmed weights."""
    macro = build_macro(build_config(), device=device)
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)
    code = decode(macro, w, torch.zeros(TINY_INPUT_NUM, dtype=torch.long, device=device))
    assert torch.equal(code.long(), torch.zeros_like(code.long()))


# ---------------------------------------------------------------------------
# to_ideal: the twin inherits the published surface
# ---------------------------------------------------------------------------


def test_to_ideal_lossless_matches_unsigned_oracle(device: torch.device) -> None:
    """``to_ideal().vec_mat_mul(adc_bits=None)`` equals the UNSIGNED integer MAC.

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
    got = ideal.vec_mat_mul(x, quantization_mode=QUANTIZATION_MODE, adc_bits=None).cpu()
    expected = ideal_mac(w_val, x, clamp=False)
    assert torch.equal(got.long(), expected)


def test_to_ideal_twin_inherits_the_published_quantization_surface(device: torch.device) -> None:
    """The twin quantizes the macro's own windows at the macro's own max bits."""
    macro = build_macro(build_config(), device=device)
    ideal = macro.to_ideal()
    assert ideal.quantization_input_ranges == macro.quantization_input_ranges
    assert ideal.adc_max_bits == macro.adc_max_bits
    assert ideal.x_value_range == macro.x_value_range
    assert ideal.w_value_range == macro.w_value_range
    # The ideal codes are the rescale reference, so the twin's factor is the identity.
    assert ideal.rescale_factor(quantization_mode=QUANTIZATION_MODE, adc_bits=TINY_ADC_BITS) == 1.0


# ---------------------------------------------------------------------------
# End-to-end: shape, monotonicity, mid-range decode
# ---------------------------------------------------------------------------


def test_end_to_end_shape_and_mac(device: torch.device) -> None:
    """Codes equal the unsigned MAC."""
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
# Energy blocks
# ---------------------------------------------------------------------------


def test_energy_channels(device: torch.device) -> None:
    """Every Fig.19 block appears under its exact name, all positive."""
    macro = build_macro(build_config(), device=device)
    w_val = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)
    x = torch.ones(TINY_INPUT_NUM, dtype=torch.long, device=device)

    macro.program(w_val)
    with NeuroxProfiler() as prof, torch.no_grad():
        macro.vec_mat_mul(x, quantization_mode=QUANTIZATION_MODE, adc_bits=TINY_ADC_BITS)
    report = prof.report(macro)
    by_name = report.energy_by_name

    # Array block: self-billed per-access caps ("array") + the two converter
    # banks' own drive rows + the macro-billed conduction channels; then the
    # readout and the flat seats.
    for key in ("array", "wl_dac", "bl_dac", ".bl_cond", ".dl_cond", "rscsa", ".mux_driver", ".timing_ctrl"):
        assert key in by_name, f"missing energy block {key!r}; have {sorted(by_name)}"
        assert by_name[key] > 0.0, f"non-positive energy block {key!r}: {by_name[key]}"


# ---------------------------------------------------------------------------
# Die ensemble crossed with an input batch
# ---------------------------------------------------------------------------

# Two dies holding DIFFERENT weights, three input vectors; every MAC stays inside
# the 4-bit code range, so the laws read codes, not saturation.
_DIE_NUM = 2
_CROSS_BATCH = 3
_CROSS_W = (((1, 2, 3, 0), (7, 0, 4, 1)), ((0, 7, 1, 5), (2, 3, 0, 6)))  # [die, in, out]
_CROSS_X = (((1, 1),), ((1, 0),), ((0, 1),))  # [batch, 1, in] — the 1 broadcasts over the dies


def _crossed_run(
    device: torch.device,
) -> tuple[Ye2023JsscCimMacro, torch.Tensor, torch.Tensor, ProfilerReport, ProfilerReport]:
    """Run one crossed ensemble call and the ``die_num * batch`` single-die runs it stands for.

    Returns:
        The ensemble macro, its codes, the single-die reference codes, and the
        two profiler reports (crossed call, then the reference runs).
    """
    config = build_config()
    ensemble = build_macro(config, device=device, inst_shape=(_DIE_NUM,))
    scalar = build_macro(config, device=device)
    w = torch.tensor(_CROSS_W, dtype=torch.long, device=device)
    x = torch.tensor(_CROSS_X, dtype=torch.long, device=device)

    with NeuroxProfiler() as prof_cross, torch.no_grad():
        ensemble.program(w)
        code_cross = ensemble.vec_mat_mul(x, quantization_mode=QUANTIZATION_MODE, adc_bits=TINY_ADC_BITS)
    report_cross = prof_cross.report(ensemble)

    code_ref = torch.empty_like(code_cross)
    with NeuroxProfiler() as prof_ref, torch.no_grad():
        for die in range(_DIE_NUM):
            scalar.program(w[die])
            for batch in range(_CROSS_BATCH):
                code_ref[batch, die] = scalar.vec_mat_mul(
                    x[batch, 0], quantization_mode=QUANTIZATION_MODE, adc_bits=TINY_ADC_BITS
                )
    return ensemble, code_cross, code_ref, report_cross, prof_ref.report(scalar)


def test_crossed_ensemble_codes_equal_the_single_die_codes(device: torch.device) -> None:
    """One crossed call returns codes bit-exact per (input, weight) pair."""
    _macro, code_cross, code_ref, _rc, _rr = _crossed_run(device)
    assert code_cross.shape == (_CROSS_BATCH, _DIE_NUM, TINY_OUTPUT_NUM)
    # The dies hold different weights, so the law is not vacuous.
    assert not torch.equal(code_cross[:, 0], code_cross[:, 1])
    assert torch.equal(code_cross.long(), code_ref.long())


def test_crossed_ensemble_bills_the_sum_of_its_dies(device: torch.device) -> None:
    """Every dynamic channel of the crossed call equals the sum over the single-die runs."""
    _macro, _cc, _cr, report_cross, report_ref = _crossed_run(device)
    cross__fJ, ref__fJ = report_cross.energy_by_name, report_ref.energy_by_name
    assert set(cross__fJ) == set(ref__fJ)
    for name, e__fJ in ref__fJ.items():
        assert cross__fJ[name] == pytest.approx(e__fJ, rel=1e-9), f"channel {name!r} does not bill per die"
    assert report_cross.total_dynamic_energy__fJ == pytest.approx(report_ref.total_dynamic_energy__fJ, rel=1e-9)


def test_static_report_seats(device: torch.device) -> None:
    """The static walk seats every configured PPA reporter (macro root + children)."""
    macro = build_macro(build_config(), device=device)
    static = {r.qualified_name: r.leakage_power__uW for r in NeuroxProfiler.collect_static(macro)}
    # Every owned block is seated, converter banks and array included.
    for seat in ("", "array", "wl_dac", "bl_dac", "rscsa", "bl_driver", "sl_driver", "mux_driver", "timing_ctrl"):
        assert seat in static, f"missing static seat {seat!r}; have {sorted(static)}"
    # The array's lattice rests at zero cell bias and holds no static conduction
    # path, so it seats area without leakage; the configured leakers do leak.
    for seat in ("", "rscsa", "bl_driver", "sl_driver", "mux_driver", "timing_ctrl"):
        assert static[seat] > 0.0, f"non-positive leakage seat {seat!r}: {static[seat]}"
