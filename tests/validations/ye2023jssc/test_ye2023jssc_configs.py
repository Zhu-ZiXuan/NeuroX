"""Shipped validations-config parse + build test — the one sanctioned disk-config test.

Asserts the paper-design artifacts under `validations/ye2023jssc/` parse, build,
and carry the modeled design point: `params.toml` (section `cim_macro`)
registry-dispatches to `Ye2023JsscCimMacroConfig` and the module builds +
fabricates at `inst_shape=()`, `policy.toml` (section `policy`) loads with
every nonideality toggle off, and `anchors.toml` parses with its two kinds of
number kept apart. No forward pass.

Beyond parsing, the shipped config is checked against the DESIGN it is supposed
to describe (laws, not tuned magnitudes):

  * the physical grid is 64 rows (logical outputs) x 128 columns = 32 inputs x
    (3 weight planes + 1 redundant SUBA4 plane),
  * the T2 table's floor row (the `V_X = 0` operating point) is
    state-independent and the seated PH0 compensation `i_ph0_comp__uA` equals
    that floor over the 352 place-value units of one row, which reproduces the
    paper's ~1 uA row leakage, so a zero-MAC access reads code 0,
  * the drive-point HRS entry saturates the measured 30 nA bound at the largest
    place value in the radix path,
  * the readout is the paper's single 4-bit operating point and its DERIVED
    access window is 66 ns,
  * the selected word line clears the step-2 WL threshold, and the IN = 0 BL
    code drives EXACTLY 0 V — the validity contract the `V_X = 0` floor
    classification rests on,
  * no mismatch / noise / jitter field is declared anywhere in either artifact.

And against the VALIDATION CONTRACT the two TOML artifacts encode:

  * every anchor value carries a provenance tag, and the `[gate]` targets are
    caliber-independent — no Fig.19 power number appears among them,
  * the `[reference]` table holds the Fig.19 picture that is reported but not
    gated (two totals, both per-pin share breakdowns, the headline efficiency),
  * the free-parameter set in `params.toml` is exactly two `[calibrated]`
    fields (the RS-CSA mirror scale and its fixed per-op energy) plus the two
    `[transcribed]` peripheral seats.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import torch._dynamo

from neurox.common import PolicyBase
from neurox.common.serialize import dict_from_file
from neurox.primitive.analog.voltage_dac import GeneralVdacConfig
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy
from neurox.works.macro.cim.ye2023jssc import (
    Ye2023JsscCimMacro,
    Ye2023JsscCimMacroConfig,
    Ye2023JsscCimMacroPolicy,
)

# tests/validations/ye2023jssc/test_ye2023jssc_configs.py -> repo root is 3 parents up.
_VALIDATIONS_DIR = Path(__file__).resolve().parents[3] / "validations" / "ye2023jssc"

# The modeled macro geometry (Fig.7(a)): 32 logical inputs x 64 logical outputs.
_INPUT_NUM = 32
_OUTPUT_NUM = 64

# Fig.19 per-block totals [uW] at the two published sparsity points.
_TOTAL_SPARSE__uW = 31.96  # p_zero_input = 0.875
_TOTAL_DENSE__uW = 82.94  # p_zero_input = 0.50

# Paper-sourced bounds the shipped physics must respect.
_I_ROW_LEAK__uA = 1.0  # measured all-off row leakage the PH0 phase cancels
_I_HRS_BOUND__uA = 0.030  # measured < 30 nA per weight-plane leakage
_T_AC__ns = 66.0  # the derived access window

# Fields no shipped artifact may declare.
_BANNED_KEYS = (
    "t_conduct__ns",
    "i_leak__uA",
    "comparator_offset_sigma__uA",
    "coupling_mismatch_sigma__uA",
    "sigma_lrs_relative",
    "sigma_hrs_relative",
)

# The sanctioned provenance vocabulary; its legend lives in docs/validation/campaigns.md.
_TAG_PATTERN = re.compile(r"\[(measured|derived|transcribed|assumed|bound-derived|calibrated)\b[^\]]*\]")

# The one key that names no physical quantity: the registry dispatch key.
_UNTAGGED_KEYS = frozenset({"_neurox_class"})

# The sanctioned free-parameter set: two solved numbers + two transcribed seats.
_CALIBRATED_KEYS = ("mirror_scale", "e_fixed_per_op__fJ")
_TRANSCRIBED_ENTRIES = (
    ("cim_macro.mux_driver_config", "leakage_per_inst__uW"),
    ("cim_macro.timing_ctrl_config", "leakage_per_inst__uW"),
)


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — `fabricate()` touches the solver-owning array; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


def _load_config() -> Ye2023JsscCimMacroConfig:
    config = CimMacroConfig.from_file(_VALIDATIONS_DIR / "params.toml", section="cim_macro")
    assert isinstance(config, Ye2023JsscCimMacroConfig)
    return config


def _assert_all_toggles_false(obj: PolicyBase, path: str = "policy") -> None:
    """Every bool field in the (nested) policy dataclass tree is False."""
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        if isinstance(value, bool):
            assert value is False, f"{path}.{field.name} is on in the all-off policy"
        elif dataclasses.is_dataclass(value):
            assert isinstance(value, PolicyBase)
            _assert_all_toggles_false(value, f"{path}.{field.name}")


def _numeric_leaves(obj: object) -> list[float]:
    """Flatten every numeric (non-bool) scalar in a nested dict / list tree."""
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        for value in obj.values():
            out.extend(_numeric_leaves(value))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            out.extend(_numeric_leaves(value))
    return out


def _all_keys(obj: object) -> set[str]:
    """Every mapping key appearing anywhere in a nested dict / list tree."""
    keys: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            keys.add(str(key))
            keys |= _all_keys(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            keys |= _all_keys(value)
    return keys


def _tagged_entries(path: Path) -> list[tuple[str, str, tuple[str, ...]]]:
    """Every `key = value` line as `(section, key, provenance tags)`.

    A key is tagged by the tags on its own line, or else by the contiguous comment
    block directly above it (a block may govern several keys and survives a table
    header). A blank line, or a key line followed by a fresh comment block, ends a
    block, so a file header cannot leak its tag vocabulary onto the keys below it.
    """
    entries: list[tuple[str, str, tuple[str, ...]]] = []
    pending: list[str] = []
    section = ""
    previous = ""
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            pending.clear()
            previous = "blank"
        elif line.startswith("#"):
            if previous != "comment":
                pending.clear()
            pending.extend(_TAG_PATTERN.findall(line))
            previous = "comment"
        elif line.startswith("["):
            section = line.strip("[]")
            previous = "table"
        else:
            key, sep, rest = line.partition("=")
            if not sep:
                continue
            inline = _TAG_PATTERN.findall(rest)
            entries.append((section, key.strip(), tuple(inline or pending)))
            previous = "key"
    return entries


# ---------------------------------------------------------------------------
# Parse + build
# ---------------------------------------------------------------------------


def test_params_config_parses_and_builds() -> None:
    """`params.toml` (section `cim_macro`) registry-dispatches and fabricates."""
    config = _load_config()
    policy = CimMacroPolicy.from_file(_VALIDATIONS_DIR / "policy.toml", section="policy")
    assert isinstance(policy, Ye2023JsscCimMacroPolicy)

    macro = CimMacro.from_config(
        config=config,
        policy=policy,
        input_num=_INPUT_NUM,
        output_num=_OUTPUT_NUM,
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    assert isinstance(macro, Ye2023JsscCimMacro)
    macro.fabricate()


def test_policy_every_toggle_false() -> None:
    """`policy.toml` loads with every nonideality toggle off (the all-off preset)."""
    policy = CimMacroPolicy.from_file(_VALIDATIONS_DIR / "policy.toml", section="policy")
    assert isinstance(policy, Ye2023JsscCimMacroPolicy)
    _assert_all_toggles_false(policy)


def test_no_nonideality_field_is_declared() -> None:
    """Neither shipped artifact declares a mismatch / noise / jitter or window field."""
    for name in ("params.toml", "policy.toml"):
        keys = _all_keys(dict_from_file(_VALIDATIONS_DIR / name))
        for banned in _BANNED_KEYS:
            assert banned not in keys, f"{name} declares {banned}"


# ---------------------------------------------------------------------------
# The modeled design point
# ---------------------------------------------------------------------------


def test_physical_grid_is_64_rows_by_128_columns() -> None:
    """32 inputs x (3 weight + 1 redundant plane) = 128 physical columns over 64 output rows."""
    config = _load_config()
    array_config = config.array_config
    assert config.max_active_num == _INPUT_NUM  # input-parallel
    assert config.w_digit_num == 3
    assert len(array_config.redundant_radix) == 1

    plane_num = len(array_config.weight_radix) + len(array_config.redundant_radix)
    macro = CimMacro.from_config(
        config=config,
        policy=CimMacroPolicy.from_file(_VALIDATIONS_DIR / "policy.toml", section="policy"),
        input_num=_INPUT_NUM,
        output_num=_OUTPUT_NUM,
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    assert isinstance(macro, Ye2023JsscCimMacro)
    assert macro.array.weight_grid_shape == (_INPUT_NUM * plane_num, _OUTPUT_NUM)
    assert _INPUT_NUM * plane_num == 128
    # The redundant plane carries no weight: the encodable range spans the weight radix only.
    assert macro.w_value_range == (0, sum(array_config.weight_radix))


def test_leakage_floor_reproduces_the_row_leakage_and_the_hrs_bound() -> None:
    """The floor entry is state-independent and covers 352 place-value units ~= 1 uA."""
    config = _load_config()
    array_config = config.array_config
    i_t2_table = config.cell_config.i_t2_table__uA

    floor_row = i_t2_table[0]
    assert len(set(floor_row)) == 1, f"the V_X = 0 floor is state-dependent: {floor_row}"

    radix_unit_num = _INPUT_NUM * sum((*array_config.weight_radix, *array_config.redundant_radix))
    assert radix_unit_num == 352
    assert floor_row[0] * radix_unit_num == pytest.approx(_I_ROW_LEAK__uA, rel=1e-3)

    # The drive-point HRS entry keeps the radix multiplier and saturates the 30 nA
    # bound at the largest place value on the radix path.
    m_max = max((*array_config.weight_radix, *array_config.redundant_radix))
    i_hrs_drive__uA = i_t2_table[1][0]
    assert i_hrs_drive__uA * m_max == pytest.approx(_I_HRS_BOUND__uA)
    assert i_hrs_drive__uA * m_max <= _I_HRS_BOUND__uA + 1e-12
    # The LRS on-current dominates the leakage by orders of magnitude.
    assert i_t2_table[1][1] > 10.0 * i_hrs_drive__uA


def test_seated_ph0_is_the_all_off_row_leakage() -> None:
    """`i_ph0_comp__uA` is a shipped macro seat that cancels the all-off row exactly."""
    config = _load_config()
    array_config = config.array_config

    radix_unit_num = _INPUT_NUM * sum((*array_config.weight_radix, *array_config.redundant_radix))
    floor__uA = config.cell_config.i_t2_table__uA[0][0]
    # The seat IS the array's own all-off row current, so a zero-MAC access
    # subtracts to exactly zero residue and reads code 0.
    assert config.i_ph0_comp__uA == pytest.approx(floor__uA * radix_unit_num, rel=1e-12)
    assert config.i_ph0_comp__uA == pytest.approx(_I_ROW_LEAK__uA, rel=1e-3)


def test_readout_is_one_4bit_point_with_a_66ns_derived_window() -> None:
    """Single 4-bit mode; `T_AC = sum(t_phase[:-1]) + t_intrinsic[-1]` = 66 ns."""
    config = _load_config()
    adc_config = config.adc_config
    assert adc_config.bits == 4
    assert len(adc_config.t_phase__ns) == adc_config.bits + 1  # PH0 + one compare phase per bit
    # The readout takes ONE reference current and weighs it by its own
    # compare-phase place values, so the source is single-tap with one row per
    # declared mode.
    reference_config = config.reference_config
    assert reference_config.tap_num == 1
    # One declared quantization mode. Its window is the readout's own full
    # scale: 2**bits codes, each worth the MAC units one current step resolves.
    assert len(config.modes) == reference_config.mode_num == 1
    mode = config.modes[0]
    lower, upper = mode.quantization_input_range
    assert lower == 0  # the scheme converts unsigned MACs only
    # The code step IS the reference current: the ladder is c * i_ref.
    i_ref__uA = reference_config.i_refs__uA[0][0]
    mac_per_code = i_ref__uA / config.cell_config.i_t2_table__uA[1][1]
    assert upper - lower + 1 == pytest.approx((1 << adc_config.bits) * mac_per_code)
    # The window step IS the physical code step, so a code already is an ideal
    # macro code: the rescale is the identity, in ideal codes, not MAC units.
    assert mode.max_bits_rescale_factor == pytest.approx(1.0)
    # The unsigned readout rises with the MAC itself, so the input code axis it
    # discriminates on is the window.
    assert mode.adc_input_code_range == mode.quantization_input_range

    # One latch delay per compare phase, each inside the phase it closes.
    assert len(adc_config.t_intrinsic__ns) == adc_config.bits
    for phase, t in enumerate(adc_config.t_intrinsic__ns):
        assert 0.0 < t < adc_config.t_phase__ns[phase + 1]
    t_ac__ns = sum(adc_config.t_phase__ns[:-1]) + adc_config.t_intrinsic__ns[-1]
    assert t_ac__ns == pytest.approx(_T_AC__ns)


def test_selected_word_line_clears_the_step2_threshold() -> None:
    """The selected WL code turns the access device on; the IN = 0 BL code is exactly 0 V."""
    config = _load_config()
    cell_config = config.cell_config
    wl_dac_config = config.wl_dac_config
    bl_dac_config = config.bl_dac_config
    assert isinstance(wl_dac_config, GeneralVdacConfig)
    assert isinstance(bl_dac_config, GeneralVdacConfig)

    # The two drives are 1-bit: one deselected / IN = 0 code and one driven code.
    assert len(wl_dac_config.code_to_signal) == len(bl_dac_config.code_to_signal) == 2
    assert wl_dac_config.code_to_signal[1] > cell_config.v_wl_on_threshold__V
    assert wl_dac_config.code_to_signal[0] <= cell_config.v_wl_on_threshold__V
    # The cell classifies its operating point on V_X = 0, so the IN = 0 code must
    # drive EXACTLY 0 V against a grounded SL: any other level would settle the
    # branch above the floor point.
    assert bl_dac_config.code_to_signal[0] == 0.0
    assert config.v_sl__V == 0.0
    assert bl_dac_config.code_to_signal[1] > 0.0
    assert bl_dac_config.code_to_signal[1] == config.array_config.v_bl_in1__V


# ---------------------------------------------------------------------------
# The validation contract
# ---------------------------------------------------------------------------


def test_gate_targets_are_caliber_independent() -> None:
    """`[gate]` carries the five gates' targets and no Fig.19 power number."""
    anchors: dict[str, Any] = dict_from_file(_VALIDATIONS_DIR / "anchors.toml")
    gate = anchors["gate"]

    # Gates 1 (golden transfer) and 4 (zero input) are closed forms: no anchor.
    assert gate["t_ac__ns"] == pytest.approx(_T_AC__ns)
    assert gate["i_tbl"]["hrs_bound__uA"] == pytest.approx(_I_HRS_BOUND__uA)
    assert len(gate["i_tbl"]["lrs_mean__uA"]) == len(gate["i_tbl"]["lrs_mean_radix"])
    assert gate["rscsa"]["energy_per_conversion__fJ"] > 0.0
    lo, hi = gate["rscsa"]["code_spread_range"]
    assert 1.0 < lo < hi

    # A caliber decides how the conduction branches map onto the Fig.19 pins, so no
    # gate target may depend on a Fig.19 power number.
    gate_leaves = _numeric_leaves(gate)
    for power in (_TOTAL_SPARSE__uW, _TOTAL_DENSE__uW, *anchors["reference"]["shares"]["p50"].values()):
        assert not any(leaf == pytest.approx(power, rel=1e-3) for leaf in gate_leaves), (
            f"the gate targets depend on the Fig.19 number {power}"
        )


def test_reference_targets_hold_the_ungated_fig19_picture() -> None:
    """`[reference]` carries the two Fig.19 totals, both share breakdowns and the headline EF."""
    anchors: dict[str, Any] = dict_from_file(_VALIDATIONS_DIR / "anchors.toml")
    reference = anchors["reference"]

    assert reference["total_macro__uW"] == pytest.approx([_TOTAL_SPARSE__uW, _TOTAL_DENSE__uW])
    assert reference["ef_tops_w"] > 0.0
    assert len(reference["point_label"]) == len(anchors["data"]["p_zero_input"]) == 2
    # Energy per output = total power * the access window.
    for total__uW, per_access__pJ in zip(reference["total_macro__uW"], reference["per_access__pJ"], strict=True):
        assert per_access__pJ == pytest.approx(total__uW * _T_AC__ns * 1e-3, rel=2e-3)
    # Both pin breakdowns cover the same four blocks and sum to the whole macro.
    shares = reference["shares"]
    assert set(shares) == {"p875", "p50"}
    assert set(shares["p875"]) == set(shares["p50"]) == {"array", "rscsa", "mux_driver", "timing_ctrl"}
    for point in shares.values():
        assert sum(point.values()) == pytest.approx(100.0, abs=0.5)


def test_every_anchor_value_carries_a_provenance_tag() -> None:
    """No number reaches the harness without a stated origin."""
    untagged = [(section, key) for section, key, tags in _tagged_entries(_VALIDATIONS_DIR / "anchors.toml") if not tags]
    assert untagged == [], f"anchors.toml values without a provenance tag: {untagged}"


def test_params_free_parameter_set_is_the_sanctioned_four() -> None:
    """Exactly two `[calibrated]` fields plus the two `[transcribed]` seats are free."""
    entries = _tagged_entries(_VALIDATIONS_DIR / "params.toml")

    calibrated = [key for _, key, tags in entries if "calibrated" in tags]
    assert calibrated == list(_CALIBRATED_KEYS), f"the calibrated set drifted: {calibrated}"

    transcribed = [(section, key) for section, key, tags in entries if "transcribed" in tags]
    assert transcribed == list(_TRANSCRIBED_ENTRIES), f"the transcribed set drifted: {transcribed}"

    untagged = [(section, key) for section, key, tags in entries if not tags and key not in _UNTAGGED_KEYS]
    assert untagged == [], f"params.toml values without a provenance tag: {untagged}"
