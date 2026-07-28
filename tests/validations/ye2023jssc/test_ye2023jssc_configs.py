"""Shipped validations-config parse + build test — the one sanctioned disk-config test.

Asserts the paper-design artifacts under ``validations/ye2023jssc/`` parse, build,
and carry the modeled design point: ``params.toml`` (section ``cim_macro``)
registry-dispatches to :class:`Ye2023JsscCimMacroConfig` and the module builds +
fabricates at ``inst_shape=()``, ``policy.toml`` (section ``policy``) loads with
every nonideality toggle off, and ``anchors.toml`` parses with its two kinds of
number kept apart. No forward pass.

Beyond parsing, the shipped config is checked against the DESIGN it is supposed
to describe (laws, not tuned magnitudes):

  * the physical grid is 64 rows (logical outputs) x 128 columns = 32 inputs x
    (3 weight planes + 1 redundant SUBA4 plane),
  * the T2 table's input-0 floor is state-independent and the derived PH0
    compensation — ``floor * 32 * sum(radix)`` over 352 place-value units —
    reproduces the paper's ~1 uA row leakage, so a zero-MAC access reads code 0,
  * the input-1 HRS entry saturates the measured 30 nA bound at the largest
    place value in the radix path,
  * the readout is the paper's single 4-bit operating point and its DERIVED
    access window is 66 ns,
  * the selected word line clears the step-2 WL threshold,
  * no mismatch / noise / jitter field is declared anywhere in either artifact.

And against the VALIDATION CONTRACT the two TOML artifacts encode:

  * every anchor value carries a provenance tag, and the ``[gate]`` targets are
    caliber-independent — no Fig.19 power number appears among them,
  * the ``[reference]`` table holds the Fig.19 picture that is reported but not
    gated (two totals, both per-pin share breakdowns, the headline efficiency),
  * the free-parameter set in ``params.toml`` is exactly three ``[calibrated]``
    fields (the C_WL knob, the RS-CSA mirror scale, its fixed per-op energy) plus
    the two ``[transcribed]`` peripheral seats.
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

# Keys that name no physical quantity: a registry dispatch key and the two
# ``(mode, bits)`` selectors of the ADC calibration entry.
_UNTAGGED_KEYS = frozenset({"_neurox_class"})
_UNTAGGED_ENTRIES = frozenset({("cim_macro.adc_calibration", "mode"), ("cim_macro.adc_calibration", "bits")})

# The sanctioned free-parameter set: three solved numbers + two transcribed seats.
_CALIBRATED_KEYS = ("c_wl__fF", "mirror_scale", "e_fixed_per_op__fJ")
_TRANSCRIBED_ENTRIES = (
    ("cim_macro.mux_driver_config", "leakage_per_inst__uW"),
    ("cim_macro.timing_ctrl_config", "leakage_per_inst__uW"),
)


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — ``fabricate()`` touches the solver-owning array; do not unroll it."""
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
    """Every ``key = value`` line as ``(section, key, provenance tags)``.

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
    """``params.toml`` (section ``cim_macro``) registry-dispatches and fabricates."""
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
    """``policy.toml`` loads with every nonideality toggle off (the all-off preset)."""
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
    """The floor entry is state-independent; PH0 == floor * 352 units ~= the 1 uA row leakage."""
    config = _load_config()
    array_config = config.array_config
    i_t2_table = config.cell_config.i_t2_table__uA

    floor_row = i_t2_table[0]
    assert len(set(floor_row)) == 1, f"input-0 floor is state-dependent: {floor_row}"

    radix_unit_num = _INPUT_NUM * sum((*array_config.weight_radix, *array_config.redundant_radix))
    assert radix_unit_num == 352
    assert floor_row[0] * radix_unit_num == pytest.approx(_I_ROW_LEAK__uA, rel=1e-3)

    # The input-1 HRS entry keeps the radix multiplier and saturates the 30 nA bound
    # at the largest place value on the radix path.
    m_max = max((*array_config.weight_radix, *array_config.redundant_radix))
    i_hrs_in1__uA = i_t2_table[1][0]
    assert i_hrs_in1__uA * m_max == pytest.approx(_I_HRS_BOUND__uA)
    assert i_hrs_in1__uA * m_max <= _I_HRS_BOUND__uA + 1e-12
    # The LRS on-current dominates the leakage by orders of magnitude.
    assert i_t2_table[1][1] > 10.0 * i_hrs_in1__uA


def test_readout_is_one_4bit_point_with_a_66ns_derived_window() -> None:
    """Single 4-bit mode; ``T_AC = sum(t_phase[:-1]) + t4_intrinsic`` = 66 ns."""
    config = _load_config()
    adc_config = config.adc_config
    assert adc_config.bits == 4
    assert len(adc_config.ref_radix) == 4
    assert len(adc_config.t_phase__ns) == adc_config.bits + 1  # PH0 + one compare phase per bit
    assert {(entry.mode, entry.bits) for entry in config.adc_calibration} == {(0, 4)}

    t_ac__ns = sum(adc_config.t_phase__ns[:-1]) + adc_config.t4_intrinsic__ns
    assert t_ac__ns == pytest.approx(_T_AC__ns)
    # The window closes at the last comparator latch, inside the last compare phase.
    assert 0.0 < adc_config.t4_intrinsic__ns < adc_config.t_phase__ns[-1]


def test_selected_word_line_clears_the_step2_threshold() -> None:
    """``v_wl_sel__V`` turns the access device on; the BL input level brackets its threshold."""
    config = _load_config()
    cell_config = config.cell_config
    assert config.v_wl_sel__V > cell_config.v_wl_on_threshold__V
    assert 0.0 < config.array_config.v_bl_in_threshold__V < config.v_bl_in1__V
    assert config.v_bl_in1__V == config.array_config.v_bl_in1__V


# ---------------------------------------------------------------------------
# The validation contract
# ---------------------------------------------------------------------------


def test_gate_targets_are_caliber_independent() -> None:
    """``[gate]`` carries the five gates' targets and no Fig.19 power number."""
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
    """``[reference]`` carries the two Fig.19 totals, both share breakdowns and the headline EF."""
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


def test_params_free_parameter_set_is_the_sanctioned_five() -> None:
    """Exactly three ``[calibrated]`` fields plus the two ``[transcribed]`` seats are free."""
    entries = _tagged_entries(_VALIDATIONS_DIR / "params.toml")

    calibrated = [key for _, key, tags in entries if "calibrated" in tags]
    assert calibrated == list(_CALIBRATED_KEYS), f"the calibrated set drifted: {calibrated}"

    transcribed = [(section, key) for section, key, tags in entries if "transcribed" in tags]
    assert transcribed == list(_TRANSCRIBED_ENTRIES), f"the transcribed set drifted: {transcribed}"

    untagged = [
        (section, key)
        for section, key, tags in entries
        if not tags and key not in _UNTAGGED_KEYS and (section, key) not in _UNTAGGED_ENTRIES
    ]
    assert untagged == [], f"params.toml values without a provenance tag: {untagged}"
