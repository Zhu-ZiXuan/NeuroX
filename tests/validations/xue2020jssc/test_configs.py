"""Shipped validations-config parse + build test — the one sanctioned disk-config test.

Asserts the paper-design artifacts under ``validations/xue2020jssc/`` parse and
build: ``params.toml`` (section ``cim_macro``) registry-dispatches to
:class:`Xue2020JsscCimMacroConfig` and the module builds + fabricates at
``inst_shape=()``, ``policy.toml`` (section ``policy``) loads with every
nonideality toggle off, and ``anchors.toml`` parses with the validation
convention keys the calibration and validation drivers depend on. No numeric MAC
assertions, no forward pass.

The shipped artifacts cover the complete composed schema: the macro contains an
``XbarArray1t1r``-configured serial-column array, so the config carries a nested
``array_config`` (linear cell, wire parasitics, and solver), uses
``sc_ratio_msb``, the DSWCT / SINWP-SC / PN-ISUB / TMCSA module configs, and the
dedicated single-tap ``cablc_vref_config`` reference source. The all-off policy
nests ``array_policy: XbarArray1t1rPolicy(cell_policy=..., solve_chunk_size=0)``
plus the ``cablc_vref_policy`` and the source-free DSWCT / SINWP-SC / PN-ISUB /
TMCSA module policies.

And against the VALIDATION CONTRACT the campaign encodes:

  * the harness is SELF-CONTAINED — it resolves the three TOML artifacts as fixed
    files beside itself and exposes only run knobs on the CLI, so no config or
    workload value can be injected at the command line,
  * every provenance tag in ``params.toml`` / ``anchors.toml`` comes from the
    authoritative legend in ``docs/validation/campaigns.md``,
  * the paired-slice aggregation law holds on a hand-built witness.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import torch._dynamo

from neurox.common import PolicyBase
from neurox.common.serialize import dict_from_file
from neurox.primitive.analog import VrefConfig, VrefPolicy
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy
from neurox.works.macro.cim.xue2020jssc import (
    DswctConfig,
    DswctPolicy,
    PnIsubConfig,
    PnIsubPolicy,
    SinwpScConfig,
    SinwpScPolicy,
    TmcsaConfig,
    TmcsaPolicy,
    Xue2020JsscCimMacro,
    Xue2020JsscCimMacroConfig,
    Xue2020JsscCimMacroPolicy,
)

# tests/validations/xue2020jssc/test_configs.py -> repo root is 3 parents up.
_VALIDATIONS_DIR = Path(__file__).resolve().parents[3] / "validations" / "xue2020jssc"

_SLICE_NAMES = ("control", "reference", "cablc", "dswct", "sinwp_sc", "pn_isub", "tmcsa")

# The sanctioned provenance vocabulary; its legend lives in docs/validation/campaigns.md.
_TAG_PATTERN = re.compile(r"\[(measured|derived|transcribed|assumed|bound-derived|calibrated)\b[^\]]*\]")
# The retired scheme-local vocabulary; no shipped artifact may still speak it.
_LEGACY_TAG_PATTERN = re.compile(r"\[(sourced|declared|adopted|uncertain)\b[^\]]*\]")

# CLI options the self-contained harness must NOT expose: a config artifact or a
# declared workload value injected at the command line would leave the campaign
# reading something other than the shipped design point.
_BANNED_CLI_OPTIONS = ("--params", "--policy", "--anchors", "--report", "--p-zero")


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — ``fabricate()`` touches the solver-owning array; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


def _assert_all_toggles_false(obj: PolicyBase, path: str = "policy") -> None:
    """Every bool field in the (nested) policy dataclass tree is False."""
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        if isinstance(value, bool):
            assert value is False, f"{path}.{field.name} is on in the all-off policy"
        elif dataclasses.is_dataclass(value):
            assert isinstance(value, PolicyBase)
            _assert_all_toggles_false(value, f"{path}.{field.name}")


def test_params_config_parses_and_builds() -> None:
    """``params.toml`` (section ``cim_macro``) registry-dispatches and fabricates."""
    config = CimMacroConfig.from_file(_VALIDATIONS_DIR / "params.toml", section="cim_macro")
    assert isinstance(config, Xue2020JsscCimMacroConfig)
    # The scheme-local readout modules are configured through their own nested
    # module configs (the macro carries no per-op PN-ISUB constant of its own).
    assert isinstance(config.dswct_config, DswctConfig)
    assert isinstance(config.sinwp_sc_config, SinwpScConfig)
    assert isinstance(config.pn_isub_config, PnIsubConfig)
    # The TMCSA phase-billing module: one PH2/PH3 window pair per ADC step,
    # each pair fitting inside its step latency (PH1/PH4 occupy the rest).
    assert isinstance(config.tmcsa_config, TmcsaConfig)
    assert len(config.tmcsa_config.t_ph2_per_step__ns) == config.adc_config.bits
    assert len(config.tmcsa_config.t_ph3_per_step__ns) == config.adc_config.bits
    for s in range(config.adc_config.bits):
        assert (
            config.tmcsa_config.t_ph2_per_step__ns[s] + config.tmcsa_config.t_ph3_per_step__ns[s]
            <= config.adc_config.step_latency__ns[s]
        )
    # Control caliber law: pure per-op (100 % dynamic) — the static seat is zero.
    assert config.control_config.leakage_per_inst__uW == 0.0
    # The CABLC reference is the macro's own dedicated single-tap source.
    assert isinstance(config.cablc_vref_config, VrefConfig)
    assert len(config.cablc_vref_config.v_refs__V) == 1
    policy = CimMacroPolicy.from_file(_VALIDATIONS_DIR / "policy.toml", section="policy")
    assert isinstance(policy, Xue2020JsscCimMacroPolicy)

    macro = CimMacro.from_config(
        config=config,
        policy=policy,
        input_num=256,
        output_num=128,
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    assert isinstance(macro, Xue2020JsscCimMacro)
    macro.fabricate()


def test_policy_every_toggle_false() -> None:
    """``policy.toml`` loads with every nonideality toggle off (the all-off preset).

    The recursive walk covers every nested child policy, including the
    ``cablc_vref_policy`` and the source-free DSWCT / SINWP-SC / PN-ISUB /
    TMCSA module policies asserted present here by type.
    """
    policy = CimMacroPolicy.from_file(_VALIDATIONS_DIR / "policy.toml", section="policy")
    assert isinstance(policy, Xue2020JsscCimMacroPolicy)
    assert isinstance(policy.cablc_vref_policy, VrefPolicy)
    assert isinstance(policy.dswct_policy, DswctPolicy)
    assert isinstance(policy.sinwp_sc_policy, SinwpScPolicy)
    assert isinstance(policy.pn_isub_policy, PnIsubPolicy)
    assert isinstance(policy.tmcsa_policy, TmcsaPolicy)
    _assert_all_toggles_false(policy)


def test_anchors_parses_with_required_convention_keys() -> None:
    """``anchors.toml`` carries the hard-gate target, Fig.18 shares, and workload convention keys."""
    anchors = dict_from_file(_VALIDATIONS_DIR / "anchors.toml")

    # Hard-gate target: the one gated number is derived from the sourced macro
    # power, sub-array count, and access rate (5.13 mW / 8 / 20 MHz = 32.06 pJ).
    target = anchors["target"]
    derived_per_access__pJ = target["total_macro__mW"] * 1000.0 / target["sub_array_num"] / target["op_frequency__MHz"]
    assert target["per_access__pJ"] == pytest.approx(derived_per_access__pJ, rel=1e-3)
    assert anchors["gate"]["hard_tolerance_relative"] == pytest.approx(0.05)

    # Fig.18 shares: every slice present; the read-path sum reconciles.
    shares = anchors["fig18_shares"]
    for name in _SLICE_NAMES:
        assert name in shares
    assert shares["read_path_sum"] == pytest.approx(
        shares["cablc"] + shares["dswct"] + shares["sinwp_sc"] + shares["pn_isub"] + shares["tmcsa"]
    )

    # Accounting conventions: the two adopted seats + the read-path physics set
    # partition the slice names.
    conventions = anchors["conventions"]
    assert set(conventions["adopt"]) | set(conventions["read_path"]) == set(_SLICE_NAMES)

    data = anchors["data"]
    assert data["weight_range"] == [-3, 3]
    assert data["input_range"] == [0, 3]
    assert "p_zero" in data


def _comment_text(path: Path) -> str:
    """Every comment line of a TOML artifact, where provenance tags live."""
    return "\n".join(line for raw in path.read_text().splitlines() if (line := raw.strip()).startswith("#"))


@pytest.mark.parametrize("name", ["params.toml", "anchors.toml"])
def test_every_provenance_tag_is_from_the_authoritative_legend(name: str) -> None:
    """No value carries a retired scheme-local tag; the campaigns.md vocabulary is the only one."""
    comments = _comment_text(_VALIDATIONS_DIR / name)
    legacy = sorted(set(_LEGACY_TAG_PATTERN.findall(comments)))
    assert legacy == [], f"{name} still uses the retired provenance tags {legacy}"
    tags = {match.group(1) for match in _TAG_PATTERN.finditer(comments)}
    assert tags, f"{name} carries no provenance tag at all"
    assert tags <= {"measured", "derived", "transcribed", "assumed", "bound-derived", "calibrated"}


def _load_validate_module():
    """Load ``validations/xue2020jssc/validate.py`` as a module (it is a script, not a package)."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("xue2020jssc_validate", _VALIDATIONS_DIR / "validate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass creation resolves the defining module
    spec.loader.exec_module(module)
    return module


def test_validate_harness_is_self_contained() -> None:
    """The three TOML artifacts are FIXED files beside the script; the CLI carries run knobs only.

    The harness resolves ``params.toml`` / ``policy.toml`` / ``anchors.toml``
    relative to itself, so a campaign run always reads the shipped design point,
    and it emits everything through ``logging`` — no bare ``print``.
    """
    validate = _load_validate_module()

    for constant, name in (
        (validate._PARAMS_PATH, "params.toml"),
        (validate._POLICY_PATH, "policy.toml"),
        (validate._ANCHORS_PATH, "anchors.toml"),
    ):
        assert constant == _VALIDATIONS_DIR / name
        assert constant.is_file()

    source = (_VALIDATIONS_DIR / "validate.py").read_text()
    for option in _BANNED_CLI_OPTIONS:
        assert f'"{option}"' not in source, f"validate.py still injects {option} on the command line"
    assert "print(" not in source, "validate.py must report through logging, not print"


def test_validate_pair_slice_aggregation_law() -> None:
    """Paired-slice caliber: each pair row sums its members' energy and its members' Fig.18 shares.

    On a hand-built witness slice list, ``paired_slices`` must return
    ``cablc+dswct`` and ``sinwp_sc+pn_isub`` rows whose dynamic / static
    energy is the member sum and whose target is ``(share_a + share_b) / 100 *
    target_total``; member slices carry no per-member target in the measured
    breakdown (the pair sum is the only well-defined comparison).
    """
    validate = _load_validate_module()

    shares = {"cablc": 14.9, "dswct": 11.5, "sinwp_sc": 8.0, "pn_isub": 3.4}
    target_total = 32.06
    slices = (
        validate.SliceEnergy(name="cablc", dynamic__pJ=3.0, static__pJ=0.5, target__pJ=0.0),
        validate.SliceEnergy(name="dswct", dynamic__pJ=2.0, static__pJ=0.25, target__pJ=0.0),
        validate.SliceEnergy(name="sinwp_sc", dynamic__pJ=1.5, static__pJ=0.0, target__pJ=0.0),
        validate.SliceEnergy(name="pn_isub", dynamic__pJ=0.75, static__pJ=0.125, target__pJ=0.0),
    )
    pairs = {s.name: s for s in validate.paired_slices(slices, shares, target_total)}
    assert set(pairs) == {"cablc+dswct", "sinwp_sc+pn_isub"}

    cd = pairs["cablc+dswct"]
    assert cd.dynamic__pJ == pytest.approx(3.0 + 2.0)
    assert cd.static__pJ == pytest.approx(0.5 + 0.25)
    assert cd.target__pJ == pytest.approx((14.9 + 11.5) / 100.0 * target_total)

    sp = pairs["sinwp_sc+pn_isub"]
    assert sp.dynamic__pJ == pytest.approx(1.5 + 0.75)
    assert sp.static__pJ == pytest.approx(0.0 + 0.125)
    assert sp.target__pJ == pytest.approx((8.0 + 3.4) / 100.0 * target_total)
