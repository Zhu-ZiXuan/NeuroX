"""Shipped validations-config parse + build test — the one sanctioned disk-config test.

Asserts the paper-design artifacts under ``validations/xue2020jssc/`` parse and
build: ``params.toml`` (section ``cim_macro``) registry-dispatches to
:class:`Xue2020JsscCimMacroConfig` and the module builds + fabricates at
``inst_shape=()``, ``policy.toml`` (section ``policy``) loads with every
nonideality toggle off, and ``anchors.toml`` parses with the validation
convention keys the Batch-2 calibration / validation solve depends on. No numeric
MAC assertions, no forward pass.

The shipped artifacts carry the restored macro schema (S4.1 / S13): the macro
composes ``XbarArray1t1r``, so the config carries a nested ``array_config`` (Linear
cell + wire parasitics + solver) and uses ``sc_ratio_msb`` / ``v_bl_clamp__V``; the
all-off policy nests ``array_policy: XbarArray1t1rPolicy(cell_policy=..., solve_chunk_size=0)``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import torch._dynamo

import neurox.works  # noqa: F401  registers every scheme class, incl. xue2020jssc
from neurox.common.serialize import dict_from_file
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy
from neurox.works.macro.cim.xue2020jssc.macro import (
    Xue2020JsscCimMacro,
    Xue2020JsscCimMacroConfig,
    Xue2020JsscCimMacroPolicy,
)

# tests/validations/xue2020jssc/test_configs.py -> repo root is 3 parents up.
_VALIDATIONS_DIR = Path(__file__).resolve().parents[3] / "validations" / "xue2020jssc"

_SLICE_NAMES = ("control", "reference", "cablc", "dswct", "sinwp_sc", "pn_isub", "tmcsa")


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — ``fabricate()`` touches the solver-owning array; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


def _assert_all_toggles_false(obj: object, path: str = "policy") -> None:
    """Every bool field in the (nested) policy dataclass tree is False."""
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        if isinstance(value, bool):
            assert value is False, f"{path}.{field.name} is on in the all-off policy"
        elif dataclasses.is_dataclass(value):
            _assert_all_toggles_false(value, f"{path}.{field.name}")


def test_params_config_parses_and_builds() -> None:
    """``params.toml`` (section ``cim_macro``) registry-dispatches and fabricates."""
    config = CimMacroConfig.from_file(_VALIDATIONS_DIR / "params.toml", section="cim_macro")
    assert isinstance(config, Xue2020JsscCimMacroConfig)
    policy = CimMacroPolicy.from_file(_VALIDATIONS_DIR / "policy.toml", section="policy")
    assert isinstance(policy, Xue2020JsscCimMacroPolicy)

    macro = CimMacro.from_config(
        config=config,
        policy=policy,
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    assert isinstance(macro, Xue2020JsscCimMacro)
    macro.fabricate()


def test_policy_every_toggle_false() -> None:
    """``policy.toml`` loads with every nonideality toggle off (the all-off preset)."""
    policy = CimMacroPolicy.from_file(_VALIDATIONS_DIR / "policy.toml", section="policy")
    assert isinstance(policy, Xue2020JsscCimMacroPolicy)
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
