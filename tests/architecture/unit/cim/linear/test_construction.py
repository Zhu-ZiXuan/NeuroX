"""Linear-unit nested configuration and policy loading."""

from __future__ import annotations

from neurox.architecture.unit.cim import CimUnitConfig, CimUnitPolicy, LinearCimUnitConfig, LinearCimUnitPolicy
from neurox.architecture.unit.cim.engine import (
    CimEngineConfig,
    CimEnginePolicy,
    DirectWeightSliceStageConfig,
    DirectWeightSliceStagePolicy,
    DirectXSliceStageConfig,
    DirectXSliceStagePolicy,
    InputActivationStageConfig,
    InputActivationStagePolicy,
)
from neurox.common.serialize import ConfigDict
from neurox.primitive.macro.cim import IdealCimMacroConfig, IdealCimMacroPolicy


def _zero_ppa() -> dict[str, float]:
    return {
        "energy_per_op__fJ": 0.0,
        "latency_per_op__ns": 0.0,
        "leakage_per_inst__uW": 0.0,
        "area_per_inst__um2": 0.0,
    }


def test_unit_config_nested_engine_deserialization() -> None:
    """Receiver-bounded deserialization resolves the unit, engine, and macro leaves from `_neurox_class`."""
    config_dict: ConfigDict = {
        "_neurox_class": "LinearCimUnitConfig",
        "area_per_inst__um2": 0.0,
        "leakage_per_inst__uW": 0.0,
        "engine": {
            "cim_macro_config": {
                "_neurox_class": "IdealCimMacroConfig",
                "input_num": 16,
                "max_active_num": 16,
                "lane_num": 1,
                "scan_num": 16,
                "leakage_per_inst__uW": 0.0,
                "area_per_inst__um2": 0.0,
                "w_digit_num": 2,
                "w_digit_radix": 2,
                "w_encoding": "true_form",
                "x_digit_num": 2,
                "x_digit_radix": 2,
                "x_encoding": "unsigned",
                "x_value_range": [0, 1],
                "w_value_range": [-3, 3],
                "rescale_factors": [1.0],
                "adc_bits": 8,
                "quantization_scheme": "zero_point",
            },
            "placement": {
                "contraction_accumulator_config": {"bit_width": 32, **_zero_ppa()},
            },
            "input_activation": {
                "phase_accumulator_config": {"bit_width": 32, **_zero_ppa()},
            },
            "weight_slice": {
                "_neurox_class": "DirectWeightSliceStageConfig",
            },
            "x_slice": {
                "_neurox_class": "DirectXSliceStageConfig",
            },
        },
    }
    config = CimUnitConfig.from_dict(config_dict)
    assert type(config) is LinearCimUnitConfig
    assert type(config.engine) is CimEngineConfig
    assert type(config.engine.cim_macro_config) is IdealCimMacroConfig
    assert type(config.engine.input_activation) is InputActivationStageConfig
    assert type(config.engine.weight_slice) is DirectWeightSliceStageConfig
    assert type(config.engine.x_slice) is DirectXSliceStageConfig


def test_unit_policy_nested_engine_deserialization() -> None:
    config_dict: ConfigDict = {
        "_neurox_class": "LinearCimUnitPolicy",
        "engine": {
            "cim_macro_policy": {
                "_neurox_class": "IdealCimMacroPolicy",
            },
            "placement": {},
            "input_activation": {},
            "weight_slice": {
                "_neurox_class": "DirectWeightSliceStagePolicy",
            },
            "x_slice": {
                "_neurox_class": "DirectXSliceStagePolicy",
            },
        },
    }
    policy = CimUnitPolicy.from_dict(config_dict)
    assert type(policy) is LinearCimUnitPolicy
    assert type(policy.engine) is CimEnginePolicy
    assert type(policy.engine.cim_macro_policy) is IdealCimMacroPolicy
    assert type(policy.engine.input_activation) is InputActivationStagePolicy
    assert type(policy.engine.weight_slice) is DirectWeightSliceStagePolicy
    assert type(policy.engine.x_slice) is DirectXSliceStagePolicy
