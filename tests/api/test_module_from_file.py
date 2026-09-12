"""File factories preserve family dispatch, validation, and caller-owned lifecycle."""

from pathlib import Path

import pytest
import torch

from neurox import cim_macro_from_file, cim_unit_from_file, diff_vadc_from_file, iadc_from_file
from neurox.architecture.unit.ideal import IdealLinearUnit
from neurox.common.module import ModuleBase
from neurox.primitive.analog.current_adc import IadcPolicy, SarIadc
from neurox.primitive.analog.diff_voltage_adc import McsSarDiffVadc
from neurox.primitive.macro.cim import IdealCimMacro

_IADC_CONFIG = """
_neurox_class = "SarIadcConfig"
bits = 3
area_per_inst__um2 = 1.0
leakage_per_inst__uW = 0.0
latency_per_bit__ns = 1.0
comparator_offset_sigma__uA = 0.5
"""
_IADC_POLICY = """
_neurox_class = "SarIadcPolicy"
comparator_offset = true
"""
_DIFF_VADC_CONFIG = """
_neurox_class = "McsSarDiffVadcConfig"
bits = 3
area_per_inst__um2 = 1.0
leakage_per_inst__uW = 0.0
latency_per_bit__ns = 1.0
c_unit__fF = 1.0
cap_mismatch_sigma_relative = 0.1
comparator_offset_sigma__V = 0.01
comparator_thermal_noise_sigma__V = 0.0
energy_per_op__fJ = 0.0
energy_per_bit__fJ = 0.0
"""
_DIFF_VADC_POLICY = """
_neurox_class = "McsSarDiffVadcPolicy"
cap_mismatch = true
comparator_offset = true
comparator_thermal_noise = false
sampling_thermal_noise = false
"""
_MACRO_CONFIG = """
_neurox_class = "IdealCimMacroConfig"
input_num = 2
lane_num = 1
scan_num = 1
max_active_num = 2
rescale_factors = [1.0]
area_per_inst__um2 = 0.0
leakage_per_inst__uW = 0.0
w_digit_num = 1
w_digit_radix = 2
w_encoding = "unsigned"
x_digit_num = 1
x_digit_radix = 2
x_encoding = "unsigned"
x_value_range = [0, 1]
w_value_range = [0, 1]
adc_bits = 2
quantization_scheme = "zero_point"
"""
_UNIT_CONFIG = """
_neurox_class = "IdealLinearUnitConfig"
area_per_inst__um2 = 0.0
leakage_per_inst__uW = 0.0
x_value_range = [0, 1]
w_value_range = [0, 1]
"""


def _files(tmp_path: Path, config: str, policy: str) -> dict[str, tuple[Path, ...]]:
    config_path = tmp_path / "config.toml"
    policy_path = tmp_path / "policy.toml"
    config_path.write_text(config)
    policy_path.write_text(policy)
    return {"config_files": (config_path,), "policy_files": (policy_path,)}


@pytest.mark.parametrize(
    ("factory", "config", "policy", "geometry", "module_type"),
    [
        (iadc_from_file, _IADC_CONFIG, _IADC_POLICY, {"inst_shape": (2,)}, SarIadc),
        (diff_vadc_from_file, _DIFF_VADC_CONFIG, _DIFF_VADC_POLICY, {"inst_shape": (2,)}, McsSarDiffVadc),
        (
            cim_macro_from_file,
            _MACRO_CONFIG,
            '_neurox_class = "IdealCimMacroPolicy"',
            {"inst_shape": (2,)},
            IdealCimMacro,
        ),
        (
            cim_unit_from_file,
            _UNIT_CONFIG,
            '_neurox_class = "IdealLinearUnitPolicy"',
            {"w_logical_shape": (1, 2), "ideal_macro": False},
            IdealLinearUnit,
        ),
    ],
    ids=["iadc", "diff_vadc", "cim_macro", "cim_unit"],
)
def test_factories_construct_registered_families_without_preparing_a_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory,
    config,
    policy,
    geometry,
    module_type,
) -> None:
    files = _files(tmp_path, config, policy)

    def unexpected_lifecycle(*args, **kwargs):
        pytest.fail("file construction must leave placement and lifecycle to the caller")

    monkeypatch.setattr(torch.nn.Module, "to", unexpected_lifecycle)
    monkeypatch.setattr(torch.nn.Module, "eval", unexpected_lifecycle)
    monkeypatch.setattr(ModuleBase, "fabricate", unexpected_lifecycle)
    monkeypatch.setattr(ModuleBase, "stamp_names", unexpected_lifecycle)
    monkeypatch.setattr(
        module_type, "program" if module_type in (IdealCimMacro, IdealLinearUnit) else "convert", unexpected_lifecycle
    )
    rng_state = torch.get_rng_state()
    module = factory(**files, **geometry, dtype=torch.float64, T__K=310.0)

    assert isinstance(module, module_type)
    assert module.training
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert all(buffer.dtype == torch.float64 for buffer in module.buffers())
    with pytest.raises(RuntimeError, match="no name stamp"):
        _ = module.qualified_name
    if "inst_shape" in geometry:
        assert module.inst_shape == geometry["inst_shape"]


def test_file_sets_merge_independently_with_first_file_priority_and_nested_sections(tmp_path: Path) -> None:
    files = _files(tmp_path, "[chip.adc]\n" + _IADC_CONFIG, "[run.adc]\n" + _IADC_POLICY)
    config_override = tmp_path / "config_override.toml"
    policy_override = tmp_path / "policy_override.toml"
    config_override.write_text("[chip.adc]\nbits = 2\n")
    policy_override.write_text("[run.adc]\ncomparator_offset = false\n")

    adc = iadc_from_file(
        config_files=(config_override, *files["config_files"]),
        policy_files=(policy_override, *files["policy_files"]),
        config_section="chip.adc",
        policy_section="run.adc",
        inst_shape=(2,),
        dtype=torch.float64,
        T__K=300.0,
    )
    assert adc.bits == 2
    assert not adc.policy.comparator_offset

    # Assemble the tree before the caller places, names, and fabricates it.
    model = torch.nn.ModuleDict({"adc": adc}).to(dtype=torch.float32)
    adc.stamp_names(qualified_name="adc")
    adc.fabricate()
    assert all(buffer.dtype == torch.float32 for buffer in model.buffers())
    assert adc.qualified_name == "adc"
    assert torch.equal(
        adc.convert(torch.tensor([0.5, 2.5]), torch.tensor([1.0, 2.0, 3.0]), active_bits=2), torch.tensor([0, 2])
    )


@pytest.mark.parametrize("which", ["config_files", "policy_files"])
def test_file_factory_rejects_an_empty_file_set(tmp_path: Path, which: str) -> None:
    files = _files(tmp_path, _IADC_CONFIG, _IADC_POLICY)
    files[which] = ()
    with pytest.raises(ValueError, match="At least one"):
        iadc_from_file(**files, inst_shape=(), dtype=torch.float64, T__K=300.0)


@pytest.mark.parametrize(
    ("config", "policy", "error", "match"),
    [
        (_IADC_CONFIG.replace("bits = 3", "bits = 0"), _IADC_POLICY, ValueError, "bits"),
        (_IADC_CONFIG, _DIFF_VADC_POLICY, TypeError, "IadcPolicy"),
        (_IADC_CONFIG, '_neurox_class = "SarIadcPolicy"', TypeError, "comparator_offset"),
    ],
)
def test_file_factory_propagates_config_and_policy_validation(tmp_path: Path, config, policy, error, match) -> None:
    files = _files(tmp_path, config, policy)
    with pytest.raises(error, match=match):
        iadc_from_file(**files, inst_shape=(), dtype=torch.float64, T__K=300.0)


def test_file_factory_rejects_an_unregistered_config_policy_pair(tmp_path: Path) -> None:
    class UnregisteredFileIadcPolicy(IadcPolicy):
        pass

    files = _files(tmp_path, _IADC_CONFIG, '_neurox_class = "UnregisteredFileIadcPolicy"')
    with pytest.raises(TypeError, match="no Iadc module registered"):
        iadc_from_file(**files, inst_shape=(), dtype=torch.float64, T__K=300.0)


def test_constructed_macro_and_unit_can_be_programmed_by_the_caller(tmp_path: Path) -> None:
    files = _files(tmp_path, _MACRO_CONFIG, '_neurox_class = "IdealCimMacroPolicy"')
    macro = cim_macro_from_file(**files, inst_shape=(), dtype=torch.float64, T__K=300.0)
    macro.program(torch.tensor([[1], [0]]))
    x = torch.tensor([[1, 1], [0, 1]])
    expected = torch.tensor([[1], [0]])
    assert torch.equal(macro.vec_mat_mul(x, quantization_mode=0, adc_active_bits=None), expected)

    files = _files(tmp_path, _UNIT_CONFIG, '_neurox_class = "IdealLinearUnitPolicy"')
    unit = cim_unit_from_file(**files, w_logical_shape=(1, 2), dtype=torch.float64, T__K=300.0, ideal_macro=False)
    unit.program(torch.tensor([[1, 0]]))
    assert torch.equal(unit.linear(x, quantization_mode=0, adc_active_bits=None), expected)
