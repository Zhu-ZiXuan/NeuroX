"""Tests for common base-class structural contracts."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import typing
from dataclasses import FrozenInstanceError, fields, is_dataclass
from inspect import Parameter, signature

import pytest
import torch.nn as nn
from torch import Tensor

import neurox
from neurox.architecture.unit.matmul_mapping import BlockSlotRouting
from neurox.common import (
    ConfigBase,
    ModuleBase,
    PolicyBase,
    RegistryMixin,
    TensorDataClassBase,
    TensorGroupMixin,
)
from neurox.common.fabricate_mixin import FabricateMixin
from neurox.common.profile_mixin import ProfileMixin
from neurox.primitive.device import MosfetConfig, MosfetPolicy
from neurox.primitive.digital import (
    AccumulatorConfig,
    AdderConfig,
    DigitalPolicy,
    ShiftAdderConfig,
    SubtractorConfig,
)

_SHARED_CONFIG_POLICY_TYPES: dict[str, tuple[type[ConfigBase], type[PolicyBase]]] = {
    "neurox.primitive.device.mosfet.Nmos": (MosfetConfig, MosfetPolicy),
    "neurox.primitive.device.mosfet.Pmos": (MosfetConfig, MosfetPolicy),
    "neurox.primitive.digital.accumulator.Accumulator": (AccumulatorConfig, DigitalPolicy),
    "neurox.primitive.digital.adder.Adder": (AdderConfig, DigitalPolicy),
    "neurox.primitive.digital.serial_accumulator.SerialAccumulator": (AccumulatorConfig, DigitalPolicy),
    "neurox.primitive.digital.shift_adder.ShiftAdder": (ShiftAdderConfig, DigitalPolicy),
    "neurox.primitive.digital.subtractor.Subtractor": (SubtractorConfig, DigitalPolicy),
}

# `dataclass(slots=True)` rebuilds the class object, and rebuilding re-runs
# `TensorDataClassBase.__init_subclass__` against a class that already carries
# the frozen `__setattr__`, which raises. A tensor-carrying dataclass that wants
# slots therefore keeps its own decorator and stays outside the hierarchy.
_TENSOR_BASE_EXEMPT: frozenset[type] = frozenset({BlockSlotRouting})


def _neurox_classes() -> set[type]:
    """Every class neurox declares, taken from the module that declares it."""
    modules = [importlib.import_module(info.name) for info in pkgutil.walk_packages(neurox.__path__, "neurox.")]
    return {
        cls
        for module in modules
        for cls in vars(module).values()
        if inspect.isclass(cls) and cls.__module__ == module.__name__
    }


def _carries_tensor(cls: type, *, visiting: frozenset[type] = frozenset()) -> bool:
    """Whether a dataclass declares a tensor field, directly or through a nested dataclass."""
    if cls in visiting:
        return False
    hints = typing.get_type_hints(cls)
    for field in fields(cls):
        annotation = hints[field.name]
        for candidate in (annotation, *typing.get_args(annotation)):
            if not isinstance(candidate, type):
                continue
            if issubclass(candidate, Tensor):
                return True
            if is_dataclass(candidate) and _carries_tensor(candidate, visiting=visiting | {cls}):
                return True
    return False


class _ParentConfig(ConfigBase):
    parent: int


class _ChildConfig(_ParentConfig):
    child: str


class _Policy(PolicyBase):
    enabled: bool


class _ValidatedPolicy(PolicyBase):
    count: int

    def validate(self) -> None:
        if self.count < 0:
            raise ValueError("count must be non-negative")


class _ValidatedConfig(ConfigBase):
    value: int

    def validate(self) -> None:
        if self.value < 0:
            raise ValueError("value must be non-negative")


class _FabricableModule(FabricateMixin, nn.Module):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.sample_count = 0

    def _sample_fabricate_mismatch(self) -> None:
        self.sample_count += 1


class _FabricableContainerHost(FabricateMixin, nn.Module):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.sequential = nn.Sequential(_FabricableModule())
        self.module_list = nn.ModuleList([nn.Sequential(_FabricableModule())])
        self.module_dict = nn.ModuleDict({"child": nn.ModuleList([_FabricableModule()])})

    def _sample_fabricate_mismatch(self) -> None:
        pass


class _ModuleConfig(ConfigBase):
    pass


class _ModulePolicy(PolicyBase):
    pass


class _ModuleRegistryRoot(
    RegistryMixin[_ModuleConfig, _ModulePolicy, "_ModuleRegistryRoot"],
    ModuleBase[_ModuleConfig, _ModulePolicy],
):
    def _sample_fabricate_mismatch(self) -> None:
        pass


@_ModuleRegistryRoot.register_neurox_module(config_type=_ModuleConfig, policy_type=_ModulePolicy)
class _RegisteredModule(_ModuleRegistryRoot):
    pass


def test_config_and_policy_subclasses_are_dataclasses() -> None:
    assert is_dataclass(_ParentConfig)
    assert is_dataclass(_ChildConfig)
    assert is_dataclass(_Policy)
    assert [field.name for field in fields(_ChildConfig)] == ["parent", "child"]


def test_every_module_has_config_and_policy() -> None:
    module_classes = {cls for cls in _neurox_classes() if issubclass(cls, ModuleBase) and cls is not ModuleBase}

    for module_class in module_classes:
        qualified_name = f"{module_class.__module__}.{module_class.__name__}"
        stem = module_class.__name__.removesuffix("Base")
        module = importlib.import_module(module_class.__module__)
        config_class, policy_class = _SHARED_CONFIG_POLICY_TYPES.get(
            qualified_name,
            (
                getattr(module, f"{stem}Config", None),
                getattr(module, f"{stem}Policy", None),
            ),
        )
        assert inspect.isclass(config_class), qualified_name
        assert issubclass(config_class, ConfigBase), qualified_name
        assert inspect.isclass(policy_class), qualified_name
        assert issubclass(policy_class, PolicyBase), qualified_name


@pytest.mark.parametrize("cls", [_ParentConfig, _ChildConfig, _Policy])
def test_config_and_policy_fields_are_keyword_only(cls: type) -> None:
    parameters = signature(cls).parameters.values()
    assert parameters
    assert all(parameter.kind is Parameter.KEYWORD_ONLY for parameter in parameters)


def test_config_and_policy_instances_are_frozen() -> None:
    config = _ChildConfig(parent=1, child="x")
    policy = _Policy(enabled=True)
    for obj, name, value in ((config, "parent", 2), (policy, "enabled", False)):
        with pytest.raises(FrozenInstanceError):
            setattr(obj, name, value)


def test_config_validation_runs_after_construction() -> None:
    assert _ValidatedConfig(value=0).value == 0
    with pytest.raises(ValueError, match="value must be non-negative"):
        _ValidatedConfig(value=-1)


def test_policy_validation_runs_after_construction() -> None:
    assert _ValidatedPolicy(count=0).count == 0
    with pytest.raises(ValueError, match="count must be non-negative"):
        _ValidatedPolicy(count=-1)


@pytest.mark.parametrize("base", [ConfigBase, PolicyBase])
def test_config_and_policy_reject_custom_post_init(base: type) -> None:
    with pytest.raises(TypeError, match=r"must implement validate\(\), not __post_init__\(\)"):

        class _InvalidStructuredInput(base):
            def __post_init__(self) -> None:
                pass


@pytest.mark.parametrize("base", [ConfigBase, PolicyBase])
def test_config_and_policy_reject_custom_init(base: type) -> None:
    with pytest.raises(TypeError, match=r"must declare dataclass fields, not __init__\(\)"):

        class _InvalidStructuredInput(base):
            def __init__(self) -> None:
                pass


def test_every_tensor_carrying_dataclass_inherits_the_tensor_base() -> None:
    for cls in _neurox_classes():
        if not is_dataclass(cls) or issubclass(cls, ConfigBase | PolicyBase) or cls in _TENSOR_BASE_EXEMPT:
            continue
        if _carries_tensor(cls):
            assert issubclass(cls, TensorDataClassBase), f"{cls.__module__}.{cls.__qualname__}"


def test_every_tensor_group_host_inherits_the_tensor_base() -> None:
    for cls in _neurox_classes():
        if issubclass(cls, TensorGroupMixin) and cls is not TensorGroupMixin:
            assert issubclass(cls, TensorDataClassBase), f"{cls.__module__}.{cls.__qualname__}"


def test_tensor_base_descendants_are_frozen_identity_dataclasses() -> None:
    for cls in _neurox_classes():
        if not issubclass(cls, TensorDataClassBase):
            continue
        name = f"{cls.__module__}.{cls.__qualname__}"
        params = cls.__dataclass_params__
        assert params.frozen is True, name
        assert params.eq is False, name
        assert all(field.kw_only for field in fields(cls)), name


def test_profile_mixin_rejects_non_module_subclass() -> None:
    with pytest.raises(TypeError, match=r"must also inherit torch\.nn\.Module"):

        class _InvalidProfileHost(ProfileMixin):
            pass


def test_fabricate_mixin_rejects_non_module_subclass() -> None:
    with pytest.raises(TypeError, match=r"must also inherit torch\.nn\.Module"):

        class _InvalidFabricateHost(FabricateMixin):
            def _sample_fabricate_mismatch(self) -> None:
                pass


def test_fabricate_mixin_accepts_module_subclass() -> None:
    module = _FabricableModule()
    module.fabricate()
    assert module.sample_count == 1


def test_fabricate_mixin_walks_pytorch_standard_containers() -> None:
    module = _FabricableContainerHost()
    module.fabricate()
    descendants = (
        module.sequential[0],
        module.module_list[0][0],
        module.module_dict["child"][0],
    )
    assert all(isinstance(child, _FabricableModule) and child.sample_count == 1 for child in descendants)


def test_module_registry_resolves_config_and_policy_instances() -> None:
    assert (
        _ModuleRegistryRoot._lookup_neurox_module(config=_ModuleConfig(), policy=_ModulePolicy()) is _RegisteredModule
    )


@pytest.mark.parametrize("inst_shape", [(0,), (2, 0, 3), (-1,)])
def test_module_base_rejects_non_positive_instance_extents(inst_shape: tuple[int, ...]) -> None:
    class _Module(ModuleBase[_ModuleConfig, _ModulePolicy]):
        def _sample_fabricate_mismatch(self) -> None:
            pass

    with pytest.raises(ValueError, match="inst_shape extents must be positive"):
        _Module(config=_ModuleConfig(), policy=_ModulePolicy(), inst_shape=inst_shape)


def test_empty_instance_shape_represents_one_instance() -> None:
    class _Module(ModuleBase[_ModuleConfig, _ModulePolicy]):
        def _sample_fabricate_mismatch(self) -> None:
            pass

    module = _Module(config=_ModuleConfig(), policy=_ModulePolicy(), inst_shape=())
    assert module.inst_count == 1
