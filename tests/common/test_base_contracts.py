"""Tests for common base-class construction and rejection contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
from inspect import Parameter, signature

import pytest
import torch
import torch.nn as nn

from neurox import fabricate
from neurox.common.module import ConfigBase, ModuleBase, PolicyBase
from neurox.common.profile_mixin import ProfileMixin
from neurox.common.registry_mixin import RegistryMixin
from neurox.common.serialize import dataclass_from_dict


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


class _ModuleConfig(ConfigBase):
    pass


class _ModulePolicy(PolicyBase):
    pass


class _ModuleRegistryRoot(
    RegistryMixin[_ModuleConfig, _ModulePolicy, "_ModuleRegistryRoot"],
    ModuleBase[_ModuleConfig, _ModulePolicy],
):
    pass


@_ModuleRegistryRoot.register_neurox_module(config_type=_ModuleConfig, policy_type=_ModulePolicy)
class _RegisteredModule(_ModuleRegistryRoot):
    pass


class _Module(ModuleBase[_ModuleConfig, _ModulePolicy]):
    pass


class _BufferedModule(ModuleBase[_ModuleConfig, _ModulePolicy]):
    def __init__(self) -> None:
        super().__init__(config=_ModuleConfig(), policy=_ModulePolicy(), inst_shape=())
        self._register_nonpersistent_buffer("anchor", torch.tensor(1.0))


class _FabricatingModule(ModuleBase[_ModuleConfig, _ModulePolicy]):
    def __init__(self, name: str, events: list[str], *children: nn.Module) -> None:
        super().__init__(config=_ModuleConfig(), policy=_ModulePolicy(), inst_shape=())
        self.name = name
        self.events = events
        self.children_ = nn.ModuleList(children)

    def _sample_fabrication_variation(self) -> None:
        self.events.append(self.name)


class _PlainWrapper(nn.Module):
    def __init__(self, child: nn.Module) -> None:
        super().__init__()
        self.child = child


def test_config_and_policy_subclasses_are_dataclasses() -> None:
    assert is_dataclass(_ParentConfig)
    assert is_dataclass(_ChildConfig)
    assert is_dataclass(_Policy)
    assert [field.name for field in fields(_ChildConfig)] == ["parent", "child"]


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
def test_config_and_policy_reject_custom_init(base: type) -> None:
    with pytest.raises(TypeError, match=r"must declare dataclass fields, not __init__\(\)"):

        class _InvalidStructuredInput(base):
            def __init__(self) -> None:
                pass


def test_profile_mixin_rejects_non_module_subclass() -> None:
    with pytest.raises(TypeError, match=r"must also inherit torch\.nn\.Module"):

        class _InvalidProfileHost(ProfileMixin):
            pass


def test_module_base_default_fabrication_is_a_noop() -> None:
    module = _Module(config=_ModuleConfig(), policy=_ModulePolicy(), inst_shape=())
    module.fabricate()


def test_module_base_registers_buffers_without_persisting_them() -> None:
    module = _BufferedModule()
    assert tuple(name for name, _ in module.named_buffers()) == ("anchor",)
    assert not module.state_dict()


def test_fabricate_walks_through_plain_module_wrappers_in_preorder() -> None:
    events: list[str] = []
    child = _FabricatingModule("child", events)
    parent = _FabricatingModule("parent", events, _PlainWrapper(child))
    fabricate(_PlainWrapper(parent))
    assert events == ["parent", "child"]


def test_fabricate_samples_a_shared_module_once() -> None:
    events: list[str] = []
    shared = _FabricatingModule("shared", events)

    class _SharedHost(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = nn.Sequential(shared)
            self.right = nn.Sequential(shared)

    fabricate(_SharedHost())
    assert events == ["shared"]


def test_module_registry_resolves_config_and_policy_instances() -> None:
    assert (
        _ModuleRegistryRoot._lookup_neurox_module(config=_ModuleConfig(), policy=_ModulePolicy()) is _RegisteredModule
    )


def test_module_registry_rejects_a_duplicate_config_policy_key() -> None:
    with pytest.raises(
        TypeError,
        match=r"config _ModuleConfig and policy _ModulePolicy already select _RegisteredModule",
    ):

        @_ModuleRegistryRoot.register_neurox_module(config_type=_ModuleConfig, policy_type=_ModulePolicy)
        class _DuplicateRegisteredModule(_ModuleRegistryRoot):
            pass


@pytest.mark.parametrize("inst_shape", [(0,), (2, 0, 3), (-1,)])
def test_module_base_rejects_non_positive_instance_extents(inst_shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="inst_shape extents must be positive"):
        _Module(config=_ModuleConfig(), policy=_ModulePolicy(), inst_shape=inst_shape)


def test_empty_instance_shape_represents_one_instance() -> None:
    module = _Module(config=_ModuleConfig(), policy=_ModulePolicy(), inst_shape=())
    assert module.inst_count == 1


@pytest.mark.parametrize("base", [ConfigBase, PolicyBase])
def test_config_and_policy_reject_an_initial_value(base: type) -> None:
    with pytest.raises(TypeError, match=r"_InvalidStructuredInput\.value carries an initial value"):

        class _InvalidStructuredInput(base):
            value: int = 3


@pytest.mark.parametrize("metric", ["area__um2", "leakage__uW"])
def test_a_profile_target_without_static_ppa_fails_when_the_metric_is_read(metric: str) -> None:
    module = _Module(config=_ModuleConfig(), policy=_ModulePolicy(), inst_shape=())
    with pytest.raises(NotImplementedError):
        getattr(module, metric)


def test_a_module_counted_at_its_owner_needs_no_static_ppa() -> None:
    class _OwnedModule(ModuleBase[_ModuleConfig, _ModulePolicy]):
        is_profile_target = False

    assert _OwnedModule(config=_ModuleConfig(), policy=_ModulePolicy(), inst_shape=()).inst_count == 1


@pytest.mark.parametrize("metric", ["area__um2", "leakage__uW"])
def test_a_module_counted_at_its_owner_refuses_static_ppa_access(metric: str) -> None:
    class _OwnedModule(ModuleBase[_ModuleConfig, _ModulePolicy]):
        is_profile_target = False

    module = _OwnedModule(config=_ModuleConfig(), policy=_ModulePolicy(), inst_shape=())
    with pytest.raises(RuntimeError, match=r"is not a profile target"):
        getattr(module, metric)


def test_a_module_counted_at_its_owner_rejects_declared_static_ppa() -> None:
    with pytest.raises(TypeError, match=r"sets is_profile_target = False but declares _area_per_inst__um2"):

        class _InvalidOwnedModule(ModuleBase[_ModuleConfig, _ModulePolicy]):
            is_profile_target = False

            @property
            def _area_per_inst__um2(self) -> float:
                return 1.0


def test_building_a_dataclass_names_the_missing_required_keys() -> None:
    with pytest.raises(
        TypeError, match=r"_ChildConfig: missing key\(s\) \['child'\]; valid fields: \['child', 'parent'\]"
    ):
        dataclass_from_dict(_ChildConfig, {"parent": 1})
