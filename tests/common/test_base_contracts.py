"""Tests for common base-class construction and rejection contracts."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

import pytest
import torch.nn as nn

from neurox import fabricate, set_temperature
from neurox.common.module import ConfigBase, ModuleBase, PolicyBase
from neurox.common.registry_mixin import RegistryMixin


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
    ModuleBase,
):
    pass


@_ModuleRegistryRoot.register_impl(config_type=_ModuleConfig, policy_type=_ModulePolicy)
class _RegisteredModule(_ModuleRegistryRoot):
    pass


class _Module(ModuleBase):
    pass


class _FabricatingModule(ModuleBase):
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


def test_config_validation_runs_after_construction() -> None:
    _ValidatedConfig(value=0)
    with pytest.raises(ValueError, match="value must be non-negative"):
        _ValidatedConfig(value=-1)


def test_policy_validation_runs_after_construction() -> None:
    _ValidatedPolicy(count=0)
    with pytest.raises(ValueError, match="count must be non-negative"):
        _ValidatedPolicy(count=-1)


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
    assert _ModuleRegistryRoot._lookup_impl(config=_ModuleConfig(), policy=_ModulePolicy()) is _RegisteredModule


class _TemperatureModule(ModuleBase):
    def __init__(self, events: list[tuple[str, float]], name: str, *children: nn.Module) -> None:
        super().__init__(config=_ModuleConfig(), policy=_ModulePolicy(), inst_shape=())
        self.events = events
        self.name = name
        self.children_ = nn.ModuleList(children)

    def _on_temperature_changed(self) -> None:
        self.events.append((self.name, self.T__K))


def test_temperature_reaches_plain_containers_and_is_local_to_subtree() -> None:
    events: list[tuple[str, float]] = []
    child = _TemperatureModule(events, "child")
    parent = _TemperatureModule(events, "parent", _PlainWrapper(child))
    sibling = _TemperatureModule(events, "sibling")
    model = nn.ModuleList([_PlainWrapper(parent), sibling])
    set_temperature(model, 350.0)
    assert sorted(events) == [("child", 350.0), ("parent", 350.0), ("sibling", 350.0)]
    events.clear()
    parent.set_temperature(320.0)
    assert sorted(events) == [("child", 320.0), ("parent", 320.0)]
    assert sibling.T__K == 350.0


def test_temperature_default_hook_and_newly_attached_child() -> None:
    parent = _Module(config=_ModuleConfig(), policy=_ModulePolicy(), inst_shape=())
    parent.set_temperature(350.0)
    child = _Module(config=_ModuleConfig(), policy=_ModulePolicy(), inst_shape=())
    parent.add_module("child", child)
    parent.set_temperature(350.0)
    assert child.T__K == parent.T__K == 350.0
