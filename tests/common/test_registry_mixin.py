"""Explicit dispatch keys, independent family registries, and conflicting registrations."""

from __future__ import annotations

import pytest

from neurox.common.module import ConfigBase, NonProfileModule, PolicyBase
from neurox.common.registry_mixin import RegistryMixin


class _Config(ConfigBase):
    pass


class _OtherConfig(_Config):
    pass


class _Policy(PolicyBase):
    pass


class _OtherPolicy(_Policy):
    pass


def test_config_and_policy_both_select_family_members() -> None:
    class Family(NonProfileModule, RegistryMixin[_Config, _Policy], base_only=True):
        pass

    @Family.register_neurox_impl(config_type=_Config, policy_type=_Policy)
    class First(Family):
        pass

    @Family.register_neurox_impl(config_type=_OtherConfig, policy_type=_Policy)
    class Second(First):
        pass

    @Family.register_neurox_impl(config_type=_Config, policy_type=_OtherPolicy)
    class Third(First):
        pass

    assert Family._lookup_impl(config=_Config(), policy=_Policy()) is First
    assert Family._lookup_impl(config=_OtherConfig(), policy=_Policy()) is Second
    assert Family._lookup_impl(config=_Config(), policy=_OtherPolicy()) is Third
    implementation = Family._lookup_impl(config=_OtherConfig(), policy=_Policy())
    instance = implementation(config=_OtherConfig(), policy=_Policy(), inst_shape=())
    assert type(instance) is Second


def test_family_bindings_are_independent_and_conflicts_preserve_the_first_registration() -> None:
    class Family(NonProfileModule, RegistryMixin[_Config, _Policy], base_only=True):
        pass

    class OtherFamily(RegistryMixin[_Config, _Policy], NonProfileModule, base_only=True):
        pass

    @Family.register_neurox_impl(config_type=_Config, policy_type=_Policy)
    class First(Family):
        pass

    @OtherFamily.register_neurox_impl(config_type=_Config, policy_type=_Policy)
    class Other(OtherFamily):
        pass

    register = Family.register_neurox_impl(config_type=_Config, policy_type=_Policy)
    assert register(First) is First

    class Second(First):
        pass

    with pytest.raises(TypeError, match="cannot rebind"):
        register(Second)
    assert Family._lookup_impl(config=_Config(), policy=_Policy()) is First
    assert OtherFamily._lookup_impl(config=_Config(), policy=_Policy()) is Other
