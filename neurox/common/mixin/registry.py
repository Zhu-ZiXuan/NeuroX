"""Typed dispatch mixins for implementation families."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from neurox.common.base import ConfigBase, ModuleBase, PolicyBase

ConfigT = TypeVar("ConfigT", bound="ConfigBase")
PolicyT = TypeVar("PolicyT", bound="PolicyBase")
ModuleT = TypeVar("ModuleT", bound="ModuleBase")


class RegistryMixin(Generic[ConfigT, PolicyT, ModuleT]):
    """Dispatch a module family from concrete config and policy types."""

    _module_registry: dict[tuple[type[ConfigT], type[PolicyT]], type[ModuleT]]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        for ancestor in cls.__mro__[1:]:
            if "_module_registry" in ancestor.__dict__:
                return
        cls._module_registry = {}

    @classmethod
    def register_neurox_module(
        cls,
        *,
        config_type: type[ConfigT],
        policy_type: type[PolicyT],
    ) -> Callable[[type[ModuleT]], type[ModuleT]]:
        """Register one concrete config-policy pair for a module class."""
        key = (config_type, policy_type)
        registry = cls._module_registry

        def _decorator(module_type: type[ModuleT]) -> type[ModuleT]:
            existing = registry.get(key)
            if existing is not None and existing is not module_type:
                raise TypeError(
                    f"config {config_type.__name__} and policy {policy_type.__name__} "
                    f"already select {existing.__name__}; cannot rebind to {module_type.__name__}."
                )
            registry[key] = module_type
            return module_type

        return _decorator

    @classmethod
    def _lookup_neurox_module(cls, *, config: ConfigT, policy: PolicyT) -> type[ModuleT]:
        """Return the module class selected by concrete config and policy types."""
        key = (type(config), type(policy))
        module_type = cls._module_registry.get(key)
        if module_type is None:
            known = ", ".join(
                sorted(
                    f"({config_type.__name__}, {policy_type.__name__})"
                    for config_type, policy_type in cls._module_registry
                )
            )
            raise TypeError(
                f"no {cls.__name__} module registered for config {type(config).__name__} "
                f"and policy {type(policy).__name__}; known: {known or '<empty>'}"
            )
        return module_type
