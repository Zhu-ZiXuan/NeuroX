"""Typed dispatch mixins for implementation families."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from .module import ConfigBase, ModuleBase, PolicyBase

ConfigT = TypeVar("ConfigT", bound=ConfigBase)
PolicyT = TypeVar("PolicyT", bound=PolicyBase)
ModuleT = TypeVar("ModuleT", bound=ModuleBase[ConfigBase, PolicyBase])


class RegistryMixin(Generic[ConfigT, PolicyT, ModuleT]):
    """Dispatch a module family from concrete config and policy types.

    The class that first mixes this in owns one registry table keyed by
    `(config type, policy type)`; every class below it in the family shares that
    same table, so one pair selects one implementation family-wide.

    Mix this in on the family base class, parameterized with the family's
    abstract config, policy, and module types, and decorate each concrete
    implementation with `register_neurox_module` for the pair it serves. The
    family base must expose a public classmethod that builds what
    `_lookup_neurox_module` returns, since callers of the family never reach the
    registry themselves.
    """

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
        """Bind one concrete config-policy pair to a module class.

        Args:
            config_type: Concrete configuration type selecting the class.
            policy_type: Concrete policy type selecting the class.

        Returns:
            Class decorator recording the binding and returning the decorated
            class unchanged. Re-decorating the same class with the same pair is
            idempotent.

        Raises:
            TypeError: Raised by the returned decorator when the pair already
                selects a different module class; rebinding is rejected.
        """
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
        """Return the module class selected by concrete config and policy types.

        Args:
            config: Configuration instance; only its type selects the class.
            policy: Policy instance; only its type selects the class.

        Returns:
            Module class registered for the pair.

        Raises:
            TypeError: No module is registered for the pair; the message lists
                the pairs the family knows.
        """
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
