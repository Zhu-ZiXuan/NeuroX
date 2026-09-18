"""Typed dispatch mixins for implementation families."""

from __future__ import annotations

from collections.abc import Callable
from typing import Self, final

from .module import ConfigBase, PolicyBase


class RegistryMixin[ConfigT: ConfigBase, PolicyT: PolicyBase]:
    """Dispatch a module family from concrete config and policy types.

    The class that first mixes this in owns one registry table keyed by
    `(config type, policy type)`; every class below it in the family shares that
    same table, so one pair selects one implementation family-wide.
    Registration occurs when the implementation module is imported; import
    implementations before resolving their config and policy pairs.

    Mix this in on the family base class, parameterized with the family's
    config and policy types, and decorate each concrete implementation with
    the family base's `register_neurox_impl` for the pair it serves. Registration and
    lookup are called on that base, so `Self` denotes the dispatched family.
    The family base exposes the public classmethod that constructs the result
    of `_lookup_impl`; implementations supply the execution behavior.
    """

    _module_registry: dict[tuple[type[ConfigT], type[PolicyT]], type[Self]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Descendants reuse the family table; resetting it would discard registrations.
        for ancestor in cls.__mro__[1:]:
            if "_module_registry" in ancestor.__dict__:
                return
        cls._module_registry = {}

    # === Public API ===

    @classmethod
    @final
    def register_neurox_impl(
        cls,
        *,
        config_type: type[ConfigT],
        policy_type: type[PolicyT],
    ) -> Callable[[type[Self]], type[Self]]:
        """Bind one concrete config-policy pair to a module class.

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

        def _decorator(module_type: type[Self]) -> type[Self]:
            existing = registry.get(key)
            if existing is not None and existing is not module_type:
                raise TypeError(
                    f"config {config_type.__name__} and policy {policy_type.__name__} "
                    f"already select {existing.__name__}; cannot rebind to {module_type.__name__}."
                )
            registry[key] = module_type
            return module_type

        return _decorator

    # === Tools for subclass and internal use ===

    @classmethod
    @final
    def _lookup_impl(cls, *, config: ConfigT, policy: PolicyT) -> type[Self]:
        """Return the module class selected by concrete config and policy types.

        Only the types select; the instances are never read.

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
