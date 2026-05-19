"""Config-type dispatch mixin for polymorphic families.

See also:
    docs/dev/architecture/config_and_construction.md
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, Generic, TypeVar, cast

CfgT = TypeVar("CfgT")
ImplT = TypeVar("ImplT", bound="ConfigDispatchMixin")


class ConfigDispatchMixin(Generic[CfgT, ImplT]):
    """Mixin providing a ``config_type -> impl_class`` registry."""

    _config_registry: ClassVar[dict[type, type]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Create a fresh family registry on the first subclass."""
        super().__init_subclass__(**kwargs)
        for ancestor in cls.__mro__[1:]:
            if "_config_registry" in ancestor.__dict__:
                return
        cls._config_registry = {}

    @classmethod
    def register_config(cls, cfg_cls: type[CfgT]) -> Callable[[type[ImplT]], type[ImplT]]:
        """Register ``cfg_cls`` for the decorated implementation."""
        registry = cls._config_registry

        def _decorator(impl_cls: type[ImplT]) -> type[ImplT]:
            existing = registry.get(cfg_cls)
            if existing is not None and existing is not impl_cls:
                raise TypeError(
                    f"config {cfg_cls.__name__} already registered to "
                    f"{existing.__name__}; cannot rebind to {impl_cls.__name__}."
                )
            registry[cfg_cls] = impl_cls
            return impl_cls

        return _decorator

    @classmethod
    def _lookup_impl(cls, cfg: CfgT) -> type[ImplT]:
        """Return the implementation registered for ``type(cfg)``."""
        impl = cls._config_registry.get(type(cfg))
        if impl is None:
            known = ", ".join(sorted(c.__name__ for c in cls._config_registry)) or "<empty>"
            raise TypeError(f"no {cls.__name__} impl registered for config type {type(cfg).__name__}; known: {known}")
        return cast("type[ImplT]", impl)
