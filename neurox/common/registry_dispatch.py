"""Key-based dispatch mixin for polymorphic families.

See also:
    docs/dev/architecture/config_and_construction.md
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, Generic, TypeVar, cast

KeyT = TypeVar("KeyT")
ImplT = TypeVar("ImplT", bound="RegistryDispatchMixin")


class RegistryDispatchMixin(Generic[KeyT, ImplT]):
    """Mixin providing a ``key -> impl_class`` registry.

    ``KeyT`` is the registered key's runtime type. Common choices:

    - ``KeyT = type[SomeConfig]`` — config-class-keyed dispatch. Callers
      that hold a config instance ``cfg`` look it up with
      ``cls._lookup_impl(type(cfg))``.
    - ``KeyT = SomeLiteral`` — string-discriminator dispatch. Callers
      pass the literal directly: ``cls._lookup_impl(encoding)``.
    """

    _impl_registry: ClassVar[dict[object, type]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Create a fresh family registry on the first subclass."""
        super().__init_subclass__(**kwargs)
        for ancestor in cls.__mro__[1:]:
            if "_impl_registry" in ancestor.__dict__:
                return
        cls._impl_registry = {}

    @classmethod
    def register_key(cls, key: KeyT) -> Callable[[type[ImplT]], type[ImplT]]:
        """Register ``key`` for the decorated implementation class."""
        registry = cls._impl_registry

        def _decorator(impl_cls: type[ImplT]) -> type[ImplT]:
            existing = registry.get(key)
            if existing is not None and existing is not impl_cls:
                raise TypeError(
                    f"key {key!r} already registered to {existing.__name__}; cannot rebind to {impl_cls.__name__}."
                )
            registry[key] = impl_cls
            return impl_cls

        return _decorator

    @classmethod
    def _lookup_impl(cls, key: KeyT) -> type[ImplT]:
        """Return the implementation registered under ``key``."""
        impl = cls._impl_registry.get(key)
        if impl is None:
            known = ", ".join(sorted(_key_repr(k) for k in cls._impl_registry)) or "<empty>"
            raise TypeError(f"no {cls.__name__} impl registered for key {_key_repr(key)}; known: {known}")
        return cast("type[ImplT]", impl)


def _key_repr(key: object) -> str:
    """Render registry keys uniformly: class names for type keys, repr otherwise."""
    return key.__name__ if isinstance(key, type) else repr(key)
