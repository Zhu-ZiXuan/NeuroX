"""Key-based dispatch mixin for polymorphic families."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

KeyT = TypeVar("KeyT")
ImplT = TypeVar("ImplT", bound="RegistryMixin")


class RegistryMixin(Generic[KeyT, ImplT]):
    """Add a typed implementation registry to a class family.

    Host requirements:
        - Bind ``KeyT`` and ``ImplT`` on the family root.
        - Keep every coexisting ``__init_subclass__`` cooperative by calling
          ``super().__init_subclass__()``.
        - Register implementation classes with :meth:`register_key`.
        - Define construction in the family root; this mixin only resolves
          classes.
    """

    _impl_registry: dict[KeyT, type[ImplT]]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        for ancestor in cls.__mro__[1:]:
            if "_impl_registry" in ancestor.__dict__:
                return
        cls._impl_registry = {}

    @classmethod
    def register_key(cls, key: KeyT) -> Callable[[type[ImplT]], type[ImplT]]:
        """Register ``key`` for the decorated implementation class.

        Args:
            key: Registry key to bind.

        Returns:
            A decorator that registers the class and returns it unchanged.

        Raises:
            TypeError: ``key`` is already bound to a different class.
        """
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
        """Return the implementation class registered under ``key``.

        Args:
            key: Registry key to resolve.

        Returns:
            The implementation class bound to ``key``.

        Raises:
            TypeError: No implementation is registered for ``key``.
        """
        impl = cls._impl_registry.get(key)
        if impl is None:
            known = ", ".join(sorted(_key_repr(k) for k in cls._impl_registry)) or "<empty>"
            raise TypeError(f"no {cls.__name__} impl registered for key {_key_repr(key)}; known: {known}")
        return impl


def _key_repr(key: object) -> str:
    """Render registry keys uniformly: class names for type keys, repr otherwise."""
    return key.__name__ if isinstance(key, type) else repr(key)
