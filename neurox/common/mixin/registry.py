"""Key-based dispatch mixin for polymorphic families.

See also:
    docs/internals/common/mixin/registry.md
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Generic, TypeVar, cast

KeyT = TypeVar("KeyT")
ImplT = TypeVar("ImplT", bound="RegistryMixin")


class RegistryMixin(Generic[KeyT, ImplT]):
    """Grant a family root a ``key -> impl_class`` registry and typed lookup.

    A polymorphic family mixes this in on its root to dispatch construction on
    either a config class or a string discriminator. The mixin owns only the
    mapping and its access — ``register_key`` to bind an impl and
    ``_lookup_impl`` to resolve one; it ships no ``from_*`` factory and no
    constructor, so building an impl and fixing the family's runtime-parameter
    signature stay with the family root.

    Host requirements:
        - Inherit ``RegistryMixin[KeyT, ImplT]`` on the family root, binding
          ``KeyT`` — the key's runtime type, ``type[SomeConfig]`` for
          config-class dispatch or a ``Literal`` for string-discriminator
          dispatch — and ``ImplT``, the family's impl base.
        - Do not hand-declare ``_impl_registry``; the mixin materialises it,
          and a manual declaration shadows the mechanism and can break sibling
          isolation.
        - Import every impl module from the family package ``__init__`` so its
          ``register_key`` runs; an un-imported impl is absent from the
          registry and surfaces only as a lookup ``TypeError``.
        - Declare the family's own factory — canonically ``from_config`` — with
          the family's runtime-parameter signature; it calls ``_lookup_impl``
          to resolve the class and supplies the arguments.

    Injected behavior:
        - ``__init_subclass__`` materialises a fresh registry for each family
          root at class-definition time: the first subclass to introduce the
          mixin gets its own dict, sibling roots get independent dicts, and
          deeper impls inherit the root's.
        - Key bindings land at import time as impl modules are imported and
          their ``register_key`` decorators run.
    """

    _impl_registry: ClassVar[dict[object, type]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for ancestor in cls.__mro__[1:]:
            if "_impl_registry" in ancestor.__dict__:
                return
        cls._impl_registry = {}

    @classmethod
    def register_key(cls, key: KeyT) -> Callable[[type[ImplT]], type[ImplT]]:
        """Register ``key`` for the decorated implementation class.

        Binds ``key`` to the decorated class in this family root's registry and
        returns the class unchanged, so it reads as ``@Family.register_key(...)``;
        the root it is invoked on selects the target registry.

        Args:
            key: The key to bind — a config class or string discriminator of
                the family's ``KeyT`` type.

        Returns:
            A decorator that registers the class and returns it unchanged.

        Raises:
            TypeError: If ``key`` is already bound to a different class;
                rebinding to the same class is a no-op.
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

        Performs no key coercion: a config-class-keyed caller materialises the
        key at the call site as ``cls._lookup_impl(type(config))``.

        Args:
            key: The key to resolve, of the family's ``KeyT`` type.

        Returns:
            The implementation class bound to ``key``.

        Raises:
            TypeError: If no impl is registered for ``key``; the message lists
                the sorted known keys.
        """
        impl = cls._impl_registry.get(key)
        if impl is None:
            known = ", ".join(sorted(_key_repr(k) for k in cls._impl_registry)) or "<empty>"
            raise TypeError(f"no {cls.__name__} impl registered for key {_key_repr(key)}; known: {known}")
        return cast("type[ImplT]", impl)


def _key_repr(key: object) -> str:
    """Render registry keys uniformly: class names for type keys, repr otherwise."""
    return key.__name__ if isinstance(key, type) else repr(key)
