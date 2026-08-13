"""Tests for the `SerializeMixin` abstract-base and receiver-bounded discriminator rules."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

import pytest

from neurox.common import SerializeMixin


@dataclass(frozen=True)
class _Fam(SerializeMixin, ABC):
    """Abstract base (declared `ABC` signal) of a polymorphic family."""


@dataclass(frozen=True)
class _LeafA(_Fam):
    pass


@dataclass(frozen=True)
class _Other(SerializeMixin, ABC):
    """A second, unrelated polymorphic family."""


@dataclass(frozen=True)
class _LeafB(_Other):
    pass


@dataclass(frozen=True)
class _ConcreteRoot(SerializeMixin):
    """Concrete class that happens to have a subclass — still buildable."""


@dataclass(frozen=True)
class _Skin(_ConcreteRoot):
    pass


@dataclass(frozen=True)
class _Box(SerializeMixin):
    """Container nesting a field typed as the abstract base `_Fam`."""

    item: _Fam


@dataclass(frozen=True)
class _UnsupportedAnnotation(SerializeMixin):
    value: complex


# --- abstract base: an undiscriminated abstract base cannot be built directly ---


def test_from_dict_abstract_base_without_discriminator_raises() -> None:
    with pytest.raises(TypeError):
        _Fam.from_dict({})


def test_from_dict_abstract_base_with_discriminator_yields_leaf() -> None:
    fam = _Fam.from_dict({"_neurox_class": "_LeafA"})
    assert isinstance(fam, _LeafA)


def test_from_dict_leaf_succeeds_without_discriminator() -> None:
    leaf = _LeafA.from_dict({})
    assert leaf == _LeafA()


# --- concrete-with-subclass: abstractness is the declared ABC signal, not
# --- subclass existence — a concrete family root stays buildable even when a
# --- subclass of it is imported elsewhere.


def test_from_dict_concrete_root_with_subclass_builds_bare() -> None:
    root = _ConcreteRoot.from_dict({})
    assert type(root) is _ConcreteRoot


def test_from_dict_concrete_root_self_discriminator_builds_root() -> None:
    root = _ConcreteRoot.from_dict({"_neurox_class": "_ConcreteRoot"})
    assert type(root) is _ConcreteRoot


def test_from_dict_concrete_root_subclass_discriminator_builds_skin() -> None:
    skin = _ConcreteRoot.from_dict({"_neurox_class": "_Skin"})
    assert type(skin) is _Skin


# --- receiver-bounded: the discriminator only resolves within the receiver's subtree ---


def test_from_dict_discriminator_outside_receiver_subtree_raises() -> None:
    with pytest.raises(TypeError):
        _Fam.from_dict({"_neurox_class": "_LeafB"})


def test_from_dict_discriminator_inside_receiver_subtree_succeeds() -> None:
    fam = _Fam.from_dict({"_neurox_class": "_LeafA"})
    assert fam == _LeafA()


# --- nested: a field typed as an abstract base enforces the same rules ---


def test_nested_field_without_discriminator_raises() -> None:
    with pytest.raises(TypeError):
        _Box.from_dict({"item": {}})


def test_nested_field_with_discriminator_yields_leaf() -> None:
    box = _Box.from_dict({"item": {"_neurox_class": "_LeafA"}})
    assert isinstance(box.item, _LeafA)


def test_unsupported_field_annotation_raises_at_deserialization() -> None:
    with pytest.raises(TypeError, match=r"_UnsupportedAnnotation\.value: unsupported field annotation"):
        _UnsupportedAnnotation.from_dict({"value": 1.0})
