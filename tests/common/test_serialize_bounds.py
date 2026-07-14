"""Tests for the ``SerializeMixin`` abstract-base and receiver-bounded discriminator rules."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from neurox.common.mixin import SerializeMixin


@dataclass(frozen=True)
class _Fam(SerializeMixin):
    """Base of a polymorphic family for the abstract-base / receiver-bounded tests."""


@dataclass(frozen=True)
class _LeafA(_Fam):
    pass


@dataclass(frozen=True)
class _Other(SerializeMixin):
    """A second, unrelated polymorphic family."""


@dataclass(frozen=True)
class _LeafB(_Other):
    pass


@dataclass(frozen=True)
class _Box(SerializeMixin):
    """Container nesting a field typed as the abstract base ``_Fam``."""

    item: _Fam


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
