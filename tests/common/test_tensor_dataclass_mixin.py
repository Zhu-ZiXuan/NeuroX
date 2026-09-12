"""Tensor dataclasses preserve immutable construction and identity equality."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from inspect import Parameter, signature

import pytest
import torch
from torch import Tensor

from neurox.common.tensor_dataclass_mixin import TensorDataClassMixin


class _Parent(TensorDataClassMixin):
    value: Tensor


class _Child(_Parent):
    count: int


def test_inherited_fields_are_frozen_and_keyword_only() -> None:
    assert [field.name for field in fields(_Child)] == ["value", "count"]
    assert all(parameter.kind is Parameter.KEYWORD_ONLY for parameter in signature(_Child).parameters.values())
    node = _Child(value=torch.ones(2), count=1)
    with pytest.raises(FrozenInstanceError):
        node.count = 2


def test_equality_and_hash_use_identity_for_tensor_fields() -> None:
    node = _Child(value=torch.ones(2), count=1)
    other = _Child(value=node.value, count=node.count)
    assert node == node
    assert node != other
    assert len({node, other}) == 2


def test_tensor_data_class_rejects_a_custom_init() -> None:
    with pytest.raises(TypeError, match=r"must declare dataclass fields, not __init__\(\)"):

        class _InvalidTensorData(TensorDataClassMixin):
            def __init__(self) -> None:
                pass


def test_tensor_data_class_rejects_a_custom_post_init() -> None:
    with pytest.raises(TypeError, match=r"carries data only; it declares fields, not __post_init__\(\)"):

        class _InvalidTensorData(TensorDataClassMixin):
            def __post_init__(self) -> None:
                pass


def test_tensor_data_class_rejects_an_initial_value() -> None:
    with pytest.raises(TypeError, match=r"_InvalidTensorData\.value carries an initial value"):

        class _InvalidTensorData(TensorDataClassMixin):
            value: int = 3
