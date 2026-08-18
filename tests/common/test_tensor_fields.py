"""Tests for the shared dataclass tensor-field traversal."""

from __future__ import annotations

import pytest

from neurox.common import walk_tensor_fields


def test_walk_tensor_fields_rejects_a_non_dataclass_instance() -> None:
    with pytest.raises(TypeError, match=r"walk_tensor_fields\(\) requires a dataclass instance"):
        walk_tensor_fields(object(), lambda tensor: tensor)
