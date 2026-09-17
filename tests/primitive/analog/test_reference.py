"""Recursive shape inference and rectangularity of reference values."""

from __future__ import annotations

from typing import Any

import pytest

from neurox.primitive.analog import ReferenceConfig


def _config(**overrides: Any) -> ReferenceConfig:
    values = {
        "values": ((1.0, -2.0), (3.0, 0.0)),
        "tolerance_sigma_relative": 0.0,
        "area_per_inst__um2": 0.0,
        "leakage_per_inst__uW": 0.0,
    }
    return ReferenceConfig(**{**values, **overrides})


@pytest.mark.parametrize(
    ("values", "shape"),
    [
        (1.0, ()),
        ((1.0, 2.0), (2,)),
        (((1.0, 2.0), (3.0, 4.0)), (2, 2)),
        ((((1.0,), (2.0,)), ((3.0,), (4.0,))), (2, 2, 1)),
    ],
)
def test_config_accepts_rectangular_arrays(values: Any, shape: tuple[int, ...]) -> None:
    assert _config(values=values).shape == shape


@pytest.mark.parametrize("values", [((1.0,), (2.0, 3.0)), (1.0, (2.0,)), (((1.0,),), ((2.0, 3.0),))])
def test_config_rejects_ragged_arrays_at_each_depth(values: Any) -> None:
    with pytest.raises(ValueError, match="values"):
        _config(values=values)
