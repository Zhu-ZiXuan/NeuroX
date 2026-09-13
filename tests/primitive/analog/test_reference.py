"""Generic reference-source shape, fabrication, and PPA contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from neurox.primitive.analog import Reference, ReferenceConfig, ReferencePolicy

_DTYPE = torch.float64


def _config(**overrides: Any) -> ReferenceConfig:
    values = {
        "values": ((1.0, -2.0), (3.0, 0.0)),
        "tolerance_sigma_relative": 0.0,
        "area_per_inst__um2": 0.0,
        "leakage_per_inst__uW": 0.0,
    }
    return ReferenceConfig(**{**values, **overrides})


def _make(
    *,
    config: ReferenceConfig | None = None,
    policy: ReferencePolicy | None = None,
    inst_shape: tuple[int, ...] = (),
) -> Reference:
    reference = Reference(
        config=_config() if config is None else config,
        policy=ReferencePolicy(tolerance=False) if policy is None else policy,
        inst_shape=inst_shape,
        dtype=_DTYPE,
    )
    reference.fabricate()
    return reference


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


@pytest.mark.parametrize("values", [(), ((),), ((1.0,), (2.0, 3.0)), (1.0, (2.0,))])
def test_config_rejects_empty_or_ragged_arrays(values: Any) -> None:
    with pytest.raises(ValueError, match="values"):
        _config(values=values)


def test_fabricated_shape_prefixes_config_shape() -> None:
    reference = _make(inst_shape=(2, 1))
    expected = torch.tensor(((1.0, -2.0), (3.0, 0.0)), dtype=_DTYPE).expand(2, 1, 2, 2)

    assert reference.values().shape == (2, 1, 2, 2)
    assert torch.equal(reference.values(), expected)


def test_scalar_reference_prefixes_only_instance_shape() -> None:
    reference = _make(config=_config(values=0.3), inst_shape=(2, 3))

    assert reference.values().shape == (2, 3)
    assert torch.equal(reference.values(), torch.full((2, 3), 0.3, dtype=_DTYPE))


def test_recursive_values_load_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "reference.toml"
    path.write_text(
        "[reference]\n"
        "values = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]\n"
        "tolerance_sigma_relative = 0.0\n"
        "area_per_inst__um2 = 0.0\n"
        "leakage_per_inst__uW = 0.0\n",
        encoding="utf-8",
    )

    config = ReferenceConfig.from_file(path, section="reference")

    assert config.values == (((1.0, 2.0), (3.0, 4.0)), ((5.0, 6.0), (7.0, 8.0)))
    assert config.shape == (2, 2, 2)
