"""Generic reference-source shape, fabrication, and PPA contracts."""

from pathlib import Path
from typing import Any

import pytest
import torch

from neurox import Profiler, Reporter, stamp_names
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
        T__K=300.0,
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


def test_config_rejects_negative_tolerance_and_ppa() -> None:
    for name in ("tolerance_sigma_relative", "area_per_inst__um2", "leakage_per_inst__uW"):
        with pytest.raises(ValueError, match=name):
            _config(**{name: -1.0})


def test_fabricated_shape_prefixes_config_shape() -> None:
    reference = _make(inst_shape=(2, 1))
    expected = torch.tensor(((1.0, -2.0), (3.0, 0.0)), dtype=_DTYPE).expand(2, 1, 2, 2)

    assert reference.values().shape == (2, 1, 2, 2)
    assert torch.equal(reference.values(), expected)


def test_scalar_reference_prefixes_only_instance_shape() -> None:
    reference = _make(config=_config(values=0.3), inst_shape=(2, 3))

    assert reference.values().shape == (2, 3)
    assert torch.equal(reference.values(), torch.full((2, 3), 0.3, dtype=_DTYPE))


def test_reads_are_stable_and_zero_survives_relative_tolerance() -> None:
    reference = _make(
        config=_config(values=(0.0, 4.0), tolerance_sigma_relative=0.1),
        policy=ReferencePolicy(tolerance=True),
    )
    first = reference.values()
    rng_state = torch.random.get_rng_state()
    second = reference.values()

    assert torch.equal(first, second)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert first[0].item() == 0.0


def test_static_ppa_and_no_dynamic_records() -> None:
    reference = _make(
        config=_config(area_per_inst__um2=2.0, leakage_per_inst__uW=0.5),
        inst_shape=(2,),
    )
    stamp_names(reference)

    assert reference.area__um2 == pytest.approx(4.0)
    assert reference.leakage__uW == pytest.approx(1.0)
    assert len(Reporter(reference).static_entries) == 1
    with Profiler() as profiler:
        reference.fabricate()
    assert profiler.records == ()


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
