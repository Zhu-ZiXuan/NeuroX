"""Iref: 2-D mode/tap bank and the PPA-only contract.

``Iref`` is a behavioural reference source: it carries static
PPA (area + leakage) and hands out the actual tap values through a snap,
but performs no computation and emits no dynamic energy or latency. It
holds a 2-D ``[mode][tap]`` bank of equal-length non-negative modes whose
mode selection is quasi-static. These tests pin, unless a test says
otherwise with the policy all-off:

- 2-D validation on the bank: equal tap lengths and non-negative taps,
  while ordering within a mode is deliberately NOT enforced — what a mode
  means is the consumer's knowledge;
- ``mode_num`` / ``tap_num`` report the bank geometry;
- ``snapshot(mode=..., shape=...)`` returns exactly the requested shape,
  with the mode axis resolved away by the source;
- with noise off the snap is exactly the fabricated nominal, draws no
  randomness, and does not drift between calls;
- a zero tap stays exactly zero under relative noise;
- static PPA equals ``per_inst * inst_count`` and is visible to the
  profiler's static walk, while ``fabricate`` + ``snapshot`` emit zero
  energy / latency events;
- a nested TOML array loads straight into the bank field; a flat TOML
  array (wrong shape for the tap bank) is rejected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from neurox.common.profiler import NeuroxProfiler
from neurox.primitive.analog.current_reference import (
    Iref,
    IrefConfig,
    IrefPolicy,
)

_TAPS = ((1.0, 5.0, 20.0), (2.0, 6.0, 25.0))
_MODE_NUM = 2
_TAP_NUM = 3
_DTYPE = torch.float64


def _config(**overrides: Any) -> IrefConfig:
    base = {
        "i_refs__uA": _TAPS,
        "tolerance_sigma_relative": 0.0,
        "noise_sigma_relative": 0.0,
        "area_per_inst__um2": 0.0,
        "leakage_per_inst__uW": 0.0,
    }
    return IrefConfig(**{**base, **overrides})


def _make(
    *,
    inst_shape: tuple[int, ...] = (),
    area: float = 0.0,
    leakage: float = 0.0,
    config: IrefConfig | None = None,
    policy: IrefPolicy | None = None,
) -> Iref:
    ref = Iref(
        config=config if config is not None else _config(area_per_inst__um2=area, leakage_per_inst__uW=leakage),
        policy=policy if policy is not None else IrefPolicy(tolerance=False, noise=False),
        inst_shape=inst_shape,
        dtype=_DTYPE,
        T__K=300.0,
    )
    ref.fabricate()
    return ref


def test_validation_rejects_bad_config() -> None:
    """Empty taps and negative sigmas/PPA are rejected."""
    for override in (
        {"i_refs__uA": ()},  # empty bank
        {"tolerance_sigma_relative": -1e-3},
        {"noise_sigma_relative": -1e-3},
        {"area_per_inst__um2": -1.0},
        {"leakage_per_inst__uW": -1.0},
    ):
        with pytest.raises(ValueError):
            _config(**override)


def test_bank_2d_validation() -> None:
    """The bank enforces equal-length non-negative modes — and nothing about order."""
    for bad in (
        ((0.6, -1.0),),  # negative tap
        ((1.0, 2.0), (1.0, 2.0, 3.0)),  # unequal tap lengths
        ((),),  # empty mode
    ):
        with pytest.raises(ValueError):
            _config(i_refs__uA=bad)
    # Ordering is the CONSUMER's law, not the source's: a decision ladder
    # must ascend, a bank of bias taps need not.
    _config(i_refs__uA=((2.0, 1.0, 3.0),))
    _config(i_refs__uA=((1.0, 1.0),))
    # A 0 tap denotes a ground/rail reference and is accepted.
    _config(i_refs__uA=((0.0, 0.6),))


def test_mode_tap_counts() -> None:
    """``mode_num`` / ``tap_num`` report the bank geometry on config and module."""
    ref = _make()
    assert ref.config.mode_num == _MODE_NUM
    assert ref.config.tap_num == _TAP_NUM
    assert ref.mode_num == _MODE_NUM
    assert ref.tap_num == _TAP_NUM


def test_snapshot_shape_is_exactly_the_requested_shape() -> None:
    """The caller names the full output shape; the source honours it verbatim."""
    ref = _make()
    for shape in ((_TAP_NUM,), (4, _TAP_NUM), (2, 3, _TAP_NUM), (1, 1, 1, _TAP_NUM)):
        assert ref.snapshot(mode=0, shape=shape).i_refs__uA.shape == shape


def test_snapshot_shape_must_end_in_tap_num() -> None:
    """A trailing axis that is not the tap axis is a caller error, not a silent reshape."""
    ref = _make()
    with pytest.raises(RuntimeError):
        ref.snapshot(mode=0, shape=(_TAP_NUM + 1,))


def test_snapshot_resolves_the_mode_axis() -> None:
    """Selecting mode ``m`` returns that mode's taps, with no mode axis left."""
    ref = _make()
    for mode, taps in enumerate(_TAPS):
        out = ref.snapshot(mode=mode, shape=(2, _TAP_NUM)).i_refs__uA
        assert out.shape == (2, _TAP_NUM)
        assert torch.equal(out, torch.tensor(taps, dtype=_DTYPE).expand(2, _TAP_NUM))


def test_all_off_snapshot_is_exactly_nominal_and_draws_nothing() -> None:
    """With noise off the snap is bit-exact nominal, repeatable, and consumes no RNG."""
    ref = _make()
    nominal = torch.tensor(_TAPS[1], dtype=_DTYPE)

    rng_state = torch.random.get_rng_state()
    snap_a = ref.snapshot(mode=1, shape=(_TAP_NUM,))
    snap_b = ref.snapshot(mode=1, shape=(_TAP_NUM,))
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert torch.equal(snap_a.i_refs__uA, nominal)
    assert torch.equal(snap_b.i_refs__uA, nominal)


def test_inst_shape_prefixes_the_requested_shape() -> None:
    """The fabricated per-instance taps right-align under the requested shape."""
    ref = _make(inst_shape=(1, 2))
    nominal = torch.tensor(_TAPS[0], dtype=_DTYPE)

    out = ref.snapshot(mode=0, shape=(1, 2, _TAP_NUM)).i_refs__uA
    assert out.shape == (1, 2, _TAP_NUM)
    assert torch.equal(out, nominal.expand(1, 2, _TAP_NUM))

    # A wider call grid prepends leading axes onto the same instance taps.
    wide = ref.snapshot(mode=0, shape=(4, 1, 2, _TAP_NUM)).i_refs__uA
    assert wide.shape == (4, 1, 2, _TAP_NUM)
    assert torch.equal(wide, nominal.expand(4, 1, 2, _TAP_NUM))


def test_zero_tap_stays_exactly_zero_under_relative_noise() -> None:
    """Relative noise is multiplicative, so an exact zero tap survives both draws."""
    ref = _make(
        config=_config(
            i_refs__uA=((0.0, 4.0),),
            tolerance_sigma_relative=0.1,
            noise_sigma_relative=0.1,
        ),
        policy=IrefPolicy(tolerance=True, noise=True),
    )
    out = ref.snapshot(mode=0, shape=(8, 2)).i_refs__uA
    assert torch.equal(out[..., 0], torch.zeros(8, dtype=_DTYPE))


def test_static_ppa_and_no_dynamic_events() -> None:
    """Static PPA scales by ``inst_count``; fabricate/snapshot emit no events."""
    ref = _make(inst_shape=(2,), area=2.0, leakage=0.5)

    assert ref.area__um2 == pytest.approx(2.0 * 2)
    assert ref.leakage__uW == pytest.approx(0.5 * 2)

    records = NeuroxProfiler.collect_static(ref)
    assert len(records) == 1
    assert records[0].area__um2 == pytest.approx(2.0 * 2)
    assert records[0].leakage_power__uW == pytest.approx(0.5 * 2)

    with NeuroxProfiler() as p:
        ref.fabricate()
        ref.snapshot(mode=0, shape=(2, _TAP_NUM))
    assert p.energy_events == []
    assert p.latency_events == []


def test_toml_nested_array_loads_as_tuple(tmp_path: Path) -> None:
    """A nested TOML array loads straight into the 2-D bank field."""
    toml = (
        "[ref]\n"
        "i_refs__uA = [[1.0, 5.0, 20.0], [2.0, 6.0, 25.0]]\n"
        "tolerance_sigma_relative = 0.01\n"
        "noise_sigma_relative = 0.002\n"
        "area_per_inst__um2 = 1.0\n"
        "leakage_per_inst__uW = 0.5\n"
    )
    path = tmp_path / "ref.toml"
    path.write_text(toml, encoding="utf-8")

    config = IrefConfig.from_file(path, section="ref")
    assert isinstance(config.i_refs__uA, tuple)
    assert config.i_refs__uA == _TAPS


def test_toml_flat_array_rejected(tmp_path: Path) -> None:
    """A flat TOML array (wrong shape for the tap bank) does not deserialize into the 2-D bank field."""
    toml = (
        "[ref]\n"
        "i_refs__uA = [1.0, 5.0, 20.0]\n"
        "tolerance_sigma_relative = 0.0\n"
        "noise_sigma_relative = 0.0\n"
        "area_per_inst__um2 = 0.0\n"
        "leakage_per_inst__uW = 0.0\n"
    )
    path = tmp_path / "ref.toml"
    path.write_text(toml, encoding="utf-8")
    with pytest.raises(TypeError):
        IrefConfig.from_file(path, section="ref")
