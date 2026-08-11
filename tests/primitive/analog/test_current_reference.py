"""Iref: 2-D mode/tap bank and the fabricate-only PPA contract.

``Iref`` is a behavioural reference source: it carries static PPA
(area + leakage) and a fabricated ``[mode][tap]`` bank read back through
:attr:`Iref.i_out__uA`, but performs no computation, samples no per-call
noise, and emits no dynamic energy or latency. Static tolerance is its only
nonideality, drawn once at fabricate time. Mode selection and broadcasting
onto a caller's own shape are the consumer's job, done by plain indexing and
a view. These tests pin, unless a test says otherwise with the policy
all-off:

- 2-D validation on the bank: equal tap lengths and non-negative taps,
  while ordering within a mode is deliberately NOT enforced — what a mode
  means is the consumer's knowledge;
- ``mode_num`` / ``tap_num`` report the bank geometry;
- ``i_out__uA`` exposes the fabricated bank verbatim at
  ``[*inst_shape, mode_num, tap_num]``, with the mode axis intact for the
  consumer to index;
- with tolerance off ``i_out__uA`` is exactly the nominal bank, is stable
  across repeated reads, and draws no randomness outside ``fabricate()``;
- a zero tap stays exactly zero under relative tolerance;
- static PPA equals ``per_inst * inst_count`` and is visible to the
  profiler's static walk, while ``fabricate`` emits zero energy events;
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
        policy=policy if policy is not None else IrefPolicy(tolerance=False),
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


def test_i_out_exposes_the_fabricated_bank_verbatim() -> None:
    """``i_out__uA`` is the whole ``[mode][tap]`` bank, mode axis intact."""
    ref = _make()
    out = ref.i_out__uA
    assert out.shape == (_MODE_NUM, _TAP_NUM)
    assert torch.equal(out, torch.tensor(_TAPS, dtype=_DTYPE))


def test_indexing_a_mode_returns_that_mode_s_taps() -> None:
    """Indexing the mode axis is the consumer's own selection, with no source involvement."""
    ref = _make()
    for mode, taps in enumerate(_TAPS):
        out = ref.i_out__uA[..., mode, :]
        assert torch.equal(out, torch.tensor(taps, dtype=_DTYPE))


def test_all_off_i_out_is_exactly_nominal_and_draws_nothing_between_reads() -> None:
    """With tolerance off ``i_out__uA`` is bit-exact nominal, repeatable, and consumes no RNG."""
    ref = _make()
    nominal = torch.tensor(_TAPS, dtype=_DTYPE)

    rng_state = torch.random.get_rng_state()
    out_a = ref.i_out__uA
    out_b = ref.i_out__uA
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert torch.equal(out_a, nominal)
    assert torch.equal(out_b, nominal)


def test_inst_shape_prefixes_the_bank() -> None:
    """The fabricated bank carries ``inst_shape`` as a leading prefix."""
    ref = _make(inst_shape=(1, 2))
    nominal = torch.tensor(_TAPS, dtype=_DTYPE)

    out = ref.i_out__uA
    assert out.shape == (1, 2, _MODE_NUM, _TAP_NUM)
    assert torch.equal(out, nominal.expand(1, 2, _MODE_NUM, _TAP_NUM))


def test_zero_tap_stays_exactly_zero_under_relative_tolerance() -> None:
    """Relative tolerance is multiplicative, so an exact zero tap survives the draw."""
    ref = _make(
        config=_config(
            i_refs__uA=((0.0, 4.0),),
            tolerance_sigma_relative=0.1,
        ),
        policy=IrefPolicy(tolerance=True),
    )
    out = ref.i_out__uA
    assert torch.equal(out[..., 0], torch.zeros_like(out[..., 0]))


def test_static_ppa_and_no_dynamic_events() -> None:
    """Static PPA scales by ``inst_count``; fabricate emits no dynamic energy events."""
    ref = _make(inst_shape=(2,), area=2.0, leakage=0.5)

    assert ref.area__um2 == pytest.approx(2.0 * 2)
    assert ref.leakage__uW == pytest.approx(0.5 * 2)

    records = NeuroxProfiler.collect_static(ref)
    assert len(records) == 1
    assert records[0].area__um2 == pytest.approx(2.0 * 2)
    assert records[0].leakage_power__uW == pytest.approx(0.5 * 2)

    with NeuroxProfiler() as p:
        ref.fabricate()
    assert p.energy_events == []


def test_toml_nested_array_loads_as_tuple(tmp_path: Path) -> None:
    """A nested TOML array loads straight into the 2-D bank field."""
    toml = (
        "[ref]\n"
        "i_refs__uA = [[1.0, 5.0, 20.0], [2.0, 6.0, 25.0]]\n"
        "tolerance_sigma_relative = 0.01\n"
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
        "area_per_inst__um2 = 0.0\n"
        "leakage_per_inst__uW = 0.0\n"
    )
    path = tmp_path / "ref.toml"
    path.write_text(toml, encoding="utf-8")
    with pytest.raises(TypeError):
        IrefConfig.from_file(path, section="ref")
