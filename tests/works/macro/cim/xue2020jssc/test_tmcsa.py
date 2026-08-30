"""Functional, timing, and energy laws for the Xue2020 TMCSA."""

from __future__ import annotations

import pytest
import torch

from neurox import Profiler, Reporter, stamp_names
from neurox.primitive.analog.current_adc import SarIadc
from neurox.works.macro.cim.xue2020jssc.tmcsa import Tmcsa, TmcsaConfig, TmcsaPolicy

_DTYPE = torch.float64
_LANE_NUM = 2
_BITS = 3
_VDD__V = 1.2
_T_PH2__NS = 0.4
_T_PH3__NS = 0.6
_ENERGY_PER_BIT__fJ = 7.0
_AREA_PER_INST__um2 = 2.0
_LEAKAGE_PER_INST__uW = 3.0
_LADDER = tuple(float(k + 1) for k in range((1 << _BITS) - 1))


def _config(
    *,
    t_ph2__ns: float = _T_PH2__NS,
    t_ph3__ns: float = _T_PH3__NS,
    energy_per_bit__fJ: float = _ENERGY_PER_BIT__fJ,
) -> TmcsaConfig:
    return TmcsaConfig(
        bits=_BITS,
        t_ph2__ns=t_ph2__ns,
        t_ph3__ns=t_ph3__ns,
        energy_per_bit__fJ=energy_per_bit__fJ,
        latency_per_bit__ns=1.0,
        comparator_offset_sigma__uA=0.0,
        area_per_inst__um2=_AREA_PER_INST__um2,
        leakage_per_inst__uW=_LEAKAGE_PER_INST__uW,
    )


def _build(*, lane_num: int = _LANE_NUM) -> Tmcsa:
    module = Tmcsa(
        config=_config(),
        policy=TmcsaPolicy(comparator_offset=False),
        inst_shape=(lane_num,),
        vdd__V=_VDD__V,
        dtype=_DTYPE,
        T__K=300.0,
    )
    module.eval()
    module.fabricate()
    stamp_names(module)
    return module


def _refs() -> torch.Tensor:
    return torch.tensor(_LADDER, dtype=_DTYPE)


def _expected_step_path(i_in: float, *, active_bits: int) -> tuple[int, list[float]]:
    code = 0
    refs: list[float] = []
    for step in range(active_bits):
        unresolved_bits = _BITS - step - 1
        boundary_rank = ((code << 1) | 1) << unresolved_bits
        i_ref = _LADDER[boundary_rank - 1]
        refs.append(i_ref)
        code = (code << 1) | int(i_in >= i_ref)
    return code, refs


def _expected_energy(i_in: torch.Tensor, *, active_bits: int) -> float:
    energy__fJ = 0.0
    for value in i_in.flatten().tolist():
        _, refs = _expected_step_path(value, active_bits=active_bits)
        for i_ref__uA in refs:
            i_common__uA = value + i_ref__uA
            energy__fJ += (
                _VDD__V * (3.0 * i_common__uA * _T_PH2__NS + 2.0 * i_common__uA * _T_PH3__NS) + _ENERGY_PER_BIT__fJ
            )
    return energy__fJ


def test_inherits_the_core_sar_conversion() -> None:
    module = _build()
    assert isinstance(module, SarIadc)
    assert not any(name for name, _ in module.named_children())


def test_shape_and_static_ppa() -> None:
    module = _build()
    assert module.inst_shape == (_LANE_NUM,)
    assert module.inst_count == _LANE_NUM
    assert module.bits == _BITS
    assert module.area__um2 == pytest.approx(_AREA_PER_INST__um2 * _LANE_NUM)
    assert module.leakage__uW == pytest.approx(_LEAKAGE_PER_INST__uW * _LANE_NUM)


def test_convert_returns_codes_and_records_the_same_bit_energy() -> None:
    module = _build()
    i_in = torch.tensor([[[1.5, 2.5], [0.5, 6.5]]], dtype=_DTYPE)

    with Profiler() as profiler, torch.no_grad():
        code = module.convert(i_in, _refs(), active_bits=_BITS)

    assert torch.equal(code, torch.tensor([[[1, 2], [0, 6]]]))
    expected__fJ = _expected_energy(i_in, active_bits=_BITS)
    reporter = Reporter(module)
    assert reporter.total_dynamic_energy__fJ(profiler) == pytest.approx(expected__fJ, rel=1e-12)
    assert reporter.by_name(profiler) == {"": pytest.approx(expected__fJ, rel=1e-12)}


def test_lowered_bits_execute_only_the_leading_steps() -> None:
    module = _build()
    active_bits = _BITS - 1
    i_in = torch.tensor([[[1.5, 2.5], [0.5, 6.5]]], dtype=_DTYPE)

    with Profiler() as profiler, torch.no_grad():
        code = module.convert(i_in, _refs(), active_bits=active_bits)

    expected_code = torch.empty_like(code)
    for index, value in enumerate(i_in.flatten().tolist()):
        expected_code.flatten()[index] = _expected_step_path(value, active_bits=active_bits)[0]
    assert torch.equal(code, expected_code)
    assert Reporter(module).total_dynamic_energy__fJ(profiler) == pytest.approx(
        _expected_energy(i_in, active_bits=active_bits),
        rel=1e-12,
    )


def test_latency_tracks_the_executed_steps() -> None:
    module = _build()
    for active_bits in range(1, _BITS + 1):
        assert module.latency__ns(active_bits=active_bits) == pytest.approx(
            active_bits * module.config.latency_per_bit__ns
        )


def test_convert_outside_profiler_still_returns_codes() -> None:
    module = _build()
    i_in = torch.tensor([[[1.5, 2.5], [0.5, 6.5]]], dtype=_DTYPE)
    code = module.convert(i_in, _refs(), active_bits=_BITS)
    assert torch.equal(code, torch.tensor([[[1, 2], [0, 6]]]))


def test_config_rejects_invalid_energy_parameters() -> None:
    for update, match in (
        ({"t_ph2__ns": -0.1}, "t_ph2__ns"),
        ({"t_ph3__ns": -0.1}, "t_ph3__ns"),
        ({"energy_per_bit__fJ": -1.0}, "energy_per_bit__fJ"),
    ):
        with pytest.raises(ValueError, match=match):
            _config(**update)


def test_config_requires_phases_to_fit_one_decision_step() -> None:
    with pytest.raises(ValueError, match=r"t_ph2__ns \+ t_ph3__ns"):
        _config(t_ph2__ns=0.8, t_ph3__ns=0.3)


def test_convert_rejects_invalid_bits() -> None:
    module = _build()
    i_in = torch.tensor([[[1.5, 2.5], [0.5, 6.5]]], dtype=_DTYPE)
    for active_bits in (0, _BITS + 1):
        with pytest.raises(ValueError, match="active_bits"):
            module.convert(i_in, _refs(), active_bits=active_bits)
