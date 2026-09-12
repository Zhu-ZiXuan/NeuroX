"""Voltage-DAC energy scales with per-code operation counts."""

from __future__ import annotations

import pytest
import torch

from neurox import Profiler, stamp_names
from neurox.api.profiler import EnergyRecord
from neurox.common.profile_mixin import ProfileMixin
from neurox.primitive.analog.voltage_dac import GeneralVdac, GeneralVdacConfig, GeneralVdacPolicy

_CODE_TO_SIGNAL__V = (0.0, 0.9)
_E_CODE_0__fJ = 3.5
_E_CODE_1__fJ = 8.25


def _build(code_to_per_op_energy__fJ: tuple[float, ...]) -> GeneralVdac:
    """Noise-free two-code DAC carrying the given per-code energy table."""
    dac = GeneralVdac(
        config=GeneralVdacConfig(
            code_to_signal=_CODE_TO_SIGNAL__V,
            drive_thermal__V=0.0,
            code_to_per_op_energy__fJ=code_to_per_op_energy__fJ,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        policy=GeneralVdacPolicy(drive_thermal=False),
        inst_shape=(),
        dtype=torch.float64,
        T__K=300.0,
    )
    dac.eval()
    dac.fabricate()
    stamp_names(dac)
    return dac


def _energy_total(records: list[EnergyRecord], module: ProfileMixin) -> float:
    """Sum the logged dynamic energy [fJ] of the records `module` emitted."""
    return sum(
        (float(r.dynamic_energy__fJ.sum()) for r in records if r.qualified_name == module.qualified_name),
        0.0,
    )


def test_conversion_bills_each_element_at_its_own_code() -> None:
    """A mixed batch costs `count0 * e0 + count1 * e1`; a zero entry costs nothing."""
    code = torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=torch.int64)
    count_1 = int(code.sum())
    count_0 = code.numel() - count_1

    dac = _build((_E_CODE_0__fJ, _E_CODE_1__fJ))
    with Profiler() as p:
        dac.convert(code)
    assert _energy_total(p.records, dac) == pytest.approx(count_0 * _E_CODE_0__fJ + count_1 * _E_CODE_1__fJ)

    free_zero = _build((0.0, _E_CODE_1__fJ))
    with Profiler() as p_free:
        free_zero.convert(code)
    assert _energy_total(p_free.records, free_zero) == pytest.approx(count_1 * _E_CODE_1__fJ)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_conversion_does_not_track_signal_table_gradients(compiled: bool) -> None:
    dac = _build((0.0, 0.0))
    dac._code_to_signal.requires_grad_()
    code = torch.tensor([0, 1], dtype=torch.int64)
    convert = torch.compile(dac.convert) if compiled else dac.convert

    with torch.enable_grad():
        signal = convert(code)
        assert torch.is_grad_enabled()

    assert not signal.requires_grad
    torch.testing.assert_close(signal, dac._code_to_signal)
