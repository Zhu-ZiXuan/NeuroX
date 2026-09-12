"""Current-DAC conversion through its shared hardware entry point."""

import pytest
import torch

from neurox.primitive.analog.current_dac import GeneralIdac, GeneralIdacConfig, GeneralIdacPolicy


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_conversion_does_not_track_signal_table_gradients(compiled: bool) -> None:
    dac = GeneralIdac(
        config=GeneralIdacConfig(
            code_to_signal=(0.0, 2.0),
            drive_thermal__uA=0.0,
            energy_per_op__fJ=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        policy=GeneralIdacPolicy(drive_thermal=False),
        inst_shape=(),
        dtype=torch.float64,
        T__K=300.0,
    )
    dac._code_to_signal.requires_grad_()
    code = torch.tensor([0, 1], dtype=torch.int64)
    convert = torch.compile(dac.convert) if compiled else dac.convert

    with torch.enable_grad():
        signal = convert(code)
        assert torch.is_grad_enabled()

    assert not signal.requires_grad
    torch.testing.assert_close(signal, dac._code_to_signal)
