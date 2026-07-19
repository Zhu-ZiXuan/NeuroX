"""IdealCimMacro emits its returned codes on the ``adc.ideal_vmm`` probe channel.

Both the lossless (``adc_bits == 0``) and quantized paths must emit exactly
the tensor they return, once per call, and stay silent without a prober.
"""

from __future__ import annotations

import torch

from neurox.common.prober import AdcProber
from neurox.primitive.analog.adc_common import AdcOperationPoint
from neurox.primitive.macro.cim.ideal import IdealCimMacro, IdealCimMacroConfig, IdealCimMacroPolicy


def _make_xbar(*, adc_max_bits: int) -> IdealCimMacro:
    config = IdealCimMacroConfig(
        col_num=3,
        row_num=4,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        x_range=(0, 1),
        w_digit_count=1,
        w_digit_radix=2,
        w_digit_range=(-1, 1),
        adc_mode_num=1,
        adc_max_bits=adc_max_bits,
    )
    xbar = IdealCimMacro(
        config=config,
        policy=IdealCimMacroPolicy(),
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    xbar.eval()
    xbar.fabricate()
    return xbar


def _program_and_input(xbar: IdealCimMacro) -> torch.Tensor:
    w = torch.tensor([[1, -1, 1, 0], [0, 1, 1, -1], [1, 1, 0, 1]], dtype=torch.int32).unsqueeze(-2)
    xbar.program(w)
    return torch.tensor([1, 0, 1, 1], dtype=torch.int32)


def test_lossless_path_emits_the_returned_dot() -> None:
    xbar = _make_xbar(adc_max_bits=0)
    x = _program_and_input(xbar)
    with AdcProber() as prober:
        out = xbar.vec_mat_mul(x, adc_operation_point=AdcOperationPoint(adc_mode=0, adc_bits=0))
    records = prober.ideal_vmm_records()
    assert len(records) == 1
    module, tensors = records[0]
    assert module is xbar
    assert torch.equal(tensors["code"], out)


def test_quantized_path_emits_the_returned_codes() -> None:
    xbar = _make_xbar(adc_max_bits=4)
    x = _program_and_input(xbar)
    with AdcProber() as prober:
        out = xbar.vec_mat_mul(x, adc_operation_point=AdcOperationPoint(adc_mode=0, adc_bits=4))
    records = prober.ideal_vmm_records()
    assert len(records) == 1
    assert torch.equal(records[0][1]["code"], out)


def test_no_prober_means_no_record_and_unchanged_numerics() -> None:
    xbar = _make_xbar(adc_max_bits=0)
    x = _program_and_input(xbar)
    silent = xbar.vec_mat_mul(x, adc_operation_point=AdcOperationPoint(adc_mode=0, adc_bits=0))
    with AdcProber() as prober:
        probed = xbar.vec_mat_mul(x, adc_operation_point=AdcOperationPoint(adc_mode=0, adc_bits=0))
    assert torch.equal(silent, probed)
    assert len(prober.ideal_vmm_records()) == 1
