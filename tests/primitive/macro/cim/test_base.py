"""The CIM-macro boundary masks valid output prefixes across caller and instance axes."""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.encoding import Encoding
from neurox.primitive.macro.cim import CimMacro, CimMacroQuantizationScheme, IdealCimMacroConfig, IdealCimMacroPolicy


class _EchoMacro(CimMacro):
    @property
    def adc_bits(self) -> int:
        return 8

    def program(self, w: Tensor) -> None:
        raise NotImplementedError

    def _vec_mat_mul_impl(
        self,
        x: Tensor,
        *,
        leading_shape: tuple[int, ...],
        quantization_mode: int,
        adc_active_bits: int | None,
        phase_mask: Tensor | None,
    ) -> Tensor:
        return x.expand(*leading_shape, self.output_num)

    def _latency_per_scan__ns(self, *, adc_active_bits: int | None) -> float:
        return 0.0


def test_effective_outputs_mask_each_mapped_access_independently(device: torch.device) -> None:
    macro = _EchoMacro(
        config=IdealCimMacroConfig(
            input_num=4,
            lane_num=2,
            scan_num=2,
            max_active_num=4,
            rescale_factors=(1.0,),
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
            w_digit_num=1,
            w_digit_radix=2,
            w_encoding=Encoding.TRUE_FORM,
            w_signed=True,
            x_digit_num=1,
            x_digit_radix=2,
            x_encoding=Encoding.UNSIGNED,
            x_value_range=(0, 1),
            w_value_range=(-1, 1),
            adc_bits=8,
            quantization_scheme=CimMacroQuantizationScheme.ZERO_POINT,
        ),
        policy=IdealCimMacroPolicy(),
        inst_shape=(2,),
        dtype=torch.float32,
    ).to(device)
    x = torch.ones((3, 2, 4), dtype=torch.int64, device=device)
    x[0, 0] = 0
    counts = torch.tensor([[0, 1], [2, 3], [4, 2]], device=device)
    actual = macro.vec_mat_mul(x, quantization_mode=0, adc_active_bits=None, effective_output_num=counts)
    expected = torch.tensor(
        [[[0, 0, 0, 0], [1, 0, 0, 0]], [[1, 1, 0, 0], [1, 1, 1, 0]], [[1, 1, 1, 1], [1, 1, 0, 0]]],
        device=device,
    )
    torch.testing.assert_close(actual, expected)
