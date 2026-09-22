"""CIM linear operator with directly owned macro and reconstruction circuits."""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.architecture.unit.cim import CimUnit, CimUnitConfig, CimUnitPolicy

from .base import LinearUnit, LinearUnitConfig, LinearUnitPolicy


class LinearCimUnitConfig(LinearUnitConfig, CimUnitConfig):
    pass


class LinearCimUnitPolicy(LinearUnitPolicy, CimUnitPolicy):
    pass


_Config = LinearCimUnitConfig
_Policy = LinearCimUnitPolicy


@LinearUnit.register_neurox_impl(config_type=_Config, policy_type=_Policy)
class LinearCimUnit(LinearUnit, CimUnit):
    """CIM linear operator using separate physical tiles without input-slot sharing."""

    config: _Config
    policy: _Policy

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        LinearUnit.__init__(
            self,
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
        )
        matrix_output_num, matrix_input_num = self._w_logical_shape
        CimUnit.__init__(
            self,
            matrix_input_num=matrix_input_num,
            matrix_output_num=matrix_output_num,
            merge=False,
            dtype=dtype,
        )

    @torch.no_grad()
    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        self._program_matrix(weight.unsqueeze(0))
        self._program_int_bias(bias, channels=self._w_logical_shape[0])

    def _linear_impl(self, input: Tensor, *, quantization_mode: int, adc_active_bits: int | None) -> Tensor:
        # Shape: [..., input] -> [..., group=1, output] -> [..., output]
        output = self._vmm(input.unsqueeze(-2), quantization_mode=quantization_mode, adc_active_bits=adc_active_bits)
        output = output.squeeze(-2)
        if self._int_bias is not None:
            output = output + self._int_bias
        return output

    def latency__ns(self, input_shape: tuple[int, ...], *, adc_active_bits: int | None) -> float:
        if not input_shape or input_shape[-1] != self.tiler.matrix_input_num or any(size < 0 for size in input_shape):
            raise ValueError("latency__ns expects [..., input_num] with nonnegative extents")
        local__ns = self._vmm_local_latency__ns(adc_active_bits=adc_active_bits)
        global__ns = self._vmm_global_latency__ns()
        return local__ns + global__ns
