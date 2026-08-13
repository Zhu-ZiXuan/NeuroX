"""LinearCimUnit — engine-backed `F.linear` replacement.

See also:
    docs/internals/architecture/unit/cim/linear.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.architecture.unit.linear import LinearUnit

from .base import CimUnit, EngineBackedCimUnit, EngineBackedCimUnitConfig, EngineBackedCimUnitPolicy


class LinearCimUnitConfig(EngineBackedCimUnitConfig):
    """Configuration for `LinearCimUnit`."""

    def validate(self) -> None:
        super().validate()


class LinearCimUnitPolicy(EngineBackedCimUnitPolicy):
    """Composite policy for `LinearCimUnit`."""


@CimUnit.register_neurox_module(config_type=LinearCimUnitConfig, policy_type=LinearCimUnitPolicy)
class LinearCimUnit(LinearUnit, EngineBackedCimUnit[LinearCimUnitConfig, LinearCimUnitPolicy]):
    """CIM-backed integer linear unit."""

    def __init__(
        self,
        *,
        config: LinearCimUnitConfig,
        policy: LinearCimUnitPolicy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_macro: bool,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_macro=ideal_macro,
        )

    def latency__ns(self, input_shape: tuple[int, ...], *, adc_bits: int | None) -> float:
        """Time the matmul this linear call lowers to.

        The unit introduces no time axis of its own; it restates the call in
        the engine's terms, which is the single plane every linear operator
        holds, matching the lowering the forward path performs. The shape says
        nothing the plane count needs, so it is not read.
        """
        del input_shape
        return self.engine.latency__ns(output_plane_num=1, adc_bits=adc_bits)

    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        if weight.dtype.is_floating_point or weight.dtype.is_complex or weight.dtype == torch.bool:
            raise TypeError(f"CIM execution requires an integer weight tensor; got dtype {weight.dtype}")
        self.engine.program(self._weight_to_matrix(weight))
        self._program_int_bias(bias, channels=self._w_logical_shape[-2])
