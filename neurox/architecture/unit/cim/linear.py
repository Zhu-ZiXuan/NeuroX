"""LinearCimUnit — engine-backed `F.linear` replacement.

See Also:
    docs/reference/architecture/unit/family.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.architecture.unit.linear import LinearUnit

from .base import CimUnit, EngineBackedCimUnit, EngineBackedCimUnitConfig, EngineBackedCimUnitPolicy


class LinearCimUnitConfig(EngineBackedCimUnitConfig):
    def validate(self) -> None:
        super().validate()


class LinearCimUnitPolicy(EngineBackedCimUnitPolicy):
    pass


_Config = LinearCimUnitConfig
_Policy = LinearCimUnitPolicy


@CimUnit.register_impl(config_type=_Config, policy_type=_Policy)
class LinearCimUnit(LinearUnit, EngineBackedCimUnit):
    """CIM-backed integer linear unit."""

    config: _Config
    policy: _Policy

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_macro: bool,
    ) -> None:
        if len(w_logical_shape) != 2:
            raise ValueError(f"w_logical_shape must be (N, K); got {w_logical_shape}")
        super().__init__(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_macro=ideal_macro,
        )

    def latency__ns(
        self,
        input_shape: tuple[int, ...],
        *,
        adc_active_bits: int | None,
    ) -> float:
        """Return the latency of the matmul this linear call lowers to.

        The unit introduces no time axis of its own; it restates the call in
        the engine's terms, which is the single plane every linear operator
        holds, matching the lowering the forward path performs. The shape says
        nothing the plane count needs, so it is not read.
        """
        del input_shape
        return self.engine.latency__ns(output_plane_num=1, adc_active_bits=adc_active_bits)

    @torch.no_grad()
    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        if weight.dtype.is_floating_point or weight.dtype.is_complex or weight.dtype == torch.bool:
            raise TypeError(f"CIM execution requires an integer weight tensor; got dtype {weight.dtype}")
        self.engine.program(self._weight_to_matrix(weight))
        self._program_int_bias(bias, channels=self._w_logical_shape[-2])
