"""LinearCimUnit — engine-backed ``F.linear`` replacement.

See also:
    docs/internals/architecture/unit/cim/linear.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.architecture.unit.linear import LinearUnit

from .base import CimUnit, EngineBackedCimUnit, EngineBackedCimUnitConfig, EngineBackedCimUnitPolicy


@dataclass(frozen=True)
class LinearCimUnitConfig(EngineBackedCimUnitConfig):
    """Configuration for :class:`LinearCimUnit`; no fields beyond the inherited set."""


@dataclass(frozen=True)
class LinearCimUnitPolicy(EngineBackedCimUnitPolicy):
    """Composite policy for :class:`LinearCimUnit`; no fields beyond the inherited set."""


@CimUnit.register_key(LinearCimUnitConfig)
class LinearCimUnit(LinearUnit, EngineBackedCimUnit):
    """CIM unit exposing the linear operator over the configured engine.

    ``linear`` (inherited from :class:`LinearUnit`) runs the inherited
    lowering template onto the engine-delegated ``_matmul`` and adds the
    programmed integer bias in the int64 accumulation domain.
    """

    config: LinearCimUnitConfig
    policy: LinearCimUnitPolicy

    def __init__(
        self,
        *,
        config: LinearCimUnitConfig,
        policy: LinearCimUnitPolicy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_xbar: bool,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_xbar=ideal_xbar,
        )
        self._init_int_bias_slot()

    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        """Write the engine's static weight state and the optional integer bias.

        Args:
            weight: Integer weight tensor whose shape matches the unit's
                ``w_logical_shape``.
            bias: Optional integer bias tensor of shape ``(N,)``; ``None``
                clears any programmed bias.
        """
        self.engine.program(self._weight_to_matrix(weight))
        self._program_int_bias(bias, channels=self._w_logical_shape[-2])
