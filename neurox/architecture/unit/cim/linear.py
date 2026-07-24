"""LinearCimUnit — engine-backed ``F.linear`` replacement.

See also:
    docs/internals/architecture/unit/cim/linear.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.architecture.unit.linear import LinearUnit

from .base import CimUnit, EngineBackedCimUnit, EngineBackedCimUnitConfig, EngineBackedCimUnitPolicy


class LinearCimUnitConfig(EngineBackedCimUnitConfig):
    """Configuration for :class:`LinearCimUnit`; no fields beyond the inherited set."""

    def validate(self) -> None:
        super().validate()
        self.validate_geometry()

    def validate_geometry(self) -> None:
        """Require uniform row blocking."""
        macro = self.engine.cim_macro_config
        if macro.row_num % macro.active_row_num != 0:
            raise ValueError(
                f"require: row_num ({macro.row_num}) % active_row_num ({macro.active_row_num}) == 0 "
                "(the linear operator reads every row; uniform row-blocking)"
            )


class LinearCimUnitPolicy(EngineBackedCimUnitPolicy):
    """Composite policy for :class:`LinearCimUnit`; no fields beyond the inherited set."""


@CimUnit.register_key(LinearCimUnitConfig)
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
        self.engine.program(self._weight_to_matrix(weight))
        self._program_int_bias(bias, channels=self._w_logical_shape[-2])
