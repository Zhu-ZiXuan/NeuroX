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

    def validate(self) -> None:
        super().validate()
        self.validate_geometry()

    def validate_geometry(self) -> None:
        """Require uniform row-blocking on the owned xbar.

        The linear operator reads every row of the weight matrix, so the
        engine's sub-phases must tile ``row_num`` into equal
        ``active_row_num`` blocks — every sub-phase then carries the same
        dot-product dynamic range (hence the same ADC calibration). A
        non-divisible geometry would leave the final block short, which is
        valid for an operator that tolerates unused rows (Conv2d) but not
        for linear.
        """
        macro = self.engine.cim_macro_config
        if macro.row_num % macro.active_row_num != 0:
            raise ValueError(
                f"require: row_num ({macro.row_num}) % active_row_num ({macro.active_row_num}) == 0 "
                "(the linear operator reads every row; uniform row-blocking)"
            )


@dataclass(frozen=True)
class LinearCimUnitPolicy(EngineBackedCimUnitPolicy):
    """Composite policy for :class:`LinearCimUnit`; no fields beyond the inherited set."""


@CimUnit.register_key(LinearCimUnitConfig)
class LinearCimUnit(LinearUnit, EngineBackedCimUnit[LinearCimUnitConfig, LinearCimUnitPolicy]):
    """CIM unit exposing the linear operator over the configured engine.

    ``linear`` (inherited from :class:`LinearUnit`) runs the inherited
    lowering template onto the engine-delegated ``_matmul`` and adds the
    programmed integer bias in the int64 accumulation domain.
    """

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
