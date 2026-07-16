"""Cross-macro Protocol shared by every concrete macro implementation.

See also:
    docs/internals/architecture/unit/matmul.md
"""

from __future__ import annotations

from typing import Protocol

from torch import Tensor

from neurox.primitive.analog.adc_common import AdcOperationPoint


class QuantMatMul(Protocol):
    """Structural contract every macro impl satisfies."""

    @property
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive integer weight range accepted by the macro."""
        ...

    @property
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive integer activation range accepted by the macro."""
        ...

    @property
    def adc_mode_num(self) -> int:
        """Number of ADC operating points the macro supports."""
        ...

    @property
    def adc_max_bits(self) -> int:
        """Maximum ``adc_bits`` value the macro supports."""
        ...

    def adc_rescale_factor(self, adc_operation_point: AdcOperationPoint) -> float:
        """Recovery-side multiplier for ``adc_operation_point``: ``M_ideal ≈ code · rescale_factor``."""
        ...

    def fabricate(self) -> None:
        """Re-sample static manufacturing variation across self and descendants."""
        ...

    def program(self, weight: Tensor) -> None:
        """Write the macro's static weight state from one logical weight tensor.

        Args:
            weight: Integer weight tensor whose shape matches
                ``self._w_logical_shape``.
        """
        ...

    def matmul(self, input: Tensor, *, adc_operation_point: AdcOperationPoint) -> Tensor:
        """Execute one integer matrix multiply against the programmed weight state.

        Matches ``torch.matmul`` semantics (pure matmul, no bias). Bias add
        and requantize live in the operator layer.

        Args:
            input: Integer activation tensor. Shape: ``[..., M, K]``.
            adc_operation_point: Runtime ADC operating point.

        Returns:
            Integer pre-requantize output tensor. Shape: ``[..., M, N]``.
        """
        ...
