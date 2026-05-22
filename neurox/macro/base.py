"""Cross-macro Protocol shared by every concrete macro implementation.

See also:
    docs/dev/modules/macro/README.md
"""

from __future__ import annotations

from typing import Protocol

from torch import Tensor


class NeuroxMacroQuantMatMul(Protocol):
    """Structural contract every macro impl satisfies.

    Attributes:
        w_value_range: Inclusive integer weight range accepted by the macro.
        x_value_range: Inclusive integer activation range accepted by the macro.
        output_rescale_factor: Ratio between the macro output scale and the
            ideal integer partial-product scale.

    Methods:
        fabricate: Resample static manufacturing variation across the macro
            tree. No arguments.
        program: Write the macro's static weight state from one logical
            integer weight tensor.
        matmul: Execute one integer matrix multiply against the programmed
            weight state.
    """

    @property
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive integer weight range accepted by the macro."""
        ...

    @property
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive integer activation range accepted by the macro."""
        ...

    @property
    def output_rescale_factor(self) -> float:
        """Ratio of the ideal partial-product max to the actual tile output max."""
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

    def matmul(self, input: Tensor) -> Tensor:
        """Execute one integer matrix multiply against the programmed weight state.

        Matches ``torch.matmul`` semantics (pure matmul, no bias). Bias add
        and requantize live in the operator layer.

        Args:
            input: Integer activation tensor. Shape: ``[..., M, K]``.

        Returns:
            Integer pre-requantize output tensor. Shape: ``[..., M, N]``.
        """
        ...
