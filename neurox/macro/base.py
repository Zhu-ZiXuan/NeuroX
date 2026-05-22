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
    def w_value_range(self) -> tuple[int, int]: ...

    @property
    def x_value_range(self) -> tuple[int, int]: ...

    @property
    def output_rescale_factor(self) -> float: ...

    def fabricate(self) -> None:
        """Resample static manufacturing variation across self and descendants."""
        ...

    def program(self, weight: Tensor) -> None:
        """Write the macro's static weight state.

        Args:
            weight: Integer weight tensor. Shape: ``[..., N, K]`` matching the
                ``w_logical_shape`` bound at construction.
        """
        ...

    def matmul(
        self,
        input: Tensor,
        bias: Tensor | None,
        rescale_multiplier: Tensor,
        rescale_rshift: Tensor,
        output_zero_point: Tensor | None,
    ) -> Tensor:
        """Execute one integer matrix multiply against the programmed weight.

        Args:
            input: Integer activation tensor. Shape: ``[..., M, K]``.
            bias: Optional integer bias tensor. Shape: ``[..., N]``.
            rescale_multiplier: Per-output fixed-point multiplier.
            rescale_rshift: Per-output right-shift amount.
            output_zero_point: Optional output zero point.

        Returns:
            Integer output tensor. Shape: ``[..., M, N]``.
        """
        ...
