"""UnitBase — root operator ABC: template skeleton over the matmul-shaped lowering.

See also:
    docs/internals/architecture/unit/base.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


def _validate_int_bias(bias: Tensor, *, channels: int) -> Tensor:
    """Validate an integer per-channel bias vector and return it as int64.

    Args:
        bias: Integer bias tensor of shape ``(channels,)``.
        channels: Expected number of output channels.

    Returns:
        ``bias`` cast to ``torch.int64`` (the accumulation domain).

    Raises:
        ValueError: ``bias`` has a non-integer dtype or a shape other than
            ``(channels,)``.
    """
    if bias.dtype.is_floating_point or bias.dtype.is_complex or bias.dtype == torch.bool:
        raise ValueError(f"require: integer bias dtype; got {bias.dtype}")
    if tuple(bias.shape) != (channels,):
        raise ValueError(f"require: bias.shape ({tuple(bias.shape)}) == ({channels},)")
    return bias.to(torch.int64)


class UnitBase(ABC):
    """Root operator ABC: template skeleton over the matmul-shaped lowering.

    Unit operators are exact-integer replicas of ``F.linear`` / ``F.conv2d``.
    There is no public matmul operator — matmul consumers are linear with
    ``bias=None``; ``engine.matmul`` remains the internal primitive name.
    A unit undoes every shape change it introduces itself, so callers see
    pure functional-operator semantics; caller-owned leading dims (batch,
    any time axis) ride through untouched.

    Bias preloads the final full-scale accumulation stage, after the last
    shift-add; it costs zero additional cycles and zero dynamic energy, and
    its static area/leakage belongs to the unit-level PPA fields.

    Axis-stack discipline (LIFO): each layer introduces its serial axis
    immediately left of the inst-aligned block and undoes exactly its own
    axis.

    Concrete hosts must be ``nn.Module`` instances and call
    :meth:`_init_int_bias_slot` in their constructor (directly or through
    an operator-specific init helper).
    """

    int_bias: Tensor | None

    # --- value-range contract ---

    @property
    @abstractmethod
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive integer weight range accepted by the unit."""
        raise NotImplementedError

    @property
    @abstractmethod
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive integer activation range accepted by the unit."""
        raise NotImplementedError

    # --- ADC operating-point surface ---

    @property
    @abstractmethod
    def adc_mode_num(self) -> int:
        """Number of supported ADC operating points; valid ``adc_mode`` values are ``[0, adc_mode_num)``."""
        raise NotImplementedError

    @property
    @abstractmethod
    def adc_max_bits(self) -> int:
        """Maximum supported ``adc_bits`` value."""
        raise NotImplementedError

    @abstractmethod
    def adc_rescale_factor(self, *, adc_mode: int, adc_bits: int) -> float:
        """Rescale factor for ``(adc_mode, adc_bits)``; raises ``KeyError`` if uncalibrated.

        Degenerate substrates accept the operating point for API uniformity
        and return ``1.0``.
        """
        raise NotImplementedError

    # --- lifecycle ---

    @abstractmethod
    def fabricate(self) -> None:
        """Re-sample static manufacturing variation across self and descendants."""
        raise NotImplementedError

    # --- protected matmul-shaped lowering primitive (substrate seam) ---

    @abstractmethod
    def _matmul(self, input: Tensor, *, adc_mode: int, adc_bits: int) -> Tensor:
        """Integer matmul against the programmed state (substrate seam).

        Args:
            input: Integer activation planes. Shape: ``[..., M, K]``.
            adc_mode: Runtime ADC operating-point index.
            adc_bits: Runtime ADC resolution.

        Returns:
            Integer pre-requantize output tensor. Shape: ``[..., M, N]``;
            leading order preserved.
        """
        raise NotImplementedError

    # --- template hook seams (identity defaults) ---

    def _weight_to_matrix(self, weight: Tensor) -> Tensor:
        """Seam 1 (program time): logical operator weight -> ``(N, K)`` matrix."""
        return weight

    def _activation_to_planes(self, input: Tensor) -> Tensor:
        """Seam 2 (call time): operator activation -> matmul-shaped planes."""
        return input

    def _undo_aggregation(self, output: Tensor) -> Tensor:
        """Seam 3 (call time): undo exactly the axes seam 2 introduced."""
        return output

    def _lower_matmul(self, input: Tensor, *, adc_mode: int, adc_bits: int) -> Tensor:
        """Template driver: seam 2 -> :meth:`_matmul` -> seam 3."""
        planes = self._activation_to_planes(input)
        y = self._matmul(planes, adc_mode=adc_mode, adc_bits=adc_bits)
        return self._undo_aggregation(y)

    # --- integer-bias slot for concrete hosts ---

    def _init_int_bias_slot(self) -> None:
        """Register the ``int_bias`` buffer slot; ``None`` until a bias is programmed.

        A ``None`` buffer is excluded from ``named_buffers()``.
        """
        assert isinstance(self, nn.Module)
        self.register_buffer("int_bias", None, persistent=False)

    def _program_int_bias(self, bias: Tensor | None, *, channels: int) -> None:
        """Store the validated int64 bias, or clear the slot with ``None``."""
        self.int_bias = None if bias is None else _validate_int_bias(bias, channels=channels)
