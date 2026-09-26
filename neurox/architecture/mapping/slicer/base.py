"""Slicer ABC for value-domain decomposition."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from neurox.primitive.digital import RadixAccumulator, RadixSummator


class Slicer(ABC):
    """Decompose integers into least-significant-first positional slices.

    Implement `slice` and consistent range, count, and radix properties. Preserve
    other input axes; recovery expects exactly `slice_num` entries on its selected
    axis. A single slice passes through without arithmetic or circuit costs.

    An optional recovery circuit remains owned and placed by its containing
    module. Without one, recovery uses tensor arithmetic without register wrap
    or circuit costs.

    Args:
        recovery_circuit: Externally owned recovery circuit, or None for tensor
            arithmetic.
    """

    def __init__(
        self,
        *,
        recovery_circuit: RadixSummator | RadixAccumulator | None = None,
    ) -> None:
        self.recovery_circuit = recovery_circuit

    # === Public API ===

    def recover(self, values: Tensor, *, dim: int, enable: Tensor | None = None) -> Tensor:
        """Recover multiple slices through a circuit or tensor arithmetic, or pass one slice through.

        Args:
            values: Slice values or linear-operation results for each slice.
                Shape: `[..., slice, ...]`.
            dim: Axis indexing slices in least-significant-first order.
            enable: Optional operand enables broadcastable to `values`.
                Ignored for a single slice.

        Returns:
            Weighted sum with the slice axis removed, preserving the dtype.
        """
        if self.slice_num == 1:
            return values.squeeze(dim)
        # Circuit topology is fixed at construction, so dispatch resolves during tracing.
        if isinstance(self.recovery_circuit, RadixAccumulator):
            return self.recovery_circuit.radix_accumulate(values, dim=dim, radix=self.slice_radix, enable=enable)
        if isinstance(self.recovery_circuit, RadixSummator):
            return self.recovery_circuit.radix_sum(values, dim=dim, radix=self.slice_radix, enable=enable)
        if enable is not None:
            values = values.where(enable, 0)
        # Static Python coefficients avoid constructing and copying a weight tensor.
        parts = values.unbind(dim=dim)
        weighted = [part * scale for part, scale in zip(parts, self.place_values, strict=True)]
        return torch.stack(weighted, dim=0).sum(dim=0, dtype=values.dtype)

    # === For subclass to implement or override ===

    @property
    @abstractmethod
    def place_values(self) -> tuple[int, ...]:
        """Positive positional coefficients, one per slice, in emitted order."""
        raise NotImplementedError

    @property
    @abstractmethod
    def has_signed_slices(self) -> bool:
        """Whether slice values can be negative."""
        raise NotImplementedError

    @property
    @abstractmethod
    def value_range(self) -> tuple[int, int]:
        """Inclusive algorithm-side integer range this slicer can encode.

        `slice` neither clamps nor rejects values outside this range.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def slice_num(self) -> int:
        """Fixed number of positional slices; encoding does not append extra slices."""
        raise NotImplementedError

    @property
    @abstractmethod
    def slice_radix(self) -> int:
        """Positional radix between adjacent slices."""
        raise NotImplementedError

    @abstractmethod
    def slice(self, x: Tensor, *, dim: int = -1) -> Tensor:
        """Decompose `x` into positional slice values.

        Args:
            x: Integer tensor to decompose.
            dim: Axis at which the slice dimension is inserted.

        Returns:
            Positional slice values with a new axis at `dim`.
            Shape: `[..., slice, ...]`.
        """
        raise NotImplementedError
