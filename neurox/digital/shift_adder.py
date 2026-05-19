"""Shift-adder for multi-digit partial-product recombination.

See also:
    docs/dev/modules/digital/README.md
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.validate import ValidateMixin
from neurox.profiler import ProfiledModule


@dataclass(frozen=True)
class ShiftAdderConfig(ValidateMixin):
    """Immutable configuration for a ShiftAdder instance.

    Attributes:
        bit_width: Signed output bit width; result wraps modulo ``2^bit_width``
            into ``[-2^(bw-1), 2^(bw-1) - 1]``.
        energy_per_op__fJ: Dynamic energy consumed per output element (fJ).
        latency_per_op__ns: Critical-path latency per operation (ns).
        leakage_per_inst__uW: Static leakage power per instance (uW).
        area_per_inst__um2: Silicon area per instance (um^2).
    """

    bit_width: int

    energy_per_op__fJ: float = 0.0

    latency_per_op__ns: float = 0.0
    leakage_per_inst__uW: float = 0.0
    area_per_inst__um2: float = 0.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_arithmetic()
        self.validate_ppa()

    def validate_arithmetic(self) -> None:
        self._require_pos(self.bit_width, "bit_width")

    def validate_ppa(self) -> None:
        self._require_nonneg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_nonneg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_nonneg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_nonneg(self.latency_per_op__ns, "latency_per_op__ns")


class ShiftAdder(nn.Module, ProfiledModule):
    """Weighted positional-sum unit for digit recombination."""

    def __init__(self, config: ShiftAdderConfig, *, name: str = "") -> None:
        nn.Module.__init__(self)
        ProfiledModule.__init__(self, name)
        self.config = config

    @property
    def area_per_inst__um2(self) -> float:
        """Area per instance in um2."""
        return self.config.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        """Leakage per instance in uW."""
        return self.config.leakage_per_inst__uW

    @property
    def latency_per_op__ns(self) -> float:
        """Latency per op in ns."""
        return self.config.latency_per_op__ns

    def fabricate(self, shape: tuple[int, ...]) -> None:
        """Sample static per-instance state over ``shape`` (re-callable).

        Args:
            shape: Per-instance fabrication shape.
        """
        self._record_inst_count(shape)

    def operate(self, x: Tensor, scale: int, dim: int = -1, init_val: Tensor | None = None) -> Tensor:
        """Compute the radix-weighted digit sum and wrap to ``bit_width`` bits.

        Args:
            x: Integer digit tensor; size of ``dim`` is the digit count.
            scale: Radix of the digit representation.
            dim: Axis indexing the digit positions.
            init_val: Optional partial-sum tensor added after the modular wrap,
                broadcastable to the output shape.

        Returns:
            Recombined sum with ``dim`` reduced.
        """

        bw = self.config.bit_width
        half = 1 << (bw - 1)
        full = 1 << bw
        scales = torch.tensor([scale**i for i in range(x.size(dim))], device=x.device, dtype=x.dtype)

        shape = [1] * x.ndim
        shape[dim] = x.size(dim)
        y = ((x * scales.view(*shape)).sum(dim=dim) + half) % full - half

        if init_val is not None:
            y = y + init_val

        dynamic_energy__fJ = torch.full_like(y, self.config.energy_per_op__fJ, dtype=torch.float32)
        self._log_dynamic(dynamic_energy__fJ, self.config.latency_per_op__ns)
        return y
