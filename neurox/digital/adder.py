"""Modular adder for CiM digital datapath stages.

Provides a simple element-wise addition with PPA accounting.  The result is
not modular-wrapped here; ``bit_width`` is stored for downstream reference
and system-level energy reporting.  Use ``Accumulator`` when a reduction sum
with explicit modular wrapping is needed.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class AdderConfig:
    """Immutable configuration for an Adder instance.

    Attributes:
        bit_width: Nominal output bit width (informational; no wrap is applied).
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


class Adder(nn.Module):
    """Element-wise integer adder with energy accounting.

    Models a hardware adder stage in the digital datapath (e.g., combining
    partial sums from multiple tiles).  No saturation or modular wrap is
    applied; the result inherits the dtype of the inputs.
    """

    def __init__(self, config: AdderConfig) -> None:
        super().__init__()
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

    def operate(self, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        """Add ``a`` and ``b`` element-wise.

        Args:
            a: Left operand.
            b: Right operand, broadcastable to ``a``.

        Returns:
            A tuple ``(y, dynamic_energy__fJ)`` where ``y = a + b`` and
            ``dynamic_energy__fJ`` is a per-output-element energy tensor
            (``shape == y.shape``) holding the dynamic energy to produce
            each output cell.  Under the current constant-energy-per-op
            model every element is ``config.energy_per_op__fJ``; future
            data-dependent energy models can replace that without
            changing this method's caller contract.  Sum via
            ``.sum()`` at the aggregation boundary to get a scalar
            total.
        """
        y = a + b
        dynamic_energy__fJ = torch.full_like(y, self.config.energy_per_op__fJ, dtype=torch.float32)
        return y, dynamic_energy__fJ
