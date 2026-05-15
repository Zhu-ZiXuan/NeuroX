"""Subtractor for bipolar weight readout in CiM crossbar macros.

In a sign-split (differential) crossbar encoding, a signed weight ``w`` is
decomposed into a positive sub-array ``w+`` and a negative sub-array ``w-``
such that ``w = w+ - w-``.  After the ADC stage digitises both sub-array
currents independently, this module computes the true signed MAC contribution
``y = a - b``, recovering the full bipolar value.

No modular wrap is applied; the result inherits the integer dtype of the
inputs.  PPA metrics are tracked per-instance for system-level estimation.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class SubtractorConfig:
    """Immutable configuration for a Subtractor instance.

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


class Subtractor(nn.Module):
    """Element-wise subtractor for sign-split crossbar readout.

    Computes ``y = a - b`` where ``a`` and ``b`` are the digitised outputs of
    the positive and negative sub-arrays respectively.  No saturation or
    modular wrap is applied; the caller is responsible for ensuring the result
    fits within the downstream bit width.
    """

    def __init__(self, config: SubtractorConfig) -> None:
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

    def fabricate(self) -> None:
        """Default no-op (Subtractor has no static fabrication state)."""
        return None

    def operate(self, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        """Subtract ``b`` from ``a`` element-wise.

        Args:
            a: Positive sub-array ADC output (minuend).
            b: Negative sub-array ADC output (subtrahend).

        Returns:
            A tuple ``(y, dynamic_energy__fJ)`` where ``y = a - b`` and
            ``dynamic_energy__fJ`` is a per-output-element energy tensor
            (``shape == y.shape``).  Callers ``.sum()`` at the
            aggregation boundary for a scalar total.
        """
        y = a - b
        dynamic_energy__fJ = torch.full_like(y, self.config.energy_per_op__fJ, dtype=torch.float32)
        return y, dynamic_energy__fJ
