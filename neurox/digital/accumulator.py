"""Digital accumulator for CiM tile reduction.

Reduces ADC output codes along one tile axis (e.g., columns within a tile)
by summing all elements and wrapping the result into the signed ``bit_width``-bit
range via modular arithmetic: ``y = (sum(x) + 2^(bw-1)) % 2^bw - 2^(bw-1)``.

The modular wrap models finite-width adder carry truncation; overflow silently
aliases rather than saturating. PPA metrics (energy, latency, leakage, area)
are tracked per-instance for system-level estimation; physical instance count
is set externally by the parent macro at fabricate time via
``ProfiledModule._record_inst_count`` and dynamic energy / latency emit
through the profiler side channel from ``operate``.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.profiler import ProfiledModule


@dataclass(frozen=True)
class AccumulatorConfig:
    """Immutable configuration for an Accumulator instance.

    Attributes:
        bit_width: Signed output bit width; result is clamped to
            ``[-2^(bw-1), 2^(bw-1) - 1]`` via modular wrap.
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


class Accumulator(nn.Module, ProfiledModule):
    """Modular adder-tree that sums ADC codes along one tile axis.

    Models a hardware adder tree with a fixed output register of ``bit_width``
    bits.  Overflow wraps via two's-complement modular arithmetic, matching
    the behavior of a synthesized ripple-carry or carry-save adder tree with
    no saturation logic.
    """

    def __init__(self, config: AccumulatorConfig, *, name: str = "") -> None:
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

    def fabricate(self) -> None:
        """No-op: Accumulator has no shape-driven state.

        Instance count is supplied by the parent macro via
        :meth:`ProfiledModule._record_inst_count` after the macro's
        own fabricate step determines the replica count.
        """
        return None

    def operate(self, x: Tensor, dim: int) -> Tensor:
        """Sum ``x`` along ``dim`` and wrap into the signed ``bit_width`` range.

        Dynamic energy and latency emit through the profiler side
        channel.

        Args:
            x: Integer-valued input tensor of ADC output codes.
            dim: Axis along which to reduce (tile columns or rows).

        Returns:
            Modular-wrapped sum with ``dim`` reduced.
        """

        bw = self.config.bit_width
        half = 1 << (bw - 1)
        full = 1 << bw
        y = (x.sum(dim) + half) % full - half

        dynamic_energy__fJ = torch.full_like(y, self.config.energy_per_op__fJ, dtype=torch.float32)
        self._log_dynamic(dynamic_energy__fJ, self.config.latency_per_op__ns)
        return y
