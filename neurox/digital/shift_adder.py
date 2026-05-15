"""Shift-adder for multi-digit CiM partial-product recombination.

When a weight or activation is decomposed into ``N`` digits of a radix-``r``
number system (e.g. two 4-bit nibbles of a radix-16 value), each digit is
processed by a separate crossbar sub-array pass.  This module recombines those
partial results by computing the weighted positional sum

    y = sum_i(x[..., i] * scale^i)   for i in 0 .. N-1

and wrapping into the signed ``bit_width``-bit range via modular arithmetic.
An optional ``init_val`` accumulates a prior partial sum, supporting chained
multi-pass recombination.

PPA metrics are tracked per-instance for system-level estimation; physical
instance count is set externally by the parent macro at fabricate time via
``ProfiledModule._record_inst_count``.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.profiler import ProfiledModule


@dataclass(frozen=True)
class ShiftAdderConfig:
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


class ShiftAdder(nn.Module, ProfiledModule):
    """Weighted positional-sum unit for digit recombination.

    Multiplies each digit slice along ``dim`` by the corresponding power of
    ``scale`` (i.e. ``scale^0, scale^1, ...``), sums the scaled digits, and
    wraps the result into the signed ``bit_width``-bit range.  Models the
    shift-and-add tree that follows the per-digit ADC readout stage in a
    bit-serial or digit-serial CiM macro.
    """

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

    def fabricate(self) -> None:
        """No-op: ShiftAdder's instance count is set by the parent macro."""
        return None

    def operate(self, x: Tensor, scale: int, dim: int = -1, init_val: Tensor | None = None) -> Tensor:
        """Compute the radix-weighted digit sum and wrap to ``bit_width`` bits.

        Each index ``i`` along ``dim`` represents digit position ``i``, weighted
        by ``scale^i``.  The final result is optionally offset by ``init_val``
        before being returned (without an additional modular wrap on that offset).
        Dynamic energy and latency emit through the profiler side channel.

        Args:
            x: Integer digit tensor.  The size of ``dim`` equals the number of
                digit positions ``N``.
            scale: Radix of the digit representation (e.g. 2 for bit-serial,
                16 for nibble-serial).
            dim: Axis indexing the digit positions (default: ``-1``).
            init_val: Optional partial-sum tensor to add after the modular wrap,
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
