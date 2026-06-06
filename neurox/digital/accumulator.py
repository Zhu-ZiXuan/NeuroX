"""Digital modular-arithmetic accumulator over an integer-tensor axis.

See also:
    docs/dev/modules/digital/README.md
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.mixin import FabricateMixin, ProfileMixin, ValidateMixin


@dataclass(frozen=True)
class AccumulatorConfig(ValidateMixin):
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

    energy_per_op__fJ: float

    latency_per_op__ns: float
    leakage_per_inst__uW: float
    area_per_inst__um2: float

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


class Accumulator(FabricateMixin, nn.Module, ProfileMixin):
    """Modular adder-tree that sums an integer tensor along one axis.

    Models a hardware adder tree with a fixed output register of ``bit_width``
    bits.  Overflow wraps via two's-complement modular arithmetic, matching
    the behavior of a synthesized ripple-carry or carry-save adder tree with
    no saturation logic.
    """

    def __init__(
        self,
        *,
        cfg: AccumulatorConfig,
        name: str,
        inst_shape: tuple[int, ...],
    ) -> None:
        nn.Module.__init__(self)
        ProfileMixin.__init__(self, name)
        self.cfg = cfg
        self._inst_shape = inst_shape
        self._log_static()

    @property
    def area_per_inst__um2(self) -> float:
        """Silicon area per instance [um^2]."""
        return self.cfg.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        """Static leakage per instance [uW]."""
        return self.cfg.leakage_per_inst__uW

    @property
    def latency_per_op__ns(self) -> float:
        """Latency per op [ns]."""
        return self.cfg.latency_per_op__ns

    def operate(self, x: Tensor, dim: int) -> Tensor:
        """Sum ``x`` along ``dim`` and wrap into the signed ``bit_width`` range.

        Dynamic energy and latency emit through the profiler side
        channel.

        Args:
            x: Integer-valued input tensor.
            dim: Axis along which to reduce.

        Returns:
            Modular-wrapped sum with ``dim`` reduced.
        """

        bw = self.cfg.bit_width
        half = 1 << (bw - 1)
        full = 1 << bw
        y = (x.sum(dim) + half) % full - half

        dynamic_energy__fJ = torch.full_like(y, self.cfg.energy_per_op__fJ, dtype=torch.float32)
        self._log_dynamic(dynamic_energy__fJ, self.cfg.latency_per_op__ns)
        return y
