"""Element-wise integer adder with energy accounting."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.mixin import FabricateMixin, ProfileMixin, ValidateMixin


@dataclass(frozen=True)
class AdderConfig(ValidateMixin):
    """Immutable configuration for an Adder instance.

    Attributes:
        bit_width: Nominal output bit width (informational; no wrap is applied).
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


class Adder(FabricateMixin, nn.Module, ProfileMixin):
    """Element-wise integer adder. No saturation or wrap."""

    def __init__(
        self,
        *,
        config: AdderConfig,
        name: str,
        inst_shape: tuple[int, ...],
    ) -> None:
        nn.Module.__init__(self)
        ProfileMixin.__init__(self, name)
        self.config = config
        self._inst_shape = inst_shape
        self._log_static()

    @property
    def area_per_inst__um2(self) -> float:
        """Silicon area per instance [um^2]."""
        return self.config.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        """Static leakage per instance [uW]."""
        return self.config.leakage_per_inst__uW

    @property
    def latency_per_op__ns(self) -> float:
        """Latency per op [ns]."""
        return self.config.latency_per_op__ns

    def operate(self, a: Tensor, b: Tensor) -> Tensor:
        """Add ``a`` and ``b`` element-wise.

        Args:
            a: Left operand.
            b: Right operand, broadcastable to ``a``.

        Returns:
            ``y = a + b``.
        """
        y = a + b
        dynamic_energy__fJ = torch.full_like(y, self.config.energy_per_op__fJ, dtype=torch.float32)
        self._log_dynamic(dynamic_energy__fJ, self.config.latency_per_op__ns)
        return y
