"""Element-wise integer subtractor for bipolar-readout aggregation.

See also:
    docs/dev/modules/digital/README.md
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.mixin import FabricateMixin, ProfileMixin, ValidateMixin


@dataclass(frozen=True)
class SubtractorConfig(ValidateMixin):
    """Immutable configuration for a Subtractor instance.

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


class Subtractor(FabricateMixin, nn.Module, ProfileMixin):
    """Element-wise integer subtractor. No saturation or wrap."""

    def __init__(
        self,
        *,
        cfg: SubtractorConfig,
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

    def operate(self, a: Tensor, b: Tensor) -> Tensor:
        """Subtract ``b`` from ``a`` element-wise.

        Args:
            a: Positive sub-array ADC output (minuend).
            b: Negative sub-array ADC output (subtrahend).

        Returns:
            ``y = a - b``.
        """
        y = a - b
        dynamic_energy__fJ = torch.full_like(y, self.cfg.energy_per_op__fJ, dtype=torch.float32)
        self._log_dynamic(dynamic_energy__fJ, self.cfg.latency_per_op__ns)
        return y
