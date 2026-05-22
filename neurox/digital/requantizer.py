"""Fixed-point multiply-shift requantizer.

See also:
    docs/dev/modules/digital/README.md
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.mixin import FabricateMixin, ProfileMixin, ValidateMixin
from neurox.common.quant import stochastic_floor_div


@dataclass(frozen=True)
class RequantizerConfig(ValidateMixin):
    """Immutable configuration for a Requantizer instance.

    Attributes:
        bit_width: Nominal output bit width (informational; the caller clips
            to this range after requantization if needed).
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


class Requantizer(FabricateMixin, nn.Module, ProfileMixin):
    """Multiply-shift requantizer for integer-MAC rescaling.

    Computes ``y = (x · multiplier) >> rshift [+ output_zero_point]``.
    Stochastic rounding adds a uniform ``[0, 1 << rshift)`` jitter before
    the shift whenever ``self.training`` is ``True``.

    Args:
        cfg: Immutable cost / bit-width configuration.
        name: Hierarchical profiler name.
        inst_shape: Per-instance fabrication shape.
    """

    def __init__(
        self,
        *,
        cfg: RequantizerConfig,
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
        """Area per instance in um2."""
        return self.cfg.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        """Leakage per instance in uW."""
        return self.cfg.leakage_per_inst__uW

    @property
    def latency_per_op__ns(self) -> float:
        """Latency per op in ns."""
        return self.cfg.latency_per_op__ns

    def operate(self, x: Tensor, multiplier: Tensor, rshift: Tensor, output_zero_point: Tensor | None) -> Tensor:
        """Compute ``y = (x · multiplier) >> rshift [+ output_zero_point]``.

        Args:
            x: Accumulated integer MAC result tensor.
            multiplier: Int32 fixed-point scale factor, broadcastable to ``x``.
            rshift: Non-negative right-shift amount, broadcastable to ``x``.
            output_zero_point: Optional zero-point offset added after the shift.

        Returns:
            Rescaled integer tensor.
        """
        y = stochastic_floor_div(
            x * multiplier,
            rshift,
            training=self.training,
        )
        if output_zero_point is not None:
            y = y + output_zero_point

        dynamic_energy__fJ = torch.full_like(y, self.cfg.energy_per_op__fJ, dtype=torch.float32)
        self._log_dynamic(dynamic_energy__fJ, self.cfg.latency_per_op__ns)
        return y
