"""Element-wise integer adder with energy accounting.

See also:
    docs/reference/primitive/digital/adder.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from .base import DigitalBase, DigitalConfig, DigitalPolicy


@dataclass(frozen=True)
class AdderConfig(DigitalConfig):
    """Immutable configuration for an Adder instance.

    Attributes:
        bit_width: Nominal output bit width (informational; no wrap is applied).
        energy_per_op__fJ: Dynamic energy consumed per output element.
        latency_per_op__ns: Per-element latency; multiplied by the
            runtime serial-op count at logging time.
    """

    bit_width: int

    energy_per_op__fJ: float
    latency_per_op__ns: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_arithmetic()
        self.validate_ppa()

    def validate_arithmetic(self) -> None:
        self._require_pos(self.bit_width, "bit_width")

    def validate_ppa(self) -> None:
        super().validate_ppa()
        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_non_neg(self.latency_per_op__ns, "latency_per_op__ns")


class Adder(DigitalBase[AdderConfig]):
    """Element-wise integer adder. No saturation or wrap."""

    def __init__(
        self,
        *,
        config: AdderConfig,
        policy: DigitalPolicy,
        inst_shape: tuple[int, ...],
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW

    def operate(self, a: Tensor, b: Tensor) -> Tensor:
        """Add ``a`` and ``b`` element-wise.

        Args:
            a: Left operand.
            b: Right operand, broadcastable to ``a``.

        Returns:
            ``y = a + b``.
        """
        y = a + b
        serial_op_count = -(-y.numel() // max(self.inst_count, 1))  # ceil(numel / inst); empty -> 0
        dynamic_energy__fJ = torch.full_like(y, self.config.energy_per_op__fJ, dtype=torch.float32)
        latency__ns = torch.tensor(
            self.config.latency_per_op__ns * serial_op_count,
            device=y.device,
            dtype=dynamic_energy__fJ.dtype,
        )
        self._log_dynamic_energy(dynamic_energy__fJ)
        self._log_latency(latency__ns)
        return y
