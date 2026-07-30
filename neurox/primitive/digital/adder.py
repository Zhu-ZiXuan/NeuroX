"""Element-wise integer adder with energy accounting.

See also:
    docs/reference/primitive/digital/adder.md
"""

import torch
from torch import Tensor

from .base import DigitalBase, DigitalConfig, DigitalPolicy


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

    def validate(self) -> None:
        super().validate()

        # --- Arithmetic ---

        self._require_pos(self.bit_width, "bit_width")

        # --- PPA ---

        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_non_neg(self.latency_per_op__ns, "latency_per_op__ns")


class Adder(DigitalBase[AdderConfig]):
    """Element-wise integer adder without saturation or wrapping.

    Args:
        config: Adder configuration.
        policy: Digital execution policy.
        inst_shape: Per-instance fabrication shape.
    """

    def __init__(
        self,
        *,
        config: AdderConfig,
        policy: DigitalPolicy,
        inst_shape: tuple[int, ...],
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._register_latency_buffer(config.latency_per_op__ns)

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    def add(self, a: Tensor, b: Tensor) -> Tensor:
        """Add ``a`` and ``b`` element-wise.

        Args:
            a: Left operand.
            b: Right operand, broadcastable to ``a``.

        Returns:
            ``y = a + b``.
        """
        y = a + b
        serial_round_count = self._count_serial_rounds(y.numel())
        latency__ns = self._latency_per_op__ns * serial_round_count
        if self._is_dynamic_energy_profile_active():
            self._record_dynamic_energy(torch.full_like(y, self.config.energy_per_op__fJ, dtype=torch.float32))
        self._record_latency(latency__ns)
        return y
