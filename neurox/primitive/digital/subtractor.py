"""Element-wise integer subtractor.

See also:
    docs/reference/primitive/digital/subtractor.md
"""

import torch
from torch import Tensor

from .base import DigitalBase, DigitalConfig, DigitalPolicy


class SubtractorConfig(DigitalConfig):
    """Immutable configuration for a Subtractor instance.

    Attributes:
        bit_width: Nominal output bit width (informational; no wrap is applied).
        energy_per_op__fJ: Dynamic energy consumed per output element.
        latency_per_op__ns: Combinational window of one subtract.
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


class Subtractor(DigitalBase[SubtractorConfig]):
    """Element-wise integer subtractor without saturation or wrapping.

    Args:
        config: Subtractor configuration.
        policy: Digital execution policy.
        inst_shape: Per-instance fabrication shape.
    """

    def __init__(
        self,
        *,
        config: SubtractorConfig,
        policy: DigitalPolicy,
        inst_shape: tuple[int, ...],
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    def subtract(self, a: Tensor, b: Tensor) -> Tensor:
        """Subtract ``b`` from ``a`` element-wise.

        Args:
            a: Minuend tensor.
            b: Subtrahend tensor (broadcast-compatible with ``a``).

        Returns:
            ``y = a - b``.
        """
        y = a - b
        if self._is_dynamic_energy_profile_active():
            # Subtractor has no caller anywhere in the execution path, so no
            # caller ever positions a space axis of this block's own inst_shape
            # inside y; inst_shape only sizes the area/leakage totals. A flat
            # per-op lump: the expanded constant holds no storage, and the
            # energy dtype is the constant's rather than the integer operand's.
            # Shape: [] -> [*y.shape]
            e_op__fJ = torch.full((), self.config.energy_per_op__fJ, dtype=torch.float32, device=y.device)
            self._record_dynamic_energy(e_op__fJ.expand(y.shape))
        return y
